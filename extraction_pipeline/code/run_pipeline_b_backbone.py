#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from case_store import CASE_STORE_DIR, UNSTRUCTURED_CASES, load_cases_by_itemid

EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = EXTRACTION_ROOT.parent
INPUT_JSONL = EXTRACTION_ROOT / "outputs" / "case_features_labels.jsonl"
SPLITS_DIR = DATASET_ROOT / "splits"
SCHEMA_PATH = EXTRACTION_ROOT / "schemas" / "pipeline_b_backbone.schema.json"
RUNS_ROOT = EXTRACTION_ROOT / "runs" / "pipeline_b_backbone"

DEFAULT_SPLITS = ["set1_primary", "set2_postcut_ood", "set3_challenging"]
WRITE_LOCK = threading.Lock()

APPNO_LINE_RE = re.compile(r"(?m)^\s*\|?\s*(\d{1,6}/\d{2})\s*$")
NAME_YEAR_RE = re.compile(r"(?m)^\s*([^\n|]{3,}?)\s*\n\s*((?:19|20)\d{2})\s*$")
TITLE_NAME_RE = re.compile(r"\b(Mr|Ms|Mrs|Miss)\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+)*)")
SINGLE_NATIONAL_RE = re.compile(
    r"by\s+an?\s+([A-Za-z-]+)\s+national,\s+(Mr|Ms|Mrs|Miss)\s+([A-Z][^,(]+)",
    re.IGNORECASE,
)
PLURAL_NATIONAL_RE = re.compile(
    r"by\s+(?:\w+\s+)?([A-Za-z-]+)\s+nationals?\b",
    re.IGNORECASE,
)
EXPLICIT_APPLICANT_COUNT_RE = re.compile(r"\((\d+)\s+applicants?\)", re.IGNORECASE)
INDIRECT_VICTIM_RE = re.compile(
    r"\b(widow|widower|mother|father|son|daughter|parents|heir|heirs|estate of|next of kin|relative of the deceased)\b",
    re.IGNORECASE,
)

# Function guide (what each def is responsible for)
# - parse_args: Parse CLI options for deterministic B-backbone execution.
# - load_json: Load JSON content from disk.
# - load_jsonl_by_itemid: Build itemid-indexed row map from JSONL input.
# - load_split_ids: Resolve case ids for a split from JSON/CSV split files.
# - dedupe_preserve_order: De-duplicate while keeping stable order.
# - make_run_dir: Create run output directory tree.
# - _clean_text: Normalize text payloads (None-safe trim/cleanup).
# - _normalize_name: Normalize applicant name tokens for matching/dedup.
# - _title_case_name: Rebuild a human-readable applicant name in title case.
# - _judgment_year: Extract judgment year used for age bucketing.
# - _age_group_from_birth_year: Compute coarse age-group label at judgment time.
# - _extract_name_year_pairs: Parse appendix blocks into (name, birth_year) pairs.
# - _extract_title_map: Infer title/honorific map for parsed names.
# - _docname_names: Parse potential person names from docname/case title.
# - _extract_num_applicants: Determine applicant count using explicit and fallback signals.
# - _extract_uniform_nationality: Detect shared nationality phrase from procedure text.
# - _appendix_text: Return normalized appendix text used by applicant parsers.
# - _build_applicants: Construct applicant records deterministically (labels, traits, demographics).
# - validate_result: Validate backbone output against schema contract.
# - load_existing_results: Load already completed itemids for resume mode.
# - write_jsonl_line: Append a thread-safe JSONL output line.
# - write_case_file: Persist per-case backbone extraction JSON.
# - run_one_case: Run deterministic applicant extraction for a single case.
# - write_per_split_results: Emit split-scoped JSONL result files.
# - main: CLI entrypoint orchestrating deterministic backbone runs.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic applicant-first Pipeline B backbone extraction.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--itemids", nargs="+", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("EXTRACTION_CONCURRENCY", "8")))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_by_itemid(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["itemid"]] = row
    return rows


def load_split_ids(split_name: str) -> list[str]:
    json_path = SPLITS_DIR / f"{split_name}_ids.json"
    if json_path.exists():
        data = load_json(json_path)
        if isinstance(data, list):
            return [str(x) for x in data]
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find split definition for {split_name}")
    with csv_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        itemid_idx = header.index("itemid")
        return [line.rstrip("\n").split(",")[itemid_idx] for line in handle if line.strip()]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def make_run_dir(run_name: str | None) -> Path:
    dirname = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cases").mkdir(exist_ok=True)
    return run_dir


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_name(text: str) -> str:
    text = re.sub(r"\b(Mr|Ms|Mrs|Miss)\b\.?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]+", " ", text)
    return " ".join(text.lower().split())


def _title_case_name(text: str) -> str:
    tokens = []
    for token in text.split():
        if token.isupper() and len(token) > 1:
            tokens.append(token.title())
        else:
            tokens.append(token)
    return " ".join(tokens)


def _judgment_year(judgementdate: str) -> int | None:
    match = re.search(r"(\d{4})", judgementdate or "")
    return int(match.group(1)) if match else None


def _age_group_from_birth_year(birth_year: int | None, judgment_year: int | None) -> str:
    if birth_year is None or judgment_year is None:
        return "unknown"
    age = judgment_year - birth_year
    if age < 0:
        return "unknown"
    if age <= 12:
        return "child"
    if age <= 17:
        return "adolescent"
    if age >= 65:
        return "elderly"
    return "adult"


def _extract_name_year_pairs(text: str) -> list[tuple[str, int]]:
    columns = [col.strip() for col in text.split("|") if col.strip()]
    candidate_cells: list[str] = []
    for col in columns:
        lowered = col.lower()
        if "applicant’s name" in lowered or "applicant's name" in lowered:
            continue
        if "application no." in lowered or "date of introduction" in lowered:
            continue
        if "amount awarded" in lowered or "substance of the complaint" in lowered or "factual information" in lowered:
            continue
        if not re.search(r"\b(?:19|20)\d{2}\b", col):
            continue
        if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", col):
            continue
        candidate_cells.append(col)

    pairs: list[tuple[str, int]] = []
    for cell in candidate_cells or [text]:
        for raw_name, year in NAME_YEAR_RE.findall(cell):
            name = _title_case_name(_clean_text(raw_name))
            if not name:
                continue
            try:
                birth_year = int(year)
            except ValueError:
                continue
            pairs.append((name, birth_year))
    deduped: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return deduped


def _extract_title_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for title, raw_name in TITLE_NAME_RE.findall(text):
        sex = "male" if title.lower() == "mr" else "female"
        full = _normalize_name(raw_name)
        if full:
            mapping[full] = sex
            mapping.setdefault(full.split()[-1], sex)
    return mapping


def _docname_names(docname: str) -> list[str]:
    match = re.search(r"CASE OF (.+?) v\.", docname or "", re.IGNORECASE)
    if not match:
        return []
    applicant_side = match.group(1).strip()
    if " AND OTHERS" in applicant_side.upper():
        return []
    parts = [part.strip() for part in re.split(r"\s+AND\s+", applicant_side, flags=re.IGNORECASE) if part.strip()]
    return [_title_case_name(part) for part in parts]


def _extract_num_applicants(row: dict[str, Any], source_row: dict[str, Any], appendix_text: str, procedure_text: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    match = EXPLICIT_APPLICANT_COUNT_RE.search(appendix_text) or EXPLICIT_APPLICANT_COUNT_RE.search(procedure_text)
    if match:
        count = int(match.group(1))
        notes.append("num_applicants from explicit '(N applicants)' marker")
        return count, notes
    applicant_pairs = _extract_name_year_pairs(appendix_text)
    if applicant_pairs:
        notes.append("num_applicants from parsed applicant rows")
        return len(applicant_pairs), notes
    source_n = source_row.get("n_applicants")
    if isinstance(source_n, int) and source_n > 0:
        notes.append("num_applicants from source metadata n_applicants")
        return source_n, notes
    source_n_text = source_row.get("n_applicants")
    if isinstance(source_n_text, str) and source_n_text.isdigit():
        notes.append("num_applicants from source metadata n_applicants")
        return int(source_n_text), notes
    docname_names = _docname_names(source_row.get("docname") or "")
    if docname_names:
        notes.append("num_applicants from docname applicant side")
        return len(docname_names), notes
    proxy = row["core_case"].get("num_applicants_proxy")
    if isinstance(proxy, int) and proxy > 0:
        notes.append("num_applicants from existing proxy")
        return proxy, notes
    notes.append("num_applicants defaulted to 1")
    return 1, notes


def _extract_uniform_nationality(procedure_text: str) -> str | None:
    single = SINGLE_NATIONAL_RE.search(procedure_text)
    if single:
        return single.group(1).title()
    plural = PLURAL_NATIONAL_RE.search(procedure_text)
    if plural:
        return plural.group(1).title()
    return None


def _appendix_text(row: dict[str, Any]) -> str:
    return _clean_text((row["claim_and_award_layer"].get("cross_validation_inputs") or {}).get("appendix_table_text") or "")


def _build_applicants(
    row: dict[str, Any],
    source_row: dict[str, Any],
    num_applicants: int,
    appendix_text: str,
    procedure_text: str,
    judgment_year: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    title_map = _extract_title_map("\n".join([appendix_text, procedure_text]))
    applicant_pairs = _extract_name_year_pairs(appendix_text)
    applicants: list[dict[str, Any]] = []
    if applicant_pairs:
        notes.append("applicant identities from appendix table rows")
        for idx, (name, birth_year) in enumerate(applicant_pairs, start=1):
            normalized = _normalize_name(name)
            parts = normalized.split()
            sex = title_map.get(normalized) or (title_map.get(parts[-1]) if parts else None) or "unknown"
            applicants.append(
                {
                    "applicant_index": idx,
                    "beneficiary_label": name,
                    "birth_year": birth_year,
                    "sex": sex,
                    "age_group": _age_group_from_birth_year(birth_year, judgment_year),
                    "nationality": None,
                }
            )
    elif SINGLE_NATIONAL_RE.search(procedure_text):
        match = SINGLE_NATIONAL_RE.search(procedure_text)
        assert match is not None
        nationality = match.group(1).title()
        sex = "male" if match.group(2).lower() == "mr" else "female"
        name = _title_case_name(_clean_text(match.group(3)))
        birth_match = re.search(r"born(?:\s+in)?\s+((?:19|20)\d{2})", procedure_text, re.IGNORECASE)
        birth_year = int(birth_match.group(1)) if birth_match else None
        notes.append("single applicant identity from formulaic procedure opening")
        applicants.append(
            {
                "applicant_index": 1,
                "beneficiary_label": name,
                "birth_year": birth_year,
                "sex": sex,
                "age_group": _age_group_from_birth_year(birth_year, judgment_year),
                "nationality": nationality,
            }
        )
    else:
        docname_names = _docname_names(source_row.get("docname") or "")
        if docname_names:
            notes.append("applicant labels from docname")
            for idx, name in enumerate(docname_names, start=1):
                applicants.append(
                    {
                        "applicant_index": idx,
                        "beneficiary_label": name,
                        "birth_year": None,
                        "sex": title_map.get(_normalize_name(name), "unknown"),
                        "age_group": "unknown",
                        "nationality": None,
                    }
                )

    uniform_nationality = _extract_uniform_nationality(procedure_text)
    if uniform_nationality:
        for applicant in applicants:
            if applicant["nationality"] is None:
                applicant["nationality"] = uniform_nationality
        notes.append("nationality applied from procedure opening")

    if len(applicants) < num_applicants:
        notes.append("applicant list padded to match num_applicants")
        for idx in range(len(applicants) + 1, num_applicants + 1):
            applicants.append(
                {
                    "applicant_index": idx,
                    "beneficiary_label": None,
                    "birth_year": None,
                    "sex": "unknown",
                    "age_group": "unknown",
                    "nationality": uniform_nationality,
                }
            )
    elif len(applicants) > num_applicants:
        applicants = applicants[:num_applicants]
        notes.append("applicant list truncated to num_applicants")

    return applicants, notes


def validate_result(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "itemid" not in result:
        errors.append("missing itemid")
    facts = result.get("facts_procedure")
    if not isinstance(facts, dict):
        errors.append("missing facts_procedure")
        return errors
    if not isinstance(facts.get("num_applicants"), int) or facts["num_applicants"] < 1:
        errors.append("facts_procedure.num_applicants must be >= 1")
    if not isinstance(facts.get("is_joint_application"), bool):
        errors.append("facts_procedure.is_joint_application must be boolean")
    elif facts["is_joint_application"] != (facts["num_applicants"] > 1):
        errors.append("facts_procedure.is_joint_application must equal (num_applicants > 1)")
    applicants = facts.get("applicants")
    if not isinstance(applicants, list):
        errors.append("facts_procedure.applicants must be a list")
    else:
        if len(applicants) != facts.get("num_applicants"):
            errors.append("facts_procedure.applicants length must equal num_applicants")
    status = facts.get("status")
    if not isinstance(status, dict) or "is_represented" not in status:
        errors.append("facts_procedure.status.is_represented missing")
    return errors


def load_existing_results(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(json.loads(line)["itemid"])
    return seen


def write_jsonl_line(path: Path, row: dict[str, Any]) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_case_file(case_dir: Path, itemid: str, payload: dict[str, Any]) -> None:
    with WRITE_LOCK:
        (case_dir / f"{itemid}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one_case(itemid: str, row: dict[str, Any], source_row: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    facts_inputs = row["facts_procedure"].get("evidence_inputs") or {}
    procedure_text = _clean_text(facts_inputs.get("procedure_text") or "")
    intro_text = _clean_text(facts_inputs.get("introduction_text") or "")
    facts_text = _clean_text(facts_inputs.get("facts_text") or "")
    appendix_text = _appendix_text(row)
    judgment_year = _judgment_year(row.get("judgementdate") or "")

    num_applicants, count_notes = _extract_num_applicants(row, source_row, appendix_text, procedure_text)
    applicants, applicant_notes = _build_applicants(row, source_row, num_applicants, appendix_text, procedure_text, judgment_year)

    if len(applicants) != num_applicants:
        num_applicants = len(applicants)

    represented = row["core_case"].get("represented")
    if represented is None:
        represented = bool(source_row.get("representedby"))

    indirect_victim_text = "\n".join([intro_text, procedure_text, facts_text])
    is_indirect_victim = bool(INDIRECT_VICTIM_RE.search(indirect_victim_text)) if indirect_victim_text else None

    result = {
        "itemid": itemid,
        "facts_procedure": {
            "num_applicants": num_applicants,
            "is_joint_application": num_applicants > 1,
            "applicants": applicants,
            "status": {
                "is_represented": represented,
            },
            "is_indirect_victim": is_indirect_victim,
            "deterministic_notes": count_notes + applicant_notes,
        },
    }
    errors = validate_result(result, schema)
    payload: dict[str, Any] = {
        "status": "success" if not errors else "failed_validation",
        "itemid": itemid,
        "attempts": 0,
        "elapsed_seconds": time.perf_counter() - start,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if errors:
        payload["errors"] = errors
    else:
        payload["result"] = result
    return payload


def write_per_split_results(run_dir: Path, split_to_ids: dict[str, list[str]], success_rows: dict[str, dict[str, Any]]) -> None:
    by_split_dir = run_dir / "by_split"
    by_split_dir.mkdir(exist_ok=True)
    for split_name, ids in split_to_ids.items():
        path = by_split_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for itemid in ids:
                row = success_rows.get(itemid)
                if row is not None:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    schema = load_json(SCHEMA_PATH)
    run_dir = make_run_dir(args.run_name)

    split_to_ids = {split: load_split_ids(split) for split in args.splits}
    if args.itemids:
        unique_ids = dedupe_preserve_order([str(x) for x in args.itemids])
        split_to_ids["manual_itemids"] = unique_ids
    else:
        unique_ids = dedupe_preserve_order([itemid for split in args.splits for itemid in split_to_ids[split]])
    if args.max_cases is not None:
        unique_ids = unique_ids[: args.max_cases]
        chosen = set(unique_ids)
        split_to_ids = {k: [x for x in v if x in chosen] for k, v in split_to_ids.items()}

    rows_by_itemid = load_jsonl_by_itemid(INPUT_JSONL)
    source_rows = load_cases_by_itemid(unique_ids, fallback_cases_json=UNSTRUCTURED_CASES, backfill_store=True)
    missing = [itemid for itemid in unique_ids if itemid not in rows_by_itemid or itemid not in source_rows]
    if missing:
        raise RuntimeError(f"{len(missing)} itemids were not found in required inputs")

    unique_results_path = run_dir / "results_unique.jsonl"
    meta_results_path = run_dir / "results_meta.jsonl"
    failures_path = run_dir / "failures.jsonl"
    run_metadata_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "summary.json"
    case_dir = run_dir / "cases"

    already_done = load_existing_results(unique_results_path) if args.resume else set()
    todo_ids = [itemid for itemid in unique_ids if itemid not in already_done]

    run_metadata = {
        "input_jsonl": str(INPUT_JSONL),
        "unstructured_cases": str(UNSTRUCTURED_CASES),
        "splits": args.splits,
        "split_counts_requested": {k: len(v) for k, v in split_to_ids.items()},
        "unique_cases_requested": len(unique_ids),
        "unique_cases_to_run": len(todo_ids),
        "resume": args.resume,
        "dry_run": args.dry_run,
        "schema": str(SCHEMA_PATH),
        "mode": "deterministic_backbone",
        "case_store_dir": str(CASE_STORE_DIR),
    }
    run_metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(run_metadata, ensure_ascii=False, indent=2))
        return

    start = time.perf_counter()
    success_rows: dict[str, dict[str, Any]] = {}
    failure_rows: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_one_case, itemid, rows_by_itemid[itemid], source_rows[itemid], schema): itemid
            for itemid in todo_ids
        }
        for future in as_completed(futures):
            itemid = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = {"status": "request_failed", "itemid": itemid, "errors": [str(exc)]}

            if payload["status"] == "success":
                result = payload["result"]
                success_rows[itemid] = result
                write_jsonl_line(unique_results_path, result)
                write_jsonl_line(
                    meta_results_path,
                    {
                        "itemid": itemid,
                        "attempts": payload.get("attempts"),
                        "elapsed_seconds": payload.get("elapsed_seconds"),
                        "usage": payload.get("usage"),
                    },
                )
                write_case_file(case_dir, itemid, result)
            else:
                failure_rows.append(payload)
                write_jsonl_line(failures_path, payload)

    write_per_split_results(run_dir, split_to_ids, success_rows)

    summary = {
        **run_metadata,
        "runtime_seconds": time.perf_counter() - start,
        "successful_cases": len(success_rows),
        "failed_cases": len(failure_rows),
        "usage_total": usage_total,
        "results_unique_jsonl": str(unique_results_path),
        "results_meta_jsonl": str(meta_results_path),
        "failures_jsonl": str(failures_path),
        "by_split_dir": str(run_dir / "by_split"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from case_store import CASE_STORE_DIR, UNSTRUCTURED_CASES, load_cases_by_itemid
from openai_compatible_client import ApiCallError, OpenAICompatibleClient
from run_pipeline_b_backbone import (
    _appendix_text,
    _build_applicants,
    _clean_text,
    _extract_num_applicants,
    _judgment_year,
)


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = EXTRACTION_ROOT.parent
INPUT_JSONL = EXTRACTION_ROOT / "outputs" / "case_features_labels.jsonl"
SPLITS_DIR = DATASET_ROOT / "splits"
PROMPT_PATH = EXTRACTION_ROOT / "prompts" / "pipeline_b_system_prompt.md"
SCHEMA_PATH = EXTRACTION_ROOT / "schemas" / "pipeline_b_facts_procedure.schema.json"
RUNS_ROOT = EXTRACTION_ROOT / "runs" / "pipeline_b"

DEFAULT_SPLITS = ["set1_primary", "set2_postcut_ood", "set3_challenging"]
WRITE_LOCK = threading.Lock()
NO_SUFFIX_RE = re.compile(r"\(No\.\s*(\d+)\)", re.IGNORECASE)
PILOT_RE = re.compile(r"\bpilot(?:-|\s)?judg", re.IGNORECASE)
REPETITIVE_RE = re.compile(r"\b(repetitive|follow-on|follow up|well-established case-law|well established case law)\b", re.IGNORECASE)
LEGAL_AID_RE = re.compile(r"\blegal aid\b", re.IGNORECASE)
PARTIAL_ADMISSIBILITY_RE = re.compile(
    r"\b(partly|partially)\s+admissible\b"
    r"|\bdismiss(?:es|ed)? the remainder\b.*\badmissib"
    r"|\b(?:remainder|rest) of the application (?:as )?inadmissible\b"
    r"|\bremaining complaints? (?:as )?inadmissible\b"
    r"|\b(?:declares?|holds?) .{0,80}inadmissible\s+and\s+the\s+remainder\b"
    r"|\badmissible and the remainder .{0,40}inadmissible\b",
    re.IGNORECASE | re.DOTALL,
)
DOMESTIC_AWARD_RE = re.compile(
    r"\b(domestic|national|district|regional|supreme|constitutional)\b.{0,120}\b(award(?:ed)?|compensation|damages|redress)\b",
    re.IGNORECASE | re.DOTALL,
)
STATE_REMEDIAL_RE = re.compile(
    r"\b(reopen(?:ed|ing)?|retrial|re-examination|fresh examination|fresh proceedings|annulled|quashed|new hearing|acknowledged|redress)\b",
    re.IGNORECASE,
)
STATE_REMEDIAL_STRONG_RE = re.compile(
    r"\b(reopen(?:ed|ing)?|retrial|re-examination|fresh examination|fresh proceedings|new hearing|remitt?ed|reinstated|pardon(?:ed)?|amend(?:ed)? legislation|legislative amendment)\b",
    re.IGNORECASE,
)
APPLICANT_CONTRIBUTION_RE = re.compile(
    r"\b(contributed to the damage|contributory conduct|award reduced due to .*conduct|own conduct contributed to|partly responsible for the damage)\b",
    re.IGNORECASE,
)
DOMESTIC_AWARD_STRICT_RE = re.compile(
    r"\b(?:district|regional|city|appeal|supreme|constitutional|administrative|military)\s+court\b[\s\S]{0,160}\b(?:awarded|granted)\b[\s\S]{0,80}\b(?:compensation|damages|redress)\b"
    r"|\b(?:awarded|granted)\b[\s\S]{0,80}\b(?:compensation|damages|redress)\b[\s\S]{0,160}\b(?:district|regional|city|appeal|supreme|constitutional|administrative|military)\s+court\b",
    re.IGNORECASE,
)
COURT_LABELS = (
    ("constitutional", re.compile(r"\bconstitutional court\b", re.IGNORECASE)),
    ("supreme", re.compile(r"\bsupreme court\b", re.IGNORECASE)),
    ("administrative", re.compile(r"\badministrative court\b", re.IGNORECASE)),
    ("military", re.compile(r"\bmilitary court\b", re.IGNORECASE)),
    ("appeal", re.compile(r"\b(?:court of appeal|appeal court|regional court)\b", re.IGNORECASE)),
    ("first_instance", re.compile(r"\b(?:district court|trial court|first-instance court|first instance court)\b", re.IGNORECASE)),
)
GUIDE = {
    "itemid": "string",
    "facts_procedure": {
        "num_applicants": "integer >= 1",
        "is_joint_application": "boolean|null, should equal (num_applicants > 1)",
        "applicants": [{
            "applicant_index": "integer >= 1",
            "beneficiary_label": "string|null",
            "birth_year": "integer|null",
            "sex": ["male", "female", "mixed", "unknown", None],
            "age_group": ["child", "adolescent", "adult", "elderly", "unknown", None],
            "nationality": "string|null",
        }],  # must contain exactly num_applicants entries
        "status": {
            "is_vulnerable": "boolean|null",
            "vulnerability_tags": [
                "minor",
                "detained",
                "disabled",
                "refugee",
                "asylum_seeker",
                "medical_vulnerability",
                "elderly_dependent",
                "victim_of_violence",
            ],
            "is_represented": "boolean|null",
        },
        "prior_cases": {"is_repeated": "boolean|null", "count": "integer|null"},
        "is_indirect_victim": "boolean|null",
        "applicant_contribution": "short extractive string|null",
        "duration_months": "number|null",
        "is_repetitive_case": "boolean|null",
        "is_pilot_judgment": "boolean|null",
        "admissibility_decision_date": "date string|null",
        "partial_admissibility": "boolean|null",
        "complaints_summary": [{"article": "string|null", "theme": "string"}],
        "legal_aid_from_coe": "boolean|null",
        "pilot_judgment_procedure": "boolean|null",
        "courts_involved": ["first_instance", "appeal", "supreme", "constitutional", "administrative", "military", "unknown"],
        "domestic_duration_months": "number|null",
        "domestic_award_prior": "boolean|null",
        "domestic_award_prior_eur": "number|null",
        "state_remedial_measures": "boolean|null",
    },
}

# Function guide (what each def is responsible for)
# - parse_args: Parse CLI arguments for batch/split/itemid execution.
# - load_json: Load JSON file content into Python objects.
# - load_jsonl_by_itemid: Read JSONL rows and index them by itemid.
# - load_split_ids: Resolve split definitions from JSON or CSV split files.
# - dedupe_preserve_order: Remove duplicates while preserving first-seen order.
# - make_run_dir: Create run directory and required subdirectories.
# - _parse_date: Parse flexible date values into datetime.
# - _month_diff: Compute approximate month span between two dates.
# - _candidate_domestic_award_eur: Heuristically detect domestic award amounts from text.
# - _courts_involved: Identify court levels mentioned in procedure text.
# - _build_deterministic_hints: Build deterministic helper signals passed to prompt/normalization.
# - _appendix_award_amounts: Extract award numbers from appendix-style award text.
# - _normalize_b_result: Normalize LLM output into stable facts_procedure contract.
# - prompt_messages: Build prompt payload (schema + evidence + hints) for B extraction.
# - validate_result: Validate B output against schema and semantic checks.
# - merge_retry_feedback: Append actionable retry guidance after validation errors.
# - load_existing_results: Load completed itemids to support resume mode.
# - write_jsonl_line: Append one JSON line safely with thread lock.
# - write_case_file: Persist per-case Pipeline B extraction artifact.
# - run_one_case: Execute one B extraction call with retry/validation loop.
# - write_per_split_results: Write split-level output JSONL files from successful rows.
# - main: CLI entrypoint orchestrating parallel Pipeline B extraction.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-based Pipeline B extraction on selected ECHR-NPD cases.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--itemids", nargs="+", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("EXTRACTION_CONCURRENCY", "8")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("EXTRACTION_MAX_RETRIES", "3")))
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


def _parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _month_diff(start_value: Any, end_value: Any) -> float | None:
    start_dt = _parse_date(start_value)
    end_dt = _parse_date(end_value)
    if not start_dt or not end_dt or end_dt < start_dt:
        return None
    return round((end_dt - start_dt).days / 30.4375, 2)


def _candidate_domestic_award_eur(text: str) -> float | None:
    match = re.search(r"\b(?:EUR|€)\s?([0-9][0-9., ]{0,20})", text, re.IGNORECASE)
    if not match:
        return None
    number_text = re.sub(r"[ ,]", "", match.group(1))
    number_text = number_text.replace(",", "")
    try:
        return float(number_text)
    except ValueError:
        return None


def _courts_involved(text: str) -> list[str]:
    found: list[str] = []
    for label, pattern in COURT_LABELS:
        if pattern.search(text):
            found.append(label)
    return found


def _build_deterministic_hints(row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    facts_inputs = row["facts_procedure"].get("evidence_inputs") or {}
    procedure_text = _clean_text(facts_inputs.get("procedure_text") or "")
    intro_text = _clean_text(facts_inputs.get("introduction_text") or "")
    facts_text = _clean_text(facts_inputs.get("facts_text") or "")
    operative_text = _clean_text(((row.get("claim_and_award_layer") or {}).get("cross_validation_inputs") or {}).get("operative_text") or "")
    appendix_text = _appendix_text(row)
    judgment_year = _judgment_year(row.get("judgementdate") or "")
    num_applicants, _ = _extract_num_applicants(row, source_row, appendix_text, procedure_text)
    applicants, _ = _build_applicants(row, source_row, num_applicants, appendix_text, procedure_text, judgment_year)
    source_text = "\n\n".join(part for part in [intro_text, procedure_text, facts_text, appendix_text, operative_text] if part)
    non_strasbourg_text = "\n\n".join(part for part in [intro_text, procedure_text, facts_text] if part)

    no_match = NO_SUFFIX_RE.search(source_row.get("docname") or "")
    repeated_count = int(no_match.group(1)) if no_match else None

    domestic_award_match = DOMESTIC_AWARD_STRICT_RE.search(non_strasbourg_text)

    return {
        "applicant_backbone": {
            "num_applicants": num_applicants,
            "is_joint_application": num_applicants > 1,
            "applicants": applicants,
            "is_represented": row["core_case"].get("represented"),
            "is_indirect_victim": row["facts_procedure"].get("is_indirect_victim"),
        },
        "timing_hints": {
            "duration_months": _month_diff(source_row.get("introductiondate"), source_row.get("judgementdate")),
            "admissibility_decision_date": source_row.get("decisiondate") or None,
        },
        "procedural_hints": {
            "is_pilot_judgment": bool(PILOT_RE.search(source_text)) or None,
            "pilot_judgment_procedure": bool(PILOT_RE.search(source_text)) or None,
            "is_repetitive_case": bool(REPETITIVE_RE.search(source_text)) or None,
            "partial_admissibility": bool(PARTIAL_ADMISSIBILITY_RE.search(source_text)) or None,
            "legal_aid_from_coe": bool(LEGAL_AID_RE.search(source_text)) or None,
        },
        "domestic_hints": {
            "courts_involved": _courts_involved(source_text),
            "domestic_award_prior": bool(domestic_award_match) or None,
            "domestic_award_prior_eur": _candidate_domestic_award_eur(domestic_award_match.group(0)) if domestic_award_match else None,
            "state_remedial_measures": bool(STATE_REMEDIAL_RE.search(source_text)) or None,
        },
        "repetition_hints": {
            "prior_cases_is_repeated": True if repeated_count and repeated_count > 1 else None,
            "prior_cases_count": repeated_count,
        },
        "applicant_contribution_hint": APPLICANT_CONTRIBUTION_RE.search(source_text).group(0) if APPLICANT_CONTRIBUTION_RE.search(source_text) else None,
}


def _appendix_award_amounts(text: str) -> set[float]:
    amounts: set[float] = set()
    for match in re.finditer(r"(?<![A-Za-z])([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)(?![A-Za-z])", text or ""):
        try:
            amounts.add(float(match.group(1).replace(",", "")))
        except ValueError:
            continue
    return amounts


def _canonical_applicant_row(applicant: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    row = dict(applicant) if isinstance(applicant, dict) else {}
    idx = row.get("applicant_index")
    if not isinstance(idx, int) or idx < 1:
        idx = fallback_index
    sex = row.get("sex")
    if sex not in {"male", "female", "mixed", "unknown", None}:
        sex = "unknown"
    age_group = row.get("age_group")
    if age_group not in {"child", "adolescent", "adult", "elderly", "unknown", None}:
        age_group = "unknown"
    nationality = row.get("nationality")
    if nationality is not None and not isinstance(nationality, str):
        nationality = None
    beneficiary_label = row.get("beneficiary_label")
    if beneficiary_label is not None and not isinstance(beneficiary_label, str):
        beneficiary_label = None
    birth_year = row.get("birth_year")
    if birth_year is not None and not isinstance(birth_year, int):
        birth_year = None
    return {
        "applicant_index": idx,
        "beneficiary_label": beneficiary_label,
        "birth_year": birth_year,
        "sex": sex,
        "age_group": age_group,
        "nationality": nationality,
    }


def _merge_applicant_backbone(
    llm_applicants: Any,
    hint_applicants: Any,
    num_applicants: int,
) -> list[dict[str, Any]]:
    base_rows: list[dict[str, Any]] = []
    if isinstance(hint_applicants, list):
        for idx, hint in enumerate(hint_applicants[:num_applicants], start=1):
            base_rows.append(_canonical_applicant_row(hint, idx))
    for idx in range(len(base_rows) + 1, num_applicants + 1):
        base_rows.append(
            {
                "applicant_index": idx,
                "beneficiary_label": f"Applicant {idx}",
                "birth_year": None,
                "sex": "unknown",
                "age_group": "unknown",
                "nationality": None,
            }
        )

    if not isinstance(llm_applicants, list):
        return base_rows

    for row_pos, raw_row in enumerate(llm_applicants, start=1):
        if not isinstance(raw_row, dict):
            continue
        candidate = _canonical_applicant_row(raw_row, row_pos)
        idx = candidate["applicant_index"]
        if not isinstance(idx, int) or not (1 <= idx <= num_applicants):
            idx = row_pos if 1 <= row_pos <= num_applicants else None
        if idx is None:
            continue
        merged = dict(base_rows[idx - 1])
        for key in ("beneficiary_label", "birth_year", "sex", "age_group", "nationality"):
            value = candidate.get(key)
            if value not in (None, "", "unknown"):
                merged[key] = value
        merged["applicant_index"] = idx
        if not merged.get("beneficiary_label"):
            merged["beneficiary_label"] = base_rows[idx - 1].get("beneficiary_label") or f"Applicant {idx}"
        if merged.get("sex") is None:
            merged["sex"] = "unknown"
        if merged.get("age_group") is None:
            merged["age_group"] = "unknown"
        base_rows[idx - 1] = merged

    for idx, row in enumerate(base_rows, start=1):
        row["applicant_index"] = idx
        if not row.get("beneficiary_label"):
            row["beneficiary_label"] = f"Applicant {idx}"
        if row.get("sex") is None:
            row["sex"] = "unknown"
        if row.get("age_group") is None:
            row["age_group"] = "unknown"
    return base_rows


def _normalize_b_result(result: dict[str, Any], row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(result))
    fp = normalized.get("facts_procedure") or {}
    hints = _build_deterministic_hints(row, source_row)
    appendix_amounts = _appendix_award_amounts(_appendix_text(row))
    facts_inputs = row["facts_procedure"].get("evidence_inputs") or {}
    intro_text = _clean_text(facts_inputs.get("introduction_text") or "")
    procedure_text = _clean_text(facts_inputs.get("procedure_text") or "")
    facts_text = _clean_text(facts_inputs.get("facts_text") or "")
    non_strasbourg_text = "\n\n".join(part for part in [intro_text, procedure_text, facts_text] if part)

    if isinstance(fp.get("applicant_contribution"), str):
        value = fp["applicant_contribution"].strip()
        if not APPLICANT_CONTRIBUTION_RE.search(value):
            fp["applicant_contribution"] = None

    normalized_num = fp.get("num_applicants")
    if isinstance(normalized_num, int) and normalized_num >= 1:
        fp["is_joint_application"] = normalized_num > 1
    else:
        hint_num = ((hints.get("applicant_backbone") or {}).get("num_applicants"))
        if isinstance(hint_num, int) and hint_num >= 1:
            fp["num_applicants"] = hint_num
            fp["is_joint_application"] = hint_num > 1
        else:
            fp["is_joint_application"] = None

    hint_backbone = hints.get("applicant_backbone") or {}
    hint_num = hint_backbone.get("num_applicants")
    if not isinstance(fp.get("num_applicants"), int) or fp.get("num_applicants", 0) < 1:
        if isinstance(hint_num, int) and hint_num >= 1:
            fp["num_applicants"] = hint_num
    num_applicants = fp.get("num_applicants")
    if isinstance(num_applicants, int) and num_applicants >= 1:
        fp["is_joint_application"] = num_applicants > 1
        fp["applicants"] = _merge_applicant_backbone(
            fp.get("applicants"),
            hint_backbone.get("applicants"),
            num_applicants,
        )

    status = fp.get("status")
    if not isinstance(status, dict):
        status = {}
    if status.get("is_represented") is None:
        hint_represented = hint_backbone.get("is_represented")
        if isinstance(hint_represented, bool):
            status["is_represented"] = hint_represented
        else:
            core_represented = row.get("core_case", {}).get("represented")
            status["is_represented"] = core_represented if isinstance(core_represented, bool) else None
    if not isinstance(status.get("vulnerability_tags"), list):
        status["vulnerability_tags"] = []
    fp["status"] = status

    if not isinstance(fp.get("courts_involved"), list):
        fp["courts_involved"] = []
    if not isinstance(fp.get("complaints_summary"), list):
        fp["complaints_summary"] = []

    domestic_hint = (hints.get("domestic_hints") or {}).get("domestic_award_prior")
    domestic_amount = fp.get("domestic_award_prior_eur")
    if domestic_hint is not True:
        if isinstance(domestic_amount, (int, float)) and float(domestic_amount) in appendix_amounts:
            fp["domestic_award_prior"] = False if fp.get("domestic_award_prior") is True else fp.get("domestic_award_prior")
            fp["domestic_award_prior_eur"] = None
        if fp.get("domestic_award_prior") is True and fp.get("domestic_award_prior_eur") is None and domestic_hint is None:
            fp["domestic_award_prior"] = None

    if fp.get("state_remedial_measures") is True and not STATE_REMEDIAL_STRONG_RE.search(non_strasbourg_text):
        fp["state_remedial_measures"] = None

    normalized["facts_procedure"] = fp
    return normalized


def prompt_messages(system_prompt: str, schema: dict[str, Any], row: dict[str, Any], source_row: dict[str, Any], include_schema_in_payload: bool) -> list[dict[str, str]]:
    core = row["core_case"]
    facts = row["facts_procedure"]
    deterministic_hints = _build_deterministic_hints(row, source_row)
    payload = {
        "case_identification": {
            "itemid": row["itemid"],
            "appno": row["appno"],
            "judgementdate": row["judgementdate"],
            "respondent_country": core.get("respondent_country"),
            "violated_articles": core.get("violated_articles"),
            "num_applicants_proxy": core.get("num_applicants_proxy"),
            "is_represented_proxy": core.get("represented"),
        },
        "task": {
            "pipeline": "B",
            "goal": "extract applicant, procedure, and domestic-context fields using deterministic hints plus explicit textual evidence",
            "schema_name": "PipelineBFactsProcedureExtraction"
        },
        "priority_fields": [
            "num_applicants",
            "is_joint_application",
            "applicants[].sex",
            "applicants[].age_group",
            "applicants[].nationality",
            "status.is_vulnerable",
            "status.is_represented",
            "is_indirect_victim",
            "prior_cases",
            "applicant_contribution",
            "duration_months",
            "is_repetitive_case",
            "is_pilot_judgment",
            "admissibility_decision_date",
            "partial_admissibility",
            "complaints_summary",
            "legal_aid_from_coe",
            "pilot_judgment_procedure",
            "courts_involved",
            "domestic_duration_months",
            "domestic_award_prior",
            "domestic_award_prior_eur",
            "state_remedial_measures"
        ],
        "deterministic_hints": deterministic_hints,
        "evidence_inputs": {
            # B extracts facts/procedure/domestic-context — it does NOT need the
            # law section. Sending the routed law text here cost 5-11k tokens
            # per case with no measurable quality benefit; removed.
            "introduction_text": (facts.get("evidence_inputs") or {}).get("introduction_text") or "",
            "procedure_text": (facts.get("evidence_inputs") or {}).get("procedure_text") or "",
            "facts_text": (facts.get("evidence_inputs") or {}).get("facts_text") or "",
            "appendix_table_text": _appendix_text(row),
        },
        "output_guide": GUIDE,
    }
    if include_schema_in_payload:
        payload["output_schema"] = schema
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def validate_result(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        import jsonschema  # type: ignore

        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(result):
            loc = ".".join(str(x) for x in err.absolute_path) or "<root>"
            errors.append(f"{loc}: {err.message}")
    except Exception:
        pass

    if "itemid" not in result:
        errors.append("missing itemid")
    if "facts_procedure" not in result:
        errors.append("missing facts_procedure")
    else:
        fp = result["facts_procedure"]
        if fp.get("num_applicants") is None or fp.get("num_applicants", 0) < 1:
            errors.append("facts_procedure.num_applicants must be >= 1")
        if not isinstance(fp.get("is_joint_application"), bool):
            errors.append("facts_procedure.is_joint_application must be boolean")
        elif fp["is_joint_application"] != (fp.get("num_applicants", 0) > 1):
            errors.append("facts_procedure.is_joint_application must equal (num_applicants > 1)")
        applicants = fp.get("applicants")
        if not isinstance(applicants, list):
            errors.append("facts_procedure.applicants must be a list")
        elif len(applicants) != fp.get("num_applicants"):
            errors.append("facts_procedure.applicants length must equal num_applicants")
    return errors


def merge_retry_feedback(messages: list[dict[str, str]], errors: list[str]) -> list[dict[str, str]]:
    base_messages = messages[:2]
    return base_messages + [{
        "role": "user",
        "content": "Your previous JSON did not validate. Fix it and return a full replacement JSON object only. Validation errors: "
        + json.dumps(errors, ensure_ascii=False),
    }]


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


def run_one_case(
    itemid: str,
    row: dict[str, Any],
    source_row: dict[str, Any],
    client: OpenAICompatibleClient,
    system_prompt: str,
    schema: dict[str, Any],
    max_retries: int = 1,  # kept for backward-compat with the standalone CLI; ignored by design
) -> dict[str, Any]:
    """Single-shot pipeline B execution.

    Returns one of:
    - {status: "success", result, usage, elapsed_seconds}
    - {status: "api_error", api_error: {kind, http_status, detail}, ...}
    - {status: "schema_validation", errors, raw_result, ...}
    """
    start = time.perf_counter()
    messages = prompt_messages(system_prompt, schema, row, source_row, include_schema_in_payload=not client.use_json_schema)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        parsed, usage = client.chat_json(messages=messages, schema=schema, schema_name="pipeline_b_facts_procedure")
    except ApiCallError as exc:
        return {
            "status": "api_error",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": time.perf_counter() - start,
            "usage": usage_total,
            "api_error": exc.to_record(),
        }

    if usage:
        for key in usage_total:
            if isinstance(usage.get(key), int):
                usage_total[key] += usage[key]
    parsed["itemid"] = itemid
    parsed = _normalize_b_result(parsed, row, source_row)
    errors = validate_result(parsed, schema)
    if errors:
        return {
            "status": "schema_validation",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": time.perf_counter() - start,
            "usage": usage_total,
            "errors": errors,
            "raw_result": parsed,
        }

    return {
        "status": "success",
        "itemid": itemid,
        "attempts": 1,
        "elapsed_seconds": time.perf_counter() - start,
        "usage": usage_total,
        "result": parsed,
    }


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
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
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

    PER_CASE_DIR = EXTRACTION_ROOT / "outputs" / "cases"
    if args.itemids and PER_CASE_DIR.exists():
        rows_by_itemid = {}
        for itemid in unique_ids:
            case_file = PER_CASE_DIR / f"{itemid}.json"
            if case_file.exists():
                rows_by_itemid[itemid] = load_json(case_file)
        missing_per_case = [itemid for itemid in unique_ids if itemid not in rows_by_itemid]
        if missing_per_case:
            print(f"[INFO] {len(missing_per_case)} cases not in per-case dir, falling back to full JSONL...")
            rows_by_itemid = load_jsonl_by_itemid(INPUT_JSONL)
    else:
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
        "splits": args.splits,
        "split_counts_requested": {k: len(v) for k, v in split_to_ids.items()},
        "unique_cases_requested": len(unique_ids),
        "unique_cases_to_run": len(todo_ids),
        "resume": args.resume,
        "dry_run": args.dry_run,
        "unstructured_cases": str(UNSTRUCTURED_CASES),
        "case_store_dir": str(CASE_STORE_DIR),
        "schema": str(SCHEMA_PATH),
        "prompt": str(PROMPT_PATH),
    }
    run_metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(run_metadata, ensure_ascii=False, indent=2))
        return

    client = OpenAICompatibleClient.from_env()
    start = time.perf_counter()
    success_rows: dict[str, dict[str, Any]] = {}
    failure_rows: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_one_case,
                itemid,
                rows_by_itemid[itemid],
                source_rows[itemid],
                client,
                system_prompt,
                schema,
                args.max_retries,
            ): itemid
            for itemid in todo_ids
        }
        for future in as_completed(futures):
            itemid = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = {"status": "request_failed", "itemid": itemid, "errors": [str(exc)]}

            usage = payload.get("usage") or {}
            for key in usage_total:
                if isinstance(usage.get(key), int):
                    usage_total[key] += usage[key]

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

    if args.resume and unique_results_path.exists():
        with unique_results_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                success_rows[row["itemid"]] = row

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

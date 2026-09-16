#!/usr/bin/env python3
"""Project explicitly reviewed canonical records into a two-CSV staging directory.

No field is inferred from LLM candidates. Cohort membership and split/view labels
are retained from the supplied published table; this never appends to that table.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from public_tables import validate

CHECKS = ("head_separation", "applicant_beneficiary_consistency", "per_applicant_sum_consistency",
          "no_claim_positive_award_incompatibility", "claim_award_consistency", "currency_normalisation",
          "operative_provision_recoverability", "bundled_award_exception")
ZERO_RATIONALES = {"finding_sufficient", "no_claim", "unsubstantiated", "rule_60_non_compliance",
                   "domestic_award_covers", "applicant_deceased_no_heir"}


def record_hash(case: dict, applicants: list) -> str:
    encoded = json.dumps({"case": case, "applicants": applicants}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_fields(row: dict, columns: list[str], label: str) -> None:
    if not isinstance(row, dict) or set(row) != set(columns):
        raise ValueError(f"{label}: provide exactly the canonical schema fields; none are inferred")
    if any(not isinstance(value, str) or not value for value in row.values()):
        raise ValueError(f"{label}: use explicit string values, including unknown where allowed")


def export(input_path: Path, cohort_path: Path, source_root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; use an isolated new staging directory")
    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    columns = {name: list(schema[name]) for name in ("case_level", "applicant_level")}
    with cohort_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cohort = {row["case_id"]: row for row in rows}
    if not cohort or len(cohort) != len(rows):
        raise ValueError("Cohort must have unique case_id values")
    source_root = source_root.resolve()
    cases, applicants, seen, sources = [], [], set(), {}
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            case, people, review = record.get("case"), record.get("applicants"), record.get("review", {})
            require_fields(case, columns["case_level"], "case")
            if not isinstance(people, list):
                raise ValueError("applicants must be an explicit list")
            for person in people:
                require_fields(person, columns["applicant_level"], "applicant")
            case_id = case["case_id"]
            if case_id in seen or case_id not in cohort:
                raise ValueError("Duplicate case or case outside the retained paper cohort")
            seen.add(case_id)
            for field in ("split", "test_view", "test_challenging_view"):
                if case[field] != cohort[case_id][field]:
                    raise ValueError("The exporter cannot alter paper splits or diagnostic membership")
            if any(person["case_id"] != case_id for person in people):
                raise ValueError("Applicant record belongs to a different case")
            if review.get("status") != "accepted" or not review.get("adjudicator") or not review.get("source_anchor"):
                raise ValueError("A source-anchored explicit review is required")
            if review.get("record_sha256") != record_hash(case, people):
                raise ValueError("Review does not bind these exact canonical fields")
            if review.get("canonical_fields_reviewed") is not True:
                raise ValueError("All canonical fields, not only targets, require review")
            checks = review.get("checks", {})
            if set(checks) != set(CHECKS):
                raise ValueError("All eight Appendix B.3 validation checks must be documented")
            for name, check in checks.items():
                if (not isinstance(check, dict) or check.get("status") not in {"pass", "not_applicable"}
                        or not str(check.get("evidence", "")).strip()):
                    raise ValueError(f"Unresolved validation check: {name}")
            source_path = review.get("source_document", "")
            source = (source_root / source_path).resolve() if source_path else source_root
            if not source.is_relative_to(source_root) or not source.is_file():
                raise ValueError("Source document is missing or outside the source root")
            digest = file_hash(source)
            if digest != review.get("source_document_sha256"):
                raise ValueError("Reviewed source document checksum mismatch")
            sources[case_id] = digest
            if Decimal(case["y_amount_eur"]) == 0 and review.get("zero_rationale") not in ZERO_RATIONALES:
                raise ValueError("A zero target requires one of the paper's six source-supported zero rationales")
            if Decimal(case["y_amount_eur"]) != Decimal(cohort[case_id]["y_amount_eur"]):
                if case["target_status"] != "corrected_amount" or not str(review.get("correction_rationale", "")).strip():
                    raise ValueError("Amount changes require corrected_amount and a source-supported correction rationale")
            names = sorted({p["applicant_name"] for p in people if p["applicant_name"] not in {"[MASKED]", "unknown"}},
                           key=len, reverse=True)
            def mask(row):
                result = dict(row)
                for key, value in result.items():
                    if key in {"case_id", "applicant_id", "split"}:
                        continue
                    for name in names:
                        value = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", "[MASKED]", value, flags=re.I)
                    result[key] = value
                return result
            cases.append(mask(case))
            for person in people:
                applicants.append({**mask(person), "applicant_name": "[MASKED]"})
    if not cases:
        raise ValueError("No reviewed records supplied")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reviewed-extraction-", dir=output.parent) as temporary:
        stage = Path(temporary) / "staged"
        (stage / "data").mkdir(parents=True)
        for name, records in (("case_level", cases), ("applicant_level", applicants)):
            with (stage / "data" / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns[name], lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
        result = validate(stage, verify_manifest=False)
        if output.exists():
            raise ValueError("Output appeared during validation; refusing to replace it")
        stage.rename(output)
    return {**result, "kind": "reviewed_paper_cohort_subset_staging", "source_sha256": sources,
            "reviewed_input_sha256": file_hash(input_path), "cohort_sha256": file_hash(cohort_path),
            "dataset_modified": False, "paper_scores_reproduced": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Reviewed canonical JSONL records")
    parser.add_argument("--cohort-case-table", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.cohort_case_table, args.source_root, args.out), indent=2))


if __name__ == "__main__":
    main()

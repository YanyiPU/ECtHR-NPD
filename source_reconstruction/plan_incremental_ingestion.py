#!/usr/bin/env python3
"""Plan an INTERNAL date-window update from a supplied metadata CSV (offline).

This is not a HUDOC search client and cannot establish source-search completeness.
It does not download cases, call a model, alter splits, or modify the case store.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from source_utils import file_hash, index_records, read_index, value_hash, write_csv, write_json


def parse_source_date(value: object) -> date:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        raise ValueError(f"Missing/invalid ISO source date {text!r}; normalize explicitly before planning") from None


def build_plan(rows: list[dict], existing: list[dict], date_from: date, date_until: date,
               *, date_field: str = "judgementdate", docx_dir: Path | None = None,
               include_known_revisions: bool = False) -> dict:
    if date_until <= date_from:
        raise ValueError("--date-until must be after --date-from")
    candidates = index_records(rows, "candidate index")
    current = index_records(existing, "existing cases")
    selected, entries = [], []
    for itemid, row in candidates.items():
        judgment_date = parse_source_date(row.get(date_field))
        in_window = date_from <= judgment_date < date_until
        old = current.get(itemid)
        if not in_window and not (include_known_revisions and old):
            continue
        source_hash = None
        path = docx_dir / f"{itemid}.docx" if docx_dir else None
        if path and path.exists():
            source_hash = file_hash(path)
        metadata_hash = value_hash(row)
        previous = (old or {}).get("source_provenance") or {}
        if old is None:
            action = "add_candidate"
        elif metadata_hash != previous.get("case_index_record_sha256"):
            action = "review_metadata_revision"
        elif source_hash is None:
            action = "metadata_unchanged_source_unverified"
        elif source_hash != previous.get("docx_sha256"):
            action = "review_document_revision"
        else:
            action = "unchanged"
        entries.append({"itemid": itemid, "judgment_date": judgment_date.isoformat(),
                        "in_date_window": in_window, "action": action,
                        "metadata_sha256": metadata_hash, "docx_sha256": source_hash})
        if action != "unchanged":
            selected.append(row)
    content = {"schema_version": 1, "visibility": "internal_only",
               "date_from_inclusive": date_from.isoformat(), "date_until_exclusive": date_until.isoformat(),
               "date_field": date_field, "include_known_revisions": include_known_revisions,
               "source_discovery": "user_supplied_index_not_live_verified",
               "source_search_completeness": "unknown",
               "automatic_dataset_acceptance": False, "split_changes": False,
               "candidate_index_sha256": value_hash(rows), "existing_records_sha256": value_hash(existing),
               "records": entries}
    content["plan_id"] = value_hash(content)
    return {"manifest": content, "selected_rows": selected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", required=True, type=Path)
    parser.add_argument("--existing-cases", type=Path, help="Current internal unstructured/cases.json; omitted for an empty store.")
    parser.add_argument("--date-from", type=date.fromisoformat, required=True)
    parser.add_argument("--date-until", type=date.fromisoformat, required=True, help="Exclusive ISO cutoff.")
    parser.add_argument("--date-field", default="judgementdate")
    parser.add_argument("--docx-dir", type=Path)
    parser.add_argument("--include-known-revisions", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    existing = json.loads(args.existing_cases.read_text(encoding="utf-8")) if args.existing_cases else []
    if not isinstance(existing, list):
        raise ValueError("--existing-cases must be a JSON array")
    rows = read_index(args.case_index)
    result = build_plan(rows, existing, args.date_from, args.date_until, date_field=args.date_field,
                        docx_dir=args.docx_dir, include_known_revisions=args.include_known_revisions)
    # Only the requested plan outputs are written; the dataset is never touched.
    write_json(args.out_dir / "ingestion_plan.json", result["manifest"])
    fields = list(rows[0]) if rows else ["itemid", args.date_field]
    write_csv(args.out_dir / "candidate_index.csv", result["selected_rows"], fields)
    print(json.dumps({"plan_id": result["manifest"]["plan_id"], "selected_candidates": len(result["selected_rows"]),
                      "dataset_changed": False, "source_search_completeness": "unknown"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an INTERNAL ECHR-OD metadata index; full metadata may identify people."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from source_utils import index_records, normalize_metadata, read_index, write_csv, write_json


SAFE_PREFIXES = ("article=", "respondent.")
SAFE_FIELDS = {
    "itemid",
    "country",
    "decisiondate",
    "doctypebranch",
    "importance",
    "introductiondate",
    "judgementdate",
    "kpdate",
    "languageisocode",
    "originatingbody",
    "originatingbody_name",
    "originatingbody_type",
    "rank",
    "separateopinion",
    "typedescription",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", required=True, type=Path, help="Released CSV with itemid column.")
    parser.add_argument("--echrod-root", required=True, type=Path, help="Local ECHR-OD/ECHROD echr_database directory.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--public-minimal", action="store_true",
                        help="Legacy reduced metadata view; NOT sufficient for extraction or guaranteed anonymous.")
    return parser.parse_args()


def read_itemids(path: Path) -> set[str]:
    return {row["itemid"] for row in read_index(path)}


def keep_field(name: str) -> bool:
    return name in SAFE_FIELDS or any(name.startswith(prefix) for prefix in SAFE_PREFIXES)


def normalize_row(row: dict[str, Any], public_minimal: bool = False) -> dict[str, Any]:
    if not public_minimal:
        # Keep originals as well as typed fields so invalid/unparsed evidence is
        # never discarded. This output is INTERNAL, not a public feature table.
        return {**normalize_metadata(row), "source_metadata": row}
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if keep_field(key):
            normalized[key] = "" if value is None else str(value)
    return normalized


def main() -> int:
    args = parse_args()
    itemids = read_itemids(args.case_index)
    if not itemids:
        raise ValueError("No requested itemids; output was not changed")
    structured_cases = args.echrod_root / "structured" / "cases.csv"
    if not structured_cases.exists():
        raise FileNotFoundError(structured_cases)

    selected: list[dict] = []

    with structured_cases.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "itemid" not in reader.fieldnames:
            raise ValueError(f"{structured_cases} must contain itemid")
        for row in reader:
            itemid = str(row.get("itemid") or "").strip()
            if itemid in itemids:
                selected.append(normalize_row(row, public_minimal=args.public_minimal))

    by_itemid = index_records(selected, str(structured_cases))
    missing = sorted(itemids - set(by_itemid))
    if missing:
        print(json.dumps({"status": "rejected", "missing_count": len(missing), "missing_itemids": missing}, indent=2))
        return 1

    fieldnames = sorted({key for row in selected for key in row})
    if "itemid" in fieldnames:
        fieldnames.remove("itemid")
        fieldnames.insert(0, "itemid")

    csv_path = args.out_dir / "echrod_metadata_subset.csv"
    write_csv(csv_path, selected, fieldnames)

    json_path = args.out_dir / "echrod_metadata_by_itemid.json"
    write_json(json_path, by_itemid)

    summary = {
        "case_index_itemids": len(itemids),
        "matched_echrod_rows": len(selected),
        "missing_itemids": sorted(itemids - set(by_itemid))[:100],
        "visibility": "internal_only" if not args.public_minimal else "reduced_metadata_not_anonymized",
        "complete_source_metadata_retained": not args.public_minimal,
        "may_contain_personal_information": True,
        "raw_text_excluded": args.public_minimal,
        "outputs": [str(csv_path), str(json_path)],
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

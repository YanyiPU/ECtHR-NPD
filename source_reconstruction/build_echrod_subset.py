#!/usr/bin/env python3
"""Build a non-text ECHR-OD/ECHROD metadata index for released itemids."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
    return parser.parse_args()


def read_itemids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "itemid" not in reader.fieldnames:
            raise ValueError(f"{path} must contain itemid")
        return {str(row.get("itemid") or "").strip() for row in reader if str(row.get("itemid") or "").strip()}


def keep_field(name: str) -> bool:
    return name in SAFE_FIELDS or any(name.startswith(prefix) for prefix in SAFE_PREFIXES)


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if keep_field(key):
            normalized[key] = "" if value is None else str(value)
    return normalized


def main() -> int:
    args = parse_args()
    itemids = read_itemids(args.case_index)
    structured_cases = args.echrod_root / "structured" / "cases.csv"
    if not structured_cases.exists():
        raise FileNotFoundError(structured_cases)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, str]] = []

    with structured_cases.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "itemid" not in reader.fieldnames:
            raise ValueError(f"{structured_cases} must contain itemid")
        for row in reader:
            itemid = str(row.get("itemid") or "").strip()
            if itemid in itemids:
                selected.append(normalize_row(row))

    fieldnames = sorted({key for row in selected for key in row})
    if "itemid" in fieldnames:
        fieldnames.remove("itemid")
        fieldnames.insert(0, "itemid")

    csv_path = args.out_dir / "echrod_metadata_subset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    json_path = args.out_dir / "echrod_metadata_by_itemid.json"
    by_itemid = {row["itemid"]: row for row in selected if row.get("itemid")}
    json_path.write_text(json.dumps(by_itemid, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "case_index_itemids": len(itemids),
        "matched_echrod_rows": len(selected),
        "missing_itemids": sorted(itemids - set(by_itemid))[:100],
        "contains_raw_text": False,
        "outputs": [str(csv_path), str(json_path)],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

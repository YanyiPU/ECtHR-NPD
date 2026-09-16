#!/usr/bin/env python3
"""Offline integrity/relational audit of an explicitly selected dataset version.

No model fitting, no scientific source certification, no publication approval.
Use --dataset-root /path/to/dataset --dataset-version corrected|paper_reference.
The root can also be the selected version's direct clean directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


class VersionError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_version(root, version):
    if version not in {"corrected", "paper_reference"}:
        raise VersionError("Explicit supported dataset version is required")
    root = Path(root).expanduser().resolve()
    if (root / "data/ecthr_npd_cases.csv").is_file():
        return root
    selected = root / ("clean" if version == "corrected" else "paper_reference/clean")
    if not (selected / "data/ecthr_npd_cases.csv").is_file():
        raise VersionError(f"Selected version missing: {selected}; no fallback is permitted")
    return selected


def safe_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise VersionError("Manifest path must be a nonempty relative string")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or relative == ".":
        raise VersionError(f"Unsafe manifest path: {relative}")
    path = root / rel
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        # Do not accept an external data file through a filesystem alias.
        raise VersionError(f"Symlinks are not permitted in a version package: {relative}")
    if root not in path.resolve().parents or not path.is_file():
        raise VersionError(f"Missing/outside version file: {relative}")
    return path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise VersionError(f"Missing/duplicate columns: {path}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise VersionError(f"Malformed CSV rows: {path}")
    return columns, rows


def unique(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        if not value or value in result:
            raise VersionError(f"Blank/duplicate {key}")
        result[value] = row
    return result


def amount(value):
    try:
        n = Decimal(str(value))
    except InvalidOperation as exc:
        raise VersionError("Invalid amount") from exc
    if not n.is_finite() or n < 0:
        raise VersionError("Amounts must be finite and nonnegative")
    return n


def flag(value):
    if str(value).lower() not in {"true", "false", "1", "0"}:
        raise VersionError("Invalid boolean flag")
    return str(value).lower() in {"true", "1"}


def verify(root, version, *, mode="audit"):
    root = resolve_version(root, version)
    mpath = root / "VERSION_MANIFEST.json"
    manifest = json.loads(mpath.read_text())
    contract = json.loads((root / "release_contract.json").read_text())
    if manifest.get("schema_version") != "ecthr-npd-version-1":
        raise VersionError("Unsupported version manifest")
    if manifest.get("dataset_version") != version or contract.get("dataset_version") != version:
        raise VersionError("Requested version differs from manifest/contract")
    if manifest.get("release_id") != contract.get("release_id") or manifest.get("counts") != contract.get("counts"):
        raise VersionError("Manifest and release contract disagree")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or "VERSION_MANIFEST.json" in files:
        raise VersionError("Invalid version integrity file map")
    for relative, h in files.items():
        if not isinstance(h, str) or len(h) != 64 or sha256(safe_file(root, relative)) != h:
            raise VersionError(f"Hash mismatch: {relative}")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and not p.name.startswith(".") and p.name != "VERSION_MANIFEST.json"}
    if actual != set(files):
        raise VersionError("Version file inventory differs from pinned manifest")
    revision = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("revision_id") != "sha256:" + revision:
        raise VersionError("Version revision digest does not match the file map")
    csv_tables = {}
    for spec in manifest.get("tables", []):
        relative = spec["path"]
        if relative in csv_tables:
            raise VersionError("Duplicate CSV table specification")
        cols, rows = read_csv(safe_file(root, relative))
        if cols != spec["columns"] or len(rows) != spec["rows"]:
            raise VersionError(f"Table schema/count differs: {relative}")
        csv_tables[relative] = (cols, rows)
    if set(csv_tables) != {p for p in files if p.endswith(".csv")}:
        raise VersionError("CSV inventory is incomplete")
    _, rows = csv_tables["data/ecthr_npd_cases.csv"]
    cases = unique(rows, "itemid")
    predictors = manifest.get("structured_predictor_columns", [])
    if len(predictors) != (48 if version == "corrected" else 50) or len(set(predictors)) != len(predictors):
        raise VersionError("Wrong version-specific predictor count or duplicate names")
    forbidden = {"perapp_unique_beneficiary_category_count", "perapp_has_joint_beneficiary_category_flag"}
    if version == "corrected" and forbidden & set(predictors):
        raise VersionError("Corrected version contains unverified allocation-derived predictors")
    counts = contract["counts"]
    if len(cases) != counts["total"] or dict(Counter(r["split"] for r in rows)) != counts["splits"]:
        raise VersionError("Canonical case/split counts differ")
    tests = [r for r in rows if r["split"] == "test"]
    if dict(Counter(r["test_view"] for r in tests)) != counts["test_views"]:
        raise VersionError("Test view counts differ")
    if sum(flag(r["test_challenging_view"]) for r in tests) != counts["challenging"]:
        raise VersionError("Challenging count differs")
    for r in rows:
        if amount(r["y_binary"]) != int(amount(r["y_amount_eur"]) > 0):
            raise VersionError("Binary/amount targets differ")
        if r["split"] != "test" and (r["test_view"] or flag(r["test_challenging_view"])):
            raise VersionError("Non-test row has evaluation view membership")
    ledger = unique(csv_tables["metadata/test_challenging_view_selector_inputs.v2.csv"][1], "itemid")
    test_ids = {r["itemid"] for r in tests}
    if set(ledger) != test_ids:
        raise VersionError("Challenging selector coverage differs from test cases")
    selected = set()
    for i, row in ledger.items():
        apps = int(row["hudoc_application_number_count"])
        articles = int(row["distinct_violated_article_code_count"])
        grand = flag(row["source_is_grand_chamber"])
        if apps < 1 or articles < 1 or grand != flag(cases[i]["is_grand_chamber"]):
            raise VersionError("Invalid/disagreeing selector inputs")
        if articles != int(cases[i]["violated_articles_count"]):
            raise VersionError("Selector/canonical article-code count differs")
        if grand or (apps > 1 and articles > 1):
            selected.add(i)
    if selected != {r["itemid"] for r in tests if flag(r["test_challenging_view"])}:
        raise VersionError("Metadata selector does not reproduce canonical Challenging flags")
    members = unique(csv_tables["metadata/test_challenging_view.v2.csv"][1], "itemid")
    if set(members) != selected:
        raise VersionError("Challenging membership record differs")
    for i, row in members.items():
        if row["split"] != "test" or row["test_view"] != cases[i]["test_view"] or not flag(row["test_challenging_view"]):
            raise VersionError("Invalid Challenging membership row")
    for split, stem in [("train", "train"), ("validation", "val"), ("test", "test")]:
        subset = {i: r for i, r in cases.items() if r["split"] == split}
        if unique(csv_tables[f"data/{split}.csv"][1], "itemid") != subset:
            raise VersionError(f"Full split rows differ from canonical: {split}")
        fcols, frows = csv_tables[f"model_inputs/structured_tree/features/{stem}.csv"]
        _, trows = csv_tables[f"model_inputs/structured_tree/targets/{stem}.csv"]
        features, targets = unique(frows, "itemid"), unique(trows, "itemid")
        if set(features) != set(subset) or set(targets) != set(subset):
            raise VersionError(f"Feature/target coverage differs: {split}")
        if [r["itemid"] for r in frows] != [r["itemid"] for r in trows]:
            raise VersionError("Feature/target row order differs")
        if fcols != ["itemid", *manifest["structured_predictor_columns"]]:
            raise VersionError("Version-specific structured predictor order differs")
        for i, row in targets.items():
            for key in ("y_amount_eur", "y_binary"):
                if amount(row[key]) != amount(cases[i][key]):
                    raise VersionError("Model targets differ from selected canonical labels")
    _, index_rows = csv_tables["splits/case_index.csv"]
    index = unique(index_rows, "itemid")
    if set(index) != set(cases):
        raise VersionError("Split-index coverage differs")
    for i, row in index.items():
        for key, value in row.items():
            if key in cases[i] and cases[i][key] != value:
                raise VersionError("Split-index value differs from canonical")
    if version == "corrected":
        _, people = csv_tables["data/applicants.csv"]
        _, allocations = csv_tables["data/allocations.csv"]
        persons = unique(people, "applicant_id")
        unique(allocations, "allocation_id")
        for row in [*people, *allocations]:
            if row["itemid"] not in cases:
                raise VersionError("Applicant/allocation has unknown case")
        for row in allocations:
            link = row["applicant_id"]
            if link and (link not in persons or persons[link]["itemid"] != row["itemid"]):
                raise VersionError("Cross-case or absent person allocation link")
    if mode == "publish":
        raise VersionError("These internal versions are not approved public-upload bundles")
    return {"status": "PASS", "dataset_version": version, "release_id": manifest["release_id"],
            "revision_id": manifest["revision_id"], "files": len(files), "counts": counts,
            "source_labels_certified": False, "historical_runs_reproduced": False,
            "public_release_approved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], required=True)
    parser.add_argument("--mode", choices=["audit", "publish"], default="audit")
    a = parser.parse_args(argv)
    try:
        print(json.dumps(verify(a.dataset_root, a.dataset_version, mode=a.mode), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as e:
        parser.exit(1, f"FAIL: {e}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the ECtHR-NPD release candidate without heavyweight dependencies.

Run from the repository root:

    python3 scripts/smoke_test_release.py --dataset-release /path/to/dataset \
        --dataset-version corrected

Audit mode checks hashes, schemas, split/view counts, cross-file keys, the
external-factor input boundary, Hub viewer upload partitions, and the frozen
Challenging-view membership record.
It intentionally returns success for a structurally sound *blocked* release
candidate. Publish mode is an explicit gate and fails until every blocker in
the release manifest is resolved:

    python3 scripts/smoke_test_release.py --dataset-release /path/to/legacy/clean \
        --dataset-version paper_reference --mode publish
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "dataset_release"
MANIFEST_PATH = RELEASE / "metadata" / "RELEASE_MANIFEST.v2.json"


class ValidationError(ValueError):
    """A structural or integrity failure in the release candidate."""


def display_path(path: Path) -> str:
    return str(path)


def release_file(relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or Path(relative) == Path("."):
        raise ValidationError(f"Unsafe manifest/schema path: {relative!r}")
    path = RELEASE / relative
    if RELEASE.resolve() not in path.resolve().parents:
        raise ValidationError(f"Manifest/schema path escapes selected dataset: {relative}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON {display_path(path)}: {error}") from error
    if not isinstance(data, dict):
        raise ValidationError(f"JSON object required: {display_path(path)}")
    return data


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValidationError(f"CSV has no header: {display_path(path)}")
            rows = list(reader)
    except OSError as error:
        raise ValidationError(f"cannot read CSV {display_path(path)}: {error}") from error
    if len(set(reader.fieldnames)) != len(reader.fieldnames) or any(None in r or None in r.values() for r in rows):
        raise ValidationError(f"CSV has duplicate columns or malformed rows: {display_path(path)}")
    return list(reader.fieldnames), rows


def require(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"missing required file: {display_path(path)}")


def require_exact_columns(path: Path, observed: list[str], expected: list[str]) -> None:
    if observed != expected:
        raise ValidationError(
            f"schema mismatch for {display_path(path)}: expected {expected}, got {observed}"
        )


def unique_index(rows: list[dict[str, str]], key: str, path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        value = (row.get(key) or "").strip()
        if not value:
            raise ValidationError(f"blank {key} in {display_path(path)}")
        if value in index:
            duplicates.append(value)
        index[value] = row
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise ValidationError(f"duplicate {key} values in {display_path(path)}: {sample}")
    return index


def as_true(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def as_int(value: str | None, *, field: str, itemid: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as error:
        raise ValidationError(f"invalid integer {field}={value!r} for {itemid}") from error


def itemid_digest(itemids: set[str]) -> str:
    payload = "".join(f"{itemid}\n" for itemid in sorted(itemids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    required = {
        "manifest_version",
        "status",
        "schema",
        "integrity",
        "expected_data",
        "challenging_view",
        "upload_exclusions",
        "publication_blockers",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValidationError(f"manifest missing keys: {', '.join(missing)}")

    schema = manifest["schema"]
    integrity = manifest["integrity"]
    if not isinstance(schema, dict) or not isinstance(integrity, dict):
        raise ValidationError("manifest schema and integrity entries must be objects")
    if integrity.get("algorithm") != "sha256" or not isinstance(integrity.get("files"), dict):
        raise ValidationError("manifest integrity section must declare SHA-256 file hashes")
    if not isinstance(schema.get("path"), str) or not isinstance(schema.get("sha256"), str):
        raise ValidationError("manifest schema entry must provide path and SHA-256")
    return manifest


def validate_hashes(manifest: dict[str, Any]) -> dict[str, Any]:
    for relative, expected in manifest["integrity"]["files"].items():
        path = release_file(relative)
        require(path)
        actual = sha256(path)
        if actual != expected:
            raise ValidationError(
                f"SHA-256 mismatch for dataset_release/{relative}: expected {expected}, got {actual}"
            )

    schema_path = release_file(manifest["schema"]["path"])
    require(schema_path)
    if sha256(schema_path) != manifest["schema"]["sha256"]:
        raise ValidationError(f"SHA-256 mismatch for {display_path(schema_path)}")
    return load_json(schema_path)


def validate_canonical_and_index(manifest: dict[str, Any], schema: dict[str, Any]) -> dict[str, dict[str, str]]:
    canonical_spec = schema.get("canonical_table")
    if not isinstance(canonical_spec, dict):
        raise ValidationError("schema missing canonical_table")
    canonical_path = release_file(canonical_spec.get("path"))
    require(canonical_path)
    canonical_columns, canonical_rows = read_csv(canonical_path)
    require_exact_columns(canonical_path, canonical_columns, list(canonical_spec.get("columns") or []))

    expected = manifest["expected_data"]
    if len(canonical_rows) != expected["canonical_row_count"]:
        raise ValidationError(f"canonical row count {len(canonical_rows)} != {expected['canonical_row_count']}")
    if len(canonical_columns) != expected["canonical_column_count"]:
        raise ValidationError(
            f"canonical column count {len(canonical_columns)} != {expected['canonical_column_count']}"
        )
    cases = unique_index(canonical_rows, "itemid", canonical_path)

    split_counts = Counter(row["split"] for row in canonical_rows)
    if dict(split_counts) != expected["split_counts"]:
        raise ValidationError(f"split counts {dict(split_counts)} != {expected['split_counts']}")

    test_rows = [row for row in canonical_rows if row["split"] == "test"]
    if any(row["split"] != "test" and row["test_view"] for row in canonical_rows):
        raise ValidationError("non-test rows must have blank test_view")
    test_views = Counter(row["test_view"] for row in test_rows)
    if dict(test_views) != expected["test_view_counts"]:
        raise ValidationError(f"test-view counts {dict(test_views)} != {expected['test_view_counts']}")

    observed_flags = {
        itemid
        for itemid, row in cases.items()
        if row["split"] == "test" and as_true(row["test_challenging_view"])
    }
    if len(observed_flags) != expected["test_challenging_flag_count"]:
        raise ValidationError(
            f"Challenging flag count {len(observed_flags)} != {expected['test_challenging_flag_count']}"
        )

    index_path = RELEASE / "splits" / "case_index.csv"
    require(index_path)
    index_columns, index_rows = read_csv(index_path)
    expected_index_columns = [
        "itemid",
        "hudoc_url",
        "judgementdate",
        "chrono_date",
        "split",
        "test_view",
        "test_challenging_view",
    ]
    require_exact_columns(index_path, index_columns, expected_index_columns)
    index = unique_index(index_rows, "itemid", index_path)
    if set(index) != set(cases):
        raise ValidationError("case_index itemids do not match canonical table")
    for itemid, row in cases.items():
        counterpart = index[itemid]
        for field in ("hudoc_url", "judgementdate", "chrono_date", "split", "test_view", "test_challenging_view"):
            if counterpart[field] != row[field]:
                raise ValidationError(f"case_index mismatch for {itemid} field {field}")

    return cases


def validate_external_factors(manifest: dict[str, Any], schema: dict[str, Any], cases: dict[str, dict[str, str]]) -> None:
    spec = schema.get("external_factor_table")
    if not isinstance(spec, dict):
        raise ValidationError("schema missing external_factor_table")
    path = release_file(spec.get("path"))
    require(path)
    columns, rows = read_csv(path)
    require_exact_columns(path, columns, list(spec.get("columns") or []))
    prohibited = set(spec.get("prohibited_columns") or [])
    present = prohibited & set(columns)
    if present:
        raise ValidationError(f"prohibited external-factor columns present: {', '.join(sorted(present))}")
    if len(rows) != manifest["expected_data"]["external_factor_row_count"]:
        raise ValidationError(f"external-factor row count {len(rows)} is unexpected")
    external = unique_index(rows, "itemid", path)
    if set(external) != set(cases):
        raise ValidationError("external-factor itemids do not match canonical table")
    for itemid, row in external.items():
        case = cases[itemid]
        for field in columns:
            if field != "itemid" and row[field] != case[field]:
                raise ValidationError(f"external-factor mismatch for {itemid} field {field}")


def validate_structured_inputs(manifest: dict[str, Any], cases: dict[str, dict[str, str]]) -> None:
    expected = manifest["expected_data"]
    # The legacy input is frozen with its own pinned CSV, not today's model
    # profile (which deliberately removed two unverified allocation fields).
    pinned_train = "model_inputs/structured_tree/features/train.csv"
    if pinned_train not in manifest["integrity"]["files"]:
        raise ValidationError("Legacy train-feature schema is not integrity-pinned")
    train_path = release_file(pinned_train)
    if sha256(train_path) != manifest["integrity"]["files"][pinned_train]:
        raise ValidationError("Legacy train-feature schema hash mismatch")
    train_header, _ = read_csv(train_path)
    if not train_header or train_header[0] != "itemid":
        raise ValidationError("Legacy train-feature schema must start with itemid")
    expected_feature_columns = train_header[1:]
    if len(expected_feature_columns) != expected["structured_feature_count_excluding_itemid"]:
        raise ValidationError("tree feature schema count does not match the release manifest")
    expected_header = ["itemid", *expected_feature_columns]
    header: list[str] | None = None
    for file_split, canonical_split in (("train", "train"), ("val", "validation"), ("test", "test")):
        feature_path = RELEASE / "model_inputs" / "structured_tree" / "features" / f"{file_split}.csv"
        target_path = RELEASE / "model_inputs" / "structured_tree" / "targets" / f"{file_split}.csv"
        require(feature_path)
        require(target_path)
        feature_columns, feature_rows = read_csv(feature_path)
        target_columns, target_rows = read_csv(target_path)
        if header is None:
            header = feature_columns
        else:
            require_exact_columns(feature_path, feature_columns, header)
        require_exact_columns(feature_path, feature_columns, expected_header)
        require_exact_columns(target_path, target_columns, ["itemid", "y_amount_eur", "y_binary"])
        feature_index = unique_index(feature_rows, "itemid", feature_path)
        target_index = unique_index(target_rows, "itemid", target_path)
        if [row["itemid"] for row in feature_rows] != [row["itemid"] for row in target_rows]:
            raise ValidationError(f"structured feature/target itemid order differs: {file_split}")
        expected_ids = {itemid for itemid, row in cases.items() if row["split"] == canonical_split}
        if set(feature_index) != expected_ids:
            raise ValidationError(f"structured feature itemids do not match {canonical_split} split")
        if set(target_index) != expected_ids:
            raise ValidationError(f"structured target itemids do not match {canonical_split} split")
        for itemid, target in target_index.items():
            if target["y_amount_eur"] != cases[itemid]["y_amount_eur"] or target["y_binary"] != cases[itemid]["y_binary"]:
                raise ValidationError(f"structured target mismatch for {itemid}")


def validate_hub_viewer_package(
    manifest: dict[str, Any], schema: dict[str, Any], cases: dict[str, dict[str, str]]
) -> None:
    """Require lossless split files and explicit feature declarations for Hub upload."""
    contract = schema.get("hub_viewer_contract")
    canonical = schema.get("canonical_table")
    if not isinstance(contract, dict) or not isinstance(canonical, dict):
        raise ValidationError("schema lacks canonical_table or hub_viewer_contract")
    declared = contract.get("required_viewer_files")
    if not isinstance(declared, list) or not declared:
        raise ValidationError("hub_viewer_contract requires non-empty required_viewer_files")
    expected_columns = list(canonical.get("columns") or [])
    for file_spec in declared:
        if not isinstance(file_spec, dict):
            raise ValidationError("hub viewer file specification must be an object")
        split = file_spec.get("split")
        relative = file_spec.get("path")
        rows_expected = file_spec.get("expected_rows")
        if not isinstance(split, str) or not isinstance(relative, str) or not isinstance(rows_expected, int):
            raise ValidationError("hub viewer file specification is incomplete")
        path = release_file(relative)
        require(path)
        columns, rows = read_csv(path)
        require_exact_columns(path, columns, expected_columns)
        if len(rows) != rows_expected:
            raise ValidationError(f"Hub viewer {split} row count {len(rows)} != {rows_expected}")
        partition = unique_index(rows, "itemid", path)
        expected_ids = {itemid for itemid, row in cases.items() if row["split"] == split}
        if set(partition) != expected_ids:
            raise ValidationError(f"Hub viewer {split} IDs do not equal canonical partition")
        for itemid, row in partition.items():
            if row != cases[itemid]:
                raise ValidationError(f"Hub viewer {split} row differs from canonical table: {itemid}")

    readme_path = RELEASE / "README.md"
    require(readme_path)
    readme = readme_path.read_text(encoding="utf-8")
    for token in ("config_name: default", "path: data/train.csv", "path: data/validation.csv", "path: data/test.csv"):
        if token not in readme:
            raise ValidationError(f"Hub viewer README declaration missing {token!r}")
    for column in expected_columns:
        if f"name: {column}" not in readme:
            raise ValidationError(f"Hub viewer README features lack canonical column {column!r}")


def validate_challenging_view_v2(manifest: dict[str, Any], cases: dict[str, dict[str, str]]) -> None:
    """Recompute the frozen 699-row view from the public count-only v2 ledger."""
    spec = manifest["challenging_view"]
    required = {
        "status",
        "view_version",
        "selector_ledger",
        "membership_record",
        "provenance_record",
        "expected_selector_ledger_row_count",
        "expected_membership_count",
        "expected_membership_itemid_sha256",
        "expected_reason_counts",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValidationError("challenging_view missing keys: " + ", ".join(missing))
    if spec["status"] != "RESOLVED_METADATA_OPERATIONALIZATION" or spec["view_version"] != "v2":
        raise ValidationError("unexpected Challenging-view v2 status or version")

    ledger_path = release_file(spec["selector_ledger"])
    membership_path = release_file(spec["membership_record"])
    record_path = release_file(spec["provenance_record"])
    require(ledger_path)
    require(membership_path)
    require(record_path)
    ledger_columns, ledger_rows = read_csv(ledger_path)
    require_exact_columns(
        ledger_path,
        ledger_columns,
        [
            "itemid",
            "hudoc_application_number_count",
            "distinct_violated_article_code_count",
            "source_is_grand_chamber",
        ],
    )
    if len(ledger_rows) != spec["expected_selector_ledger_row_count"]:
        raise ValidationError("unexpected Challenging selector-ledger row count")
    ledger = unique_index(ledger_rows, "itemid", ledger_path)
    test_cases = {itemid: row for itemid, row in cases.items() if row["split"] == "test"}
    if set(ledger) != set(test_cases):
        raise ValidationError("Challenging selector ledger itemids do not equal canonical test IDs")

    selected: set[str] = set()
    expected_reasons: dict[str, str] = {}
    for itemid, row in ledger.items():
        application_count = as_int(
            row["hudoc_application_number_count"], field="hudoc_application_number_count", itemid=itemid
        )
        article_count = as_int(
            row["distinct_violated_article_code_count"], field="distinct_violated_article_code_count", itemid=itemid
        )
        if application_count < 1 or article_count < 1:
            raise ValidationError(f"invalid non-positive Challenging selector count for {itemid}")
        case = test_cases[itemid]
        if article_count != as_int(case["violated_articles_count"], field="violated_articles_count", itemid=itemid):
            raise ValidationError(f"selector article-code count disagrees with canonical table for {itemid}")
        source_grand = as_true(row["source_is_grand_chamber"])
        if source_grand != as_true(case["is_grand_chamber"]):
            raise ValidationError(f"selector Grand-Chamber flag disagrees with canonical table for {itemid}")
        multi = application_count > 1 and article_count > 1
        if source_grand or multi:
            selected.add(itemid)
            expected_reasons[itemid] = (
                "grand_chamber;multi_hudoc_appno_and_distinct_violated_article_codes"
                if source_grand and multi
                else "grand_chamber"
                if source_grand
                else "multi_hudoc_appno_and_distinct_violated_article_codes"
            )

    canonical_flags = {
        itemid for itemid, row in test_cases.items() if as_true(row["test_challenging_view"])
    }
    if selected != canonical_flags:
        raise ValidationError("v2 selector ledger does not reproduce canonical Challenging flags")
    if len(selected) != spec["expected_membership_count"]:
        raise ValidationError("unexpected Challenging selected-member count")
    if itemid_digest(selected) != spec["expected_membership_itemid_sha256"]:
        raise ValidationError("Challenging member itemid digest mismatch")

    member_columns, member_rows = read_csv(membership_path)
    require_exact_columns(
        membership_path,
        member_columns,
        ["itemid", "split", "test_view", "test_challenging_view", "challenge_reason", "view_version"],
    )
    membership = unique_index(member_rows, "itemid", membership_path)
    if set(membership) != selected:
        raise ValidationError("v2 membership record IDs do not match the recomputed selector")
    reason_counts: Counter[str] = Counter()
    for itemid, row in membership.items():
        case = test_cases[itemid]
        if (
            row["split"] != "test"
            or row["test_view"] != case["test_view"]
            or not as_true(row["test_challenging_view"])
            or row["view_version"] != "v2"
            or row["challenge_reason"] != expected_reasons[itemid]
        ):
            raise ValidationError(f"v2 membership row does not match canonical/selector data for {itemid}")
        reason_counts[row["challenge_reason"]] += 1
    if dict(reason_counts) != spec["expected_reason_counts"]:
        raise ValidationError("v2 Challenging reason counts do not match the manifest")

    record = load_json(record_path)
    if record.get("view_version") != "v2" or record.get("n") != len(selected):
        raise ValidationError("v2 Challenging provenance record has unexpected version or member count")
    if record.get("membership", {}).get("canonical_itemid_sha256") != itemid_digest(selected):
        raise ValidationError("v2 Challenging provenance record itemid digest mismatch")
    if record.get("membership", {}).get("reason_counts") != dict(reason_counts):
        raise ValidationError("v2 Challenging provenance record reason counts mismatch")


def validate_upload_exclusions(manifest: dict[str, Any], schema: dict[str, Any]) -> None:
    """Ensure the explicit Hub package contract cannot accidentally include audit-only files."""
    exclusions = manifest["upload_exclusions"]
    if not isinstance(exclusions, list):
        raise ValidationError("upload_exclusions must be a list")
    excluded_paths: set[str] = set()
    for artifact in exclusions:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValidationError("upload_exclusions entries require a path")
        excluded_paths.add(str(artifact["path"]))
        release_file(artifact["path"])
    included = set(manifest["integrity"]["files"])
    if included & excluded_paths:
        raise ValidationError("upload exclusions overlap integrity-listed files")
    required_metadata = set(schema.get("hub_viewer_contract", {}).get("required_metadata", []))
    for relative in required_metadata:
        release_file(relative)
    if required_metadata & excluded_paths:
        raise ValidationError("upload exclusions overlap Hub required metadata")


def publication_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("status") != "READY_FOR_PUBLICATION":
        errors.append(f"manifest status is {manifest.get('status')!r}, not 'READY_FOR_PUBLICATION'")
    open_blockers = [
        str(blocker.get("id") or "unnamed")
        for blocker in manifest["publication_blockers"]
        if str(blocker.get("status") or "").upper() != "RESOLVED"
    ]
    if open_blockers:
        errors.append("open publication blockers: " + ", ".join(open_blockers))
    return errors


def run(mode: str, *, dataset_release: Path, dataset_version: str) -> int:
    global RELEASE, MANIFEST_PATH
    from verify_dataset_version import resolve_version, verify
    if mode not in {"audit", "publish"}:
        raise ValidationError("Unsupported audit mode")
    RELEASE = resolve_version(dataset_release, dataset_version)
    if (RELEASE / "VERSION_MANIFEST.json").is_file():
        print(json.dumps(verify(RELEASE, dataset_version, mode=mode), indent=2))
        return 0
    if dataset_version != "paper_reference":
        raise ValidationError("Legacy release is paper_reference only; corrected version manifest required")
    MANIFEST_PATH = RELEASE / "metadata" / "RELEASE_MANIFEST.v2.json"
    require(MANIFEST_PATH)
    manifest = validate_manifest(load_json(MANIFEST_PATH))
    schema = validate_hashes(manifest)
    cases = validate_canonical_and_index(manifest, schema)
    validate_external_factors(manifest, schema, cases)
    validate_structured_inputs(manifest, cases)
    validate_hub_viewer_package(manifest, schema, cases)
    validate_challenging_view_v2(manifest, cases)
    validate_upload_exclusions(manifest, schema)

    blockers = publication_errors(manifest)
    print("PASS: release integrity audit")
    print(f"  canonical rows: {manifest['expected_data']['canonical_row_count']}")
    print(f"  Challenging flag rows: {manifest['expected_data']['test_challenging_flag_count']}")
    print(f"  Challenging v2 selector rows: {manifest['challenging_view']['expected_selector_ledger_row_count']}")
    if blockers:
        print("  publication status: BLOCKED")
        for blocker in blockers:
            print(f"    - {blocker}")
    else:
        print("  publication status: READY")
    if mode == "publish" and blockers:
        raise ValidationError("publish gate failed; resolve the blockers above before upload")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-release", type=Path, required=True, help="Dataset root or explicitly selected clean directory; no fallback")
    parser.add_argument("--dataset-version", choices=("corrected", "paper_reference"), required=True)
    parser.add_argument(
        "--mode",
        choices=("audit", "publish"),
        default="audit",
        help="audit validates integrity; publish additionally enforces release-clearance decisions",
    )
    args = parser.parse_args(argv)
    try:
        return run(args.mode, dataset_release=args.dataset_release, dataset_version=args.dataset_version)
    except (ValidationError, ValueError, OSError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the auditable ECtHR-NPD Challenging-view v2 release records.

The submitted-study view is an evaluation-only, overlapping test subset. Its
historical metadata selector is deliberately *not* the public person-level
``num_applicants`` or finding-level ``num_violations_found`` proxy:

    Grand Chamber OR (multiple HUDOC application numbers AND multiple
    distinct recorded violated-article codes).

This utility accepts the frozen HUDOC metadata export used by the study and
writes a privacy-minimised selector ledger (counts only) plus a versioned
membership record. It never writes raw application numbers into the release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICALIZATION = "UTF-8, lexicographically sorted itemid values, one itemid per line, trailing newline"
VIEW_VERSION = "v2"
LEDGER_NAME = "test_challenging_view_selector_inputs.v2.csv"
MEMBERSHIP_NAME = "test_challenging_view.v2.csv"
RECORD_NAME = "test_challenging_view.v2.json"

REASON_GRAND = "grand_chamber"
REASON_MULTI = "multi_hudoc_appno_and_distinct_violated_article_codes"
REASON_BOTH = f"{REASON_GRAND};{REASON_MULTI}"


class ViewBuildError(ValueError):
    """Raised when frozen source metadata does not support the current view."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def itemid_digest(itemids: set[str]) -> str:
    payload = "".join(f"{itemid}\n" for itemid in sorted(itemids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "itemid" not in reader.fieldnames:
            raise ViewBuildError(f"{path} must be a CSV with an itemid column")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            itemid = str(row.get("itemid") or "").strip()
            if not itemid:
                raise ViewBuildError(f"{path} has a blank itemid")
            if itemid in rows:
                raise ViewBuildError(f"{path} has duplicate itemid {itemid!r}")
            rows[itemid] = {str(key): str(value or "") for key, value in row.items()}
    return rows


def semicolon_tokens(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def source_is_grand_chamber(row: dict[str, str]) -> bool:
    return row.get("decision_body_category", "").strip() == "Grand Chamber" or "GRAND" in row.get(
        "doctypebranch", ""
    ).upper()


def as_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def require_source_columns(source_rows: dict[str, dict[str, str]]) -> None:
    sample = next(iter(source_rows.values()), None)
    if sample is None:
        raise ViewBuildError("source metadata has no rows")
    required = {"appno", "violated_articles", "decision_body_category", "doctypebranch"}
    missing = sorted(required - set(sample))
    if missing:
        raise ViewBuildError("source metadata lacks required columns: " + ", ".join(missing))


def build_records(
    canonical_rows: dict[str, dict[str, str]], source_rows: dict[str, dict[str, str]],
    *, expected_counts: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    test_rows = {itemid: row for itemid, row in canonical_rows.items() if row.get("split") == "test"}
    expected_test = expected_counts["splits"]["test"]
    expected_challenging = expected_counts["challenging"]
    if len(test_rows) != expected_test:
        raise ViewBuildError(f"expected {expected_test} test rows, found {len(test_rows)}")
    require_source_columns(source_rows)
    missing = sorted(set(test_rows) - set(source_rows))
    if missing:
        raise ViewBuildError("source metadata is missing test itemids: " + ", ".join(missing[:10]))

    ledger: list[dict[str, str]] = []
    membership: list[dict[str, str]] = []
    selected: set[str] = set()
    expected_flags = {
        itemid for itemid, row in test_rows.items() if as_true(row.get("test_challenging_view", ""))
    }
    reasons: Counter[str] = Counter()
    source_grand_disagreements: list[str] = []

    for itemid in sorted(test_rows):
        canonical = test_rows[itemid]
        source = source_rows[itemid]
        raw_codes = semicolon_tokens(source["violated_articles"])
        distinct_codes = sorted(set(raw_codes))
        canonical_codes = sorted(set(semicolon_tokens(canonical.get("violated_articles", ""))))
        if canonical_codes != distinct_codes:
            raise ViewBuildError(
                f"violated-article codes do not match source metadata for {itemid}: "
                f"canonical={canonical_codes}, source={distinct_codes}"
            )
        appno_count = len(set(semicolon_tokens(source["appno"])))
        if appno_count < 1:
            raise ViewBuildError(f"source metadata contains no HUDOC application number for {itemid}")
        is_grand = source_is_grand_chamber(source)
        if is_grand != as_true(canonical.get("is_grand_chamber", "")):
            source_grand_disagreements.append(itemid)
        multi = appno_count > 1 and len(distinct_codes) > 1
        selected_now = is_grand or multi
        ledger.append(
            {
                "itemid": itemid,
                "hudoc_application_number_count": str(appno_count),
                "distinct_violated_article_code_count": str(len(distinct_codes)),
                "source_is_grand_chamber": str(is_grand),
            }
        )
        if selected_now:
            selected.add(itemid)
            reason = REASON_BOTH if is_grand and multi else REASON_GRAND if is_grand else REASON_MULTI
            reasons[reason] += 1
            membership.append(
                {
                    "itemid": itemid,
                    "split": "test",
                    "test_view": canonical["test_view"],
                    "test_challenging_view": "True",
                    "challenge_reason": reason,
                    "view_version": VIEW_VERSION,
                }
            )

    if source_grand_disagreements:
        raise ViewBuildError(
            "source Grand Chamber determination disagrees with canonical is_grand_chamber for: "
            + ", ".join(source_grand_disagreements[:10])
        )
    if selected != expected_flags:
        only_selector = sorted(selected - expected_flags)
        only_flags = sorted(expected_flags - selected)
        raise ViewBuildError(
            "historical selector does not reproduce canonical Challenging flags; "
            f"selector_only={only_selector[:10]}, flag_only={only_flags[:10]}"
        )
    if len(selected) != expected_challenging:
        raise ViewBuildError(f"selector produced {len(selected)} rows, not version-contract count {expected_challenging}")
    expected_reason_counts = expected_counts.get("challenging_reason_counts")
    if expected_reason_counts is not None and dict(reasons) != expected_reason_counts:
        raise ViewBuildError(f"unexpected Challenging reason counts: {dict(reasons)}")

    summary = {
        "test_row_count": len(test_rows),
        "selected_count": len(selected),
        "selected_itemid_sha256": itemid_digest(selected),
        "reason_counts": {key: reasons[key] for key in sorted(reasons)},
    }
    return ledger, membership, summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_release_records(
    release: Path,
    source_metadata: Path,
    output_dir: Path,
    ledger: list[dict[str, str]],
    membership: list[dict[str, str]],
    summary: dict[str, Any],
) -> None:
    release = release.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    # Never rewrite historical/corrected package metadata, including via ../
    # or symlink aliases. A clean directory's enclosing dataset stays protected.
    protected = release.parent if release.name == "clean" else release
    if protected.name == "paper_reference":
        protected = protected.parent
    if output_dir == protected or protected in output_dir.parents:
        raise ViewBuildError("Output must be outside the selected dataset package")
    if output_dir.exists():
        raise ViewBuildError("Output must be a fresh, nonexistent external directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = output_dir / LEDGER_NAME
    membership_path = output_dir / MEMBERSHIP_NAME
    record_path = output_dir / RECORD_NAME
    write_csv(
        ledger_path,
        [
            "itemid",
            "hudoc_application_number_count",
            "distinct_violated_article_code_count",
            "source_is_grand_chamber",
        ],
        ledger,
    )
    write_csv(
        membership_path,
        ["itemid", "split", "test_view", "test_challenging_view", "challenge_reason", "view_version"],
        membership,
    )
    canonical_path = release / "data" / "ecthr_npd_cases.csv"
    record = {
        "record_type": "versioned Challenging-view selector and membership record",
        "view_name": "test_challenging_view",
        "view_version": VIEW_VERSION,
        "base_population": "locked test split",
        "overlapping": True,
        "n": summary["selected_count"],
        "dataset_version": summary.get("dataset_version"),
        "release_id": summary.get("release_id"),
        "application_count_definition": "distinct nonempty semicolon-delimited HUDOC application numbers",
        "selector_logic": {
            "operator": "OR",
            "grand_chamber": "source decision_body_category == 'Grand Chamber' OR 'GRAND' in source doctypebranch.upper()",
            "multi_hudoc_appno_and_distinct_violated_article_codes": "hudoc_application_number_count > 1 AND distinct_violated_article_code_count > 1",
        },
        "terminology_mapping": {
            "multiple_applicants": "HUDOC application-number multiplicity in the study metadata; not the public person-level num_applicants field",
            "multiple_violations": "distinct recorded violated-article-code multiplicity; not the public finding-level num_violations_found field",
        },
        "evaluation_only": {
            "selector_ledger_is_not_a_model_input": True,
            "test_challenging_view_is_not_a_model_input": True,
            "raw_application_numbers_are_not_redistributed": True,
        },
        "source_metadata": {
            "file_name": source_metadata.name,
            "sha256": sha256(source_metadata),
            "required_source_columns": ["itemid", "appno", "violated_articles", "decision_body_category", "doctypebranch"],
            "public_derivative": LEDGER_NAME,
        },
        "generator": {
            "path": "scripts/build_challenging_view_v2.py",
            "sha256": sha256(Path(__file__)),
        },
        "canonical_table": {
            "path": "data/ecthr_npd_cases.csv",
            "sha256": sha256(canonical_path),
            "flag_filter": "split == 'test' AND test_challenging_view == True",
        },
        "selector_ledger": {
            "path": LEDGER_NAME,
            "sha256": sha256(ledger_path),
            "row_count": summary["test_row_count"],
        },
        "membership": {
            "path": MEMBERSHIP_NAME,
            "sha256": sha256(membership_path),
            "canonical_itemid_sha256": summary["selected_itemid_sha256"],
            "canonicalization": CANONICALIZATION,
            "reason_counts": summary["reason_counts"],
        },
    }
    with record_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ECtHR-NPD Challenging-view v2 metadata records")
    parser.add_argument("--source-metadata", required=True, help="Frozen HUDOC metadata CSV containing appno and violated_articles")
    parser.add_argument("--dataset-release", required=True, help="Dataset root or selected clean directory")
    parser.add_argument("--dataset-version", required=True, choices=("corrected", "paper_reference"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", help="Fresh external output directory; never writes selected dataset metadata")
    action.add_argument("--check-only", action="store_true", help="Validate source metadata against canonical flags without writing files")
    args = parser.parse_args(argv)

    from verify_dataset_version import resolve_version
    from smoke_test_release import run
    run("audit", dataset_release=Path(args.dataset_release), dataset_version=args.dataset_version)
    release = resolve_version(Path(args.dataset_release), args.dataset_version)
    source_metadata = Path(args.source_metadata).expanduser().resolve()
    canonical_path = release / "data" / "ecthr_npd_cases.csv"
    if not canonical_path.is_file():
        raise ViewBuildError(f"missing canonical table: {canonical_path}")
    if not source_metadata.is_file():
        raise ViewBuildError(f"missing source metadata: {source_metadata}")
    contract_path = release / "release_contract.json"
    if (release / "VERSION_MANIFEST.json").is_file():
        contract = json.loads(contract_path.read_text())
        counts = contract["counts"]
        release_id = contract["release_id"]
    else:
        manifest = json.loads((release / "metadata/RELEASE_MANIFEST.v2.json").read_text())
        counts = {"splits": manifest["expected_data"]["split_counts"],
                  "challenging": manifest["expected_data"]["test_challenging_flag_count"],
                  "challenging_reason_counts": manifest["challenging_view"]["expected_reason_counts"]}
        release_id = manifest["release_candidate_id"]
    ledger, membership, summary = build_records(read_csv(canonical_path), read_csv(source_metadata), expected_counts=counts)
    summary.update(dataset_version=args.dataset_version, release_id=release_id)
    if not args.check_only:
        write_release_records(release, source_metadata, Path(args.output_dir), ledger, membership, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"VIEW BUILD REFUSED: {error}") from error

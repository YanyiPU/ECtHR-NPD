#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import argparse
import sys
from pathlib import Path
from statistics import median


KB_DIR = Path(__file__).resolve().parent
if str(KB_DIR) not in sys.path:
    sys.path.insert(0, str(KB_DIR))
from prior_contract import TABLE_NAMES, articles as normalise_prior_articles, cohort_pin, file_sha256, read_training_inputs, verify_prior_bundle, bind_selected_training_inputs
PACKAGE_ROOT = KB_DIR.parent
EMPIRICAL_DIR = Path(__file__).resolve().parent / "modules" / "empirical"
ARTICLE_OUT = EMPIRICAL_DIR / "article_award_distribution_train.csv"
COUNTRY_OUT = EMPIRICAL_DIR / "country_award_distribution_train.csv"
ARTICLE_COUNTRY_OUT = EMPIRICAL_DIR / "article_country_award_distribution_train.csv"
LEGACY_ARTICLE_OUT = EMPIRICAL_DIR / "article_single_violation_stats_train.csv"
MANIFEST_OUT = EMPIRICAL_DIR / "PRIOR_MANIFEST.v1.json"
PROVENANCE_COUNT_FIELDS = (
    "zero_reason_count",
    "amount_direct_count",
    "proxy_keep72_count",
)
PROVENANCE_SOURCE_VALUES = {
    "zero_reason",
    "amount_direct",
    "proxy_keep72",
}


def parse_float(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_articles(value: object) -> list[str]:
    return normalise_prior_articles(value)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row.get("itemid") or ""): row for row in csv.DictReader(handle)}


def normalize_country(row: dict[str, str]) -> str:
    return str(row.get("country_alpha2") or row.get("respondent_country") or "").strip().upper()


def provenance_status(sources: list[str | None], missing_source_status: str) -> str:
    """State whether every contributing label supplied a usable y_source value."""
    if not sources:
        return "unavailable_no_labeled_rows"
    if any(not source for source in sources):
        # A complete supplied table can still have an unmatched contributing row.
        return (
            missing_source_status
            if missing_source_status != "available_from_supplied_y_source"
            else "unavailable_missing_y_source"
        )
    if any(source not in PROVENANCE_SOURCE_VALUES for source in sources):
        return "unavailable_unrecognized_y_source_values"
    return "available_from_supplied_y_source"


def bucket_record(
    bucket_id: str,
    values: list[float],
    sources: list[str | None],
    missing_source_status: str,
) -> dict[str, object]:
    positives = [value for value in values if value > 0]
    amount_values = positives or values
    p25 = percentile(amount_values, 0.25)
    p75 = percentile(amount_values, 0.75)
    zero_count = sum(1 for value in values if value <= 0)
    source_status = provenance_status(sources, missing_source_status)
    record: dict[str, object] = {
        "bucket": bucket_id,
        "sample_count": len(values),
        "positive_count": len(positives),
        "zero_count": zero_count,
        "zero_rate": round(zero_count / len(values), 6),
        "median_positive_or_all": round(float(median(amount_values)), 6),
        "iqr_positive_or_all": round(p75 - p25, 6),
        "p10_positive_or_all": round(percentile(amount_values, 0.10), 6),
        "p90_positive_or_all": round(percentile(amount_values, 0.90), 6),
        "median_all": round(float(median(values)), 6),
        "p10_all": round(percentile(values, 0.10), 6),
        "p90_all": round(percentile(values, 0.90), 6),
        "label_provenance_status": source_status,
    }
    if source_status == "available_from_supplied_y_source":
        record.update(
            {
                "zero_reason_count": sum(1 for source in sources if source == "zero_reason"),
                "amount_direct_count": sum(1 for source in sources if source == "amount_direct"),
                "proxy_keep72_count": sum(1 for source in sources if source == "proxy_keep72"),
            }
        )
    else:
        # Empty CSV cells mean unknown, never a zero historical count.
        record.update({field: None for field in PROVENANCE_COUNT_FIELDS})
    return record


def write_distribution(
    path: Path,
    key_fields: list[str],
    buckets: dict[tuple[str, ...], list[tuple[float, str | None]]],
    missing_source_status: str,
) -> set[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    emitted_statuses: set[str] = set()
    fieldnames = [
        *key_fields,
        "sample_count",
        "positive_count",
        "zero_count",
        "zero_rate",
        "median_positive_or_all",
        "iqr_positive_or_all",
        "p10_positive_or_all",
        "p90_positive_or_all",
        "median_all",
        "p10_all",
        "p90_all",
        "label_provenance_status",
        *PROVENANCE_COUNT_FIELDS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(buckets):
            values = [value for value, _ in buckets[key]]
            sources = [source for _, source in buckets[key]]
            record = bucket_record("|".join(key), values, sources, missing_source_status)
            emitted_statuses.add(str(record["label_provenance_status"]))
            row = {field: key[idx] for idx, field in enumerate(key_fields)}
            row.update({field: record[field] for field in fieldnames if field not in key_fields})
            writer.writerow(row)
    return emitted_statuses


def write_legacy_article_distribution(article_buckets: dict[tuple[str, ...], list[tuple[float, str | None]]], path: Path = LEGACY_ARTICLE_OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["article", "sample_count", "zero_rate", "median", "iqr", "p10", "p90"],
        )
        writer.writeheader()
        for (article,) in sorted(article_buckets):
            values = [value for value, _ in article_buckets[(article,)]]
            positives = [value for value in values if value > 0]
            amount_values = positives or values
            p25 = percentile(amount_values, 0.25)
            p75 = percentile(amount_values, 0.75)
            writer.writerow(
                {
                    "article": article,
                    "sample_count": len(values),
                    "zero_rate": round(sum(1 for value in values if value <= 0) / len(values), 6),
                    "median": round(float(median(amount_values)), 6),
                    "iqr": round(p75 - p25, 6),
                    "p10": round(percentile(amount_values, 0.10), 6),
                    "p90": round(percentile(amount_values, 0.90), 6),
                }
            )


def label_table_provenance_status(labels: dict[str, dict[str, str]]) -> str:
    if not labels:
        return "unavailable_no_train_labels"
    if not any("y_source" in row for row in labels.values()):
        return "unavailable_y_source_column_absent"
    if any(not str(row.get("y_source") or "").strip() for row in labels.values()):
        return "unavailable_y_source_incomplete"
    if any(str(row.get("y_source") or "").strip() not in PROVENANCE_SOURCE_VALUES for row in labels.values()):
        return "unavailable_unrecognized_y_source_values"
    return "available_from_supplied_y_source"


def write_manifest(
    *,
    using_default_train_labels: bool,
    input_label_status: str,
    emitted_bucket_statuses: set[str],
    output_dir: Path,
    case_rows_path: Path,
    target_rows_path: Path,
    dataset_release: Path,
    dataset_provenance: dict | None = None,
) -> None:
    """Record the exact public-only reconstruction boundary for prior tables."""
    table_status = (
        next(iter(emitted_bucket_statuses))
        if len(emitted_bucket_statuses) == 1
        else "mixed_bucket_provenance_status"
    )
    manifest = {
        "manifest_version": "2.0.0",
        "artifact_scope": "train_only_empirical_priors",
        "builder": "agent_knowledge_base/build_train_article_priors.py",
        "input_mode": "selected_version_targets" if using_default_train_labels else "version_bound_user_supplied_train_labels",
        "public_default_inputs": {
            "case_rows": "<selected-clean-root>/data/ecthr_npd_cases.csv",
            "train_targets": "<selected-clean-root>/model_inputs/structured_tree/targets/train.csv",
            "target_fields_used_for_amount_statistics": ["itemid", "y_amount_eur"],
        },
        "label_provenance": {
            "field": "y_source",
            "recognized_values": sorted(PROVENANCE_SOURCE_VALUES),
            "input_label_schema_status": input_label_status,
            "table_status": table_status,
            "emitted_bucket_statuses": sorted(emitted_bucket_statuses),
            "public_default_target_schema": "does_not_include_y_source",
            "count_fields": list(PROVENANCE_COUNT_FIELDS),
            "count_field_rule": (
                "Counts are populated only when every contributing input label has a non-empty "
                "y_source in the recognized taxonomy. Otherwise they are blank and must be treated "
                "as unavailable, not zero."
            ),
            "reconstruction_boundary": (
                "Public default targets can reproduce amount and zero-rate statistics, but cannot "
                "reproduce historical label-source/provenance counts."
            ),
        },
        "tables": [
            ARTICLE_OUT.name,
            COUNTRY_OUT.name,
            ARTICLE_COUNTRY_OUT.name,
            LEGACY_ARTICLE_OUT.name,
        ],
    }
    contract_path = dataset_release / "release_contract.json"
    manifest["dataset_release_id"] = (json.loads(contract_path.read_text()).get("release_id")
                                      if contract_path.exists() else "unversioned_explicit_inputs")
    manifest["dataset_version"] = (dataset_provenance or {}).get("dataset_version", "unversioned_explicit_inputs")
    manifest["dataset_provenance"] = dataset_provenance
    manifest["cohort_pin"] = cohort_pin(case_rows_path, target_rows_path)
    manifest["input_files"] = {
        "case_rows": {"path": str(case_rows_path), "sha256": file_sha256(case_rows_path)},
        "train_targets": {"path": str(target_rows_path), "sha256": file_sha256(target_rows_path)},
    }
    manifest["artifact_sha256"] = {name: file_sha256(output_dir / name) for name in TABLE_NAMES}
    manifest["builder_source_sha256"] = {"build_train_article_priors.py": file_sha256(Path(__file__)),
                                         "prior_contract.py": file_sha256(KB_DIR / "prior_contract.py")}
    manifest["scope_boundary"] = "Rebuilds article, country, article-country and legacy-article statistics only; not the full normative/external KB."
    manifest["training_id_policy"] = "exact train-only case/target coverage; no case-row label fallback"
    (output_dir / MANIFEST_OUT.name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only empirical priors from a dataset_release directory.")
    parser.add_argument("--dataset-release", type=Path, default=None)
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], default="corrected")
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--case-rows", type=Path, default=None, help="Full case table or explicit train-only case table")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh candidate directory; never overwrite active/frozen priors")
    args = parser.parse_args()

    dataset_release, train_cases, train_labels, provenance = bind_selected_training_inputs(
        args.dataset_release, args.case_rows, args.train_labels, dataset_version=args.dataset_version)
    if args.output_dir.resolve() == EMPIRICAL_DIR.resolve():
        parser.error("active/frozen empirical directory cannot be a build destination; use a fresh candidate")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty; existing prior bundles are immutable")
    cases_by_id, labels = read_training_inputs(train_cases, train_labels)
    status = label_table_provenance_status(labels)
    article_buckets: dict[tuple[str, ...], list[tuple[float, str | None]]] = {}
    country_buckets: dict[tuple[str, ...], list[tuple[float, str | None]]] = {}
    article_country_buckets: dict[tuple[str, ...], list[tuple[float, str | None]]] = {}

    for itemid, row in cases_by_id.items():
        label = labels[itemid]
        amount = float(label["y_amount_eur"])
        source = str((label or {}).get("y_source") or "").strip() or None
        articles = split_articles(row.get("violated_articles"))
        if not articles:
            continue
        country = normalize_country(row)
        if country:
            country_buckets.setdefault((country,), []).append((amount, source))
        for article in articles:
            article_buckets.setdefault((article,), []).append((amount, source))
            if country:
                article_country_buckets.setdefault((article, country), []).append((amount, source))

    emitted_bucket_statuses = set()
    emitted_bucket_statuses.update(write_distribution(args.output_dir / ARTICLE_OUT.name, ["article"], article_buckets, status))
    emitted_bucket_statuses.update(write_distribution(args.output_dir / COUNTRY_OUT.name, ["country_alpha2"], country_buckets, status))
    emitted_bucket_statuses.update(
        write_distribution(args.output_dir / ARTICLE_COUNTRY_OUT.name, ["article", "country_alpha2"], article_country_buckets, status)
    )
    write_legacy_article_distribution(article_buckets, args.output_dir / LEGACY_ARTICLE_OUT.name)
    write_manifest(
        using_default_train_labels=args.train_labels is None,
        input_label_status=status,
        emitted_bucket_statuses=emitted_bucket_statuses,
        output_dir=args.output_dir,
        case_rows_path=train_cases,
        target_rows_path=train_labels,
        dataset_release=dataset_release,
        dataset_provenance=provenance,
    )
    verified = verify_prior_bundle(args.output_dir, train_cases, train_labels)
    print(json.dumps({"output_dir": str(args.output_dir), "verification": verified,
                      "label_provenance_status": status}, indent=2))


if __name__ == "__main__":
    main()

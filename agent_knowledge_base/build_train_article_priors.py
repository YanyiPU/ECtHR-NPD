#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import argparse
from pathlib import Path
from statistics import median


KB_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = KB_DIR.parent
DEFAULT_DATASET_RELEASE = Path(os.environ.get("ECTHR_NPD_DATASET_RELEASE", str(PACKAGE_ROOT / "dataset_release")))
EMPIRICAL_DIR = Path(__file__).resolve().parent / "modules" / "empirical"
ARTICLE_OUT = EMPIRICAL_DIR / "article_award_distribution_train.csv"
COUNTRY_OUT = EMPIRICAL_DIR / "country_award_distribution_train.csv"
ARTICLE_COUNTRY_OUT = EMPIRICAL_DIR / "article_country_award_distribution_train.csv"
LEGACY_ARTICLE_OUT = EMPIRICAL_DIR / "article_single_violation_stats_train.csv"


def parse_float(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_articles(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip().upper() for part in text.split(";") if part.strip()]


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


def bucket_record(bucket_id: str, values: list[float], sources: list[str]) -> dict[str, object]:
    positives = [value for value in values if value > 0]
    amount_values = positives or values
    p25 = percentile(amount_values, 0.25)
    p75 = percentile(amount_values, 0.75)
    zero_count = sum(1 for value in values if value <= 0)
    return {
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
        "zero_reason_count": sum(1 for source in sources if source == "zero_reason"),
        "amount_direct_count": sum(1 for source in sources if source == "amount_direct"),
        "proxy_keep72_count": sum(1 for source in sources if source == "proxy_keep72"),
    }


def write_distribution(path: Path, key_fields: list[str], buckets: dict[tuple[str, ...], list[tuple[float, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "zero_reason_count",
        "amount_direct_count",
        "proxy_keep72_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(buckets):
            values = [value for value, _ in buckets[key]]
            sources = [source for _, source in buckets[key]]
            record = bucket_record("|".join(key), values, sources)
            row = {field: key[idx] for idx, field in enumerate(key_fields)}
            row.update({field: record[field] for field in fieldnames if field not in key_fields})
            writer.writerow(row)


def write_legacy_article_distribution(article_buckets: dict[tuple[str, ...], list[tuple[float, str]]]) -> None:
    LEGACY_ARTICLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with LEGACY_ARTICLE_OUT.open("w", newline="", encoding="utf-8") as handle:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only empirical priors from a dataset_release directory.")
    parser.add_argument("--dataset-release", type=Path, default=DEFAULT_DATASET_RELEASE)
    parser.add_argument("--train-labels", type=Path, default=None)
    args = parser.parse_args()

    dataset_release = args.dataset_release
    train_cases = dataset_release / "data" / "ecthr_npd_cases.csv"
    train_labels = args.train_labels or dataset_release / "model_inputs" / "structured_tree" / "targets" / "train.csv"
    labels = load_labels(train_labels)
    article_buckets: dict[tuple[str, ...], list[tuple[float, str]]] = {}
    country_buckets: dict[tuple[str, ...], list[tuple[float, str]]] = {}
    article_country_buckets: dict[tuple[str, ...], list[tuple[float, str]]] = {}

    with train_cases.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if row.get("split_role") not in {"", "train"} and row.get("split") not in {"", "train"}:
            continue
        label = labels.get(str(row.get("itemid") or ""))
        amount = parse_float((label or row).get("y_amount_eur"))
        if amount is None:
            continue
        source = str((label or {}).get("y_source") or "public_release_label")
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

    write_distribution(ARTICLE_OUT, ["article"], article_buckets)
    write_distribution(COUNTRY_OUT, ["country_alpha2"], country_buckets)
    write_distribution(ARTICLE_COUNTRY_OUT, ["article", "country_alpha2"], article_country_buckets)
    write_legacy_article_distribution(article_buckets)

    print(f"Wrote {ARTICLE_OUT}")
    print(f"Wrote {COUNTRY_OUT}")
    print(f"Wrote {ARTICLE_COUNTRY_OUT}")
    print(f"Wrote legacy compatibility file {LEGACY_ARTICLE_OUT}")


if __name__ == "__main__":
    main()

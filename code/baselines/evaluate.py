#!/usr/bin/env python3
"""Canonical, schema-checked evaluator for ECtHR-NPD amount predictions.

The public baseline families emit different, documented column names. This
module accepts those names, aligns predictions by ``itemid``, and calculates
the full EUR and zero-award metric suite on *all* examples. It deliberately
does not substitute a missing prediction or target with zero: that would make
an incomplete output look like a valid evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "ecthr-npd-evaluation-v1"
ITEMID_ALIASES = ("itemid", "case_id", "id")
TARGET_ALIASES = ("y_amount_eur", "target_award_eur", "award_eur", "target")
PREDICTION_ALIASES = ("predicted_award_eur", "prediction", "y_pred", "predicted_amount_eur")


class EvaluationSchemaError(ValueError):
    """Raised when a predictions/targets artifact cannot be evaluated safely."""


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    artifact = Path(path)
    suffix = artifact.suffix.lower()
    if suffix == ".csv":
        with artifact.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with artifact.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise EvaluationSchemaError(f"{artifact}:{line_number} is not a JSON object")
                    records.append(row)
        return records
    if suffix == ".json":
        value = load_json(artifact)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return [dict(row) for row in value]
        if isinstance(value, dict):
            for container_key in ("predictions", "targets", "rows", "data"):
                rows = value.get(container_key)
                if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                    return [dict(row) for row in rows]
        raise EvaluationSchemaError(
            f"{artifact} must contain a list of record objects, or a predictions/targets/rows/data list."
        )
    raise EvaluationSchemaError(f"Unsupported artifact extension {suffix!r}; use .csv, .jsonl, or .json.")


def _first_present(row: Mapping[str, Any], aliases: Sequence[str], *, field: str, artifact: str) -> Any:
    for key in aliases:
        if key in row and row[key] not in (None, ""):
            return row[key]
    raise EvaluationSchemaError(
        f"{artifact} record is missing {field}; accepted columns are {', '.join(aliases)}."
    )


def _records_by_itemid(
    records: Iterable[Mapping[str, Any]],
    *,
    value_aliases: Sequence[str],
    value_name: str,
    artifact: str,
) -> dict[str, float]:
    by_itemid: dict[str, float] = {}
    for position, row in enumerate(records):
        itemid = str(_first_present(row, ITEMID_ALIASES, field="itemid", artifact=artifact)).strip()
        if not itemid:
            raise EvaluationSchemaError(f"{artifact} record {position} has a blank itemid")
        if itemid in by_itemid:
            raise EvaluationSchemaError(f"{artifact} contains duplicate itemid {itemid!r}")
        raw_value = _first_present(row, value_aliases, field=value_name, artifact=artifact)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise EvaluationSchemaError(
                f"{artifact} itemid {itemid!r} has non-numeric {value_name}: {raw_value!r}"
            ) from exc
        if not np.isfinite(value):
            raise EvaluationSchemaError(f"{artifact} itemid {itemid!r} has non-finite {value_name}: {raw_value!r}")
        by_itemid[itemid] = value
    if not by_itemid:
        raise EvaluationSchemaError(f"{artifact} contains no usable records")
    return by_itemid


def align_prediction_records(
    prediction_records: Iterable[Mapping[str, Any]],
    ground_truth_records: Iterable[Mapping[str, Any]],
    *,
    prediction_artifact: str = "predictions",
    ground_truth_artifact: str = "ground truth",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return exactly aligned arrays, rejecting incomplete or duplicate artifacts."""
    predictions = _records_by_itemid(
        prediction_records,
        value_aliases=PREDICTION_ALIASES,
        value_name="prediction",
        artifact=prediction_artifact,
    )
    targets = _records_by_itemid(
        ground_truth_records,
        value_aliases=TARGET_ALIASES,
        value_name="target",
        artifact=ground_truth_artifact,
    )
    if predictions.keys() != targets.keys():
        only_predictions = sorted(predictions.keys() - targets.keys())
        only_targets = sorted(targets.keys() - predictions.keys())
        details: list[str] = []
        if only_predictions:
            details.append("prediction-only=" + ", ".join(only_predictions[:5]))
        if only_targets:
            details.append("target-only=" + ", ".join(only_targets[:5]))
        raise EvaluationSchemaError("itemid sets do not match (" + "; ".join(details) + ")")
    itemids = sorted(targets)
    return (
        np.asarray([targets[itemid] for itemid in itemids], dtype=float),
        np.asarray([predictions[itemid] for itemid in itemids], dtype=float),
        itemids,
    )


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties without taking a SciPy dependency."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        return np.array([], dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return pearson_r(rankdata_average(y_true), rankdata_average(y_pred))


def _f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float | int]:
    true_positive = y_true > threshold
    predicted_positive = y_pred > threshold
    true_zero = ~true_positive
    predicted_zero = ~predicted_positive

    tp = int((true_positive & predicted_positive).sum())
    tn = int((true_zero & predicted_zero).sum())
    fp = int((true_zero & predicted_positive).sum())
    fn = int((true_positive & predicted_zero).sum())

    positive_precision = tp / (tp + fp) if tp + fp else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    zero_precision = tn / (tn + fn) if tn + fn else 0.0
    zero_recall = tn / (tn + fp) if tn + fp else 0.0

    return {
        "positive_threshold_eur": float(threshold),
        "zero_positive_accuracy": float((true_positive == predicted_positive).mean()) if y_true.size else 0.0,
        "positive_precision": float(positive_precision),
        "positive_recall": float(positive_recall),
        "positive_f1": _f1(positive_precision, positive_recall),
        "zero_precision": float(zero_precision),
        "zero_recall": float(zero_recall),
        "zero_f1": _f1(zero_precision, zero_recall),
        "true_positive": tp,
        "true_zero": tn,
        "false_positive": fp,
        "false_zero": fn,
    }


def compute_eur_metrics(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """Compute EUR regression metrics across all cases, including zero awards."""
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    if true_values.shape != predicted_values.shape:
        raise ValueError(f"y_true shape {true_values.shape} != y_pred shape {predicted_values.shape}")
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("EUR metrics require finite y_true and y_pred values")
    if not true_values.size:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "medae": 0.0,
            "p95ae": 0.0,
            "r2": 0.0,
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "mae_positive_only": 0.0,
            "mape_positive_eur_plus_one": 0.0,
            "mape": 0.0,
            "num_samples": 0,
            "num_positive": 0,
            "num_zero": 0,
            "num_nonzero": 0,
        }

    errors = predicted_values - true_values
    absolute_errors = np.abs(errors)
    positive_mask = true_values > 0.5
    total_sum_squares = float(np.sum((true_values - true_values.mean()) ** 2))
    r2 = 0.0 if total_sum_squares == 0.0 else float(1.0 - np.sum(errors**2) / total_sum_squares)
    metrics: dict[str, float | int] = {
        "mae": float(absolute_errors.mean()),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "medae": float(np.median(absolute_errors)),
        "p95ae": float(np.quantile(absolute_errors, 0.95)),
        "r2": r2,
        "pearson_r": pearson_r(true_values, predicted_values),
        "spearman_rho": spearman_rho(true_values, predicted_values),
        "mae_positive_only": float(absolute_errors[positive_mask].mean()) if positive_mask.any() else 0.0,
        "mape_positive_eur_plus_one": (
            float((absolute_errors[positive_mask] / (true_values[positive_mask] + 1.0)).mean() * 100.0)
            if positive_mask.any()
            else 0.0
        ),
        "num_samples": int(true_values.size),
        "num_positive": int(positive_mask.sum()),
        "num_zero": int((~positive_mask).sum()),
    }
    # Explicit aliases make both the paper-table vocabulary and old baseline
    # consumer code unambiguous during the transition.
    metrics.update(
        {
            "mape": metrics["mape_positive_eur_plus_one"],
            "mae_all": metrics["mae"],
            "rmse_all": metrics["rmse"],
            "rmse_all_appendix_only": metrics["rmse"],
            "median_ae_all": metrics["medae"],
            "p95_ae_all": metrics["p95ae"],
            "num_nonzero": metrics["num_positive"],
        }
    )
    return metrics


def evaluate_arrays(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return the canonical regression plus zero/positive classification metrics."""
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    metrics = compute_eur_metrics(true_values, predicted_values)
    metrics.update(_classification_metrics(true_values, predicted_values, threshold))
    return metrics


def _read_bin_edges(path: str | Path) -> list[float]:
    config = load_json(path)
    if isinstance(config, list):
        edges = config
    elif isinstance(config, dict):
        edges = config.get("bin_edges", config.get("edges", config.get("thresholds")))
    else:
        edges = None
    if not isinstance(edges, list) or not edges:
        raise EvaluationSchemaError("bin scheme must be a JSON list or object with bin_edges/edges/thresholds")
    numeric = [float(edge) for edge in edges]
    if numeric != sorted(numeric) or len(set(numeric)) != len(numeric):
        raise EvaluationSchemaError("bin scheme edges must be strictly increasing")
    return numeric


def compute_ordinal_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    bin_edges: Sequence[float],
) -> dict[str, float | dict[str, float]]:
    """Evaluate amount-derived ordinal bins; never compare predictions to themselves."""
    true_bins = np.digitize(np.asarray(y_true, dtype=float), np.asarray(bin_edges, dtype=float), right=False)
    predicted_bins = np.digitize(np.asarray(y_pred, dtype=float), np.asarray(bin_edges, dtype=float), right=False)
    if not true_bins.size:
        return {"bin_accuracy": 0.0, "bin_within_1": 0.0, "bin_within_2": 0.0, "per_bin_recall": {}}
    per_bin_recall: dict[str, float] = {}
    for bin_number in range(len(bin_edges) + 1):
        mask = true_bins == bin_number
        if mask.any():
            per_bin_recall[f"bin_{bin_number}"] = float((predicted_bins[mask] == bin_number).mean())
    differences = np.abs(predicted_bins - true_bins)
    return {
        "bin_accuracy": float((predicted_bins == true_bins).mean()),
        "bin_within_1": float((differences <= 1).mean()),
        "bin_within_2": float((differences <= 2).mean()),
        "per_bin_recall": per_bin_recall,
    }


def evaluate_predictions(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    bin_scheme_path: str | Path | None = None,
    *,
    threshold: float = 0.5,
    dataset_release: str | Path | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    from data.data_loader import validate_selected_targets, file_sha256
    prediction_records = _read_records(predictions_path)
    truth_records = _read_records(ground_truth_path)
    y_true, y_pred, itemids = align_prediction_records(
        prediction_records,
        truth_records,
        prediction_artifact=str(predictions_path),
        ground_truth_artifact=str(ground_truth_path),
    )
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "itemid_count": len(itemids),
        "metrics": evaluate_arrays(y_true, y_pred, threshold=threshold),
        "dataset_provenance": {"dataset_version": "unversioned_artifact_evaluation"},
        "input_sha256": {"predictions": file_sha256(predictions_path), "ground_truth": file_sha256(ground_truth_path),
                          **({"bin_scheme": file_sha256(bin_scheme_path)} if bin_scheme_path else {})},
    }
    if dataset_release is not None or dataset_version is not None:
        metrics["dataset_provenance"] = validate_selected_targets(itemids, y_true, dataset_release,
                                                                   dataset_version=dataset_version or "corrected")
        metrics["prediction_version_binding"] = "ground_truth_checked_against_selected_version; prediction_training_provenance_not_certified"
    if bin_scheme_path:
        metrics["ordinal_metrics"] = compute_ordinal_metrics(y_true, y_pred, _read_bin_edges(bin_scheme_path))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "metrics.json"
    if output_path.exists():
        raise ValueError("metrics.json already exists; choose a fresh output directory")
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Metrics saved to {output_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ECtHR-NPD amount predictions with a canonical schema")
    parser.add_argument("--predictions", required=True, help="Prediction .csv, .jsonl, or .json artifact")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth .csv, .jsonl, or .json artifact")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--bin-scheme", default=None, help="Optional JSON bin-edge list/object for ordinal metrics")
    parser.add_argument("--positive-threshold-eur", type=float, default=0.5)
    parser.add_argument("--dataset-release", default=None, help="Optional explicit dataset root to bind ground truth to a version")
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], default=None,
                        help="Explicit audit/evaluation version; without selection this is labelled unversioned artifact evaluation")
    args = parser.parse_args()
    evaluate_predictions(
        args.predictions,
        args.ground_truth,
        args.output_dir,
        args.bin_scheme,
        threshold=args.positive_threshold_eur,
        dataset_release=args.dataset_release,
        dataset_version=args.dataset_version,
    )


if __name__ == "__main__":
    main()

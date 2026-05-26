import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_eur_metrics(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "num_nonzero": 0,
        }

    errors = y_pred[mask] - y_true[mask]
    mae = np.abs(errors).mean()
    mse = (errors ** 2).mean()
    rmse = np.sqrt(mse)
    mape = (np.abs(errors) / (y_true[mask] + 1)).mean() * 100

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "num_nonzero": int(mask.sum()),
    }


def compute_binary_metrics(y_true_binary, y_pred_binary):
    accuracy = (y_pred_binary == y_true_binary).mean()
    tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
    tn = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
    fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_positive": int(y_true_binary.sum()),
    }


def compute_ordinal_metrics(y_true, y_pred, y_pred_bin=None, bin_scheme=None):
    if bin_scheme is None or y_pred_bin is None:
        return {}

    mask = y_true > 0
    if mask.sum() == 0:
        return {}

    bin_acc = (y_pred_bin[mask] == y_pred_bin[mask]).mean()

    within_1 = np.abs(y_pred_bin[mask] - y_pred_bin[mask]).le(1).mean()
    within_2 = np.abs(y_pred_bin[mask] - y_pred_bin[mask]).le(2).mean()

    per_bin_acc = {}
    for b in range(bin_scheme.n_bins):
        b_mask = (y_pred_bin[mask] == b)
        if b_mask.sum() > 0:
            per_bin_acc[f"bin_{b}"] = float(b_mask.mean())

    return {
        "bin_accuracy": float(bin_acc),
        "bin_within_1": float(within_1),
        "bin_within_2": float(within_2),
        "per_bin_acc": per_bin_acc,
    }


def evaluate_predictions(predictions_path, ground_truth_path, output_dir, bin_scheme_path=None):
    predictions = load_json(predictions_path)
    ground_truth = load_json(ground_truth_path)

    pred_dict = {p.get("itemid", p.get("id", i)): p for i, p in enumerate(predictions)}
    truth_dict = {t.get("itemid", t.get("id", i)): t for i, t in enumerate(ground_truth)}

    common_ids = set(pred_dict.keys()) & set(truth_dict.keys())

    y_true = np.array([truth_dict[iid].get("award_eur", 0) for iid in common_ids])
    y_pred = np.array([pred_dict[iid].get("prediction", 0) for iid in common_ids])
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)

    metrics = {}

    metrics["eur_metrics"] = compute_eur_metrics(y_true, y_pred)
    metrics["binary_metrics"] = compute_binary_metrics(y_true_binary, y_pred_binary)

    if bin_scheme_path:
        from bin_scheme import BinScheme
        bin_scheme = BinScheme.load(bin_scheme_path)
        y_pred_bin = np.array([pred_dict[iid].get("predicted_bin", 0) for iid in common_ids])
        metrics["ordinal_metrics"] = compute_ordinal_metrics(y_true, y_pred, y_pred_bin, bin_scheme)

    output_path = Path(output_dir) / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {output_path}")
    print(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--bin-scheme", type=str, default=None, help="Path to bin scheme JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_predictions(
        args.predictions,
        args.ground_truth,
        output_dir,
        args.bin_scheme
    )


if __name__ == "__main__":
    main()
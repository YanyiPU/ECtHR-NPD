#!/usr/bin/env python3
"""Train strict CatBoost, XGBoost, or LightGBM pure-regression tree baselines.

The current tree setting is direct regression on log1p non-pecuniary
award amounts. No separate zero/positive classifier is fit. Validation
and test metrics are reported after a train-only fit and are not used
for early stopping or model selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CATBOOST_IMPORT_ERROR: Exception | None = None
try:
    from catboost import CatBoostRegressor, Pool
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    CatBoostRegressor = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment]
    CATBOOST_IMPORT_ERROR = exc

XGBOOST_IMPORT_ERROR: Exception | None = None
try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    xgb = None  # type: ignore[assignment]
    XGBOOST_IMPORT_ERROR = exc

LIGHTGBM_IMPORT_ERROR: Exception | None = None
try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    lgb = None  # type: ignore[assignment]
    LIGHTGBM_IMPORT_ERROR = exc

BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from data.data_loader import (
    encode_categorical_columns,
    input_contract_summary,
    load_structured_tree_splits,
    validate_dataset_release,
)


TREE_SETTINGS = {
    "setting_name": "strict_trainonly_50_feature_tree_regression",
    "target": "y_amount_eur",
    "target_transform": "log1p(y_amount_eur), inverse expm1 to EUR, clipped at 0",
    "optimization_policy": (
        "train-only fit; validation/test metrics are reported only and are not used "
        "for early stopping or model selection"
    ),
    "positive_threshold_eur": 0.5,
    "xgboost_params": {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "random_state": 42,
    },
    "catboost_params": {
        "iterations": 700,
        "learning_rate": 0.04,
        "depth": 4,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "CPU",
    },
    "lightgbm_params": {
        "n_estimators": 900,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "min_child_samples": 20,
        "reg_lambda": 5.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return pearson_r(rankdata_average(y_true), rankdata_average(y_pred))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    binary_true = (y_true > threshold).astype(int)
    binary_pred = (y_pred > threshold).astype(int)
    positive_mask = binary_true == 1

    tp = int(((binary_pred == 1) & (binary_true == 1)).sum())
    fp = int(((binary_pred == 1) & (binary_true == 0)).sum())
    fn = int(((binary_pred == 0) & (binary_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "mae_all": float(np.abs(y_pred - y_true).mean()) if y_true.size else 0.0,
        "rmse_all_appendix_only": float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else 0.0,
        "mae_positive_only": (
            float(np.abs(y_pred[positive_mask] - y_true[positive_mask]).mean())
            if int(positive_mask.sum())
            else 0.0
        ),
        "zero_positive_accuracy": float((binary_pred == binary_true).mean()) if y_true.size else 0.0,
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_rho": spearman_rho(y_true, y_pred),
        "num_samples": int(y_true.size),
        "num_positive": int(binary_true.sum()),
    }


def prepare_catboost_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = X.copy()
    categorical_cols: list[str] = []
    for column in frame.columns:
        if str(frame[column].dtype) in {"object", "category", "string"} or frame[column].dtype == bool:
            categorical_cols.append(column)
            frame[column] = frame[column].fillna("<MISSING>").astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame, categorical_cols


def fit_catboost(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> Any:
    if CATBOOST_IMPORT_ERROR is not None or CatBoostRegressor is None or Pool is None:
        raise RuntimeError("catboost is not installed") from CATBOOST_IMPORT_ERROR
    params = dict(TREE_SETTINGS["catboost_params"])
    params["random_seed"] = seed
    X_cb, categorical_cols = prepare_catboost_frame(X_train)
    cat_indices = [list(X_cb.columns).index(col) for col in categorical_cols]
    model = CatBoostRegressor(**params)
    model.fit(Pool(X_cb, np.log1p(y_train), cat_features=cat_indices))
    return {"model": model, "categorical_cols": categorical_cols, "feature_columns": list(X_cb.columns)}


def predict_catboost(fitted: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    frame, _ = prepare_catboost_frame(X)
    cat_indices = [list(frame.columns).index(col) for col in fitted["categorical_cols"]]
    log_pred = np.asarray(fitted["model"].predict(Pool(frame, cat_features=cat_indices)), dtype=float)
    return np.expm1(np.maximum(log_pred, 0.0))


def fit_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> Any:
    if XGBOOST_IMPORT_ERROR is not None or xgb is None:
        raise RuntimeError("xgboost is not installed") from XGBOOST_IMPORT_ERROR
    params = dict(TREE_SETTINGS["xgboost_params"])
    params["random_state"] = seed
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, np.log1p(y_train), verbose=False)
    return model


def fit_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> Any:
    if LIGHTGBM_IMPORT_ERROR is not None or lgb is None:
        raise RuntimeError("lightgbm is not installed") from LIGHTGBM_IMPORT_ERROR
    params = dict(TREE_SETTINGS["lightgbm_params"])
    params["random_state"] = seed
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, np.log1p(y_train))
    return model


def save_prediction_rows(path: Path, split_name: str, itemids: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rows = [
        {
            "split": split_name,
            "itemid": str(itemid),
            "target_award_eur": float(actual),
            "predicted_award_eur": float(predicted),
            "target_positive": int(actual > TREE_SETTINGS["positive_threshold_eur"]),
            "predicted_positive": int(predicted > TREE_SETTINGS["positive_threshold_eur"]),
        }
        for itemid, actual, predicted in zip(itemids, y_true, y_pred)
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def add_test_views(metrics: dict[str, Any], itemids: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, cases: pd.DataFrame) -> None:
    frame = pd.DataFrame({"itemid": itemids.astype(str), "y_true": y_true, "y_pred": y_pred})
    frame = frame.merge(cases[["itemid", "split", "test_view", "test_challenging_view"]], on="itemid", how="left")
    test = frame[frame["split"] == "test"]
    for name, subset in {
        "test_ID": test[test["test_view"] == "ID"],
        "test_OOD": test[test["test_view"] == "OOD"],
        "test_challenging": test[pd.to_numeric(test["test_challenging_view"], errors="coerce").fillna(0).astype(bool)],
    }.items():
        metrics[name] = evaluate_predictions(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict pure-regression CatBoost/XGBoost/LightGBM on ECtHR-NPD")
    parser.add_argument("--dataset-release", type=str, default=None, help="Path to dataset_release directory")
    parser.add_argument("--model", choices=["catboost", "xgboost", "lightgbm"], required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/tree_models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    validation = validate_dataset_release(args.dataset_release)
    if validation["status"] != "PASS":
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))

    splits = load_structured_tree_splits(args.dataset_release)
    train = splits["train"]
    y_train = train.y_amount_eur.to_numpy(dtype=float)

    if args.model == "catboost":
        fitted = fit_catboost(train.X, y_train, args.seed)
        predict_fn = lambda X: predict_catboost(fitted, X)
        model_metadata = {
            "model": "catboost",
            "params": {**TREE_SETTINGS["catboost_params"], "random_seed": args.seed},
            "categorical_columns": fitted["categorical_cols"],
        }
    elif args.model == "xgboost":
        encoded = encode_categorical_columns(train.X, splits["validation"].X, splits["test"].X)
        encoded_by_split = {"train": encoded[0], "validation": encoded[1], "test": encoded[2]}
        fitted = fit_xgboost(encoded_by_split["train"], y_train, args.seed)
        predict_fn = lambda X: np.expm1(np.maximum(np.asarray(fitted.predict(X), dtype=float), 0.0))
        model_metadata = {
            "model": "xgboost",
            "params": {**TREE_SETTINGS["xgboost_params"], "random_state": args.seed},
            "categorical_encoding": "train-only vocabulary; unknown validation/test categories encoded as -1",
        }
    else:
        encoded = encode_categorical_columns(train.X, splits["validation"].X, splits["test"].X)
        encoded_by_split = {"train": encoded[0], "validation": encoded[1], "test": encoded[2]}
        fitted = fit_lightgbm(encoded_by_split["train"], y_train, args.seed)
        predict_fn = lambda X: np.expm1(np.maximum(np.asarray(fitted.predict(X), dtype=float), 0.0))
        model_metadata = {
            "model": "lightgbm",
            "params": {**TREE_SETTINGS["lightgbm_params"], "random_state": args.seed},
            "categorical_encoding": "train-only vocabulary; unknown validation/test categories encoded as -1",
        }

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}

    for split_name, split in splits.items():
        X = encoded_by_split[split_name] if args.model in {"xgboost", "lightgbm"} else split.X
        y_true = split.y_amount_eur.to_numpy(dtype=float)
        y_pred = predict_fn(X)
        metrics[split_name] = evaluate_predictions(y_true, y_pred)
        if split_name == "test":
            add_test_views(metrics, split.itemids, y_true, y_pred, split.cases)
        save_prediction_rows(output_dir / f"{split_name}_predictions.json", split.name, split.itemids, y_true, y_pred)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "setting_name": TREE_SETTINGS["setting_name"],
                "task": "pure_regression",
                "dataset_loader": "baselines.data.data_loader",
                "dataset_validation": validation,
                "input_contract": input_contract_summary(),
                "tree_settings": TREE_SETTINGS,
                **model_metadata,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

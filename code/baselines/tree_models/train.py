#!/usr/bin/env python3
"""Train schema-checked strict tree baselines for the ECtHR-NPD protocol.

The paper uses direct log1p EUR regression, training-split imputation, and
validation MAE (median AE tie-break) to select a candidate configuration.
The historical candidate list is not distributed, so a paper-protocol run
requires the user to supply it explicitly; ``--example-fixed-parameters``
is intentionally labelled as an example rather than a reproduction.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from importlib import metadata as importlib_metadata
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
except Exception as exc:  # pragma: no cover - optional native dependency can fail to load
    CatBoostRegressor = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment]
    CATBOOST_IMPORT_ERROR = exc

XGBOOST_IMPORT_ERROR: Exception | None = None
try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - optional native dependency can fail to load
    xgb = None  # type: ignore[assignment]
    XGBOOST_IMPORT_ERROR = exc

LIGHTGBM_IMPORT_ERROR: Exception | None = None
try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - optional native dependency can fail to load
    lgb = None  # type: ignore[assignment]
    LIGHTGBM_IMPORT_ERROR = exc

BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from data.data_loader import (
    DatasetReleaseError,
    encode_categorical_columns,
    input_contract_summary,
    load_structured_tree_splits,
    validate_dataset_release,
    file_sha256,
)
from evaluate import evaluate_arrays
from data.public_adapter import is_public_root, feature_schema as public_feature_schema


TREE_SETTING_NAME = "corrected_strict_trainonly_48_feature_tree_regression"
TREE_SETTINGS = {
    "setting_name": TREE_SETTING_NAME,
    "profile_status": "corrected_internal_48_not_certified_historical_X1",
    "historical_reproduction": False,
    "target": "y_amount_eur",
    "target_transform": "log1p(y_amount_eur), inverse expm1 to EUR, clipped at 0",
    "optimization_policy": "select candidate by validation MAE, median AE tie-break; refit selected candidate on train only",
    "historical_candidate_manifest": "not distributed; required for a paper-protocol reproduction",
    "positive_threshold_eur": 0.5,
}

TREE_SETTING_DIR = (
    Path(__file__).resolve().parents[3]
    / "model_settings"
    / "tree"
    / "strict_trainonly_50_feature_tree_regression"  # historical config location, not current run identity
)
TREE_TRAINING_CONFIG_PATH = TREE_SETTING_DIR / "training_config.json"
TREE_MODELS = ("xgboost", "catboost", "lightgbm")


@lru_cache(maxsize=None)
def load_tree_training_config(setting_dir: Path = TREE_SETTING_DIR) -> dict[str, Any]:
    """Load the single source of truth for the released tree settings.

    The historical environment record is not merely documentation: defaults
    for all estimators are taken directly from this file.  Fail before a run
    if a maintainer edits it into an internally inconsistent configuration.
    ``null`` estimator options (currently ``early_stopping_rounds``) mean
    that the option is deliberately omitted from the constructor.
    """
    config_path = setting_dir / "training_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetReleaseError(f"Missing tree training configuration: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetReleaseError(f"Invalid JSON in tree training configuration: {config_path}") from exc
    if not isinstance(config, dict):
        raise DatasetReleaseError(f"Tree training configuration must be an object: {config_path}")

    required = {
        "models",
        "target",
        "task",
        "numeric_imputation",
        "positive_threshold_eur",
        "categorical_handling",
        "seed",
        "device_requested",
        "device_used",
        "packages",
        "historical_candidate_manifest",
        *(f"{model_name}_params" for model_name in TREE_MODELS),
    }
    missing = sorted(required - set(config))
    if missing:
        raise DatasetReleaseError(f"Tree training configuration missing keys: {', '.join(missing)}")
    if config["models"] != list(TREE_MODELS):
        raise DatasetReleaseError(f"Tree training models must be {list(TREE_MODELS)}, got {config['models']!r}")
    if config["target"] != TREE_SETTINGS["target"] or config["task"] != "pure_regression":
        raise DatasetReleaseError("Tree training configuration target/task does not match the released protocol")
    if config["numeric_imputation"] != "training-split median; an entirely missing training feature is an error":
        raise DatasetReleaseError("Tree training configuration must require training-split median imputation")
    if config["positive_threshold_eur"] != TREE_SETTINGS["positive_threshold_eur"]:
        raise DatasetReleaseError("Tree training configuration positive threshold does not match the released protocol")
    categorical_handling = config["categorical_handling"]
    if not isinstance(categorical_handling, dict) or categorical_handling != {
        "catboost": "native categorical pools",
        "xgboost_lightgbm": "train-only one-hot with explicit missing and unknown-category dummies",
    }:
        raise DatasetReleaseError("Tree training configuration categorical handling does not match the submitted protocol")
    if config["device_requested"] != "cpu" or config["device_used"] != "cpu":
        raise DatasetReleaseError("Released tree training configuration must explicitly use CPU")
    if not isinstance(config["seed"], int):
        raise DatasetReleaseError("Tree training configuration seed must be an integer")
    if not isinstance(config["packages"], dict):
        raise DatasetReleaseError("Tree training configuration packages must be an object")
    historical_manifest = config["historical_candidate_manifest"]
    if not isinstance(historical_manifest, dict) or historical_manifest.get("status") != "not_released":
        raise DatasetReleaseError("Tree training configuration must accurately mark the historical candidate manifest unavailable")
    for model_name in TREE_MODELS:
        params = config[f"{model_name}_params"]
        if not isinstance(params, dict) or not params:
            raise DatasetReleaseError(f"Tree training configuration {model_name}_params must be a non-empty object")
        if params.get("early_stopping_rounds") is not None:
            raise DatasetReleaseError(
                f"{model_name}_params early_stopping_rounds must be null in the released fixed configuration"
            )
    return config


def installed_tree_runtime_versions() -> dict[str, str]:
    """Record the runtime actually used, alongside the frozen historical record."""
    versions = {"python": sys.version}
    for distribution in ("numpy", "pandas", "catboost", "xgboost", "lightgbm", "scikit-learn", "joblib"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_tree_feature_schema(setting_dir: Path = TREE_SETTING_DIR) -> dict[str, Any]:
    """Load the release schema used to reject stale/misordered feature files."""
    schema_path = setting_dir / "feature_columns.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetReleaseError(f"Missing tree feature schema: {schema_path}") from exc
    expected = schema.get("feature_columns")
    categorical = schema.get("categorical_columns")
    if not isinstance(expected, list) or not expected or not all(isinstance(column, str) for column in expected):
        raise DatasetReleaseError(f"Invalid feature_columns in {schema_path}")
    if schema.get("feature_count") != len(expected):
        raise DatasetReleaseError(f"feature_count does not match feature_columns in {schema_path}")
    if not isinstance(categorical, list) or not set(categorical).issubset(expected):
        raise DatasetReleaseError(f"Invalid categorical_columns in {schema_path}")
    return schema


def validate_tree_feature_schema(splits: dict[str, Any], schema: dict[str, Any]) -> None:
    """Require each split to use the exact current versioned column order.

    The corrected internal schema has 48 predictors and is not certified X1.
    The old directory name is retained only for path compatibility.
    """
    expected = list(schema["feature_columns"])
    for split_name, split in splits.items():
        actual = list(split.X.columns)
        if actual != expected:
            missing = [column for column in expected if column not in actual]
            unexpected = [column for column in actual if column not in expected]
            raise DatasetReleaseError(
                f"{split_name} tree feature schema mismatch; missing={missing}, unexpected={unexpected}, "
                "or column order differs from model_settings/tree/.../feature_columns.json"
            )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Backward-compatible local name backed by the shared canonical evaluator."""
    return dict(evaluate_arrays(y_true, y_pred, threshold=threshold))


def prepare_catboost_frame(
    X: pd.DataFrame,
    *,
    categorical_columns: list[str] | None = None,
    numeric_fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Apply configured categoricals and training-median numeric imputation."""
    frame = X.copy()
    configured = set(categorical_columns or ())
    unknown_configured = configured - set(frame.columns)
    if unknown_configured:
        raise DatasetReleaseError("Configured categorical features absent from frame: " + ", ".join(sorted(unknown_configured)))
    categorical_cols: list[str] = []
    fill_values: dict[str, float] = {}
    for column in frame.columns:
        is_categorical = (
            column in configured
            or str(frame[column].dtype) in {"object", "category", "string"}
            or frame[column].dtype == bool
        )
        if is_categorical:
            categorical_cols.append(column)
            frame[column] = frame[column].fillna("<MISSING>").astype(str)
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if numeric_fill_values is None:
            fill_value = values.median(skipna=True)
            if pd.isna(fill_value):
                raise DatasetReleaseError(
                    f"Cannot median-impute numeric tree feature {column!r}: entirely missing in training."
                )
            fill_values[column] = float(fill_value)
        else:
            if column not in numeric_fill_values:
                raise DatasetReleaseError(f"Missing fitted median for numeric tree feature {column!r}")
            fill_values[column] = float(numeric_fill_values[column])
        frame[column] = values.fillna(fill_values[column])
    return frame, categorical_cols, fill_values


def _merged_params(model_name: str, params: dict[str, Any] | None, seed: int) -> dict[str, Any]:
    if model_name not in TREE_MODELS:
        raise DatasetReleaseError(f"Unknown tree model: {model_name}")
    # Defaults are loaded from model_settings/.../training_config.json rather
    # than duplicated in code.  Do not pass deliberate JSON nulls into the
    # third-party estimators.
    configured = load_tree_training_config()[f"{model_name}_params"]
    supplied = dict(params or {})
    immutable_by_model = {
        "xgboost": ("device", "tree_method", "early_stopping_rounds"),
        "catboost": ("task_type", "allow_writing_files", "early_stopping_rounds"),
        "lightgbm": ("device_type", "early_stopping_rounds"),
    }
    for key in immutable_by_model[model_name]:
        if key in supplied and supplied[key] != configured.get(key):
            raise DatasetReleaseError(
                f"Candidate parameters may not override frozen {model_name} protocol field "
                f"{key!r}: expected {configured.get(key)!r}, got {supplied[key]!r}"
            )
    seed_key = "random_seed" if model_name == "catboost" else "random_state"
    if seed_key in supplied and supplied[seed_key] != seed:
        raise DatasetReleaseError(
            f"Candidate parameters may not override the requested reproducibility seed {seed_key!r}={seed}"
        )

    merged = {
        key: value
        for key, value in configured.items()
        if value is not None
    }
    merged.update(supplied)
    # Deliberate JSON nulls must never reach third-party constructors.
    merged = {key: value for key, value in merged.items() if value is not None}
    merged[seed_key] = seed
    if model_name == "catboost":
        merged["allow_writing_files"] = False
    return merged


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    *,
    params: dict[str, Any] | None = None,
    categorical_columns: list[str] | None = None,
) -> dict[str, Any]:
    if CATBOOST_IMPORT_ERROR is not None or CatBoostRegressor is None or Pool is None:
        raise RuntimeError("catboost is unavailable in this environment") from CATBOOST_IMPORT_ERROR
    X_cb, categorical_cols, fill_values = prepare_catboost_frame(X_train, categorical_columns=categorical_columns)
    cat_indices = [list(X_cb.columns).index(column) for column in categorical_cols]
    model = CatBoostRegressor(**_merged_params("catboost", params, seed))
    model.fit(Pool(X_cb, np.log1p(y_train), cat_features=cat_indices))
    return {
        "model": model,
        "categorical_cols": categorical_cols,
        "feature_columns": list(X_cb.columns),
        "numeric_fill_values": fill_values,
    }


def predict_catboost(fitted: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    expected_columns = list(fitted["feature_columns"])
    if list(X.columns) != expected_columns:
        raise DatasetReleaseError("CatBoost prediction frame feature order does not match the fitted training frame")
    frame, categorical_cols, _ = prepare_catboost_frame(
        X,
        categorical_columns=list(fitted["categorical_cols"]),
        numeric_fill_values=dict(fitted["numeric_fill_values"]),
    )
    cat_indices = [list(frame.columns).index(column) for column in categorical_cols]
    log_pred = np.asarray(fitted["model"].predict(Pool(frame, cat_features=cat_indices)), dtype=float)
    return np.maximum(np.expm1(log_pred), 0.0)


def fit_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, seed: int, *, params: dict[str, Any] | None = None) -> Any:
    if XGBOOST_IMPORT_ERROR is not None or xgb is None:
        raise RuntimeError("xgboost is unavailable in this environment") from XGBOOST_IMPORT_ERROR
    model = xgb.XGBRegressor(**_merged_params("xgboost", params, seed))
    model.fit(X_train, np.log1p(y_train), verbose=False)
    return model


def fit_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray, seed: int, *, params: dict[str, Any] | None = None) -> Any:
    if LIGHTGBM_IMPORT_ERROR is not None or lgb is None:
        raise RuntimeError("lightgbm is unavailable in this environment") from LIGHTGBM_IMPORT_ERROR
    model = lgb.LGBMRegressor(**_merged_params("lightgbm", params, seed))
    model.fit(X_train, np.log1p(y_train))
    return model


def load_selection_candidates(path: str | Path, model_name: str) -> list[dict[str, Any]]:
    """Load a user-supplied validation-selection candidate manifest.

    Supported shapes are ``[{"name": ..., "params": {...}}]`` and
    ``{"candidates": [...]}``; a top-level model key is also accepted for a
    shared manifest. The release does not contain the historical manifest, so
    this parser is intentionally strict rather than guessing one.
    """
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: Any = payload
    if isinstance(payload, dict):
        candidates = payload.get("candidates", payload.get(model_name))
    if not isinstance(candidates, list) or not candidates:
        raise DatasetReleaseError(
            f"{manifest_path} must contain a non-empty candidate list (or a {model_name!r} list)."
        )
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise DatasetReleaseError(f"Candidate {index} in {manifest_path} is not an object")
        params = candidate["params"] if "params" in candidate else {
            key: value for key, value in candidate.items() if key != "name"
        }
        if not isinstance(params, dict):
            raise DatasetReleaseError(f"Candidate {index} in {manifest_path} has non-object params")
        name = str(candidate.get("name", f"candidate_{index:03d}"))
        if name in names:
            raise DatasetReleaseError(f"Duplicate candidate name {name!r} in {manifest_path}")
        names.add(name)
        normalized.append({"name": name, "params": dict(params)})
    return normalized


def select_candidate_by_validation(
    candidates: list[dict[str, Any]],
    *,
    fit_candidate: Any,
    predict_candidate: Any,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select lowest validation MAE, resolving ties with validation MedAE."""
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        fitted = fit_candidate(dict(candidate["params"]))
        predictions = np.asarray(predict_candidate(fitted, X_validation), dtype=float)
        metrics = evaluate_arrays(y_validation, predictions, threshold=threshold)
        scored.append(
            {
                "name": candidate["name"],
                "params": dict(candidate["params"]),
                "validation_mae": float(metrics["mae"]),
                "validation_medae": float(metrics["medae"]),
                "validation_metrics": metrics,
            }
        )
    selected = min(scored, key=lambda row: (row["validation_mae"], row["validation_medae"], row["name"]))
    return {"name": selected["name"], "params": selected["params"]}, scored


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
    parser = argparse.ArgumentParser(description="Train schema-checked ECtHR-NPD tree regressors")
    parser.add_argument("--dataset-release", type=str, default=None, help="Path to dataset_release directory")
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], default="corrected")
    parser.add_argument("--model", choices=["catboost", "xgboost", "lightgbm"], required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/tree_models")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for an example run; paper-protocol runs use the frozen configuration seed.",
    )
    parser.add_argument(
        "--selection-candidates",
        type=str,
        default=None,
        help="JSON candidate manifest; required for a paper-protocol validation-selection run.",
    )
    parser.add_argument(
        "--example-fixed-parameters",
        action="store_true",
        help="Run one fixed parameter set as an example only; it is not a reproduction of the reported selection.",
    )
    args = parser.parse_args()
    if not args.dataset_release:
        parser.error("supply --dataset-release pointing to the public package root")
    output_dir = Path(args.output_dir) / args.model
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory must be empty; do not overwrite an existing experiment")

    if bool(args.selection_candidates) == bool(args.example_fixed_parameters):
        parser.error(
            "supply --selection-candidates for the paper protocol, or explicitly opt into "
            "--example-fixed-parameters; the historical candidate manifest is not released."
        )

    run_input_hashes = {"training_config": file_sha256(TREE_TRAINING_CONFIG_PATH),
                       **({"selection_candidates": file_sha256(args.selection_candidates)} if args.selection_candidates else {})}
    training_config = load_tree_training_config()
    seed = int(training_config["seed"]) if args.seed is None else args.seed
    if args.selection_candidates and seed != int(training_config["seed"]):
        parser.error(
            "paper-protocol candidate selection requires the frozen training_config.json seed; "
            "omit --seed or use --example-fixed-parameters for an exploratory override."
        )

    set_seed(seed)
    validation = validate_dataset_release(args.dataset_release, dataset_version=args.dataset_version)
    if validation["status"] != "PASS":
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))
    splits = load_structured_tree_splits(args.dataset_release, dataset_version=args.dataset_version)
    feature_schema = public_feature_schema() if is_public_root(args.dataset_release) else load_tree_feature_schema()
    validate_tree_feature_schema(splits, feature_schema)
    train = splits["train"]
    validation_split = splits["validation"]
    y_train = train.y_amount_eur.to_numpy(dtype=float)
    y_validation = validation_split.y_amount_eur.to_numpy(dtype=float)
    categorical_columns = list(feature_schema["categorical_columns"])

    if args.selection_candidates:
        candidates = load_selection_candidates(args.selection_candidates, args.model)
        selection_provenance: dict[str, Any] = {
            "status": "user_supplied_candidate_manifest",
            "path": str(Path(args.selection_candidates)),
            "rule": "lowest validation MAE; median AE tie-break",
        }
    else:
        candidates = [{"name": "fixed_example", "params": {}}]
        selection_provenance = {
            "status": "example_only_not_reported_result_reproduction",
            "reason": "The historical validation candidate manifest is not included in the release.",
        }

    if args.model == "catboost":
        fit_for_selection = lambda params: fit_catboost(
            train.X, y_train, seed, params=params, categorical_columns=categorical_columns
        )
        predict_for_selection = lambda fitted, X: predict_catboost(fitted, X)
        selected, selection_rows = select_candidate_by_validation(
            candidates,
            fit_candidate=fit_for_selection,
            predict_candidate=predict_for_selection,
            X_validation=validation_split.X,
            y_validation=y_validation,
            threshold=TREE_SETTINGS["positive_threshold_eur"],
        )
        fitted = fit_for_selection(selected["params"])
        predict_fn = lambda X: predict_catboost(fitted, X)
        model_metadata = {
            "model": "catboost",
            "params": _merged_params("catboost", selected["params"], seed),
            "categorical_columns": fitted["categorical_cols"],
            "numeric_imputation": "training-split median",
        }
    elif args.model == "xgboost":
        encoded = encode_categorical_columns(
            train.X,
            validation_split.X,
            splits["test"].X,
            categorical_columns=categorical_columns,
        )
        encoded_by_split = {"train": encoded[0], "validation": encoded[1], "test": encoded[2]}
        fit_for_selection = lambda params: fit_xgboost(encoded_by_split["train"], y_train, seed, params=params)
        predict_for_selection = lambda fitted, X: np.maximum(
            np.expm1(np.asarray(fitted.predict(X), dtype=float)), 0.0
        )
        selected, selection_rows = select_candidate_by_validation(
            candidates,
            fit_candidate=fit_for_selection,
            predict_candidate=predict_for_selection,
            X_validation=encoded_by_split["validation"],
            y_validation=y_validation,
            threshold=TREE_SETTINGS["positive_threshold_eur"],
        )
        fitted = fit_for_selection(selected["params"])
        predict_fn = lambda X: np.expm1(np.maximum(np.asarray(fitted.predict(X), dtype=float), 0.0))
        model_metadata = {
            "model": "xgboost",
            "params": _merged_params("xgboost", selected["params"], seed),
            "categorical_encoding": "train-only one-hot with explicit missing and unknown-category dummies",
            "numeric_imputation": "training-split median",
        }
    else:
        encoded = encode_categorical_columns(
            train.X,
            validation_split.X,
            splits["test"].X,
            categorical_columns=categorical_columns,
        )
        encoded_by_split = {"train": encoded[0], "validation": encoded[1], "test": encoded[2]}
        fit_for_selection = lambda params: fit_lightgbm(encoded_by_split["train"], y_train, seed, params=params)
        predict_for_selection = lambda fitted, X: np.maximum(
            np.expm1(np.asarray(fitted.predict(X), dtype=float)), 0.0
        )
        selected, selection_rows = select_candidate_by_validation(
            candidates,
            fit_candidate=fit_for_selection,
            predict_candidate=predict_for_selection,
            X_validation=encoded_by_split["validation"],
            y_validation=y_validation,
            threshold=TREE_SETTINGS["positive_threshold_eur"],
        )
        fitted = fit_for_selection(selected["params"])
        predict_fn = lambda X: np.expm1(np.maximum(np.asarray(fitted.predict(X), dtype=float), 0.0))
        model_metadata = {
            "model": "lightgbm",
            "params": _merged_params("lightgbm", selected["params"], seed),
            "categorical_encoding": "train-only one-hot with explicit missing and unknown-category dummies",
            "numeric_imputation": "training-split median",
        }

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}

    for split_name, split in splits.items():
        X = encoded_by_split[split_name] if args.model in {"xgboost", "lightgbm"} else split.X
        y_true = split.y_amount_eur.to_numpy(dtype=float)
        y_pred = predict_fn(X)
        metrics[split_name] = evaluate_predictions(y_true, y_pred, TREE_SETTINGS["positive_threshold_eur"])
        if split_name == "test":
            add_test_views(metrics, split.itemids, y_true, y_pred, split.cases)
        save_prediction_rows(output_dir / f"{split_name}_predictions.json", split.name, split.itemids, y_true, y_pred)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "setting_name": "public_structured_tree_regression" if is_public_root(args.dataset_release) else TREE_SETTINGS["setting_name"],
                "task": "pure_regression",
                "dataset_loader": "baselines.data.data_loader",
                "dataset_validation": validation,
                "run_input_sha256": run_input_hashes,
                "feature_schema": feature_schema,
                "input_contract": feature_schema if is_public_root(args.dataset_release) else input_contract_summary(),
                "tree_settings": {**TREE_SETTINGS, "setting_name": "public_structured_tree_regression",
                                  "profile_status": "new_public_table_representation"} if is_public_root(args.dataset_release) else TREE_SETTINGS,
                "training_config": training_config,
                "effective_seed": seed,
                "runtime_versions": installed_tree_runtime_versions(),
                "selection": {**selection_provenance, "selected": selected, "candidate_scores": selection_rows},
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

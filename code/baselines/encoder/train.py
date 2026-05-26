#!/usr/bin/env python3
"""Train strict pure-regression encoder baselines for ECtHR-NPD.

This script is intentionally text-source agnostic. Use `--text-inputs`
for strict Article-41-free case texts. If omitted, the loader serializes
the public strict inputs so the code remains runnable from the release
without redistributing raw judgments.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

TRANSFORMERS_IMPORT_ERROR: Exception | None = None
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
except ImportError as exc:  # pragma: no cover - optional encoder dependency
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]
    TRANSFORMERS_IMPORT_ERROR = exc


BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from encoder.data_loader import load_encoder_splits


ENCODER_SETTINGS = {
    "task": "pure_regression",
    "target": "y_amount_eur",
    "target_transform": "log1p(y_amount_eur), inverse expm1 to EUR, clipped at 0",
    "input_contract": "safe metadata, violated articles, case facts text or serialized strict inputs, external factors",
    "strict_exclusions": [
        "Article 41/Article 50 compensation text",
        "operative award clauses",
        "claimed amounts",
        "direct award snippets",
        "target-derived fields",
    ],
    "supported_settings": [
        "modernbert_base_8k_pure_regression",
        "legallongformer_2x4k_pure_regression",
        "modernbert_latefusion_x0_locked50",
        "modernbert_latefusion_x1_applicant",
        "modernbert_latefusion_x2_reasoning",
        "legallongformer_latefusion_x1_applicant",
    ],
}


class EncoderDataset(Dataset):  # type: ignore[misc]
    def __init__(self, frame: Any, tokenizer: Any, max_length: int) -> None:
        self.itemids = frame["itemid"].astype(str).tolist()
        self.labels = np.log1p(frame["y_amount_eur"].to_numpy(dtype=float)).astype("float32")
        self.encodings = tokenizer(
            frame["text"].astype(str).tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def rankdata_average(values: np.ndarray) -> np.ndarray:
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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_pos = y_true > threshold
    pred_pos = y_pred > threshold
    tp = int((true_pos & pred_pos).sum())
    fp = int((~true_pos & pred_pos).sum())
    fn = int((true_pos & ~pred_pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "mae_all": float(np.abs(y_pred - y_true).mean()) if y_true.size else 0.0,
        "rmse_all_appendix_only": float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else 0.0,
        "mae_positive_only": float(np.abs(y_pred[true_pos] - y_true[true_pos]).mean()) if true_pos.any() else 0.0,
        "zero_positive_accuracy": float((true_pos == pred_pos).mean()) if y_true.size else 0.0,
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_rho": pearson_r(rankdata_average(y_true), rankdata_average(y_pred)),
        "num_samples": int(y_true.size),
        "num_positive": int(true_pos.sum()),
    }


def predict_eur(trainer: Any, dataset: Any) -> np.ndarray:
    output = trainer.predict(dataset)
    logits = np.asarray(output.predictions).reshape(-1)
    return np.expm1(np.maximum(logits, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict encoder pure-regression baseline")
    parser.add_argument("--dataset-release", default=None)
    parser.add_argument("--text-inputs", default=None, help="Optional strict Article-41-free CSV/JSONL keyed by itemid")
    parser.add_argument("--text-field", default="combined_input_text_with_violated_articles")
    parser.add_argument("--model-name-or-path", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", default="outputs/encoder")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError("transformers and torch are required for encoder training") from TRANSFORMERS_IMPORT_ERROR

    set_seed(args.seed)
    splits = load_encoder_splits(args.dataset_release, text_inputs=args.text_inputs, text_field=args.text_field)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        problem_type="regression",
    )

    train_ds = EncoderDataset(splits["train"], tokenizer, args.max_length)
    eval_ds = EncoderDataset(splits["validation"], tokenizer, args.max_length)
    test_ds = EncoderDataset(splits["test"], tokenizer, args.max_length)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        report_to=[],
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()

    metrics: dict[str, Any] = {}
    for name, dataset in {"validation": eval_ds, "test": test_ds}.items():
        y_true = splits[name]["y_amount_eur"].to_numpy(dtype=float)
        y_pred = predict_eur(trainer, dataset)
        metrics[name] = evaluate(y_true, y_pred)

    metadata = {
        "encoder_settings": ENCODER_SETTINGS,
        "model_name_or_path": args.model_name_or_path,
        "max_length": args.max_length,
        "text_inputs": "user_supplied_strict_text" if args.text_inputs else "serialized_public_strict_inputs",
        "split_rows": {name: int(len(frame)) for name, frame in splits.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example-only strict encoder training scaffold for ECtHR-NPD.

The released generic Trainer scaffold does not implement the paper's original
chunking/late-fusion architectures or ship their checkpoints and historical
text inputs. It can be run only with an explicit example-only acknowledgement;
it must not be presented as reproduction of a reported encoder result. The
separate train_paper.py now implements the specified core for explicit NEW runs.
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
except Exception as exc:  # pragma: no cover - optional native dependency can fail to load
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
from data.data_loader import dataset_provenance, require_corrected_version
from evaluate import evaluate_arrays


ENCODER_SETTINGS = {
    "release_status": "example_only_not_a_reported_encoder_reproduction",
    "reproduction_blockers": [
        "original historical implementation/provenance is not recovered; train_paper.py is a new implementation",
        "reported checkpoints are not included",
        "historical strict FACTS/text inputs are not included",
        "reported prediction artifacts are not included",
    ],
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
    "historical_setting_records_not_implemented_by_this_scaffold": [
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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    return dict(evaluate_arrays(y_true, y_pred, threshold=threshold))


def predict_eur(trainer: Any, dataset: Any) -> np.ndarray:
    output = trainer.predict(dataset)
    logits = np.asarray(output.predictions).reshape(-1)
    return np.expm1(np.maximum(logits, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict encoder pure-regression baseline")
    parser.add_argument("--dataset-release", default=None)
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], default="corrected")
    parser.add_argument("--text-inputs", default=None, help="Optional strict Article-41-free CSV/JSONL keyed by itemid")
    parser.add_argument("--text-field", default="combined_input_text_with_violated_articles")
    parser.add_argument("--model-name-or-path", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", default="outputs/encoder")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--example-only",
        action="store_true",
        help="Acknowledge that this generic scaffold is not the reported encoder implementation or checkpoint.",
    )
    args = parser.parse_args()

    if not args.example_only:
        parser.error(
            "The released generic encoder scaffold cannot reproduce a reported encoder result. "
            "Pass --example-only only for a new illustrative run."
        )

    require_corrected_version(args.dataset_version)
    provenance = dataset_provenance(args.dataset_release, dataset_version=args.dataset_version,
                                    input_files={"text_inputs": args.text_inputs})
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory must be empty; do not overwrite an experiment")
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError("transformers and torch are required for encoder training") from TRANSFORMERS_IMPORT_ERROR

    set_seed(args.seed)
    splits = load_encoder_splits(args.dataset_release, text_inputs=args.text_inputs, text_field=args.text_field, dataset_version=args.dataset_version)
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
        "dataset_provenance": provenance,
        "encoder_settings": ENCODER_SETTINGS,
        "run_status": "example_only_not_a_reported_encoder_reproduction",
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

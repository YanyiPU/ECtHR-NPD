#!/usr/bin/env python3
"""Explicit NEW encoder experiment implementing the specified Table 22 core.

Requires supplied strict FACTS, local backbone files and explicit hyperparameters.
Does not fetch a model, infer historical settings, or claim historical reproduction.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np

BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))
from encoder.paper_model import LogTargetScaler, PaperEncoderRegressor, encode_chunks
from encoder.data_loader import load_user_text_inputs
from feature_profiles import TrainOnlyPreprocessor
from retrieval.tabular_knn import load_inputs
from retrieval.bm25_pfme_knn import normalize_protocol_split, read_rows, read_targets
from evaluate import evaluate_arrays
from data.data_loader import require_corrected_version, validate_selected_targets


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-release", required=True, help="Explicit dataset root or selected clean root")
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], default="corrected")
    parser.add_argument("--case-index", required=True, help="CSV: itemid,split,judgementdate,violated_articles")
    parser.add_argument("--targets", required=True, help="Separate full-cohort itemid,y_amount_eur CSV")
    parser.add_argument("--text-inputs", required=True, help="Strict FACTS CSV/JSONL; no serialized feature substitute")
    parser.add_argument("--text-field", default="facts_text")
    parser.add_argument("--architecture", choices=["modernbert", "legal_longformer"], required=True)
    parser.add_argument("--model-name-or-path", required=True, help="Existing local backbone directory; never downloads")
    parser.add_argument("--features", default=None)
    parser.add_argument("--profile", default=None, help="Explicit X0--X3 column/lineage JSON for late fusion")
    parser.add_argument("--mlp-activation", choices=["relu", "gelu"], default=None)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acknowledge-new-run", action="store_true")
    args = parser.parse_args()
    require_corrected_version(args.dataset_version)
    if not args.acknowledge_new_run:
        parser.error("explicit --acknowledge-new-run required; this is not a recovered historical implementation")
    if bool(args.features) != bool(args.profile) or bool(args.features) != bool(args.mlp_activation):
        parser.error("late fusion requires --features, --profile and --mlp-activation together")
    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("positive learning-rate and epochs required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Table 22 effective batch=16 preset currently supports one process only")
    if not Path(args.model_name_or_path).is_dir():
        parser.error("model-name-or-path must be a pre-existing local backbone directory")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; existing runs are immutable")
    index = read_rows(Path(args.case_index))
    ids = [str(row["itemid"]).strip() for row in index]
    if not ids or len(set(ids)) != len(ids) or any(not itemid for itemid in ids):
        raise ValueError("case-index itemids must be nonempty and unique")
    split_indices = {name: [i for i, row in enumerate(index) if normalize_protocol_split(row["split"]) == name]
                     for name in ("train", "validation", "test")}
    if any(not indices for indices in split_indices.values()):
        raise ValueError("all three explicit splits are required")
    labels = read_targets(Path(args.targets))
    text = load_user_text_inputs(args.text_inputs, args.text_field).set_index("itemid")["text"].to_dict()
    if set(ids) != set(labels) or set(ids) != set(text):
        raise ValueError("case-index/target/strict FACTS coverage must match exactly")
    provenance = validate_selected_targets(ids, [labels[itemid] for itemid in ids], args.dataset_release,
        dataset_version=args.dataset_version, splits=[row["split"] for row in index], require_all=True)
    scaler = LogTargetScaler.fit([labels[ids[i]] for i in split_indices["train"]])
    structured, preprocessing, profile = None, None, None
    if args.features:
        profile = json.loads(Path(args.profile).read_text())
        if profile.get("profile") not in {"X0", "X1", "X2", "X3", "public_structured"}:
            raise ValueError("late-fusion requires an explicit X0--X3 profile")
        _, feature_rows = load_inputs(index, read_rows(Path(args.features)), profile)
        prep = TrainOnlyPreprocessor().fit([feature_rows[i] for i in split_indices["train"]], profile)
        structured, preprocessing = prep.transform(feature_rows).astype("float32"), prep.manifest()
    import torch
    from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Table 22 preset requires bf16-capable CUDA; no silent precision substitution")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.model_name_or_path, local_files_only=True)
    expected_type = "modernbert" if args.architecture == "modernbert" else "longformer"
    if getattr(backbone.config, "model_type", None) != expected_type:
        raise ValueError(f"architecture requires {expected_type} backbone config")
    model = PaperEncoderRegressor(backbone, structured_dim=0 if structured is None else structured.shape[1],
                                  activation=args.mlp_activation)
    class Cases(torch.utils.data.Dataset):
        def __init__(self, indices):
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, position):
            i = self.indices[position]
            record = encode_chunks(text[ids[i]], tokenizer, args.architecture)
            record = {key: torch.tensor(value, dtype=torch.float32 if key == "chunk_mask" else torch.long)
                      for key, value in record.items()}
            record["labels"] = torch.tensor(float(scaler.transform([labels[ids[i]]])[0]), dtype=torch.float32)
            if structured is not None:
                record["structured_features"] = torch.tensor(structured[i])
            return record
    datasets = {name: Cases(indices) for name, indices in split_indices.items()}
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=str(output / "trainer"), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, bf16=True, gradient_checkpointing=True, weight_decay=0.01,
        warmup_ratio=0.06, lr_scheduler_type="linear", optim="adamw_torch", report_to=[], seed=args.seed,
        save_strategy="epoch", eval_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="mae_eur", greater_is_better=False, remove_unused_columns=False)
    def metrics(prediction):
        truth = scaler.inverse(prediction.label_ids)
        predicted = scaler.inverse(np.asarray(prediction.predictions).reshape(-1))
        return {"mae_eur": float(np.abs(truth - predicted).mean())}
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
                      data_collator=default_data_collator, compute_metrics=metrics)
    # Record immutable input/config evidence BEFORE running, including unknowns.
    manifest = {"status": "new_implementation_not_historical_reproduction", "arguments": vars(args),
        "dataset_provenance": provenance,
        "paper": "submitted PDF Table 22, p24", "target_scaler_train_only": asdict(scaler),
        "profile": profile, "preprocessing": preprocessing,
        "new_choices_not_specified_in_paper": ["MLP activation", "empty chunk masking", "validation MAE checkpoint selection",
                                               "target standard deviation ddof=0", "numeric scaling for fusion"],
        "historical_blockers": ["original weights/checkpoints", "original FACTS hashes", "hyperparameter selection records",
                                "LegalBERT initialization provenance", "original profile/run/prediction mapping"],
        "input_sha256": {name: file_sha256(getattr(args, name))
                         for name in ["case_index", "targets", "text_inputs", "features", "profile"] if getattr(args, name)},
        "local_backbone_files_sha256": {str(path.relative_to(args.model_name_or_path)): file_sha256(path)
            for path in sorted(Path(args.model_name_or_path).rglob("*")) if path.is_file()},
        "torch_version": torch.__version__, "tokenizer_class": type(tokenizer).__name__}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    trainer.train()
    result = {}
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["itemid", "split", "predicted_award_eur"])
        writer.writeheader()
        for name in ["validation", "test"]:
            predicted = scaler.inverse(np.asarray(trainer.predict(datasets[name]).predictions).reshape(-1))
            truth = [labels[ids[i]] for i in split_indices[name]]
            result[name] = dict(evaluate_arrays(truth, predicted))
            writer.writerows({"itemid": ids[i], "split": name, "predicted_award_eur": float(value)}
                             for i, value in zip(split_indices[name], predicted, strict=True))
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

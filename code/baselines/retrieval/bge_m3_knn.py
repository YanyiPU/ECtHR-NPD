#!/usr/bin/env python3
"""Strict BGE-M3 dense/sparse KNN retrieval baseline.

This script mirrors the BGE-M3 retrieval settings without redistributing
raw judgment text, embeddings, neighbor traces, or predictions. Users
must supply strict Article-41-free text inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

FLAGEMBEDDING_IMPORT_ERROR: Exception | None = None
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError as exc:  # pragma: no cover - optional retrieval dependency
    BGEM3FlagModel = None  # type: ignore[assignment]
    FLAGEMBEDDING_IMPORT_ERROR = exc


BGE_M3_SETTINGS = {
    "methods": ["bge_m3_dense", "bge_m3_sparse"],
    "input_policy": "text_a41_free",
    "text_field": "combined_input_text_with_violated_articles",
    "optional_external_factors": "append respondent-state/year external factors by itemid when supplied",
    "retrieval_corpus": "train split only",
    "temporal_filter": "candidate_judgement_date < target_judgement_date",
    "k_grid": [1, 3, 5, 10, 15, 20, 25, 30, 50],
    "aggregators": ["mean", "median"],
    "max_length": 2048,
    "positive_threshold_eur": 0.5,
}


def parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y"):
        try:
            return datetime.strptime(text[:10] if fmt != "%Y" else text[:4], fmt)
        except ValueError:
            continue
    return None


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_rows(path) if path.suffix.lower() == ".jsonl" else read_csv_rows(path)


def read_targets(path: Path) -> dict[str, float]:
    return {
        str(row["itemid"]): float(row["y_amount_eur"])
        for row in read_csv_rows(path)
        if row.get("itemid") and row.get("y_amount_eur") not in (None, "")
    }


def read_external_factors(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {
        str(row["itemid"]): {
            key: value
            for key, value in row.items()
            if key != "itemid" and value not in (None, "")
        }
        for row in read_csv_rows(path)
        if row.get("itemid")
    }


def append_external_factors(text: str, itemid: str, external_factors: dict[str, dict[str, Any]]) -> str:
    factors = external_factors.get(str(itemid))
    if not factors:
        return text
    serialized = json.dumps(factors, ensure_ascii=False, sort_keys=True)
    return f"{text}\n\nExternal factors: {serialized}".strip()


def row_date(row: dict[str, Any]) -> datetime | None:
    return parse_date(row.get("judgementdate") or row.get("judgment_date") or row.get("chrono_date"))


def aggregate(values: list[float], method: str) -> float:
    if not values:
        return 0.0
    if method == "mean":
        return float(sum(values) / len(values))
    if method == "median":
        return float(median(values))
    raise ValueError(f"unknown aggregator: {method}")


def encode_texts(model_name_or_path: str, texts: list[str], mode: str, batch_size: int, max_length: int) -> Any:
    if FLAGEMBEDDING_IMPORT_ERROR is not None or BGEM3FlagModel is None:
        raise RuntimeError("FlagEmbedding is not installed") from FLAGEMBEDDING_IMPORT_ERROR
    model = BGEM3FlagModel(model_name_or_path, use_fp16=True)
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=mode == "dense",
        return_sparse=mode == "sparse",
        return_colbert_vecs=False,
    )
    return output["dense_vecs"] if mode == "dense" else output["lexical_weights"]


def dense_scores(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    corpus_norm = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12)
    query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
    return corpus_norm @ query_norm


def sparse_score(query: dict[str, float], candidate: dict[str, float]) -> float:
    if len(query) > len(candidate):
        query, candidate = candidate, query
    return float(sum(weight * candidate.get(token, 0.0) for token, weight in query.items()))


def run_knn(
    rows: list[dict[str, Any]],
    train_targets: dict[str, float],
    representations: Any,
    *,
    mode: str,
    k_grid: list[int],
    aggregators: list[str],
) -> dict[str, list[dict[str, Any]]]:
    train_indices = [
        idx for idx, row in enumerate(rows)
        if str(row.get("split") or "") == "train" and str(row.get("itemid") or "") in train_targets
    ]
    outputs: dict[str, list[dict[str, Any]]] = {f"{mode}_k{k}_{agg}": [] for k in k_grid for agg in aggregators}

    dense_corpus = np.asarray([representations[idx] for idx in train_indices], dtype=float) if mode == "dense" else None
    for idx, target in enumerate(rows):
        split = str(target.get("split") or "")
        if split not in {"val", "validation", "test"}:
            continue
        target_itemid = str(target.get("itemid") or "")
        target_date = row_date(target)
        allowed_positions = [
            pos for pos, train_idx in enumerate(train_indices)
            if str(rows[train_idx].get("itemid") or "") != target_itemid
            and row_date(rows[train_idx]) is not None
            and target_date is not None
            and row_date(rows[train_idx]) < target_date
        ]
        if mode == "dense":
            scores = dense_scores(np.asarray(representations[idx], dtype=float), dense_corpus)
            ranked_positions = sorted(allowed_positions, key=lambda pos: scores[pos], reverse=True)
        else:
            ranked_positions = sorted(
                allowed_positions,
                key=lambda pos: sparse_score(representations[idx], representations[train_indices[pos]]),
                reverse=True,
            )
        ranked_itemids = [str(rows[train_indices[pos]].get("itemid") or "") for pos in ranked_positions]
        for k in k_grid:
            values = [train_targets[itemid] for itemid in ranked_itemids[:k] if itemid in train_targets]
            for agg in aggregators:
                outputs[f"{mode}_k{k}_{agg}"].append(
                    {
                        "split": "validation" if split == "val" else split,
                        "itemid": target_itemid,
                        "predicted_award_eur": aggregate(values, agg),
                        "neighbors_used": len(values),
                    }
                )
    return outputs


def write_outputs(outputs: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        with (output_dir / f"bge_m3_knn_{name}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "itemid", "predicted_award_eur", "neighbors_used"])
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict BGE-M3 dense/sparse KNN")
    parser.add_argument("--documents", required=True, help="Strict text-input CSV or JSONL")
    parser.add_argument("--train-targets", required=True, help="Train target CSV with itemid,y_amount_eur")
    parser.add_argument("--model-name-or-path", default="BAAI/bge-m3")
    parser.add_argument("--mode", choices=["dense", "sparse"], required=True)
    parser.add_argument("--text-field", default=BGE_M3_SETTINGS["text_field"])
    parser.add_argument("--external-factors", type=Path, default=None, help="Optional economic covariate CSV keyed by itemid")
    parser.add_argument("--output-dir", default="outputs/retrieval/bge_m3_knn")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=BGE_M3_SETTINGS["max_length"])
    parser.add_argument("--k-grid", nargs="+", type=int, default=BGE_M3_SETTINGS["k_grid"])
    parser.add_argument("--aggregators", nargs="+", default=BGE_M3_SETTINGS["aggregators"])
    args = parser.parse_args()

    rows = read_rows(Path(args.documents))
    external_factors = read_external_factors(args.external_factors)
    texts = [
        append_external_factors(str(row.get(args.text_field) or ""), str(row.get("itemid") or ""), external_factors)
        for row in rows
    ]
    representations = encode_texts(args.model_name_or_path, texts, args.mode, args.batch_size, args.max_length)
    outputs = run_knn(
        rows,
        read_targets(Path(args.train_targets)),
        representations,
        mode=args.mode,
        k_grid=args.k_grid,
        aggregators=args.aggregators,
    )
    output_dir = Path(args.output_dir) / args.mode
    write_outputs(outputs, output_dir)
    (output_dir / "run_settings.json").write_text(
        json.dumps(
            {
                **BGE_M3_SETTINGS,
                "mode": args.mode,
                "external_factors_appended": bool(external_factors),
                "split_rows": Counter(str(row.get("split") or "") for row in rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

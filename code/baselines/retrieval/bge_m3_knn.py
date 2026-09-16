#!/usr/bin/env python3
"""Paper-protocol BGE-M3 FACTS KNN baselines for ECtHR-NPD.

This runner uses only strict FACTS text, filters train candidates by time and
shared violated Article, and falls back to the training median when there is
no eligible reference. It does not serialize external/audit fields into text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:  # supports both ``python file.py`` and package-style imports in tests
    from .bm25_pfme_knn import (
        PAPER_AGGREGATOR,
        PAPER_K,
        CaseDocument,
        assert_train_document_target_alignment,
        eligible_train_indices,
        load_documents,
        normalize_eval_splits,
        normalize_protocol_split,
        prediction_row_from_ranked_neighbors,
        read_targets,
        training_median,
        validate_paper_protocol_coverage,
        validate_retrieval_selection,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from bm25_pfme_knn import (  # type: ignore[no-redef]
        PAPER_AGGREGATOR,
        PAPER_K,
        CaseDocument,
        assert_train_document_target_alignment,
        eligible_train_indices,
        load_documents,
        normalize_eval_splits,
        normalize_protocol_split,
        prediction_row_from_ranked_neighbors,
        read_targets,
        training_median,
        validate_paper_protocol_coverage,
        validate_retrieval_selection,
    )


FLAGEMBEDDING_IMPORT_ERROR: Exception | None = None
try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as exc:  # pragma: no cover - optional native dependency
    BGEM3FlagModel = None  # type: ignore[assignment]
    FLAGEMBEDDING_IMPORT_ERROR = exc


BGE_M3_SETTINGS = {
    "methods": ["bge_m3_dense", "bge_m3_sparse"],
    "document_field": "facts_text",
    "input_policy": "strict Article-41-free FACTS only; no external-factor serialization",
    "retrieval_corpus": "train split only",
    "temporal_filter": "candidate_judgement_date < target_judgement_date",
    "article_filter": "candidate and target share at least one violated Article",
    "selected_k": PAPER_K,
    "selected_aggregator": PAPER_AGGREGATOR,
    "empty_eligible_set": "training-split median y_amount_eur",
    "max_length": 2048,
}


def encode_texts(model_name_or_path: str, texts: list[str], mode: str, batch_size: int, max_length: int) -> Any:
    if FLAGEMBEDDING_IMPORT_ERROR is not None or BGEM3FlagModel is None:
        raise RuntimeError("FlagEmbedding is unavailable in this environment") from FLAGEMBEDDING_IMPORT_ERROR
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
    documents: list[CaseDocument],
    train_targets: dict[str, float],
    representations: Any,
    *,
    mode: str,
    eval_splits: set[str],
) -> list[dict[str, Any]]:
    """Run the final BGE K=20 median protocol on supplied representations."""
    if len(representations) != len(documents):
        raise ValueError("representation count does not match document count")
    # Match BM25's guard so the BGE corpus cannot silently omit a train
    # document merely because its target ID is absent (or vice versa).
    assert_train_document_target_alignment(documents, train_targets)
    normalized_eval_splits = normalize_eval_splits(eval_splits)
    train_indices = [
        index
        for index, document in enumerate(documents)
        if normalize_protocol_split(document.split) == "train"
    ]
    if not train_indices:
        raise ValueError("No train-split documents with training targets")
    train_documents = [documents[index] for index in train_indices]
    fallback_value = training_median({document.itemid: train_targets[document.itemid] for document in train_documents})
    dense_corpus = np.asarray([representations[index] for index in train_indices], dtype=float) if mode == "dense" else None
    outputs: list[dict[str, Any]] = []

    for index, target in enumerate(documents):
        if normalize_protocol_split(target.split) not in normalized_eval_splits:
            continue
        eligible = eligible_train_indices(train_documents, target)
        if mode == "dense":
            assert dense_corpus is not None
            scores = dense_scores(np.asarray(representations[index], dtype=float), dense_corpus)
            ranked_positions = sorted(eligible, key=lambda position: scores[position], reverse=True)
            ranked = [(train_documents[position], float(scores[position])) for position in ranked_positions]
        elif mode == "sparse":
            ranked_positions = sorted(
                eligible,
                key=lambda position: sparse_score(representations[index], representations[train_indices[position]]),
                reverse=True,
            )
            ranked = [
                (
                    train_documents[position],
                    sparse_score(representations[index], representations[train_indices[position]]),
                )
                for position in ranked_positions
            ]
        else:
            raise ValueError(f"Unsupported BGE retrieval mode {mode!r}")
        outputs.append(
            prediction_row_from_ranked_neighbors(
                target,
                ranked,
                train_targets,
                fallback_value=fallback_value,
            )
        )
    return outputs


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    import csv

    fields = [
        "split",
        "itemid",
        "predicted_award_eur",
        "neighbors_used",
        "neighbor_itemids",
        "neighbor_scores",
        "fallback_reason",
    ]
    with (output_dir / f"bge_m3_{mode}_facts_knn_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-protocol BGE-M3 FACTS KNN")
    parser.add_argument("--documents", required=True, help="Strict FACTS CSV/JSONL with split, date, and violated_articles")
    parser.add_argument("--train-targets", required=True, help="Train target CSV/JSONL with itemid,y_amount_eur")
    parser.add_argument("--model-name-or-path", default="BAAI/bge-m3")
    parser.add_argument("--mode", choices=["dense", "sparse"], required=True)
    parser.add_argument("--output-dir", default="outputs/retrieval/bge_m3_knn")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=BGE_M3_SETTINGS["max_length"])
    parser.add_argument("--eval-splits", nargs="+", default=["val", "validation", "test"])
    parser.add_argument(
        "--dataset-release",
        default=None,
        help="Explicit dataset root or selected clean root; no legacy auto-discovery.",
    )
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], default="corrected")
    parser.add_argument(
        "--example-subset-only",
        action="store_true",
        help=(
            "Permit a deliberately incomplete example subset. This is not a paper-protocol run "
            "or a reproduction of reported results."
        ),
    )
    args = parser.parse_args()

    documents = load_documents(Path(args.documents))
    targets = read_targets(Path(args.train_targets))
    eval_splits = normalize_eval_splits(args.eval_splits)
    coverage = validate_retrieval_selection(documents, targets, eval_splits=eval_splits,
        dataset_release=args.dataset_release, dataset_version=args.dataset_version,
        example_subset_only=args.example_subset_only, input_files={"documents": args.documents, "train_targets": args.train_targets})
    destination = Path(args.output_dir) / args.mode
    if destination.exists() and any(destination.iterdir()):
        parser.error("output directory must be empty; do not overwrite a run")
    representations = encode_texts(
        args.model_name_or_path,
        [document.text for document in documents],
        args.mode,
        args.batch_size,
        args.max_length,
    )
    rows = run_knn(
        documents,
        targets,
        representations,
        mode=args.mode,
        eval_splits=eval_splits,
    )
    output_dir = Path(args.output_dir) / args.mode
    write_outputs(rows, output_dir, args.mode)
    (output_dir / "run_settings.json").write_text(
        json.dumps(
            {
                **BGE_M3_SETTINGS,
                "mode": args.mode,
                "model_name_or_path": args.model_name_or_path,
                "split_rows": Counter(document.split for document in documents),
                "coverage_validation": coverage,
                "protocol_status": (
                    "example_subset_only_not_paper_reproduction"
                    if args.example_subset_only
                    else "full_public_split_coverage_validated_not_historical_result_reproduction"
                ),
                "historical_artifacts": {
                    "status": "not_released",
                    "missing": ["historical FACTS corpus", "historical embeddings", "validation selection trace", "reported predictions"],
                    "result": "This run executes the final reported protocol but is not asserted to reproduce reported numbers.",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Metadata/structured kNN NEW-run entry point with Table 21 filters.

An explicit feature/lineage manifest and distance metric are mandatory because
the paper does not establish their complete historical values. No text input,
model downloads or paid provider are needed. Default coverage is the full cohort.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))
from feature_profiles import TrainOnlyPreprocessor, validate_profile
from retrieval.bm25_pfme_knn import (
    CaseDocument, assert_train_document_target_alignment, eligible_train_indices,
    normalize_eval_splits, normalize_protocol_split, normalize_violated_articles,
    parse_date, prediction_row_from_ranked_neighbors, read_rows, read_targets,
    training_median, validate_paper_protocol_coverage,
    validate_retrieval_selection,
)


def load_inputs(metadata_rows, feature_rows, profile):
    columns = validate_profile(profile)
    allowed_metadata = {"itemid", "split", "judgementdate", "violated_articles"}
    features, documents = {}, []
    for row in feature_rows:
        if set(row) != set(columns) | {"itemid"}:
            raise ValueError("feature rows must contain exactly itemid and manifest columns")
        itemid = str(row["itemid"]).strip()
        if not itemid or itemid in features:
            raise ValueError("blank/duplicate feature itemid")
        features[itemid] = row
    seen = set()
    for row in metadata_rows:
        if set(row) != allowed_metadata:
            raise ValueError("metadata must contain exactly itemid, split, judgementdate, violated_articles")
        itemid = str(row["itemid"]).strip()
        if not itemid or itemid in seen:
            raise ValueError("blank/duplicate metadata itemid")
        seen.add(itemid)
        try:
            date = datetime.strptime(str(row["judgementdate"]), "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("judgementdate must be a valid full YYYY-MM-DD, not a guessed year") from exc
        articles = normalize_violated_articles(row["violated_articles"])
        if date is None or not articles:
            raise ValueError("missing/invalid date or violated Articles; do not silently infer eligibility")
        documents.append(CaseDocument(itemid, normalize_protocol_split(row["split"]), date, "", articles))
    if seen != set(features):
        raise ValueError("metadata/features ID coverage mismatch")
    return documents, [features[document.itemid] for document in documents]


def run_knn(documents, feature_rows, train_targets, profile, *, metric, eval_splits):
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("choose and record an explicit metric: euclidean or cosine")
    if len(documents) != len(feature_rows):
        raise ValueError("document/feature row count mismatch")
    assert_train_document_target_alignment(documents, train_targets)
    train_indices = [i for i, doc in enumerate(documents) if doc.split == "train"]
    preprocessor = TrainOnlyPreprocessor().fit([feature_rows[i] for i in train_indices], profile)
    matrix = preprocessor.transform(feature_rows)
    train_docs = [documents[i] for i in train_indices]
    corpus = matrix[train_indices]
    fallback = training_median(train_targets)
    outputs = []
    for i, target in enumerate(documents):
        if target.split not in normalize_eval_splits(eval_splits):
            continue
        eligible = eligible_train_indices(train_docs, target)
        if metric == "euclidean":
            scores = -np.linalg.norm(corpus - matrix[i], axis=1)
        else:
            scores = (corpus @ matrix[i]) / np.maximum(np.linalg.norm(corpus, axis=1) * np.linalg.norm(matrix[i]), 1e-12)
        ranked = sorted(eligible, key=lambda pos: (-scores[pos], train_docs[pos].itemid))
        outputs.append(prediction_row_from_ranked_neighbors(target, [(train_docs[pos], scores[pos]) for pos in ranked],
                                                          train_targets, fallback_value=fallback))
    return outputs, preprocessor.manifest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--profile", required=True, help="JSON with profile, columns, categorical_columns, lineage")
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--metric", required=True, choices=["euclidean", "cosine"])
    parser.add_argument("--dataset-release", default=None)
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], default="corrected")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--example-subset-only", action="store_true")
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    if profile.get("profile") not in {"metadata_knn", "structured_knn"}:
        parser.error("use a metadata_knn or structured_knn profile")
    docs, rows = load_inputs(read_rows(Path(args.metadata)), read_rows(Path(args.features)), profile)
    targets = read_targets(Path(args.train_targets))
    coverage = validate_retrieval_selection(docs, targets, eval_splits={"validation", "test"},
        dataset_release=args.dataset_release, dataset_version=args.dataset_version,
        example_subset_only=args.example_subset_only,
        input_files={name: getattr(args, name) for name in ["metadata", "features", "profile", "train_targets"]})
    predictions, preprocessing = run_knn(docs, rows, targets, profile, metric=args.metric, eval_splits={"validation", "test"})
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; do not overwrite a run")
    output.mkdir(parents=True, exist_ok=True)
    if predictions:
        with (output / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    manifest = {"status": "new_implementation_not_historical_reproduction", "profile": profile,
                "metric": args.metric, "k": 20, "aggregation": "median", "tie_break": "itemid_ascending",
                "coverage": coverage, "preprocessing": preprocessing,
                "input_sha256": {name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
                                 for name in ["metadata", "features", "profile", "train_targets"]},
                "unknown_historical_fields": ["distance_metric", "feature_profile", "preprocessing", "selection_trace"]}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

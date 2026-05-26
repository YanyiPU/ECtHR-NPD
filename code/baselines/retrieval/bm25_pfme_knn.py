#!/usr/bin/env python3
"""Strict BM25 PFME-KNN retrieval baseline.

This reproduces the release setting at the code level without shipping
raw judgment text. Users must supply strict redacted/procedure-facts
case inputs separately. The retrieval corpus is train-only, and every
candidate reference is temporally prior to the target case.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable


STRICT_DOCUMENT_KEYS = (
    "procedure_facts_text",
    "oracle_violated_articles",
    "metadata",
    "external_factors",
    "extracted_hints",
)

FORBIDDEN_DOCUMENT_KEYS = (
    "itemid",
    "article41_structured_hints",
    "article_41",
    "article41",
    "just_satisfaction",
    "operative",
    "claim",
    "claimed",
    "award",
    "safe_non_pec",
    "target",
    "y_amount",
    "y_binary",
)

BM25_SETTINGS = {
    "setting_name": "strict_bm25_pfme_knn",
    "baseline": "BM25 PFME-KNN",
    "mode": "strict_procedure_facts_plus_metadata_extracted",
    "document_source": "case_inputs without itemid",
    "included_case_input_keys": list(STRICT_DOCUMENT_KEYS),
    "excluded_case_input_keys": ["itemid", "article41_structured_hints"],
    "query_token_policy": "deduplicate query tokens before BM25 scoring, then cap to max_query_terms",
    "max_query_terms": 512,
    "k_grid": [1, 3, 5, 10, 15, 20, 25, 30],
    "aggregators": ["mean", "median"],
    "temporal_filter": "candidate_judgement_date < target_judgement_date",
    "retrieval_corpus": "train split only",
    "positive_threshold_eur": 0.5,
}


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


@dataclass(frozen=True)
class CaseDocument:
    itemid: str
    split: str
    judgementdate: datetime | None
    text: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


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


def stable_serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def forbidden_key_visible(key: str) -> bool:
    lowered = str(key or "").lower()
    return any(fragment in lowered for fragment in FORBIDDEN_DOCUMENT_KEYS)


def assemble_strict_document(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in STRICT_DOCUMENT_KEYS:
        if forbidden_key_visible(key):
            raise ValueError(f"forbidden strict document key: {key}")
        value = row.get(key)
        if value in (None, ""):
            continue
        parts.append(stable_serialize(value))
    return "\n".join(parts)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_targets(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row["itemid"]): float(row["y_amount_eur"])
            for row in csv.DictReader(handle)
            if row.get("itemid") and row.get("y_amount_eur") not in (None, "")
        }


def load_documents(path: Path) -> list[CaseDocument]:
    documents: list[CaseDocument] = []
    for row in read_rows(path):
        itemid = str(row.get("itemid") or "").strip()
        if not itemid:
            continue
        documents.append(
            CaseDocument(
                itemid=itemid,
                split=str(row.get("split") or "").strip(),
                judgementdate=parse_date(row.get("judgementdate") or row.get("judgment_date") or row.get("chrono_date")),
                text=assemble_strict_document(row),
            )
        )
    return documents


class BM25Index:
    def __init__(self, documents: list[CaseDocument], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        doc_freq: Counter[str] = Counter()

        for document in documents:
            counts = Counter(tokenize(document.text))
            self.term_freqs.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            doc_freq.update(counts.keys())

        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        total = len(documents)
        self.idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query_terms: Iterable[str], allowed_indices: set[int]) -> list[tuple[int, float]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx in allowed_indices:
                tf = self.term_freqs[idx].get(term, 0)
                if tf <= 0:
                    continue
                dl = self.doc_lengths[idx] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[idx] += idf * tf * (self.k1 + 1.0) / denom
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def query_terms(document: CaseDocument, max_terms: int) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(document.text):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def aggregate(values: list[float], method: str) -> float:
    if not values:
        return 0.0
    if method == "mean":
        return float(sum(values) / len(values))
    if method == "median":
        return float(median(values))
    raise ValueError(f"unknown aggregator: {method}")


def run_retrieval(
    documents: list[CaseDocument],
    train_targets: dict[str, float],
    *,
    eval_splits: set[str],
    k_grid: list[int],
    aggregators: list[str],
    max_query_terms: int,
) -> dict[str, list[dict[str, Any]]]:
    train_documents = [doc for doc in documents if doc.split == "train" and doc.itemid in train_targets]
    index = BM25Index(train_documents)
    outputs: dict[str, list[dict[str, Any]]] = {f"k{k}_{agg}": [] for k in k_grid for agg in aggregators}

    for target in documents:
        if target.split not in eval_splits:
            continue
        allowed = {
            idx
            for idx, candidate in enumerate(train_documents)
            if candidate.itemid != target.itemid
            and candidate.judgementdate is not None
            and target.judgementdate is not None
            and candidate.judgementdate < target.judgementdate
        }
        ranked = index.score(query_terms(target, max_query_terms), allowed)
        ranked_ids = [train_documents[idx].itemid for idx, _score in ranked]
        for k in k_grid:
            neighbor_values = [train_targets[itemid] for itemid in ranked_ids[:k] if itemid in train_targets]
            for agg in aggregators:
                outputs[f"k{k}_{agg}"].append(
                    {
                        "split": target.split,
                        "itemid": target.itemid,
                        "predicted_award_eur": aggregate(neighbor_values, agg),
                        "neighbors_used": len(neighbor_values),
                    }
                )
    return outputs


def write_outputs(outputs: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        path = output_dir / f"bm25_pfme_knn_{name}_predictions.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "itemid", "predicted_award_eur", "neighbors_used"])
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict BM25 PFME-KNN on user-supplied strict case inputs")
    parser.add_argument("--documents", required=True, help="Strict case-input CSV or JSONL with train/val/test split column")
    parser.add_argument("--train-targets", required=True, help="Train target CSV with itemid,y_amount_eur")
    parser.add_argument("--output-dir", default="outputs/retrieval/bm25_pfme_knn")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"])
    parser.add_argument("--k-grid", nargs="+", type=int, default=BM25_SETTINGS["k_grid"])
    parser.add_argument("--aggregators", nargs="+", default=BM25_SETTINGS["aggregators"])
    parser.add_argument("--max-query-terms", type=int, default=BM25_SETTINGS["max_query_terms"])
    args = parser.parse_args()

    documents = load_documents(Path(args.documents))
    train_targets = read_targets(Path(args.train_targets))
    outputs = run_retrieval(
        documents,
        train_targets,
        eval_splits=set(args.eval_splits),
        k_grid=args.k_grid,
        aggregators=args.aggregators,
        max_query_terms=args.max_query_terms,
    )
    output_dir = Path(args.output_dir)
    write_outputs(outputs, output_dir)
    (output_dir / "run_settings.json").write_text(
        json.dumps({**BM25_SETTINGS, "split_rows": Counter(doc.split for doc in documents)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

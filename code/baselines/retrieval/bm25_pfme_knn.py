#!/usr/bin/env python3
"""Paper-protocol BM25S FACTS KNN baseline for ECtHR-NPD.

The public release intentionally does not include the historical FACTS text,
candidate-selection trace, predictions, or index. A user can supply strict
FACTS inputs to run the protocol, but the script never labels that new run as
a reproduction of the reported historical result.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable


BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from data.data_loader import (resolve_dataset_release, load_release_contract, require_corrected_version,
                              dataset_provenance, validate_selected_targets, file_sha256)


PAPER_TEXT_FIELD = "facts_text"
PAPER_K = 20
PAPER_AGGREGATOR = "median"
BM25S_VERSION = "0.3.8"
FORBIDDEN_DOCUMENT_KEY_FRAGMENTS = (
    "article41",
    "article_41",
    "article 41",
    "article50",
    "article_50",
    "just_satisfaction",
    "operative",
    "claim",
    "claimed",
    "award",
    "safe_non_pec",
    "safe_total",
    "target",
    "y_amount",
    "y_binary",
    "raw_extractor",
    "appno",
    "ecli",
)

BM25_SETTINGS = {
    "setting_name": "strict_bm25_pfme_knn",
    "baseline": "BM25S FACTS KNN",
    "backend": f"bm25s=={BM25S_VERSION}",
    "document_field": PAPER_TEXT_FIELD,
    "retrieval_corpus": "train split only",
    "temporal_filter": "candidate_judgement_date < target_judgement_date",
    "article_filter": "candidate and target share at least one violated Article",
    "selected_k": PAPER_K,
    "selected_aggregator": PAPER_AGGREGATOR,
    "empty_eligible_set": "training-split median y_amount_eur",
}


class PaperProtocolCoverageError(ValueError):
    """Raised when supplied retrieval documents are not the full fixed cohort."""


SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def normalize_protocol_split(value: Any, *, context: str = "split") -> str:
    """Normalize a split label and reject labels outside the public protocol."""
    raw = str(value or "").strip().lower()
    normalized = SPLIT_ALIASES.get(raw)
    if normalized is None:
        raise PaperProtocolCoverageError(
            f"{context} {value!r} is not one of train, val/validation, or test"
        )
    return normalized


def normalize_eval_splits(eval_splits: Iterable[str]) -> set[str]:
    """Normalize requested validation/test labels for retrieval evaluation."""
    normalized = {normalize_protocol_split(value, context="evaluation split") for value in eval_splits}
    if not normalized:
        raise PaperProtocolCoverageError("at least one evaluation split must be requested")
    unsupported = normalized - {"validation", "test"}
    if unsupported:
        raise PaperProtocolCoverageError(
            "paper-protocol retrieval evaluation supports validation and test only; got "
            + ", ".join(sorted(unsupported))
        )
    return normalized


@dataclass(frozen=True)
class CaseDocument:
    itemid: str
    split: str
    judgementdate: datetime | None
    text: str
    violated_articles: frozenset[str]


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


def normalize_violated_articles(value: Any) -> frozenset[str]:
    """Normalize CSV/JSON article lists for the mandatory shared-Article filter."""
    if value is None:
        return frozenset()
    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return frozenset()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text
        if isinstance(parsed, str):
            parsed = re.split(r"[;,|]", parsed)
    if not isinstance(parsed, (list, tuple, set, frozenset)):
        parsed = [parsed]
    normalized = {
        re.sub(r"\s+", "", str(article)).upper().replace("_", "-")
        for article in parsed
        if str(article).strip()
    }
    return frozenset(normalized)


def forbidden_key_visible(key: str) -> bool:
    lowered = str(key or "").lower()
    return any(fragment in lowered for fragment in FORBIDDEN_DOCUMENT_KEY_FRAGMENTS)


def assert_strict_document_row(row: dict[str, Any]) -> None:
    bad = sorted(key for key in row if forbidden_key_visible(key))
    if bad:
        raise ValueError("strict retrieval document contains forbidden keys: " + ", ".join(bad))
    missing = [key for key in ("itemid", "split", PAPER_TEXT_FIELD, "violated_articles") if key not in row]
    if missing:
        raise ValueError("strict retrieval document is missing required columns: " + ", ".join(missing))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(row)
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_targets(path: Path) -> dict[str, float]:
    targets: dict[str, float] = {}
    for row_number, row in enumerate(read_rows(path), start=1):
        itemid = str(row.get("itemid") or "").strip()
        if not itemid:
            raise PaperProtocolCoverageError(f"train targets row {row_number} has a blank itemid")
        if row.get("y_amount_eur") in (None, ""):
            raise PaperProtocolCoverageError(f"train targets {itemid!r} has a blank y_amount_eur")
        if itemid in targets:
            raise PaperProtocolCoverageError(f"duplicate target itemid {itemid!r}")
        try:
            targets[itemid] = float(row["y_amount_eur"])
        except (TypeError, ValueError) as exc:
            raise PaperProtocolCoverageError(
                f"train targets {itemid!r} has a non-numeric y_amount_eur"
            ) from exc
    if not targets:
        raise PaperProtocolCoverageError("train targets contain no itemid,y_amount_eur rows")
    if any(not math.isfinite(value) or value < 0 for value in targets.values()):
        raise PaperProtocolCoverageError("train targets must be finite, nonnegative EUR")
    return targets


def load_documents(path: Path) -> list[CaseDocument]:
    documents: list[CaseDocument] = []
    itemids: set[str] = set()
    for row in read_rows(path):
        assert_strict_document_row(row)
        itemid = str(row["itemid"]).strip()
        if not itemid:
            raise ValueError("strict retrieval document contains a blank itemid")
        if itemid in itemids:
            raise ValueError(f"duplicate document itemid {itemid!r}")
        text = str(row[PAPER_TEXT_FIELD] or "").strip()
        if not text:
            raise ValueError(f"strict retrieval document {itemid!r} has empty {PAPER_TEXT_FIELD}")
        itemids.add(itemid)
        documents.append(
            CaseDocument(
                itemid=itemid,
                split=normalize_protocol_split(row["split"], context=f"document {itemid!r} split"),
                judgementdate=parse_date(row.get("judgementdate") or row.get("judgment_date") or row.get("chrono_date")),
                text=text,
                violated_articles=normalize_violated_articles(row.get("violated_articles")),
            )
        )
    if not documents:
        raise ValueError(f"no strict retrieval documents in {path}")
    return documents


def _document_itemids_by_split(documents: Iterable[CaseDocument]) -> dict[str, set[str]]:
    """Collect IDs without allowing a duplicate to be hidden by set conversion."""
    itemids_by_split = {"train": set(), "validation": set(), "test": set()}
    seen_itemids: set[str] = set()
    for document in documents:
        itemid = str(document.itemid or "").strip()
        if not itemid:
            raise PaperProtocolCoverageError("retrieval documents contain a blank itemid")
        if itemid in seen_itemids:
            raise PaperProtocolCoverageError(f"duplicate document itemid {itemid!r}")
        seen_itemids.add(itemid)
        split = normalize_protocol_split(document.split, context=f"document {itemid!r} split")
        itemids_by_split[split].add(itemid)
    return itemids_by_split


def _describe_id_difference(actual: set[str], expected: set[str]) -> str:
    """Make cohort mismatches actionable without dumping a whole split."""
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = [f"actual={len(actual)}", f"expected={len(expected)}"]
    if missing:
        details.append(f"missing={len(missing)} (for example: {', '.join(missing[:5])})")
    if extra:
        details.append(f"unexpected={len(extra)} (for example: {', '.join(extra[:5])})")
    return "; ".join(details)


def _require_exact_itemids(actual: set[str], expected: set[str], *, context: str) -> None:
    if actual != expected:
        raise PaperProtocolCoverageError(f"{context} IDs do not match exactly: {_describe_id_difference(actual, expected)}")


def assert_train_document_target_alignment(
    documents: Iterable[CaseDocument], train_targets: dict[str, float]
) -> dict[str, set[str]]:
    """Require one document for every training target and no orphan train text.

    Both retrieval backends call this guard before indexing.  Without it, the
    old ``itemid in train_targets`` filter silently excluded document rows and
    changed the retrieval corpus.
    """
    itemids_by_split = _document_itemids_by_split(documents)
    target_ids = {str(itemid).strip() for itemid in train_targets}
    if "" in target_ids:
        raise PaperProtocolCoverageError("train targets contain a blank itemid")
    _require_exact_itemids(
        itemids_by_split["train"],
        target_ids,
        context="train retrieval documents versus train targets",
    )
    return itemids_by_split


def canonical_split_itemids(dataset_release: str | Path | None = None, *, dataset_version="corrected") -> dict[str, set[str]]:
    """Read canonical fixed split IDs from the public release case index."""
    require_corrected_version(dataset_version)
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    from data.public_adapter import is_public_root, canonical_cases
    if is_public_root(release):
        rows = canonical_cases(release)
        return {name: set(rows.loc[rows["split"] == name, "itemid"])
                for name in ("train", "validation", "test")}
    cases_path = release / "data" / "ecthr_npd_cases.csv"
    canonical = {"train": set(), "validation": set(), "test": set()}
    seen_itemids: set[str] = set()
    for row_number, row in enumerate(read_rows(cases_path), start=1):
        itemid = str(row.get("itemid") or "").strip()
        if not itemid:
            raise PaperProtocolCoverageError(f"canonical case index row {row_number} has a blank itemid")
        if itemid in seen_itemids:
            raise PaperProtocolCoverageError(f"canonical case index has duplicate itemid {itemid!r}")
        seen_itemids.add(itemid)
        split = normalize_protocol_split(row.get("split"), context=f"canonical case {itemid!r} split")
        canonical[split].add(itemid)
    if not any(canonical.values()):
        raise PaperProtocolCoverageError(f"canonical case index contains no rows: {cases_path}")
    expected = load_release_contract(release)["counts"]["splits"]
    if {name: len(ids) for name, ids in canonical.items()} != expected:
        raise PaperProtocolCoverageError("canonical split counts differ from selected contract")
    return canonical


def validate_paper_protocol_coverage(
    documents: list[CaseDocument],
    train_targets: dict[str, float],
    *,
    eval_splits: Iterable[str],
    dataset_release: str | Path | None = None,
    dataset_version: str = "corrected",
) -> dict[str, Any]:
    """Validate full-cohort paper-protocol coverage before a retrieval run.

    The public release fixes the train/validation/test membership.  A default
    command-line run must cover every train target, then every requested
    validation/test ID, exactly once.  Tiny demonstrations can only bypass
    this check through the explicitly non-reproduction ``--example-subset-only``
    option in the runners.
    """
    requested_splits = normalize_eval_splits(eval_splits)
    document_ids = assert_train_document_target_alignment(documents, train_targets)
    canonical_ids = canonical_split_itemids(dataset_release, dataset_version=dataset_version)
    provenance = validate_selected_targets(list(train_targets), list(train_targets.values()), dataset_release,
                                            dataset_version=dataset_version, splits=["train"] * len(train_targets))
    target_ids = {str(itemid).strip() for itemid in train_targets}
    _require_exact_itemids(
        target_ids,
        canonical_ids["train"],
        context="train targets versus canonical public train split",
    )
    for split in sorted(requested_splits):
        _require_exact_itemids(
            document_ids[split],
            canonical_ids[split],
            context=f"{split} retrieval documents versus canonical public {split} split",
        )
    return {
        "train_document_rows": len(document_ids["train"]),
        "train_target_rows": len(target_ids),
        **{f"{split}_document_rows": len(document_ids[split]) for split in sorted(requested_splits)},
        "coverage_mode": "full_fixed_public_splits",
        "dataset_provenance": provenance,
    }


def validate_retrieval_selection(documents, targets, *, eval_splits, dataset_release=None,
                                 dataset_version="corrected", example_subset_only=False, input_files=None):
    """Explicit examples may be unversioned, but selected examples cannot mix labels."""
    require_corrected_version(dataset_version)
    from data.public_adapter import is_public_root, canonical_cases
    selected_root = dataset_release or os.environ.get("ECTHR_NPD_DATASET_RELEASE")
    if is_public_root(selected_root):
        canonical_rows = canonical_cases(selected_root).set_index("itemid")
        for document in documents:
            if document.itemid not in canonical_rows.index:
                raise PaperProtocolCoverageError("Retrieval document is outside the public dataset")
            row = canonical_rows.loc[document.itemid]
            if document.split != row["split"] or document.judgementdate != parse_date(row["judgementdate"]) or document.violated_articles != normalize_violated_articles(row["violated_articles"]):
                raise PaperProtocolCoverageError("Retrieval split/date/Articles differ from the public dataset; eligibility cannot be overridden")
    if not example_subset_only:
        result = validate_paper_protocol_coverage(documents, targets, eval_splits=eval_splits,
                                                  dataset_release=dataset_release, dataset_version=dataset_version)
    else:
        actual = assert_train_document_target_alignment(documents, targets)
        result = {"coverage_mode": "example_subset_only_not_paper_reproduction"}
        if dataset_release or os.environ.get("ECTHR_NPD_DATASET_RELEASE"):
            canonical = canonical_split_itemids(dataset_release, dataset_version=dataset_version)
            if any(actual[split] - canonical[split] for split in actual):
                raise PaperProtocolCoverageError("example IDs/splits differ from selected dataset")
            result["dataset_provenance"] = validate_selected_targets(list(targets), list(targets.values()), dataset_release,
                                                        dataset_version=dataset_version, splits=["train"] * len(targets))
        else:
            result["dataset_provenance"] = {"dataset_version": "unversioned_example", "training_status": "synthetic_or_external_example_only"}
    result["dataset_provenance"]["run_input_sha256"] = {name: file_sha256(path) for name, path in (input_files or {}).items() if path is not None}
    return result


def eligible_train_indices(train_documents: list[CaseDocument], target: CaseDocument) -> set[int]:
    """Apply both required eligibility rules before selecting KNN neighbors."""
    if target.judgementdate is None or not target.violated_articles:
        return set()
    return {
        index
        for index, candidate in enumerate(train_documents)
        if candidate.itemid != target.itemid
        and candidate.judgementdate is not None
        and candidate.judgementdate < target.judgementdate
        and bool(candidate.violated_articles & target.violated_articles)
    }


def training_median(train_targets: dict[str, float]) -> float:
    if not train_targets:
        raise ValueError("Cannot calculate retrieval fallback: no training targets")
    return float(median(train_targets.values()))


def prediction_row_from_ranked_neighbors(
    target: CaseDocument,
    ranked_neighbors: Iterable[tuple[CaseDocument, float]],
    train_targets: dict[str, float],
    *,
    fallback_value: float,
    k: int = PAPER_K,
) -> dict[str, Any]:
    selected = list(ranked_neighbors)[:k]
    # All ranked candidates originate from the train corpus, whose IDs were
    # required to equal ``train_targets`` before indexing.  Direct lookup is
    # intentional: a broken invariant must fail loudly, never drop a neighbor.
    values = [train_targets[document.itemid] for document, _score in selected]
    fallback = not values
    return {
        "split": "validation" if target.split == "val" else target.split,
        "itemid": target.itemid,
        "predicted_award_eur": float(median(values)) if values else float(fallback_value),
        "neighbors_used": len(values),
        "neighbor_itemids": json.dumps([document.itemid for document, _score in selected]),
        "neighbor_scores": json.dumps([float(score) for _document, score in selected]),
        "fallback_reason": "no_eligible_temporal_shared_article_neighbor" if fallback else "",
    }


def require_bm25s() -> Any:
    try:
        installed_version = importlib.metadata.version("bm25s")
        import bm25s
    except Exception as exc:  # pragma: no cover - optional package
        raise RuntimeError(
            f"The paper BM25 baseline requires bm25s=={BM25S_VERSION}; install that exact package before running."
        ) from exc
    if installed_version != BM25S_VERSION:
        raise RuntimeError(
            f"The paper BM25 baseline requires bm25s=={BM25S_VERSION}, but {installed_version} is installed."
        )
    return bm25s


def rank_all_bm25s(bm25s: Any, index: Any, query_text: str, corpus_size: int) -> list[tuple[int, float]]:
    """Retrieve the full train corpus, then enforce eligibility without a rank cutoff leak."""
    query_tokens = bm25s.tokenize([query_text])
    indices, scores = index.retrieve(query_tokens, k=corpus_size)
    return [(int(index_), float(score)) for index_, score in zip(indices[0], scores[0]) if int(index_) >= 0]


def run_retrieval(
    documents: list[CaseDocument],
    train_targets: dict[str, float],
    *,
    eval_splits: set[str],
    bm25s_module: Any | None = None,
) -> list[dict[str, Any]]:
    # Do this in the backend-independent runner as well as the CLI.  It keeps
    # library callers from silently shrinking the train corpus via an inner
    # membership filter.
    assert_train_document_target_alignment(documents, train_targets)
    normalized_eval_splits = normalize_eval_splits(eval_splits)
    train_documents = [document for document in documents if normalize_protocol_split(document.split) == "train"]
    if not train_documents:
        raise PaperProtocolCoverageError("No train-split documents with training targets")
    bm25s = bm25s_module or require_bm25s()
    index = bm25s.BM25()
    index.index(bm25s.tokenize([document.text for document in train_documents]))
    fallback_value = training_median({document.itemid: train_targets[document.itemid] for document in train_documents})
    outputs: list[dict[str, Any]] = []
    for target in documents:
        if normalize_protocol_split(target.split) not in normalized_eval_splits:
            continue
        eligible = eligible_train_indices(train_documents, target)
        ranked = rank_all_bm25s(bm25s, index, target.text, len(train_documents))
        eligible_ranked = [
            (train_documents[index], score)
            for index, score in ranked
            if index in eligible
        ]
        outputs.append(
            prediction_row_from_ranked_neighbors(
                target,
                eligible_ranked,
                train_targets,
                fallback_value=fallback_value,
            )
        )
    return outputs


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "itemid",
        "predicted_award_eur",
        "neighbors_used",
        "neighbor_itemids",
        "neighbor_scores",
        "fallback_reason",
    ]
    with (output_dir / "bm25s_facts_knn_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-protocol BM25S FACTS KNN baseline")
    parser.add_argument("--documents", required=True, help="Strict FACTS CSV/JSONL with split, date, and violated_articles")
    parser.add_argument("--train-targets", required=True, help="Train target CSV/JSONL with itemid,y_amount_eur")
    parser.add_argument("--output-dir", default="outputs/retrieval/bm25s_facts_knn")
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
    if Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        parser.error("output directory must be empty; do not overwrite a run")
    rows = run_retrieval(documents, targets, eval_splits=eval_splits)
    output_dir = Path(args.output_dir)
    write_outputs(rows, output_dir)
    (output_dir / "run_settings.json").write_text(
        json.dumps(
            {
                **BM25_SETTINGS,
                "split_rows": Counter(document.split for document in documents),
                "coverage_validation": coverage,
                "protocol_status": (
                    "example_subset_only_not_paper_reproduction"
                    if args.example_subset_only
                    else "full_public_split_coverage_validated_not_historical_result_reproduction"
                ),
                "historical_artifacts": {
                    "status": "not_released",
                    "missing": ["historical FACTS corpus", "historical BM25S index", "validation selection trace", "reported predictions"],
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

#!/usr/bin/env python3
"""Encoder input loader for ECtHR-NPD baselines.

The public release does not redistribute raw HUDOC text. Encoder runs can
therefore use either user-supplied strict Article-41-free text inputs or a
serialized version of the public strict inputs: safe metadata, violated
articles, case facts as structured features, and external factors.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from data.data_loader import (
    assert_no_forbidden_strict_columns,
    load_external_factors,
    load_structured_tree_splits,
)


FORBIDDEN_TEXT_INPUT_TERMS = (
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
    "raw_extractor",
    "target",
    "y_amount",
    "y_binary",
    "case_name",
    "appno",
    "ecli",
)


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


def assert_no_forbidden_keys(row: dict[str, Any]) -> None:
    bad = [
        key for key in row
        if any(term in str(key).lower() for term in FORBIDDEN_TEXT_INPUT_TERMS)
    ]
    if bad:
        raise ValueError(f"strict encoder text input contains forbidden keys: {sorted(bad)}")


def load_user_text_inputs(path: str | Path, text_field: str) -> pd.DataFrame:
    """Load strict user text without deduplicating or dropping any records.

    Coverage against the fixed public splits is checked by
    :func:`load_encoder_splits`.  This reader deliberately fails on malformed
    rows first, so a typo, duplicate, or empty text cannot turn into a silent
    change in the number of evaluated cases.
    """
    rows = read_rows(Path(path))
    records: list[dict[str, Any]] = []
    seen_itemids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"strict encoder text input row {row_number} is not an object")
        assert_no_forbidden_keys(row)
        if "itemid" not in row:
            raise ValueError(f"strict encoder text input row {row_number} is missing itemid")
        if text_field not in row:
            raise ValueError(
                f"strict encoder text input row {row_number} is missing requested text field {text_field!r}"
            )
        itemid = str(row.get("itemid") or "").strip()
        text = str(row.get(text_field) or "").strip()
        if not itemid:
            raise ValueError(f"strict encoder text input row {row_number} has a blank itemid")
        if itemid in seen_itemids:
            raise ValueError(f"duplicate strict encoder text itemid {itemid!r}")
        if not text:
            raise ValueError(
                f"strict encoder text input {itemid!r} has an empty requested text field {text_field!r}"
            )
        seen_itemids.add(itemid)
        records.append({"itemid": itemid, "text": text})
    if not records:
        raise ValueError(f"no usable strict encoder text rows in {path}")
    return pd.DataFrame(records)


def _expected_text_itemids_by_split(splits: dict[str, Any]) -> dict[str, set[str]]:
    """Return exact public target IDs for each fixed encoder split."""
    expected: dict[str, set[str]] = {}
    for split_name, split in splits.items():
        itemids = split.targets["itemid"].astype(str).str.strip()
        if (itemids == "").any():
            raise ValueError(f"{split_name} encoder targets contain a blank itemid")
        if itemids.duplicated().any():
            raise ValueError(f"{split_name} encoder targets contain duplicate itemid values")
        expected[split_name] = set(itemids)
    return expected


def assert_complete_user_text_coverage(text_frame: pd.DataFrame, splits: dict[str, Any]) -> None:
    """Reject unknown or missing text rather than letting an inner join drop it.

    User-provided text must cover every fixed train/validation/test target
    exactly once.  The full check belongs here, before labels are joined to
    text, because an inner join would otherwise make incomplete evaluation
    look valid.
    """
    if text_frame["itemid"].duplicated().any():
        duplicates = text_frame.loc[text_frame["itemid"].duplicated(), "itemid"].astype(str).tolist()
        raise ValueError(f"duplicate strict encoder text itemid(s): {duplicates[:5]}")

    expected_by_split = _expected_text_itemids_by_split(splits)
    expected_all = set().union(*expected_by_split.values())
    actual = set(text_frame["itemid"].astype(str).str.strip())
    unknown = sorted(actual - expected_all)
    if unknown:
        raise ValueError(
            "strict encoder text inputs contain itemid(s) outside the public fixed splits: "
            + ", ".join(unknown[:10])
        )

    missing_messages: list[str] = []
    for split_name, expected in expected_by_split.items():
        missing = sorted(expected - actual)
        if missing:
            missing_messages.append(f"{split_name}={len(missing)} (for example: {', '.join(missing[:5])})")
    if missing_messages:
        raise ValueError("strict encoder text inputs are incomplete by split: " + "; ".join(missing_messages))


def serialize_record(record: dict[str, Any]) -> str:
    assert_no_forbidden_strict_columns(record.keys())
    audit_only = {"split", "test_view", "test_challenging_view", "hudoc_url"} & set(record)
    if audit_only:
        raise ValueError(
            "strict encoder serialization received audit-only fields: " + ", ".join(sorted(audit_only))
        )
    clean = {
        key: value
        for key, value in record.items()
        if key != "itemid" and value not in (None, "")
    }
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def load_serialized_strict_inputs(dataset_release: str | Path | None = None, *, dataset_version="corrected") -> pd.DataFrame:
    splits = load_structured_tree_splits(dataset_release, dataset_version=dataset_version)
    external = load_external_factors(dataset_release, dataset_version=dataset_version).copy()
    external["itemid"] = external["itemid"].astype(str)
    frames: list[pd.DataFrame] = []
    for split_name, split in splits.items():
        features = split.features.copy()
        features["itemid"] = features["itemid"].astype(str)
        # Merge only allow-listed factors that are not already part of the
        # strict feature matrix. This prevents duplicate/raw source columns
        # from being silently serialized into encoder text.
        external_to_merge = external[[
            column for column in external.columns if column == "itemid" or column not in features.columns
        ]]
        features = features.merge(external_to_merge, on="itemid", how="left", validate="one_to_one")
        text_rows = [
            {
                "itemid": str(row["itemid"]),
                "text": serialize_record(row.to_dict()),
            }
            for _, row in features.iterrows()
        ]
        frame = pd.DataFrame(text_rows)
        frame["split"] = split_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_encoder_splits(
    dataset_release: str | Path | None = None,
    *,
    text_inputs: str | Path | None = None,
    text_field: str = "combined_input_text_with_violated_articles",
    dataset_version: str = "corrected",
) -> dict[str, pd.DataFrame]:
    splits = load_structured_tree_splits(dataset_release, dataset_version=dataset_version)
    if text_inputs:
        text_frame = load_user_text_inputs(text_inputs, text_field)
        assert_complete_user_text_coverage(text_frame, splits)
    else:
        text_frame = load_serialized_strict_inputs(dataset_release, dataset_version=dataset_version)[["itemid", "text"]]
    text_frame["itemid"] = text_frame["itemid"].astype(str)

    out: dict[str, pd.DataFrame] = {}
    for split_name, split in splits.items():
        labels = split.targets[["itemid", "y_amount_eur", "y_binary"]].copy()
        labels["itemid"] = labels["itemid"].astype(str)
        cases = split.cases[["itemid", "split", "test_view", "test_challenging_view"]].copy()
        cases["itemid"] = cases["itemid"].astype(str)
        frame = labels.merge(cases, on="itemid", how="left", validate="one_to_one")
        # ``how='left'`` preserves every target row.  Text coverage was
        # validated above, and this explicit check prevents any future change
        # from reintroducing the old inner-join row loss.
        frame = frame.merge(text_frame, on="itemid", how="left", validate="one_to_one")
        if frame["text"].isna().any():
            missing = frame.loc[frame["text"].isna(), "itemid"].astype(str).tolist()
            raise ValueError(
                f"{split_name} encoder split has target row(s) with no text after coverage validation: {missing[:10]}"
            )
        if len(frame) != len(labels):
            raise ValueError(
                f"{split_name} encoder split changed row count during text assembly: {len(frame)} != {len(labels)}"
            )
        frame["y_amount_eur"] = pd.to_numeric(frame["y_amount_eur"], errors="raise")
        frame["y_binary"] = pd.to_numeric(frame["y_binary"], errors="raise").astype(int)
        out[split_name] = frame[["itemid", "split", "text", "y_amount_eur", "y_binary", "test_view", "test_challenging_view"]]
    return out

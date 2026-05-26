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

from data.data_loader import load_external_factors, load_structured_tree_splits


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
    rows = read_rows(Path(path))
    records: list[dict[str, Any]] = []
    for row in rows:
        assert_no_forbidden_keys(row)
        itemid = str(row.get("itemid") or "").strip()
        text = str(row.get(text_field) or "").strip()
        if not itemid or not text:
            continue
        records.append({"itemid": itemid, "text": text})
    if not records:
        raise ValueError(f"no usable strict encoder text rows in {path}")
    return pd.DataFrame(records).drop_duplicates("itemid", keep="first")


def serialize_record(record: dict[str, Any]) -> str:
    clean = {
        key: value
        for key, value in record.items()
        if key != "itemid" and value not in (None, "")
    }
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def load_serialized_strict_inputs(dataset_release: str | Path | None = None) -> pd.DataFrame:
    splits = load_structured_tree_splits(dataset_release)
    external = load_external_factors(dataset_release).copy()
    external["itemid"] = external["itemid"].astype(str)
    frames: list[pd.DataFrame] = []
    for split_name, split in splits.items():
        features = split.features.copy()
        features["itemid"] = features["itemid"].astype(str)
        features = features.merge(external, on="itemid", how="left", suffixes=("", "_external"))
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
) -> dict[str, pd.DataFrame]:
    splits = load_structured_tree_splits(dataset_release)
    if text_inputs:
        text_frame = load_user_text_inputs(text_inputs, text_field)
    else:
        text_frame = load_serialized_strict_inputs(dataset_release)[["itemid", "text"]]
    text_frame["itemid"] = text_frame["itemid"].astype(str)

    out: dict[str, pd.DataFrame] = {}
    for split_name, split in splits.items():
        labels = split.targets[["itemid", "y_amount_eur", "y_binary"]].copy()
        labels["itemid"] = labels["itemid"].astype(str)
        cases = split.cases[["itemid", "split", "test_view", "test_challenging_view"]].copy()
        cases["itemid"] = cases["itemid"].astype(str)
        frame = labels.merge(cases, on="itemid", how="left", validate="one_to_one")
        frame = frame.merge(text_frame, on="itemid", how="inner", validate="one_to_one")
        frame["y_amount_eur"] = pd.to_numeric(frame["y_amount_eur"], errors="raise")
        frame["y_binary"] = pd.to_numeric(frame["y_binary"], errors="raise").astype(int)
        out[split_name] = frame[["itemid", "split", "text", "y_amount_eur", "y_binary", "test_view", "test_challenging_view"]]
    return out

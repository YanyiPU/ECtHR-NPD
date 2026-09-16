#!/usr/bin/env python3
"""Prepare new experiments from the single readable ECtHR-NPD dataset.

No model calls/downloads. Derived inputs must live outside the public package.
Identifiers and labels are never inserted into serialized model inputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT / "baselines"))
from data import public_adapter as public
from encoder.data_loader import read_rows
from evaluate import evaluate_arrays


def _records(frame):
    return [{key: None if isinstance(value, float) and math.isnan(value) else value
             for key, value in row.items()} for row in frame.to_dict("records")]


def _csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@lru_cache(maxsize=None)
def _prompt_blocks(path):
    blocks = re.findall(r"```text\n(.*?)\n```", path.read_text(encoding="utf-8"), flags=re.S)
    if len(blocks) != 2:
        raise ValueError(f"Expected a system and user template in {path}")
    return blocks


def _render_prompt(path, values):
    blocks = _prompt_blocks(path)
    return [{"role": role, "content": block.format(**values)}
            for role, block in zip(("system", "user"), blocks)]


def reviewed_facts(path):
    """Require review evidence; raw reconstructed sections are not certified inputs."""
    facts = {}
    for row in read_rows(Path(path)):
        itemid = str(row.get("case_id") or row.get("itemid") or "").strip()
        if not itemid or itemid in facts:
            raise ValueError("Blank/duplicate reviewed FACTS identifier")
        if row.get("case_id") and row.get("itemid") and row["case_id"] != row["itemid"]:
            raise ValueError("Conflicting reviewed FACTS identifiers")
        if row.get("review_status") != "accepted_prediction_input" or not re.fullmatch(r"[0-9a-fA-F]{64}", str(row.get("source_document_sha256", ""))) or not str(row.get("source_anchor", "")).strip():
            raise ValueError("FACTS requires accepted_prediction_input review, source_document_sha256 and source_anchor")
        text = str(row.get("facts_text") or "").strip()
        if not text:
            raise ValueError("Empty reviewed FACTS text")
        facts[itemid] = text
    return facts


def prepare(dataset_root, output_dir, *, facts_inputs=None):
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("Derived experiment inputs must be outside the public dataset root")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; do not overwrite experiment artifacts")
    cases = public.canonical_cases(root)
    features = public.feature_frame(cases)
    rows, feature_rows = _records(cases), _records(features)
    index = [{key: row[key] for key in ("itemid", "split", "judgementdate", "violated_articles")} for row in rows]
    targets = [{key: row[key] for key in ("itemid", "y_amount_eur", "y_binary")} for row in rows]
    facts = None
    if facts_inputs:
        facts = reviewed_facts(facts_inputs)
        if set(facts) != set(cases["itemid"]):
            raise ValueError("Audited FACTS input must cover every public case exactly once")
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "case_index.csv", list(index[0]), index)
    _csv(output / "targets.csv", list(targets[0]), targets)
    _csv(output / "strict_features.csv", list(feature_rows[0]), feature_rows)
    for split in ("train", "validation", "test"):
        chosen = [target for row, target in zip(rows, targets) if row["split"] == split]
        _csv(output / f"{split}_targets.csv", list(targets[0]), chosen)
    train_cases = [{**feature, "split": row["split"], "judgementdate": row["judgementdate"]}
                   for row, feature in zip(rows, feature_rows) if row["split"] == "train"]
    _csv(output / "train_cases.csv", list(train_cases[0]), train_cases)
    profile = public.feature_schema()
    _json(output / "feature_profile.json", profile)
    _json(output / "knn_profile.json", {**profile, "profile": "structured_knn"})
    serialized, agent_cases, zero_prompts, cot_prompts = [], [], [], []
    prompt_root = CODE_ROOT.parent / "prompts"
    cot_schema = json.loads((prompt_root / "cot/static_cot_serialized_features_schema.json").read_text())
    for row, feature in zip(rows, feature_rows):
        model_features = {key: value for key, value in feature.items() if key != "itemid"}
        text = json.dumps(model_features, ensure_ascii=False, sort_keys=True, allow_nan=False)
        serialized.append({"itemid": row["itemid"], "serialized_features": text})
        # The controller receives identity/date for joins, but only the strict
        # serialized projection is visible as combined_input_text.
        agent_cases.append({**feature, "judgementdate": row["judgementdate"], "combined_input_text": text})
        if row["split"] != "train":
            values = {"provided_violated_articles": row["violated_articles"], "combined_input_text": text,
                      "output_schema": json.dumps(cot_schema)}
            for destination, template in ((zero_prompts, "zero_shot/vanilla_serialized_features_prompt.md"),
                                          (cot_prompts, "cot/static_cot_serialized_features_prompt.md")):
                destination.append({"itemid": row["itemid"], "split": row["split"],
                    "messages": _render_prompt(prompt_root / template, values)})
    _jsonl(output / "serialized_inputs.jsonl", serialized)
    _jsonl(output / "agent_cases.jsonl", agent_cases)
    _jsonl(output / "zero_shot_requests.jsonl", zero_prompts)
    _jsonl(output / "cot_requests.jsonl", cot_prompts)
    if facts is not None:
        _jsonl(output / "retrieval_documents.jsonl", [{**row, "facts_text": facts[row["itemid"]]} for row in index])
        _jsonl(output / "encoder_facts.jsonl", [{"itemid": row["itemid"], "facts_text": facts[row["itemid"]]} for row in index])
    manifest = {"status": "new_experiment_inputs_not_historical_reproduction",
                "dataset_provenance": public.provenance(root, {"facts_inputs": facts_inputs}),
                "feature_profile": profile, "identity": "itemid is an unchanged alias of public HUDOC case_id; dropped before fitting/prompt rendering",
                "text_status": "user_supplied_audited_FACTS" if facts is not None else "structured_serialization_only_not_FACTS",
                "implementation_sha256": {str(path.relative_to(CODE_ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__), Path(public.__file__), prompt_root / "zero_shot/vanilla_serialized_features_prompt.md",
                                 prompt_root / "cot/static_cot_serialized_features_prompt.md", prompt_root / "cot/static_cot_serialized_features_schema.json"]},
                "prompt_execution": "Requests prepared only. Send messages only; never send itemid/split wrapper. Record model snapshot, provider, access date, seed42 when supported, temperature0 when supported, max output4096, disable tools, and validate response schema.",
                "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()}}
    _json(output / "preparation_manifest.json", manifest)
    return {"case_rows": len(rows), "feature_count": len(public.FEATURES), "output_dir": str(output)}


def constants(dataset_root, output_dir):
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("Run outputs must be outside the public dataset root")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    cases = public.canonical_cases(root)
    train = cases.loc[cases["split"] == "train", "y_amount_eur"].astype(float).to_numpy()
    positive = train[train > 0]
    values = {"zero": 0., "train_median": float(np.median(train)), "train_mean": float(train.mean())}
    if len(positive):
        values.update(train_positive_median=float(np.median(positive)), train_positive_mean=float(positive.mean()))
    output.mkdir(parents=True, exist_ok=True)
    result = {"dataset_provenance": public.provenance(root), "constants_eur": values, "metrics": {}}
    for name, amount in values.items():
        metrics = {}
        for view, selected in (("validation", cases["split"] == "validation"), ("test", cases["split"] == "test"),
                ("ID", cases["test_view"] == "ID"), ("OOD", cases["test_view"] == "OOD"),
                ("Challenging", cases["test_challenging_view"] == 1)):
            target = cases.loc[selected, "y_amount_eur"].astype(float).to_numpy()
            if len(target):
                metrics[view] = evaluate_arrays(target, np.full(len(target), amount))
        result["metrics"][name] = metrics
    _json(output / "constants_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "constants"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--facts-inputs", type=Path, help="Reviewed complete case_id,facts_text,review_status,source_document_sha256,source_anchor CSV/JSONL")
    args = parser.parse_args()
    if args.command == "constants" and args.facts_inputs:
        parser.error("--facts-inputs is only used by prepare")
    result = prepare(args.dataset_root, args.output_dir, facts_inputs=args.facts_inputs) if args.command == "prepare" else constants(args.dataset_root, args.output_dir)
    if args.command == "constants":
        result = {"output": str(args.output_dir / "constants_metrics.json"), "constants_eur": result["constants_eur"],
                  "test_mae_eur": {name: metrics["test"]["mae"] for name, metrics in result["metrics"].items()}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

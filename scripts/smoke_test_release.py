#!/usr/bin/env python3
"""Smoke test a shared ECtHR-NPD release folder.

Run from the release root:

    python scripts/smoke_test_release.py

This test intentionally avoids heavyweight model dependencies. It verifies
that the packaged public dataset, loaders, split counts, and sanitized release
layout are internally consistent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
EXPECTED_SPLITS = {"train": 10217, "validation": 1461, "test": 2897}
FORBIDDEN_FILE_NAMES = {
    "CHECK" + "SUMS.sha256",
    "FILE" + "_INVENTORY.csv",
    "RELEASE" + "_SCOPE.md",
    "anonymization" + "_check.json",
    "field" + "_inventory.csv",
    "mani" + "fest.json",
}
FORBIDDEN_PATH_PARTS = {"raw", "outputs", "archive", "logs", "__pycache__"}
FORBIDDEN_LITERAL_MARKERS = [
    "mani" + "fest",
    "CHECK" + "SUMS",
    "FILE" + "_INVENTORY",
    "RELEASE" + "_SCOPE",
    "au" + "thor" + " information",
    "affili" + "ation",
    "au" + "thor" + "_name",
    "au" + "thor" + "_email",
    "institu" + "tion",
    "univer" + "sity",
    "/" + "Users",
    "pp" + "ziz",
    "yan" + "yi",
    "api" + "_key_path",
    "qwen" + "_api",
]
FORBIDDEN_REGEX_MARKERS = [
    r"sk-[A-Za-z0-9_-]{20,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"Bearer [A-Za-z0-9._-]{16,}",
]
FORBIDDEN_TEXT = re.compile(
    "|".join([re.escape(marker) for marker in FORBIDDEN_LITERAL_MARKERS] + FORBIDDEN_REGEX_MARKERS),
    re.IGNORECASE,
)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def require(path: str) -> Path:
    resolved = ROOT / path
    if not resolved.exists():
        fail(f"missing required path: {path}")
    return resolved


def scan_layout() -> None:
    for rel in [
        "dataset_release/data/ecthr_npd_cases.csv",
        "dataset_release/model_inputs/external_factors/economic_covariates.csv",
        "code/baselines/data/data_loader.py",
        "code/baselines/tree_models/train.py",
        "code/baselines/retrieval/bm25_pfme_knn.py",
        "code/baselines/retrieval/bge_m3_knn.py",
        "code/baselines/encoder/data_loader.py",
        "source_reconstruction/download_hudoc_judgments.py",
        "extraction_pipeline/code/holistic_extractor.py",
        "requirements.txt",
    ]:
        require(rel)

    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        lower_parts = {part.lower() for part in Path(rel).parts}
        if path.name in FORBIDDEN_FILE_NAMES or ("mani" + "fest") in path.name.lower():
            fail(f"forbidden file present: {rel}")
        if lower_parts & FORBIDDEN_PATH_PARTS:
            fail(f"forbidden path part in: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if FORBIDDEN_TEXT.search(text):
                fail(f"forbidden text marker in: {rel}")


def smoke_loaders() -> None:
    sys.path.insert(0, str(ROOT / "code"))
    from baselines.data.data_loader import load_external_factors, load_structured_tree_splits
    from baselines.encoder.data_loader import load_encoder_splits

    dataset_root = ROOT / "dataset_release"
    tree = load_structured_tree_splits(dataset_root)
    encoder = load_encoder_splits(dataset_root)
    external = load_external_factors(dataset_root)

    tree_counts = {name: len(split.X) for name, split in tree.items()}
    encoder_counts = {name: len(frame) for name, frame in encoder.items()}
    if tree_counts != EXPECTED_SPLITS:
        fail(f"unexpected tree split counts: {tree_counts}")
    if encoder_counts != EXPECTED_SPLITS:
        fail(f"unexpected encoder split counts: {encoder_counts}")
    if len(external) != 14575:
        fail(f"unexpected external factor rows: {len(external)}")
    for split in tree.values():
        if split.X.shape[1] != 50:
            fail(f"unexpected structured feature count: {split.X.shape[1]}")


def main() -> int:
    scan_layout()
    smoke_loaders()
    print("PASS: release smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline adversarial version/target/hash boundaries; synthetic records only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1] / "code/baselines"
sys.path.insert(0, str(BASE))
from data.data_loader import (DatasetReleaseError, SAFE_METADATA_COLUMNS, VIOLATED_ARTICLE_COLUMNS,
    SERIALIZED_CASE_FACT_COLUMNS, STRUCTURED_EXTERNAL_FACTOR_COLUMNS, UNVERIFIED_ALLOCATION_FEATURES,
    resolve_dataset_release, load_structured_tree_split, load_legacy_audit_split,
    dataset_provenance, file_sha256, validate_selected_targets)
from retrieval.bm25_pfme_knn import CaseDocument, validate_retrieval_selection
from evaluate import evaluate_predictions


def fixture(root, version="corrected"):
    release = root / ("clean" if version == "corrected" else "paper_reference/clean")
    contract = {"dataset_version": version, "release_id": "synthetic_" + version,
        "counts": {"total": 4, "splits": {"train": 1, "validation": 1, "test": 2},
                   "test_views": {"ID": 1, "OOD": 1}, "challenging": 1}}
    release.mkdir(parents=True)
    (release / "release_contract.json").write_text(json.dumps(contract))
    columns = ["itemid", *SAFE_METADATA_COLUMNS, *VIOLATED_ARTICLE_COLUMNS,
               *SERIALIZED_CASE_FACT_COLUMNS, *STRUCTURED_EXTERNAL_FACTOR_COLUMNS]
    if version == "paper_reference": columns += sorted(UNVERIFIED_ALLOCATION_FEATURES)
    amount = 100 if version == "corrected" else 90
    cases = [{"itemid": id_, "split": split, "y_amount_eur": value, "y_binary": int(value > 0)}
             for id_, split, value in [("a", "train", amount), ("b", "validation", 0), ("c", "test", 200), ("d", "test", 0)]]
    def write(path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    write(release / "data/ecthr_npd_cases.csv", list(cases[0]), cases)
    for split, stem in [("train", "train"), ("validation", "val"), ("test", "test")]:
        rows = [row for row in cases if row["split"] == split]
        write(release / f"model_inputs/structured_tree/features/{stem}.csv", columns,
              [{**dict.fromkeys(columns, 1), "itemid": row["itemid"]} for row in rows])
        write(release / f"model_inputs/structured_tree/targets/{stem}.csv", ["itemid", "y_amount_eur", "y_binary"],
              [{key: row[key] for key in ("itemid", "y_amount_eur", "y_binary")} for row in rows])
    return release


def manifest(release):
    files = {str(path.relative_to(release)): file_sha256(path) for path in release.rglob("*") if path.is_file() and path.name != "VERSION_MANIFEST.json"}
    contract = json.loads((release / "release_contract.json").read_text())
    value = {**contract, "schema_version": "ecthr-npd-version-1", "files": files,
        "revision_id": "sha256:" + hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    (release / "VERSION_MANIFEST.json").write_text(json.dumps(value))


class BaselineVersionTests(unittest.TestCase):
    def test_missing_selection_and_unversioned_frozen_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DatasetReleaseError, "no legacy"):
                resolve_dataset_release()
        with self.assertRaises(DatasetReleaseError):
            resolve_dataset_release(BASE.parents[1] / "dataset_release")

    def test_explicit_versions_preserve_48_vs_50_and_forbid_strict_paper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); corrected = fixture(root); paper = fixture(root, "paper_reference")
            self.assertEqual(load_structured_tree_split("train", root).X.shape, (1, 48))
            with self.assertWarns(UserWarning):
                audit = load_legacy_audit_split("train", root)
            self.assertEqual(audit.features.shape, (1, 51))
            with self.assertRaises(DatasetReleaseError): _ = audit.X
            with self.assertRaises(DatasetReleaseError): load_structured_tree_split("train", root, dataset_version="paper_reference", validate=False)
            with self.assertRaises(DatasetReleaseError): resolve_dataset_release(paper)
            with self.assertRaises(DatasetReleaseError): resolve_dataset_release(corrected, dataset_version="paper_reference")

    def test_validate_false_cannot_disable_target_or_leakage_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            release = fixture(Path(directory))
            target = release / "model_inputs/structured_tree/targets/train.csv"
            target.write_text("itemid,y_amount_eur,y_binary\na,999,1\n")
            with self.assertWarns(UserWarning), self.assertRaisesRegex(DatasetReleaseError, "differs"):
                load_structured_tree_split("train", release, validate=False)
            target.write_text("itemid,y_amount_eur,y_binary\na,100,1\n")
            feature = release / "model_inputs/structured_tree/features/train.csv"
            feature.write_text(feature.read_text().replace("country_alpha2", "award_amount"))
            with self.assertWarns(UserWarning), self.assertRaises(DatasetReleaseError):
                load_structured_tree_split("train", release, validate=False)

    def test_hash_is_content_addressed_and_manifest_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); release = fixture(root); manifest(release)
            clone = root / "clone"; shutil.copytree(release, clone)
            self.assertEqual(dataset_provenance(release)["dataset_fingerprint_sha256"], dataset_provenance(clone)["dataset_fingerprint_sha256"])
            feature = clone / "model_inputs/structured_tree/features/train.csv"
            feature.write_text(feature.read_text() + "\n")
            with self.assertRaisesRegex(DatasetReleaseError, "hash mismatch"): dataset_provenance(clone)

    def test_manifest_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); release = fixture(root); manifest(release)
            path = release / "VERSION_MANIFEST.json"; value = json.loads(path.read_text())
            value["files"]["../private.txt"] = "0" * 64; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(DatasetReleaseError, "Unsafe"): dataset_provenance(release)

    def test_retrieval_subset_flag_does_not_allow_other_version_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            docs = [CaseDocument("a", "train", None, "invented facts", frozenset({"3"})), CaseDocument("c", "test", None, "invented facts", frozenset({"3"}))]
            with self.assertRaisesRegex(DatasetReleaseError, "differs"):
                validate_retrieval_selection(docs, {"a": 90}, eval_splits={"test"}, dataset_release=root, example_subset_only=True)
            result = validate_retrieval_selection(docs, {"a": 100}, eval_splits={"test"}, dataset_release=root, example_subset_only=True)
            self.assertEqual(result["dataset_provenance"]["dataset_version"], "corrected")

    def test_evaluator_rejects_mixed_version_truth_and_records_paper_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root); fixture(root, "paper_reference")
            pred = root / "pred.csv"; truth = root / "truth.csv"
            pred.write_text("itemid,predicted_award_eur\na,95\n"); truth.write_text("itemid,y_amount_eur\na,90\n")
            with self.assertRaisesRegex(DatasetReleaseError, "differs"):
                evaluate_predictions(pred, truth, root / "bad", dataset_release=root, dataset_version="corrected")
            result = evaluate_predictions(pred, truth, root / "audit", dataset_release=root, dataset_version="paper_reference")
            self.assertEqual(result["dataset_provenance"]["dataset_version"], "paper_reference")
            self.assertFalse((root / "bad/metrics.json").exists())

    def test_split_mixing_is_rejected_even_when_amount_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            with self.assertRaisesRegex(DatasetReleaseError, "splits differ"):
                validate_selected_targets(["a"], [100], root, splits=["test"])

    def test_evaluation_rejects_case_vs_split_target_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); release = fixture(root)
            (release / "model_inputs/structured_tree/targets/train.csv").write_text("itemid,y_amount_eur,y_binary\na,90,1\n")
            with self.assertRaisesRegex(DatasetReleaseError, "differs"):
                validate_selected_targets(["a"], [100], root)


if __name__ == "__main__": unittest.main()

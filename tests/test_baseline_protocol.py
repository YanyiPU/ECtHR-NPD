"""Regression tests for safe, paper-protocol baseline release behavior."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


RELEASE_STAGING = Path(__file__).resolve().parents[1]
BASELINES = RELEASE_STAGING / "code" / "baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from data.data_loader import DatasetReleaseError, DatasetSplit, UNVERIFIED_ALLOCATION_FEATURES, encode_categorical_columns, load_external_factors, load_structured_tree_splits
from encoder.data_loader import load_encoder_splits, load_serialized_strict_inputs, serialize_record
from evaluate import EvaluationSchemaError, align_prediction_records, evaluate_arrays, evaluate_predictions
from retrieval.bge_m3_knn import run_knn
from retrieval.bm25_pfme_knn import (
    CaseDocument,
    PaperProtocolCoverageError,
    assert_train_document_target_alignment,
    eligible_train_indices,
    normalize_violated_articles,
    prediction_row_from_ranked_neighbors,
    validate_paper_protocol_coverage,
)
from tree_models.train import (
    _merged_params,
    load_tree_feature_schema,
    load_tree_training_config,
    load_selection_candidates,
    prepare_catboost_frame,
    select_candidate_by_validation,
    validate_tree_feature_schema,
)


DATASET_RELEASE = RELEASE_STAGING / "dataset_release"


def historical_frames_for_projection_test():
    """Read-only test fixture, NOT a loader bypass or approved training dataset."""
    cases = pd.read_csv(DATASET_RELEASE / "data/ecthr_npd_cases.csv")
    return {name: DatasetSplit(name,
        pd.read_csv(DATASET_RELEASE / f"model_inputs/structured_tree/features/{stem}.csv"),
        pd.read_csv(DATASET_RELEASE / f"model_inputs/structured_tree/targets/{stem}.csv"), cases, DATASET_RELEASE)
        for name, stem in [("train", "train"), ("validation", "val"), ("test", "test")]}


def document(itemid: str, split: str, date: str, articles: str) -> CaseDocument:
    return CaseDocument(
        itemid=itemid,
        split=split,
        judgementdate=datetime.strptime(date, "%Y-%m-%d"),
        text=f"facts for {itemid}",
        violated_articles=normalize_violated_articles(articles),
    )


class InputIsolationTests(unittest.TestCase):
    def test_external_factors_are_projected_to_allow_list(self) -> None:
        # Projection is unit-tested with a labelled synthetic contract; no legacy fallback.
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            (release / "data").mkdir()
            (release / "data/ecthr_npd_cases.csv").write_text("itemid,split\nx,train\n")
            (release / "release_contract.json").write_text(json.dumps({"dataset_version": "corrected", "release_id": "synthetic",
                "counts": {"total": 1, "splits": {"train": 1, "validation": 0, "test": 0}, "test_views": {"ID": 0, "OOD": 0}, "challenging": 0}}))
            folder = release / "model_inputs/external_factors"
            folder.mkdir(parents=True)
            (folder / "economic_covariates.csv").write_text("itemid,gdp_per_capita_log1p,gdp_constant_2015_log1p,respondent_state,split\nx,1,2,A,train\n")
            external = load_external_factors(release)
        self.assertEqual(
            list(external.columns),
            ["itemid", "gdp_per_capita_log1p", "gdp_constant_2015_log1p"],
        )
        self.assertEqual(len(external), 1)
        self.assertNotIn("respondent_state", external.columns)
        self.assertNotIn("split", external.columns)

    def test_encoder_serialization_rejects_audit_metadata(self) -> None:
        with self.assertRaises(ValueError):
            serialize_record({"itemid": "x", "test_view": "OOD", "country_alpha2": "at"})

    @unittest.skipUnless((DATASET_RELEASE / 'data/ecthr_npd_cases.csv').is_file(), 'Optional private historical-fixture integration test')
    def test_public_serialized_encoder_inputs_exclude_audit_columns(self) -> None:
        # The frozen fixture intentionally retains its original 50 columns.
        # It must now fail the strict provenance gate; simulate the approved
        # two-column projection in memory, without rewriting historical data.
        with self.assertRaises(DatasetReleaseError):
            load_serialized_strict_inputs(DATASET_RELEASE)
        splits = historical_frames_for_projection_test()
        splits = {name: replace(split, features=split.features.drop(columns=sorted(UNVERIFIED_ALLOCATION_FEATURES)))
                  for name, split in splits.items()}
        external = pd.read_csv(DATASET_RELEASE / "model_inputs/external_factors/economic_covariates.csv")[["itemid", "gdp_per_capita_log1p", "gdp_constant_2015_log1p"]]
        with patch("encoder.data_loader.load_structured_tree_splits", return_value=splits), patch("encoder.data_loader.load_external_factors", return_value=external):
            serialized = load_serialized_strict_inputs(DATASET_RELEASE)
        self.assertEqual(len(serialized), 14575)
        self.assertFalse(serialized["text"].str.contains("test_view|test_challenging_view|hudoc_url", regex=True).any())

    def test_numeric_imputation_is_training_median_and_forced_categories_are_one_hot_encoded(self) -> None:
        train = pd.DataFrame({"case_importance": [1, 2, 1], "numeric": [1.0, np.nan, 5.0]})
        validation = pd.DataFrame({"case_importance": [3], "numeric": [np.nan]})
        encoded_train, encoded_validation = encode_categorical_columns(
            train, validation, categorical_columns=["case_importance"]
        )
        self.assertNotIn("case_importance", encoded_train.columns)
        self.assertIn("case_importance__<UNKNOWN>", encoded_train.columns)
        self.assertEqual(encoded_train["case_importance__<UNKNOWN>"].sum(), 0)
        self.assertEqual(encoded_validation.loc[0, "case_importance__<UNKNOWN>"], 1)
        self.assertEqual(encoded_validation.loc[0, "numeric"], 3.0)

    @staticmethod
    def _fake_encoder_splits() -> dict[str, SimpleNamespace]:
        cases = pd.DataFrame(
            [
                {"itemid": "train-case", "split": "train", "test_view": "", "test_challenging_view": 0},
                {"itemid": "validation-case", "split": "validation", "test_view": "", "test_challenging_view": 0},
                {"itemid": "test-case", "split": "test", "test_view": "ID", "test_challenging_view": 0},
            ]
        )
        return {
            "train": SimpleNamespace(
                targets=pd.DataFrame([{"itemid": "train-case", "y_amount_eur": 1.0, "y_binary": 1}]),
                cases=cases,
            ),
            "validation": SimpleNamespace(
                targets=pd.DataFrame([{"itemid": "validation-case", "y_amount_eur": 0.0, "y_binary": 0}]),
                cases=cases,
            ),
            "test": SimpleNamespace(
                targets=pd.DataFrame([{"itemid": "test-case", "y_amount_eur": 2.0, "y_binary": 1}]),
                cases=cases,
            ),
        }

    def _load_user_encoder_inputs(self, rows: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "texts.jsonl"
            text_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with patch("encoder.data_loader.load_structured_tree_splits", return_value=self._fake_encoder_splits()):
                return load_encoder_splits(text_inputs=text_path, text_field="facts_text")

    def test_user_encoder_text_rejects_duplicate_itemids_instead_of_deduplicating(self) -> None:
        rows = [
            {"itemid": "train-case", "facts_text": "first"},
            {"itemid": "train-case", "facts_text": "duplicate"},
            {"itemid": "validation-case", "facts_text": "validation"},
            {"itemid": "test-case", "facts_text": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate strict encoder text itemid"):
            self._load_user_encoder_inputs(rows)

    def test_user_encoder_text_rejects_unknown_ids_and_missing_split_coverage(self) -> None:
        complete_rows = [
            {"itemid": "train-case", "facts_text": "train"},
            {"itemid": "validation-case", "facts_text": "validation"},
            {"itemid": "test-case", "facts_text": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "outside the public fixed splits"):
            self._load_user_encoder_inputs([*complete_rows, {"itemid": "unknown-case", "facts_text": "extra"}])
        with self.assertRaisesRegex(ValueError, "incomplete by split: test=1"):
            self._load_user_encoder_inputs(complete_rows[:-1])

    def test_user_encoder_text_full_coverage_preserves_every_target_row(self) -> None:
        splits = self._load_user_encoder_inputs(
            [
                {"itemid": "train-case", "facts_text": "train"},
                {"itemid": "validation-case", "facts_text": "validation"},
                {"itemid": "test-case", "facts_text": "test"},
            ]
        )
        self.assertEqual({name: len(frame) for name, frame in splits.items()}, {
            "train": 1,
            "validation": 1,
            "test": 1,
        })


class EvaluatorTests(unittest.TestCase):
    def test_all_case_metrics_include_zero_f1(self) -> None:
        metrics = evaluate_arrays(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(metrics["mae"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(metrics["zero_f1"]), 0.8)
        for name in ("rmse", "medae", "p95ae", "r2", "pearson_r", "spearman_rho", "positive_f1"):
            self.assertIn(name, metrics)

    def test_alias_schema_aligns_by_itemid_not_input_order(self) -> None:
        y_true, y_pred, itemids = align_prediction_records(
            [{"itemid": "b", "prediction": 4}, {"itemid": "a", "predicted_award_eur": 1}],
            [{"itemid": "a", "y_amount_eur": 0}, {"itemid": "b", "target_award_eur": 3}],
        )
        self.assertEqual(itemids, ["a", "b"])
        self.assertEqual(y_true.tolist(), [0.0, 3.0])
        self.assertEqual(y_pred.tolist(), [1.0, 4.0])

    def test_duplicate_or_incomplete_artifacts_fail_closed(self) -> None:
        with self.assertRaises(EvaluationSchemaError):
            align_prediction_records(
                [{"itemid": "a", "prediction": 1}, {"itemid": "a", "prediction": 2}],
                [{"itemid": "a", "y_amount_eur": 1}],
            )
        with self.assertRaises(EvaluationSchemaError):
            align_prediction_records(
                [{"itemid": "a", "prediction": 1}],
                [{"itemid": "b", "y_amount_eur": 1}],
            )

    def test_file_evaluator_accepts_tree_prediction_json_and_target_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.json"
            targets = root / "targets.csv"
            predictions.write_text(
                json.dumps(
                    [
                        {"itemid": "case-a", "target_award_eur": 0, "predicted_award_eur": 0},
                        {"itemid": "case-b", "target_award_eur": 100, "predicted_award_eur": 80},
                    ]
                ),
                encoding="utf-8",
            )
            targets.write_text("itemid,y_amount_eur\ncase-a,0\ncase-b,100\n", encoding="utf-8")
            result = evaluate_predictions(predictions, targets, root / "metrics")
            self.assertEqual(result["schema_version"], "ecthr-npd-evaluation-v1")
            self.assertAlmostEqual(float(result["metrics"]["mae"]), 10.0)
            self.assertTrue((root / "metrics" / "metrics.json").is_file())


class TreeProtocolTests(unittest.TestCase):
    def test_estimator_defaults_come_from_the_frozen_training_config(self) -> None:
        config = load_tree_training_config()
        self.assertEqual(config["models"], ["xgboost", "catboost", "lightgbm"])
        self.assertEqual(config["device_used"], "cpu")
        self.assertIsNone(config["xgboost_params"]["early_stopping_rounds"])
        self.assertEqual(_merged_params("xgboost", {}, 42)["device"], "cpu")
        self.assertEqual(_merged_params("lightgbm", {}, 42)["device_type"], "cpu")
        self.assertNotIn("early_stopping_rounds", _merged_params("catboost", {}, 42))
        with self.assertRaisesRegex(DatasetReleaseError, "frozen xgboost protocol field"):
            _merged_params("xgboost", {"device": "cuda"}, 42)
        with self.assertRaisesRegex(DatasetReleaseError, "reproducibility seed"):
            _merged_params("lightgbm", {"random_state": 0}, 42)

    @unittest.skipUnless((DATASET_RELEASE / 'data/ecthr_npd_cases.csv').is_file(), 'Optional private historical-fixture integration test')
    def test_corrected_tree_schema_rejects_frozen_and_accepts_exact_projection(self) -> None:
        schema = load_tree_feature_schema()
        with self.assertRaises(DatasetReleaseError):
            load_structured_tree_splits(DATASET_RELEASE)
        splits = historical_frames_for_projection_test()
        with self.assertRaises(DatasetReleaseError):
            validate_tree_feature_schema(splits, schema)
        splits = {name: replace(split, features=split.features.drop(columns=sorted(UNVERIFIED_ALLOCATION_FEATURES)))
                  for name, split in splits.items()}
        validate_tree_feature_schema(splits, schema)
        self.assertEqual(schema["feature_count"], 48)
        self.assertFalse(set(schema["feature_columns"]) & UNVERIFIED_ALLOCATION_FEATURES)
        self.assertEqual(list(splits["train"].X.columns), schema["feature_columns"])

    def test_catboost_preparation_fits_and_reuses_training_median(self) -> None:
        train = pd.DataFrame({"country_alpha2": ["at", "fr"], "numeric": [2.0, 6.0]})
        validation = pd.DataFrame({"country_alpha2": ["de"], "numeric": [np.nan]})
        _, categories, fills = prepare_catboost_frame(train, categorical_columns=["country_alpha2"])
        prepared_validation, _, _ = prepare_catboost_frame(
            validation, categorical_columns=categories, numeric_fill_values=fills
        )
        self.assertEqual(prepared_validation.loc[0, "numeric"], 4.0)

    def test_candidate_selection_uses_medae_as_mae_tiebreak(self) -> None:
        candidates = [{"name": "worse_median", "params": {"key": "worse"}}, {"name": "better_median", "params": {"key": "better"}}]
        predictions = {
            "worse": np.array([1.0, 1.0, 1.0, 99.0]),
            "better": np.array([0.0, 0.0, 0.0, 96.0]),
        }
        selected, scores = select_candidate_by_validation(
            candidates,
            fit_candidate=lambda params: params["key"],
            predict_candidate=lambda fitted, _x: predictions[fitted],
            X_validation=pd.DataFrame({"x": [1, 2, 3, 4]}),
            y_validation=np.array([0.0, 0.0, 0.0, 100.0]),
        )
        self.assertEqual(selected["name"], "better_median")
        self.assertEqual(len(scores), 2)

    def test_candidate_manifest_rejects_missing_list_and_strips_name_from_direct_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "candidates.json"
            manifest.write_text(json.dumps([{"name": "one", "max_depth": 3}]), encoding="utf-8")
            candidates = load_selection_candidates(manifest, "xgboost")
            self.assertEqual(candidates, [{"name": "one", "params": {"max_depth": 3}}])
            manifest.write_text(json.dumps({"candidates": []}), encoding="utf-8")
            with self.assertRaises(DatasetReleaseError):
                load_selection_candidates(manifest, "xgboost")


class RetrievalProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_six = document("train-six", "train", "2020-01-01", "6")
        self.train_eight = document("train-eight", "train", "2019-01-01", "8")
        self.target_six = document("target-six", "test", "2021-01-01", "6")
        self.target_none = document("target-none", "test", "2021-01-01", "14")
        self.validation_six = document("validation-six", "validation", "2020-06-01", "6")
        self.targets = {"train-six": 100.0, "train-eight": 200.0}

    def _canonical_release(self, root: Path) -> Path:
        release = root / "dataset_release"
        (release / "data").mkdir(parents=True)
        (release / "data" / "ecthr_npd_cases.csv").write_text(
            "itemid,split,y_amount_eur,y_binary\n"
            "train-six,train,100,1\n"
            "train-eight,train,200,1\n"
            "validation-six,validation,0,0\n"
            "target-six,test,100,1\n"
            "target-none,test,0,0\n",
            encoding="utf-8",
        )
        (release / "release_contract.json").write_text(json.dumps({"dataset_version": "corrected", "release_id": "synthetic",
            "counts": {"total": 5, "splits": {"train": 2, "validation": 1, "test": 2}, "test_views": {"ID": 1, "OOD": 1}, "challenging": 0}}))
        return release

    def test_shared_article_and_temporal_filter_then_median_fallback(self) -> None:
        eligible = eligible_train_indices([self.train_six, self.train_eight], self.target_six)
        self.assertEqual(eligible, {0})
        row = prediction_row_from_ranked_neighbors(
            self.target_six,
            [(self.train_six, 0.9)],
            self.targets,
            fallback_value=150.0,
        )
        self.assertEqual(row["predicted_award_eur"], 100.0)
        self.assertEqual(row["neighbors_used"], 1)
        fallback = prediction_row_from_ranked_neighbors(
            self.target_none, [], self.targets, fallback_value=150.0
        )
        self.assertEqual(fallback["predicted_award_eur"], 150.0)
        self.assertEqual(fallback["fallback_reason"], "no_eligible_temporal_shared_article_neighbor")

    def test_bge_dense_knn_applies_article_filter_and_fallback(self) -> None:
        documents = [self.train_six, self.train_eight, self.target_six, self.target_none]
        rows = run_knn(
            documents,
            self.targets,
            np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.0, 1.0]]),
            mode="dense",
            eval_splits={"test"},
        )
        by_itemid = {row["itemid"]: row for row in rows}
        self.assertEqual(by_itemid["target-six"]["predicted_award_eur"], 100.0)
        self.assertEqual(by_itemid["target-none"]["predicted_award_eur"], 150.0)
        self.assertEqual(by_itemid["target-none"]["neighbors_used"], 0)

    def test_retrieval_train_document_target_mismatch_fails_before_indexing(self) -> None:
        with self.assertRaisesRegex(PaperProtocolCoverageError, "train retrieval documents versus train targets"):
            assert_train_document_target_alignment(
                [self.train_six, self.target_six],
                self.targets,
            )
        with self.assertRaisesRegex(PaperProtocolCoverageError, "train retrieval documents versus train targets"):
            run_knn(
                [self.train_six, self.target_six],
                self.targets,
                np.array([[1.0, 0.0], [0.9, 0.1]]),
                mode="dense",
                eval_splits={"test"},
            )

    def test_paper_retrieval_coverage_requires_complete_canonical_evaluation_splits(self) -> None:
        documents = [
            self.train_six,
            self.train_eight,
            self.validation_six,
            self.target_six,
            self.target_none,
        ]
        with tempfile.TemporaryDirectory() as directory:
            release = self._canonical_release(Path(directory))
            coverage = validate_paper_protocol_coverage(
                documents,
                self.targets,
                eval_splits={"validation", "test"},
                dataset_release=release,
            )
            self.assertEqual(coverage["coverage_mode"], "full_fixed_public_splits")
            self.assertEqual(coverage["test_document_rows"], 2)
            with self.assertRaisesRegex(PaperProtocolCoverageError, "validation retrieval documents versus canonical"):
                validate_paper_protocol_coverage(
                    [self.train_six, self.train_eight, self.target_six, self.target_none],
                    self.targets,
                    eval_splits={"validation", "test"},
                    dataset_release=release,
                )


if __name__ == "__main__":
    unittest.main()

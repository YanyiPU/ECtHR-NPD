"""Offline regression checks for the readable-table experiment bridge."""
import csv
from dataclasses import replace
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/baselines"))
sys.path.insert(0, str(ROOT / "agent_knowledge_base"))
import prepare_experiments as prep
from data import public_adapter as public
from data.data_loader import load_structured_tree_splits, validate_selected_targets, encode_categorical_columns
from feature_profiles import validate_profile
from retrieval.tabular_knn import load_inputs, run_knn
from retrieval.bm25_pfme_knn import validate_retrieval_selection
from prior_contract import bind_selected_training_inputs
from evaluate import align_prediction_records
import react_orchestrator as react


def fixture(root):
    cases = []
    for index, (split, amount, state) in enumerate((("train", "0", "gb"), ("train", "100", "gb"),
                                                  ("validation", "999999", "fr"), ("test", "50", "fr"))):
        row = dict.fromkeys(public.PUBLIC.CASE_COLUMNS, "unknown")
        row.update(case_id=f"001-{100000 + index}", split=split, test_view="ID" if split == "test" else "not_applicable",
            test_challenging_view="yes" if split == "test" else "no", judgment_date=f"{2010 + index}-01-01",
            judgment_year=str(2010 + index), respondent_state="Test state", country_alpha2=state,
            hudoc_decision_body="CHAMBER", case_importance="low", has_separate_opinion="no", represented="yes",
            num_applicants="1", hudoc_application_count="1", num_violations_found="1", violated_articles="3",
            violated_articles_count="1", violation_type="substantive", violation_duration_months="12",
            applicant_sex="unknown", applicant_age_group="adult", applicant_birth_years="1980",
            applicant_nationality_scope="single", gdp_per_capita_current_usd="10000", gdp_constant_2015_usd="100000",
            y_amount_eur=amount, y_binary=str(int(float(amount) > 0)), beneficiary_type="FORBIDDEN_SENTINEL",
            has_joint_beneficiary="FORBIDDEN_SENTINEL", target_status="FORBIDDEN_SENTINEL")
        cases.append(row)
    (root / "data").mkdir(parents=True)
    for name, fields, rows in (("case_level", list(cases[0]), cases), ("applicant_level", ["case_id", "npd_award_eur"], [])):
        with (root / f"data/{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    return cases, []


class UnifiedExperimentTests(unittest.TestCase):
    def test_controller_shipped_client_fallback_with_mock_transport(self):
        with patch.dict(os.environ, {"TEST_NPD_KEY": "synthetic-not-a-real-key"}, clear=True):
            client = react.build_client("https://example.invalid/v1", "mock-model", None, "TEST_NPD_KEY", False)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": '{"award_eur": 50}'}}]}).encode()
        with patch("urllib.request.urlopen", return_value=response) as transport:
            result, _ = client.chat_json(messages=[{"role": "user", "content": "Synthetic input"}],
                                         schema={"type": "object"}, schema_name="synthetic")
        self.assertEqual(result["award_eur"], 50)
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(json.loads(request.data)["model"], "mock-model")

    def test_public_projection_has_no_identifier_target_allocation_or_split_predictors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            tables = fixture(root)
            with patch.object(public.PUBLIC, "load_tables", return_value=tables):
                splits = load_structured_tree_splits(root)
                self.assertEqual(splits["train"].X.shape, (2, 20))
                self.assertEqual(list(splits["train"].X), list(public.FEATURES))
                self.assertNotIn("FORBIDDEN_SENTINEL", splits["train"].X.to_json())
                self.assertEqual(splits["test"].itemids.tolist(), ["001-100003"])
                self.assertEqual(validate_profile(public.feature_schema()), list(public.FEATURES))
                validate_selected_targets(["001-100003"], [50], root, splits=["test"])
                with self.assertRaisesRegex(ValueError, "differs"):
                    validate_selected_targets(["001-100003"], [51], root)
                encoded_train, encoded_test = encode_categorical_columns(splits["train"].X, splits["test"].X,
                                                                          categorical_columns=public.CATEGORICAL)
                unknown = [c for c in encoded_test if c.startswith("country_alpha2__<UNKNOWN>")]
                self.assertEqual(encoded_test[unknown[0]].tolist(), [1])
                self.assertEqual(encoded_train[unknown[0]].tolist(), [0, 0])

    def test_preparation_feeds_retrieval_and_pinned_priors_without_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out = Path(directory) / "dataset", Path(directory) / "inputs"
            tables = fixture(root)
            with patch.object(public.PUBLIC, "load_tables", return_value=tables):
                prep.prepare(root, out)
                with (out / "case_index.csv").open() as handle: metadata = list(csv.DictReader(handle))
                with (out / "strict_features.csv").open() as handle: features = list(csv.DictReader(handle))
                profile = json.loads((out / "knn_profile.json").read_text())
                documents, features = load_inputs(metadata, features, profile)
                targets = {"001-100000": 0., "001-100001": 100.}
                validate_retrieval_selection(documents, targets, eval_splits={"validation", "test"}, dataset_release=root)
                changed = [replace(documents[0], judgementdate=datetime(2099, 1, 1)), *documents[1:]]
                with self.assertRaisesRegex(ValueError, "eligibility"):
                    validate_retrieval_selection(changed, targets, eval_splits={"validation", "test"}, dataset_release=root)
                predictions, _ = run_knn(documents, features, targets, profile, metric="euclidean", eval_splits={"validation", "test"})
                self.assertEqual(len(predictions), 2)
                binding = bind_selected_training_inputs(root, out / "train_cases.csv", out / "train_targets.csv")
                self.assertEqual(binding[3]["dataset_version"], "unified")
                manifest = json.loads((out / "preparation_manifest.json").read_text())
                self.assertEqual(manifest["text_status"], "structured_serialization_only_not_FACTS")
                self.assertFalse((out / "encoder_facts.jsonl").exists())
                requests = [json.loads(line) for line in (out / "zero_shot_requests.jsonl").read_text().splitlines()]
                for request in requests:
                    text = json.dumps(request["messages"])
                    self.assertNotIn(request["itemid"], text)
                    self.assertNotIn("FORBIDDEN_SENTINEL", text)
                    self.assertNotIn("999999", text)
                    self.assertNotIn("one third", text)
                with self.assertRaisesRegex(ValueError, "outside"):
                    prep.prepare(root, root / "experiment_inputs")

    def test_constants_only_estimate_on_train_and_evaluator_accepts_case_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            tables = fixture(root)
            with patch.object(public.PUBLIC, "load_tables", return_value=tables):
                result = prep.constants(root, Path(directory) / "result")
                self.assertEqual(result["constants_eur"]["train_mean"], 50.)
                self.assertEqual(result["constants_eur"]["train_median"], 50.)
                self.assertIn("Challenging", result["metrics"]["zero"])
            truth, pred, ids = align_prediction_records([{"case_id": "001-100003", "prediction": 50}],
                                                       [{"case_id": "001-100003", "y_amount_eur": 50}])
            self.assertEqual(ids, ["001-100003"])

    def test_facts_requires_review_evidence_and_exact_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.jsonl"
            row = {"case_id": "001-100003", "facts_text": "Reviewed factual summary."}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "review"):
                prep.reviewed_facts(path)
            row.update(review_status="accepted_prediction_input", source_document_sha256="a" * 64, source_anchor="FACTS §§1–10")
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(prep.reviewed_facts(path)[row["case_id"]], row["facts_text"])
            root = Path(directory) / "dataset"
            tables = fixture(root)
            with patch.object(public.PUBLIC, "load_tables", return_value=tables), self.assertRaisesRegex(ValueError, "every public case"):
                prep.prepare(root, Path(directory) / "inputs", facts_inputs=path)


if __name__ == "__main__":
    unittest.main()

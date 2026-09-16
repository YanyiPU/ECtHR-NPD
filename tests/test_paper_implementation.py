"""Offline contract tests: invented cases, tiny backbones, no downloads/API."""
from __future__ import annotations
import importlib.util
import json
import csv
import io
import subprocess
import tempfile
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
BASELINES = RELEASE / "code" / "baselines"
sys.path.insert(0, str(BASELINES))
from encoder.paper_model import LogTargetScaler, PaperEncoderRegressor, encode_chunks, torch

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
if HAS_PANDAS:
    from feature_profiles import TrainOnlyPreprocessor, validate_profile
    from retrieval.tabular_knn import load_inputs, run_knn
    from retrieval.bm25_pfme_knn import CaseDocument
    from data.data_loader import DatasetReleaseError, input_contract_summary, load_release_contract, load_structured_tree_split, resolve_dataset_release

spec = importlib.util.spec_from_file_location("paper_contract_react", RELEASE / "agent_knowledge_base" / "react_orchestrator.py")
REACT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = REACT
spec.loader.exec_module(REACT)


class PaperMathTests(unittest.TestCase):
    def test_target_train_standardization_and_inverse(self):
        scaler = LogTargetScaler.fit([0, 99, 9999])
        z = scaler.transform([0, 99, 9999])
        self.assertAlmostEqual(float(z.mean()), 0)
        self.assertAlmostEqual(float(z.std()), 1)
        np.testing.assert_allclose(scaler.inverse(z), [0, 99, 9999], atol=1e-8)
        self.assertEqual(float(scaler.inverse([-100])[0]), 0)

    def test_target_validation_and_constant_scale(self):
        self.assertEqual(LogTargetScaler.fit([0, 0]).scale, 1)
        for values in [[], [-1], [float("nan")], [float("inf")]]:
            with self.assertRaises(ValueError):
                LogTargetScaler.fit(values)

    def test_chunk_lengths_boundaries_and_masks(self):
        class Tokenizer:
            def num_special_tokens_to_add(self, pair=False): return 2
            def encode(self, text, **kwargs): return list(range(int(text)))[:kwargs["max_length"]]
            def prepare_for_model(self, tokens, **kwargs):
                sequence = [9001] + tokens + [9002]
                length = kwargs["max_length"]
                return {"input_ids": sequence + [0] * (length-len(sequence)),
                        "attention_mask": [1] * len(sequence) + [0] * (length-len(sequence))}
        tok = Tokenizer()
        long = encode_chunks("9000", tok, "legal_longformer")
        self.assertEqual([len(chunk) for chunk in long["input_ids"]], [4096, 4096])
        self.assertEqual(long["input_ids"][1][1], 4094)
        self.assertEqual(long["chunk_mask"], [1, 1])
        self.assertEqual(long["global_attention_mask"][1][0], 1)
        short = encode_chunks("10", tok, "legal_longformer")
        self.assertEqual(short["chunk_mask"], [1, 0])
        modern = encode_chunks("9000", tok, "modernbert")
        self.assertEqual(len(modern["input_ids"]), 1)
        self.assertEqual(len(modern["input_ids"][0]), 8192)


@unittest.skipIf(torch is None, "PyTorch not installed in this interpreter")
class PaperTorchTests(unittest.TestCase):
    def backbone(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=2)
            def forward(self, input_ids, attention_mask, **kwargs):
                return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1, 1, 2))
        return TinyBackbone()

    def test_mean_cls_pooling_and_smooth_l1(self):
        model = PaperEncoderRegressor(self.backbone())
        with torch.no_grad():
            model.regression.weight.fill_(0.5)
            model.regression.bias.zero_()
        ids = torch.tensor([[[2, 9], [6, 8]]])
        output = model(ids, torch.ones_like(ids), torch.tensor([[1., 1.]]), labels=torch.tensor([0.]))
        self.assertEqual(output["logits"].item(), 4.)
        self.assertEqual(output["loss"].item(), 3.5)
        output["loss"].backward()
        self.assertIsNotNone(model.regression.weight.grad)

    def test_empty_chunk_mask(self):
        model = PaperEncoderRegressor(self.backbone())
        with torch.no_grad():
            model.regression.weight.fill_(0.5)
            model.regression.bias.zero_()
        ids = torch.tensor([[[2, 9], [100, 8]]])
        self.assertEqual(model(ids, torch.ones_like(ids), torch.tensor([[1., 0.]]))["logits"].item(), 2.)

    def test_late_fusion_dimensions_and_required_input(self):
        model = PaperEncoderRegressor(self.backbone(), structured_dim=3, activation="relu")
        self.assertEqual(model.structured[0].out_features, 256)
        self.assertEqual(model.structured[2].out_features, 768)
        self.assertEqual(model.regression.in_features, 770)
        ids = torch.ones((2, 1, 3), dtype=torch.long)
        result = model(ids, torch.ones_like(ids), torch.ones(2, 1), structured_features=torch.ones(2, 3))
        self.assertEqual(tuple(result["logits"].shape), (2,))
        with self.assertRaises(ValueError):
            model(ids, torch.ones_like(ids), torch.ones(2, 1))


@unittest.skipUnless(HAS_PANDAS, "pandas unavailable in this interpreter")
class PaperTabularTests(unittest.TestCase):
    def profile(self):
        return {"profile": "metadata_knn", "columns": ["importance", "country"], "categorical_columns": ["country"],
                "lineage": {key: {"source_group": "metadata", "source_reference": "synthetic"} for key in ["importance", "country"]}}

    def test_train_only_numeric_and_unknown_category(self):
        prep = TrainOnlyPreprocessor().fit([{"importance": 1, "country": "A"}, {"importance": 3, "country": "B"}], self.profile())
        transformed = prep.transform([{"importance": "", "country": "NEW"}])
        np.testing.assert_array_equal(transformed, [[0, 0, 0]])
        self.assertEqual(prep.statistics["importance"]["median"], 2)

    def test_lineage_and_auxiliary_leakage_rejected(self):
        for column in ["itemid", "test_view", "perapp_has_joint_beneficiary_category_flag", "claim_non_pec_eur"]:
            profile = self.profile()
            profile["columns"].append(column)
            with self.assertRaises(ValueError): validate_profile(profile)
        profile = self.profile()
        profile["lineage"]["importance"]["source_group"] = "allocation"
        with self.assertRaises(ValueError): validate_profile(profile)
        allowed = json.dumps(input_contract_summary()["allowed_input_groups"])
        for column in ["perapp_unique_beneficiary_category_count", "perapp_has_joint_beneficiary_category_flag"]:
            self.assertNotIn(column, allowed)
            self.assertTrue(REACT.key_is_blocked(column, REACT.STRICT_REACT))
            self.assertTrue(REACT.reference_feature_key_is_blocked(column, REACT.STRICT_REACT))

    def test_knn_filters_median_and_fallback(self):
        docs = [CaseDocument("early", "train", datetime(2000, 1, 1), "", frozenset({"3"})),
                CaseDocument("future", "train", datetime(2020, 1, 1), "", frozenset({"3"})),
                CaseDocument("other_article", "train", datetime(2000, 1, 1), "", frozenset({"6"})),
                CaseDocument("target", "test", datetime(2010, 1, 1), "", frozenset({"3"})),
                CaseDocument("no_neighbors", "validation", datetime(1990, 1, 1), "", frozenset({"3"}))]
        features = [{"importance": i, "country": "A"} for i in [1, 2, 3, 1, 1]]
        rows, _ = run_knn(docs, features, {"early": 100, "future": 900, "other_article": 500}, self.profile(),
                          metric="euclidean", eval_splits={"validation", "test"})
        self.assertEqual(rows[0]["predicted_award_eur"], 100)
        self.assertEqual(json.loads(rows[0]["neighbor_itemids"]), ["early"])
        self.assertEqual(rows[1]["predicted_award_eur"], 500)
        self.assertTrue(rows[1]["fallback_reason"])

    def test_input_id_alignment_and_extra_features_rejected(self):
        metadata = [{"itemid": "a", "split": "train", "judgementdate": "2000-01-01", "violated_articles": "3"}]
        feature = [{"itemid": "b", "importance": 1, "country": "A"}]
        with self.assertRaises(ValueError): load_inputs(metadata, feature, self.profile())
        feature[0]["itemid"] = "a"
        feature[0]["secret"] = "must not be serialized"
        with self.assertRaises(ValueError): load_inputs(metadata, feature, self.profile())

    def test_versioned_counts_and_target_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            features = root / "model_inputs" / "structured_tree" / "features"
            targets = root / "model_inputs" / "structured_tree" / "targets"
            features.mkdir(parents=True)
            targets.mkdir(parents=True)
            contract = {"dataset_version": "corrected", "release_id": "synthetic_corrected", "counts": {"total": 4,
                        "splits": {"train": 1, "validation": 1, "test": 2},
                        "test_views": {"ID": 1, "OOD": 1}, "challenging": 1}}
            (root / "release_contract.json").write_text(json.dumps(contract))
            (root / "data" / "ecthr_npd_cases.csv").write_text("itemid,split,y_amount_eur,y_binary\na,train,100,1\nb,validation,0,0\nc,test,200,1\nd,test,300,1\n")
            from data.data_loader import SAFE_METADATA_COLUMNS, VIOLATED_ARTICLE_COLUMNS, SERIALIZED_CASE_FACT_COLUMNS, STRUCTURED_EXTERNAL_FACTOR_COLUMNS
            columns = [*SAFE_METADATA_COLUMNS, *VIOLATED_ARTICLE_COLUMNS, *SERIALIZED_CASE_FACT_COLUMNS, *STRUCTURED_EXTERNAL_FACTOR_COLUMNS]
            (features / "train.csv").write_text("itemid," + ",".join(columns) + "\na," + ",".join(["1"] * len(columns)) + "\n")
            (targets / "train.csv").write_text("itemid,y_amount_eur,y_binary\na,100,1\n")
            self.assertEqual(load_release_contract(root)["counts"]["total"], 4)
            self.assertEqual(len(load_structured_tree_split("train", root).features), 1)
            (targets / "train.csv").write_text("itemid,y_amount_eur,y_binary\na,999,1\n")
            with self.assertRaisesRegex(DatasetReleaseError, "differs"):
                load_structured_tree_split("train", root)
            contract["counts"]["total"] = 5
            (root / "release_contract.json").write_text(json.dumps(contract))
            with self.assertRaises(DatasetReleaseError): load_release_contract(root)
            with self.assertRaises(DatasetReleaseError): resolve_dataset_release(root / "missing")

    def test_actual_tabular_cli_with_invented_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.csv").write_text("itemid,split,judgementdate,violated_articles\na,train,2000-01-01,3\nb,test,2010-01-01,3\n")
            (root / "features.csv").write_text("itemid,importance,country\na,1,A\nb,2,A\n")
            (root / "targets.csv").write_text("itemid,y_amount_eur\na,100\n")
            (root / "profile.json").write_text(json.dumps(self.profile()))
            result = subprocess.run([sys.executable, "-B", str(BASELINES / "retrieval" / "tabular_knn.py"),
                "--metadata", str(root / "metadata.csv"), "--features", str(root / "features.csv"),
                "--train-targets", str(root / "targets.csv"), "--profile", str(root / "profile.json"),
                "--metric", "euclidean", "--example-subset-only", "--output-dir", str(root / "output")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "output" / "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]["predicted_award_eur"]), 100.)
            self.assertEqual(json.loads((root / "output" / "run_manifest.json").read_text())["status"],
                             "new_implementation_not_historical_reproduction")


class PaperReactTests(unittest.TestCase):
    def test_preset_and_conflicting_override(self):
        with patch.object(sys, "argv", ["react", "--case_file", "synthetic.json", "--paper_preset"]):
            args = REACT.parse_args()
        self.assertEqual((args.max_steps, args.top_k, args.inference_mode), (12, 5, "few_shot"))
        with patch.object(sys, "argv", ["react", "--case_file", "synthetic.json", "--paper_preset", "--top_k", "10"]), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit): REACT.parse_args()

    def test_table25_whitelist_and_required_actions(self):
        state = {"paper_preset": True, "react_mode": REACT.STRICT_REACT, "inference_mode": "few_shot", "target_context_policy": "lazy"}
        actions = REACT.allowed_actions_for_state(state)
        self.assertEqual(len(actions), 10)
        self.assertNotIn("load_relevant_modules", actions)
        self.assertNotIn("assess_zero_positive_evidence", actions)
        completed = {"inspect_case", "query_target_information", "search_modules", "load_module", "resolve_empirical_priors",
                     "retrieve_train_references", "assess_aggregation_pattern"}
        self.assertEqual(REACT.missing_required_actions(state, completed), [])

    def test_reference_union_is_total_budget_not_perclass(self):
        rows = [{"itemid": str(i), "split": "train"} for i in range(10)]
        labels = {str(i): {"amount": i} for i in range(10)}
        def candidate(case, row, label, *args):
            return {"row": row, "y_binary": int(row["itemid"]) % 2, "sim_score": 10-int(row["itemid"])}
        def ref(value, *args, **kwargs): return {"itemid": value["row"]["itemid"]}
        with patch.object(REACT, "build_reference_candidate", side_effect=candidate), \
             patch.object(REACT, "apply_domain_filters", side_effect=lambda candidates, n: (candidates, [])), \
             patch.object(REACT, "candidate_to_trace", side_effect=ref), \
             patch.object(REACT, "candidate_to_reference", side_effect=ref):
            result = REACT.retrieve_train_references({"itemid": "target", "judgementdate": "2025-01-01", "violated_articles": ["3"]}, rows, labels, 5)
        all_ids = {r["itemid"] for key in ["reference_cases", "positive_reference_cases", "zero_reference_cases"] for r in result[key]}
        self.assertEqual(len(result["reference_cases"]), 5)
        self.assertEqual(len(all_ids), 5)
        self.assertEqual(len(result["positive_reference_cases"]), 3)
        self.assertEqual(len(result["zero_reference_cases"]), 2)
        with self.assertRaises(ValueError): REACT.retrieve_train_references({}, [], {}, 0)

    def test_diagnostics_required_and_nan_rejected(self):
        state = {"paper_preset": True, "react_mode": REACT.STRICT_REACT, "leakage_audit": {"passed": True}}
        for value in [float("nan"), float("inf"), -1]:
            result = REACT.execute_action(state, {"action": "final_predict", "action_input": {"award_eur": value}})
            self.assertIn("error", result)
        with patch.object(REACT, "selected_target_snapshot", return_value={}), patch.object(REACT, "target_claim_cap_from_snapshot", return_value={}):
            valid = {"award_eur": 100., "rationale_summary": "Synthetic summary", "zero_positive_decision": "positive",
                     "aggregation_scale_decision": "single case", "uncertainty": "high"}
            result = REACT.execute_action(state, {"action": "final_predict", "action_input": valid})
        self.assertEqual(result["prediction"], valid)
        invalid = dict(valid, zero_positive_decision="zero")
        self.assertIn("error", REACT.execute_action(state, {"action": "final_predict", "action_input": invalid}))

    def test_mocked_paper_loop_preserves_diagnostics_without_api(self):
        state = {"paper_preset": True, "react_mode": REACT.STRICT_REACT, "inference_mode": "few_shot",
                 "itemid": "synthetic", "target_context_policy": "lazy", "leakage_audit": {"passed": True}}
        prediction = {"award_eur": 0., "rationale_summary": "Synthetic summary", "zero_positive_decision": "zero",
                      "aggregation_scale_decision": "case level", "uncertainty": "high"}
        action_names = ["inspect_case", "query_target_information", "search_modules", "load_module", "resolve_empirical_priors",
                        "retrieve_train_references", "assess_aggregation_pattern", "final_predict"]
        actions = [({"thought_summary": "Synthetic", "action": name,
                     "action_input": prediction if name == "final_predict" else {}}, {}) for name in action_names]
        original = REACT.execute_action
        def execute(state, action, **kwargs):
            return original(state, action, **kwargs) if action["action"] == "final_predict" else {}
        with patch.object(REACT, "chat_json_with_retry", side_effect=actions), \
             patch.object(REACT, "execute_action", side_effect=execute), \
             patch.object(REACT, "selected_target_snapshot", return_value={}), \
             patch.object(REACT, "target_claim_cap_from_snapshot", return_value={}):
            result = REACT.run_live_trace(state, None, 12, 0.0)
        self.assertEqual(result["prediction"], prediction)
        self.assertEqual(len(result["events"]), 8)

    def test_paper_dry_trace_contains_only_whitelisted_actions(self):
        state = {"paper_preset": True, "react_mode": REACT.STRICT_REACT, "inference_mode": "few_shot",
                 "itemid": "synthetic", "top_k": 5, "target_context_policy": "lazy", "leakage_audit": {"passed": True}}
        with patch.object(REACT, "execute_action", return_value={}):
            trace = REACT.deterministic_dry_run_trace(state)
        self.assertTrue({event["action"] for event in trace["events"]} <= set(REACT.allowed_actions_for_state(state)))
        self.assertLessEqual(len(trace["events"]), 12)
        self.assertIsNone(trace["prediction"])


class PriorPinTests(unittest.TestCase):
    def make_inputs(self, root):
        cases = root / "cases.csv"
        targets = root / "targets.csv"
        cases.write_text("itemid,split,split_role,judgementdate,country_alpha2,violated_articles,y_amount_eur\na,train,,2000-01-01,AA,3,100\nb,test,,2020-01-01,BB,6,999999\n")
        targets.write_text("itemid,y_amount_eur\na,100\n")
        return cases, targets

    def make_bundle(self, root, cases, targets):
        bundle = root / "priors"
        (root / "data").mkdir()
        (root / "data/ecthr_npd_cases.csv").write_text(cases.read_text())
        target_dir = root / "model_inputs/structured_tree/targets"
        target_dir.mkdir(parents=True)
        (target_dir / "train.csv").write_text(targets.read_text())
        (root / "release_contract.json").write_text(json.dumps({"dataset_version": "corrected", "release_id": "synthetic",
            "counts": {"total": 2, "splits": {"train": 1, "validation": 0, "test": 1}, "test_views": {"ID": 1, "OOD": 0}, "challenging": 0}}))
        result = subprocess.run([sys.executable, "-B", str(RELEASE / "agent_knowledge_base" / "build_train_article_priors.py"),
            "--dataset-release", str(root), "--case-rows", str(cases), "--train-labels", str(targets), "--output-dir", str(bundle)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return bundle

    def test_builder_uses_only_train_and_pins_all_four_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, targets = self.make_inputs(root)
            bundle = self.make_bundle(root, cases, targets)
            verified = REACT.PRIOR_CONTRACT.verify_prior_bundle(bundle, cases, targets)
            self.assertEqual(verified["cohort_pin"]["train_count"], 1)
            self.assertEqual(len(verified["artifact_sha256"]), 4)
            with (bundle / "country_award_distribution_train.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["country_alpha2"], "AA")
            self.assertEqual(float(rows[0]["median_all"]), 100.)
            train_only = root / "train.csv"
            train_only.write_text("\n".join(cases.read_text().splitlines()[:2]) + "\n")
            self.assertEqual(REACT.PRIOR_CONTRACT.cohort_pin(train_only, targets), verified["cohort_pin"])

    def test_real_state_assembly_keeps_controller_report_internal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, targets = self.make_inputs(root)
            bundle = self.make_bundle(root, cases, targets)
            target = {"itemid": "synthetic-target", "judgementdate": "2025-01-01", "country_alpha2": "AA",
                      "violated_articles": ["3"], "num_applicants": 1,
                      "combined_input_text": "FACTS: This is an invented test case."}
            state = REACT.build_state(RELEASE / "agent_knowledge_base", target, REACT.STRICT_REACT, "zero_shot", 5,
                                      cases, targets, "lazy", "lazy", prior_dir=bundle)
            self.assertTrue(state["leakage_audit"]["passed"], state["leakage_audit"])
            self.assertNotIn("target_redaction_report", state["case_inputs"])
            self.assertIn("target_final_award_text_redaction_count", state["internal_target_redaction_report"])

    def test_target_metadata_and_artifact_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, targets = self.make_inputs(root)
            bundle = self.make_bundle(root, cases, targets)
            original = cases.read_text()
            cases.write_text(original.replace(",AA,", ",CC,"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                REACT.PRIOR_CONTRACT.verify_prior_bundle(bundle, cases, targets)
            cases.write_text(original.replace(",100\n", ",200\n"))
            targets.write_text("itemid,y_amount_eur\na,200\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                REACT.PRIOR_CONTRACT.verify_prior_bundle(bundle, cases, targets)
            cases.write_text(original)
            targets.write_text("itemid,y_amount_eur\na,100\n")
            table = bundle / "article_award_distribution_train.csv"
            table.write_text(table.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "artifact"):
                REACT.PRIOR_CONTRACT.verify_prior_bundle(bundle, cases, targets)

    def test_legacy_manifest_is_rejected_before_state_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, targets = self.make_inputs(root)
            bundle = root / "priors"
            bundle.mkdir()
            (bundle / "PRIOR_MANIFEST.v1.json").write_text('{"manifest_version":"1.0.0"}')
            with self.assertRaisesRegex(ValueError, "legacy/unpinned"):
                REACT.build_state(RELEASE / "agent_knowledge_base", {}, REACT.STRICT_REACT, "few_shot", 5,
                                  cases, targets, "lazy", "lazy", prior_dir=bundle)

    def test_prior_version_marker_and_selected_cohort_are_both_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, targets = self.make_inputs(root)
            bundle = self.make_bundle(root, cases, targets)
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                REACT.PRIOR_CONTRACT.verify_prior_bundle(bundle, cases, targets, dataset_version="paper_reference")
            targets.write_text("itemid,y_amount_eur\na,90\n")
            cases.write_text(cases.read_text().replace(",100\n", ",90\n"))
            with self.assertRaisesRegex(ValueError, "selected dataset"):
                REACT.PRIOR_CONTRACT.bind_selected_training_inputs(root, cases, targets)

    def test_prior_binary_inconsistency_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, targets = self.make_inputs(root)
            targets.write_text("itemid,y_amount_eur,y_binary\na,100,0\n")
            with self.assertRaisesRegex(ValueError, "binary"):
                REACT.PRIOR_CONTRACT.read_training_inputs(cases, targets)

    def test_react_prediction_export_is_evaluator_compatible_and_keeps_provenance(self):
        from evaluate import _read_records, align_prediction_records
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction.json"
            result = {"itemid": "invented", "prediction": {"award_eur": 100., "uncertainty": "high"},
                      "implementation_manifest": {"dataset_provenance": {"dataset_version": "corrected"}}}
            REACT.write_outputs(result, None, str(output))
            truth, predicted, ids = align_prediction_records(_read_records(output), [{"itemid": "invented", "y_amount_eur": 90.}])
            self.assertEqual(ids, ["invented"])
            self.assertEqual(predicted.tolist(), [100.])
            self.assertEqual(json.loads(output.read_text())["implementation_manifest"], result["implementation_manifest"])

    def test_missing_duplicate_or_nontrain_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, targets = self.make_inputs(root)
            for content in ["itemid,y_amount_eur\n", "itemid,y_amount_eur\na,100\na,100\n",
                            "itemid,y_amount_eur\na,100\nb,999999\n", "itemid,y_amount_eur\na,NaN\n"]:
                targets.write_text(content)
                with self.assertRaises(ValueError): REACT.PRIOR_CONTRACT.read_training_inputs(cases, targets)
            self.assertEqual(REACT.PRIOR_CONTRACT.articles('["3", "3", "6"]'), ["3", "6"])


if __name__ == "__main__":
    unittest.main()

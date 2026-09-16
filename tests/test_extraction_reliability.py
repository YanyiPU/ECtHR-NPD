"""Offline control-flow regressions; synthetic fixtures, never provider calls.

Run with the extraction requirements installed:
python3 -B -m unittest discover -s tests -p test_extraction_reliability.py -v
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error


CODE = Path(__file__).resolve().parents[1] / "extraction_pipeline" / "code"
sys.path.insert(0, str(CODE))
import case_store
import holistic_extractor as holistic
import openai_compatible_client as api
import run_pipeline_b_backbone as backbone
import run_pipeline_b_extraction as pipeline_b
import run_pipeline_c_backbone as pipeline_c
import run_pipeline_e_extraction as pipeline_e


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def args(**changes):
    values = dict(itemids=["a", "b"], splits=["nonexistent"], max_cases=None,
                  run_name="synthetic", dry_run=False, resume=False, concurrency=1,
                  max_retries=0, regex_only=False)
    values.update(changes)
    return SimpleNamespace(**values)


class CaseStoreTests(unittest.TestCase):
    def test_large_object_does_not_stall(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            dump(path, [{"itemid": "large", "text": "x" * (2 * 1024 * 1024)}, {"itemid": "last"}])
            # Timeout catches the former infinite loop without hanging the suite.
            code = ("import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; "
                    "from case_store import iter_cases_from_cases_json; "
                    "print([x['itemid'] for x in iter_cases_from_cases_json(Path(sys.argv[2]))])")
            result = subprocess.run([sys.executable, "-B", "-c", code, str(CODE), str(path)],
                                    capture_output=True, text=True, timeout=10, check=True)
            self.assertIn("['large', 'last']", result.stdout)

    def test_truncated_array_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text('[{"itemid":"a"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(case_store.iter_cases_from_cases_json(path, chunk_size=4))

    def test_store_and_fallback_merge_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump(root / "store" / "a.json", {"itemid": "a", "value": "preferred"})
            dump(root / "cases.json", [{"itemid": "a", "value": "old"}, {"itemid": "b"}])
            rows = case_store.load_cases_by_itemid(["a", "b"], root / "store", root / "cases.json", False)
            self.assertEqual(set(rows), {"a", "b"})
            self.assertEqual(rows["a"]["value"], "preferred")
            self.assertFalse((root / "store" / "b.json").exists())


class SchemaTests(unittest.TestCase):
    def test_missing_dependency_fails_closed(self):
        with mock.patch.dict(sys.modules, {"jsonschema": None}):
            with self.assertRaisesRegex(RuntimeError, "requires jsonschema"):
                api.schema_errors({}, {})

    def test_invalid_schema_is_fatal(self):
        from jsonschema.exceptions import SchemaError
        with self.assertRaises(SchemaError):
            api.schema_errors({}, {"type": "not_a_json_schema_type"})

    def test_every_stage_uses_full_schema_validation(self):
        schema = {"type": "object", "additionalProperties": False}
        validators = [backbone.validate_result, pipeline_b.validate_result,
                      pipeline_c.validate_llm_result, pipeline_e.validate_result,
                      holistic.validate_legal_analysis_result]
        for validate in validators:
            with self.subTest(stage=validate.__module__):
                self.assertTrue(any("Additional properties" in e for e in validate({"unexpected": True}, schema)))

    def test_schema_dependency_checked_before_paid_call(self):
        client = mock.Mock()
        with mock.patch.dict(sys.modules, {"jsonschema": None}):
            with self.assertRaises(RuntimeError):
                pipeline_b.run_one_case("a", {}, {}, client, "", {}, max_retries=0)
        client.chat_json.assert_not_called()


class ApplicantCountTests(unittest.TestCase):
    def test_equal_length_unlinked_awards_never_link_by_row_order(self):
        b = {"facts_procedure": {"num_applicants": 2, "applicants": [
            {"applicant_index": 1, "beneficiary_label": "Person A"},
            {"applicant_index": 2, "beneficiary_label": "Person B"}]}}
        rows = [{"beneficiary_label": None, "applicant_index": None, "head": "non_pecuniary", "eur_amount": n} for n in (10, 20)]
        result = holistic._repair_award_rows(rows, b)
        self.assertEqual(result, rows)
        self.assertTrue(all(row["applicant_index"] is None for row in result))

    def test_fewer_awards_than_people_do_not_create_or_link_missing_rows(self):
        b = {"facts_procedure": {"num_applicants": 4, "applicants": []}}
        rows = [{"beneficiary_label": None, "applicant_index": None, "head": "non_pecuniary", "eur_amount": 100}]
        self.assertEqual(holistic._repair_award_rows(rows, b), rows)

    def test_joint_group_awards_keep_amount_once_and_record_source_index_separately(self):
        for label in ("applicants jointly", "Person A and Person B", "the family"):
            with self.subTest(label=label):
                row = {"beneficiary_label": label, "applicant_index": 1, "head": "non_pecuniary", "eur_amount": 100}
                c = {"article_41_extraction": {"award_per_applicant": [row]}, "final_awards": {"total_eur": 100}}
                result = holistic._repair_c_per_applicant(c, {"facts_procedure": {"num_applicants": 2, "applicants": []}})
                self.assertEqual(len(result["article_41_extraction"]["award_per_applicant"]), 1)
                self.assertIsNone(result["article_41_extraction"]["award_per_applicant"][0]["applicant_index"])
                self.assertEqual(result["article_41_extraction"]["award_per_applicant"][0]["eur_amount"], 100)
                audit = result["per_applicant_mapping_audit"][0]
                self.assertEqual(audit["source_applicant_index"], 1)
                self.assertIsNone(audit["verified_applicant_id"])
                self.assertTrue(audit["identity_review_required"])

    def test_candidate_source_index_never_fills_person_name_from_position(self):
        row = {"beneficiary_label": None, "applicant_index": 1, "head": "non_pecuniary", "eur_amount": 100}
        b = {"facts_procedure": {"num_applicants": 1, "applicants": [{"applicant_index": 1, "beneficiary_label": "Person A"}]}}
        self.assertEqual(holistic._repair_award_rows([row], b), [row])

    def test_application_count_is_not_a_person_count(self):
        row = {"core_case": {"num_applicants_proxy": 7, "num_application_numbers": 7}}
        count, notes = backbone._extract_num_applicants(row, {"docname": "EXAMPLE AND OTHERS v. STATE"}, "", "")
        self.assertIsNone(count)
        self.assertIn("unknown", notes[0])
        self.assertIsNone(pipeline_c._claim_num_applicants(row))

    def test_person_count_evidence_still_works(self):
        row = {"core_case": {"num_applicants_proxy": 1}}
        self.assertEqual(backbone._extract_num_applicants(row, {"n_applicants": 12}, "", "")[0], 12)
        self.assertEqual(pipeline_c._claim_num_applicants({**row, "facts_procedure": {"num_applicants": 12}}), 12)

    def test_missing_count_backbone_is_not_a_successful_single_applicant(self):
        row = {"facts_procedure": {}, "core_case": {}, "claim_and_award_layer": {}}
        result = backbone.run_one_case("a", row, {}, {})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("result", result)


class RetryTests(unittest.TestCase):
    def test_actual_b_stage_retries_schema_failure(self):
        source = {"itemid": "a", "n_applicants": 1, "content": {"document": {}}}
        scaffold = holistic._build_scaffold(source, {}, len)
        client = mock.Mock(use_json_schema=False)
        client.chat_json.side_effect = [
            ({"facts_procedure": {"num_applicants": 1}}, {"total_tokens": 3}),
            ({"marker": True, "facts_procedure": {"num_applicants": 1}}, {"total_tokens": 4}),
        ]
        payload = pipeline_b.run_one_case("a", scaffold, source, client, "synthetic", {"required": ["marker"]}, max_retries=1)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["attempts"], 2)
        self.assertEqual(payload["usage"]["total_tokens"], 7)
        self.assertEqual(client.chat_json.call_count, 2)

    def test_transient_retry_budget_and_usage(self):
        count = []

        @api.retry_extraction
        def stage(client, schema, max_retries=2):
            count.append(1)
            return {"status": "api_error", "api_error": {"kind": "connect"},
                    "usage": {"total_tokens": 2}}

        with mock.patch.object(api.time, "sleep") as sleep:
            output = stage(mock.Mock(), {}, 2)
        self.assertEqual(len(count), 3)
        self.assertEqual(output["attempts"], 3)
        self.assertEqual(output["usage"]["total_tokens"], 6)
        self.assertEqual(sleep.call_args_list, [mock.call(1), mock.call(2)])

    def test_auth_and_payload_errors_do_not_retry(self):
        for kind in ("http_4xx", "payload", "no_json"):
            with self.subTest(kind=kind):
                count = []

                @api.retry_extraction
                def stage(client, max_retries=3):
                    count.append(1)
                    return {"status": "api_error", "api_error": {"kind": kind}}

                self.assertEqual(stage(mock.Mock())["attempts"], 1)
                self.assertEqual(len(count), 1)

    def test_schema_retry_sends_feedback_without_mutating_shared_client(self):
        client = mock.Mock()
        client.chat_json.side_effect = [({"bad": 1}, {"total_tokens": 5}), ({"good": 1}, {"total_tokens": 7})]

        @api.retry_extraction
        def stage(client, max_retries=1):
            parsed, usage = client.chat_json(messages=[{"role": "user", "content": "fixture"}])
            if "bad" in parsed:
                return {"status": "schema_validation", "errors": ["good is required"], "usage": usage}
            return {"status": "success", "result": parsed, "usage": usage}

        result = stage(client)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["usage"]["total_tokens"], 12)
        self.assertEqual(len(client.chat_json.call_args_list[0].kwargs["messages"]), 1)
        second_messages = client.chat_json.call_args_list[1].kwargs["messages"]
        self.assertIn("good is required", second_messages[-1]["content"])
        self.assertEqual(second_messages[-2]["role"], "assistant")

    def test_combined_stage_does_not_double_count(self):
        count = []

        @api.retry_extraction
        def stage(client, max_retries=1):
            count.append(1)
            return ({"status": "success", "usage": {"total_tokens": 9}},
                    {"status": "schema_validation" if len(count) == 1 else "success", "errors": ["invalid"]})

        d, e = stage(mock.Mock())
        self.assertEqual(d["usage"]["total_tokens"], 18)
        self.assertEqual(e["usage"]["total_tokens"], 0)
        self.assertEqual(e["status"], "success")

    def test_invalid_budget_rejected(self):
        @api.retry_extraction
        def stage(client, max_retries=0):
            self.fail("invalid budget must fail before stage")

        for budget in (-1, 11, True, 1.5):
            with self.assertRaises(ValueError):
                stage(mock.Mock(), budget)

    def test_chat_post_has_no_hidden_retry(self):
        client = api.OpenAICompatibleClient("https://example.invalid", "synthetic", "synthetic")
        error = urllib.error.HTTPError("https://example.invalid", 429, "fixture", {}, io.BytesIO(b"limited"))
        with mock.patch.object(api.urllib.request, "urlopen", side_effect=error) as request:
            with self.assertRaises(api.ApiCallError) as caught:
                client._post_json("/chat/completions", {})
        self.assertEqual(caught.exception.kind, "http_429")
        self.assertEqual(request.call_count, 1)


class EntryPointTests(unittest.TestCase):
    def test_invalid_c_candidate_is_not_compacted_into_awards(self):
        source = {"itemid": "a", "appno": ["1/20"], "conclusion": [], "article": [], "content": {"document": {}}}
        b = {"status": "success", "result": {"facts_procedure": {"num_applicants": 3, "applicants": []}}}
        c = {"status": "schema_validation", "raw_result": {"awards": {"non_pecuniary": {"eur_amount": 999}}},
             "errors": ["synthetic invalid candidate"], "awards_regex": {}}
        d = {"status": "success", "result": {"legal_analysis": {}}}
        e = {"status": "success", "result": {"reasoning_layer": {}}}
        with mock.patch.object(holistic, "run_pipeline_b_case", return_value=b), \
             mock.patch.object(holistic, "run_pipeline_c_case", return_value=c) as c_call, \
             mock.patch.object(holistic, "run_pipeline_de_combined_case", return_value=(d, e)):
            result = holistic.run_one_case("a", source, {}, len, mock.Mock(), {}, {}, {}, {}, "", "", "", "", mock.Mock(), 0)
        self.assertEqual(result["status"], "partial_success")
        self.assertIsNone(result["result"]["final_awards"])
        self.assertFalse(result["layers"]["pipeline_c_llm"]["stage_meta"]["used_in_combined_output"])
        self.assertEqual(c_call.call_args.args[1]["facts_procedure"]["num_applicants"], 3)

    def test_b_e_mixed_inputs_keep_per_case_only_ids_and_skip_splits(self):
        for module in (pipeline_b, pipeline_e):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
                root = Path(tmp)
                dump(root / "outputs" / "cases" / "a.json", {"itemid": "a", "origin": "per_case"})
                input_path = root / "input.jsonl"
                jsonl(input_path, [{"itemid": "b"}])
                stack.enter_context(mock.patch.object(module, "EXTRACTION_ROOT", root))
                stack.enter_context(mock.patch.object(module, "INPUT_JSONL", input_path))
                stack.enter_context(mock.patch.object(module, "RUNS_ROOT", root / "runs"))
                stack.enter_context(mock.patch.object(module, "parse_args", return_value=args(dry_run=True)))
                split = stack.enter_context(mock.patch.object(module, "load_split_ids", side_effect=AssertionError("historical split requested")))
                if module is pipeline_b:
                    stack.enter_context(mock.patch.object(module, "load_cases_by_itemid", return_value={"a": {}, "b": {}}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                module.main()
                split.assert_not_called()

    def test_b_backbone_resume_preserves_earlier_split_rows(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            jsonl(root / "input.jsonl", [{"itemid": "a"}, {"itemid": "b"}])
            jsonl(root / "runs" / "synthetic" / "results_unique.jsonl", [{"itemid": "a", "preserved": True}])
            stack.enter_context(mock.patch.object(backbone, "RUNS_ROOT", root / "runs"))
            stack.enter_context(mock.patch.object(backbone, "INPUT_JSONL", root / "input.jsonl"))
            stack.enter_context(mock.patch.object(backbone, "parse_args", return_value=args(resume=True)))
            stack.enter_context(mock.patch.object(backbone, "load_cases_by_itemid", return_value={"a": {}, "b": {}}))
            stage = stack.enter_context(mock.patch.object(backbone, "run_one_case", return_value={"status": "success", "result": {"itemid": "b"}}))
            stack.enter_context(mock.patch.object(backbone, "load_split_ids", side_effect=AssertionError("historical split requested")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            backbone.main()
            rows = backbone.load_jsonl_by_itemid(root / "runs" / "synthetic" / "by_split" / "manual_itemids.jsonl")
            self.assertEqual(set(rows), {"a", "b"})
            self.assertTrue(rows["a"]["preserved"])
            self.assertEqual(stage.call_count, 1)

    def test_c_regex_only_actual_cli_needs_no_api_configuration(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            row = {"itemid": "a", "core_case": {}, "claim_and_award_layer": {"cross_validation_inputs": {}}}
            jsonl(root / "input.jsonl", [row])
            stack.enter_context(mock.patch.object(pipeline_c, "RUNS_ROOT", root / "runs"))
            stack.enter_context(mock.patch.object(pipeline_c, "INPUT_JSONL", root / "input.jsonl"))
            stack.enter_context(mock.patch.object(pipeline_c, "parse_args", return_value=args(itemids=["a"], regex_only=True)))
            api_factory = stack.enter_context(mock.patch.object(api.OpenAICompatibleClient, "from_env", side_effect=AssertionError("API not allowed")))
            stack.enter_context(mock.patch.object(pipeline_c, "load_split_ids", side_effect=AssertionError("historical split requested")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            pipeline_c.main()
            api_factory.assert_not_called()
            rows = pipeline_c.load_jsonl_by_itemid(root / "runs" / "synthetic" / "results_unique.jsonl")
            self.assertTrue(rows["a"]["cross_validation"]["flag_for_review"])
            self.assertIsNone(rows["a"]["cross_validation"]["non_pec_match"])
            self.assertEqual(rows["a"]["extraction_mode"], "deterministic_regex")
            self.assertEqual(rows["a"]["final_awards"]["non_pecuniary_source"], "not_found_in_extraction")

    def test_holistic_resume_excludes_partial_and_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for itemid, status in (("a", "success"), ("b", "partial_success"), ("c", "request_failed")):
                dump(root / f"{itemid}.meta.json", {"itemid": itemid, "status": status})
            self.assertEqual(holistic.load_completed_ids(root), {"a"})

    def test_holistic_partial_is_quarantined_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            stack.enter_context(mock.patch.object(holistic, "RUNS_ROOT", root / "runs"))
            stack.enter_context(mock.patch.object(holistic, "CASES_OUTPUT", root / "cases"))
            stack.enter_context(mock.patch.object(holistic, "parse_args", return_value=args(itemids=["a"])))
            stack.enter_context(mock.patch.object(holistic, "load_cases_by_itemid", return_value={"a": {}}))
            stack.enter_context(mock.patch.object(holistic, "load_cases_core_lookup", return_value={}))
            stack.enter_context(mock.patch.object(holistic, "build_token_counter", return_value=(len, "test")))
            stack.enter_context(mock.patch.object(api.OpenAICompatibleClient, "from_env", return_value=mock.Mock()))
            stack.enter_context(mock.patch.object(holistic, "run_one_case", return_value={
                "itemid": "a", "status": "partial_success", "layers": {"merged": {"itemid": "a"}}}))
            write = stack.enter_context(mock.patch.object(holistic, "write_case_result"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaises(SystemExit) as failed:
                holistic.main()
            self.assertEqual(failed.exception.code, 1)
            write.assert_not_called()
            self.assertTrue((root / "runs" / "synthetic" / "quarantine" / "a.json").exists())
            summary = json.loads((root / "runs" / "synthetic" / "summary.json").read_text())
            self.assertEqual(summary["completed_cases"], 0)
            self.assertEqual(summary["requires_review_or_retry"], 1)
            self.assertEqual(holistic.load_completed_ids(root / "runs" / "synthetic"), set())


if __name__ == "__main__":
    unittest.main()

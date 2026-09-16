"""Offline regression tests; synthetic documents only, no model/network calls."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from datetime import date
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_reconstruction"))
sys.path.insert(0, str(ROOT / "extraction_pipeline" / "code"))

import build_echrod_subset as metadata_export
import build_extraction_layers as scaffold
import download_hudoc_judgments as downloader
import plan_incremental_ingestion as planner
import prepare_extraction_case_store as preparation
import source_utils as utils


def docx_bytes(text: str = "Synthetic judgment") -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                         f'<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')
    return buffer.getvalue()


class MetadataTests(unittest.TestCase):
    def test_internal_export_retains_identity_and_structured_evidence(self):
        original = {"itemid": "001-1", "appno": "123/20;124/20", "representedby": '["Synthetic counsel"]',
                    "conclusion": '[{"type":"violation","article":"3"}]', "article": "['3', '6']", "docname": "Synthetic case"}
        result = metadata_export.normalize_row(original)
        self.assertEqual(result["representedby"], ["Synthetic counsel"])
        self.assertEqual(result["article"], ["3", "6"])
        self.assertEqual(result["conclusion"][0]["article"], "3")
        self.assertEqual(result["source_metadata"], original)

    def test_unparsed_conclusion_is_unknown_not_no_violation(self):
        result = utils.normalize_metadata({"conclusion": "Violation of Article 3"})
        self.assertIsNone(result["conclusion"])
        self.assertEqual(result["__conclusion"], "Violation of Article 3")
        self.assertTrue(result["metadata_normalization_issues"])

    def test_missing_metadata_not_zero_or_one(self):
        core = scaffold.make_core_row({"itemid": "001-1"}, None, None, [])
        for field in ("num_applicants_proxy", "num_application_numbers", "num_violations_found", "represented",
                      "conclusion_count", "challenging_metadata_eligible", "has_separate_opinion", "has_mixed_outcome"):
            self.assertIsNone(core[field], field)

    def test_selector_uses_distinct_metadata_codes_and_app_numbers(self):
        row = {"itemid": "001-1", "appno": "123/20;124/20;123/20", "conclusion": [
            {"type": "violation", "article": "6-1", "base_article": "6"},
            {"type": "violation", "article": "6-3", "base_article": "6"},
            {"type": "violation", "article": "6-3", "base_article": "6"}], "representedby": []}
        core = scaffold.make_core_row(row, None, None, [])
        self.assertEqual(core["num_application_numbers"], 2)
        self.assertEqual(core["num_distinct_violated_article_codes"], 2)
        self.assertEqual(core["num_violations_found"], 3)
        self.assertTrue(core["challenging_metadata_eligible"])
        self.assertFalse(core["represented"])

    def test_duplicate_findings_do_not_create_challenging_case(self):
        row = {"itemid": "001-1", "doctypebranch": "CHAMBER", "appno": ["123/20", "124/20"], "conclusion": [
            {"type": "violation", "article": "3"}, {"type": "violation", "article": "3"}]}
        self.assertFalse(scaffold.make_core_row(row, None, None, [])["challenging_metadata_eligible"])

    def test_grand_chamber_selector_arm_survives_missing_other_metadata(self):
        row = {"itemid": "001-1", "doctypebranch": "GRAND CHAMBER"}
        core = scaffold.make_core_row(row, None, None, [])
        self.assertTrue(core["challenging_metadata_eligible"])
        self.assertIsNone(core["multi_application_multi_violation_condition"])

    def test_one_hot_mentioned_articles_not_used_as_violations(self):
        row = utils.normalize_metadata({"itemid": "001-1", "appno": "123/20;124/20", "article=3": "1", "article=6": "1"})
        core = scaffold.make_core_row(row, None, None, [])
        self.assertEqual(core["mentioned_articles"], ["3", "6"])
        self.assertIsNone(core["challenging_metadata_eligible"])

    def test_conflicting_metadata_requires_explicit_precedence(self):
        row, metadata = {"itemid": "001-1", "appno": "123/20"}, {"itemid": "001-1", "appno": "124/20"}
        with self.assertRaisesRegex(ValueError, "conflicting metadata"):
            preparation.combine_metadata(row, metadata)
        merged, conflicts = preparation.combine_metadata(row, metadata, "echrod")
        self.assertEqual(merged["appno"], "124/20")
        self.assertEqual(conflicts, ["appno"])

    def test_historical_core_is_optional_but_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "core.json"
            self.assertEqual(scaffold.load_optional_core(path), {})
            utils.write_json(path, [{"itemid": "001-1"}, {"itemid": "001-1"}])
            with self.assertRaises(ValueError):
                scaffold.load_optional_core(path)


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        self.index = self.root / "index.csv"
        self.workspace = self.root / "workspace"

    def args(self, overwrite=False):
        return argparse.Namespace(case_index=self.index, hudoc_docx_dir=self.docs, out_root=self.workspace,
                                  echrod_metadata=None, limit=None, overwrite=overwrite, metadata_precedence="error", resume_interrupted=False)

    def batch(self, rows):
        utils.write_csv(self.index, rows, list(rows[0]))
        for row in rows:
            (self.docs / f"{row['itemid']}.docx").write_bytes(docx_bytes())

    def ingest(self, overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(preparation, "parse_args", return_value=self.args(overwrite)):
            return preparation.main()

    def test_append_preserves_old_order_and_is_idempotent(self):
        self.batch([{"itemid": "001-2"}, {"itemid": "001-1"}])
        self.assertEqual(self.ingest(), 0)
        self.batch([{"itemid": "001-3"}])
        self.assertEqual(self.ingest(), 0)
        path = self.workspace / "unstructured" / "cases.json"
        first_hash = utils.file_hash(path)
        self.assertEqual([row["itemid"] for row in json.loads(path.read_text())], ["001-2", "001-1", "001-3"])
        self.assertEqual(self.ingest(), 0)
        self.assertEqual(first_hash, utils.file_hash(path))
        self.assertEqual(json.loads((path.parent / "summary.json").read_text())["unchanged_cases"], 1)

    def test_changed_document_rejected_without_overwrite(self):
        self.batch([{"itemid": "001-1"}])
        self.ingest()
        aggregate = self.workspace / "unstructured" / "cases.json"
        before = aggregate.read_bytes()
        (self.docs / "001-1.docx").write_bytes(docx_bytes("Revised synthetic judgment"))
        with self.assertRaisesRegex(ValueError, "Source conflict"):
            self.ingest()
        self.assertEqual(before, aggregate.read_bytes())
        self.assertEqual(self.ingest(overwrite=True), 0)
        self.assertNotEqual(before, aggregate.read_bytes())

    def test_partial_missing_batch_does_not_commit(self):
        self.batch([{"itemid": "001-1"}, {"itemid": "001-2"}])
        (self.docs / "001-2.docx").unlink()
        self.assertEqual(self.ingest(), 1)
        self.assertFalse((self.workspace / "unstructured" / "cases.json").exists())
        self.assertFalse((self.workspace / "unstructured" / "cases_by_itemid" / "001-1.json").exists())

    def test_duplicate_index_and_unsafe_itemid_rejected(self):
        self.batch([{"itemid": "001-1"}, {"itemid": "001-1"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.ingest()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            utils.index_records([{"itemid": "../escape"}])

    def test_unknown_aggregate_per_item_conflict_is_not_overwritten(self):
        self.batch([{"itemid": "001-1"}])
        self.ingest()
        case_path = self.workspace / "unstructured" / "cases_by_itemid" / "001-1.json"
        altered = json.loads(case_path.read_text())
        altered["appno"] = "999/20"
        utils.write_json(case_path, altered)
        with self.assertRaisesRegex(ValueError, "Aggregate/per-item conflict"):
            self.ingest(overwrite=True)

    def test_derived_appendix_cache_not_a_source_conflict(self):
        self.batch([{"itemid": "001-1"}])
        self.ingest()
        case_path = self.workspace / "unstructured" / "cases_by_itemid" / "001-1.json"
        altered = json.loads(case_path.read_text())
        altered["appendix_table_text"] = "Derived cache"
        utils.write_json(case_path, altered)
        self.assertEqual(self.ingest(), 0)

    def test_atomic_write_failure_preserves_original(self):
        path = self.root / "keep.json"
        path.write_bytes(b"original")
        with patch.object(utils.os, "replace", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                utils.atomic_write(path, b"new")
        self.assertEqual(path.read_bytes(), b"original")
        self.assertEqual(list(self.root.glob(".keep.json.*.tmp")), [])

    def test_interrupted_update_recovers_only_with_same_hash_verified_batch(self):
        self.batch([{"itemid": "001-1"}])
        self.ingest()
        self.batch([{"itemid": "001-1"}, {"itemid": "001-3"}, {"itemid": "001-2"}])
        (self.docs / "001-1.docx").write_bytes(docx_bytes("Revised source"))
        real_write = preparation.write_json
        def interrupted_write(path, payload):
            if path.name == "cases.json":
                raise OSError("synthetic interruption before aggregate replacement")
            real_write(path, payload)
        with patch.object(preparation, "write_json", side_effect=interrupted_write):
            with self.assertRaises(OSError):
                self.ingest(overwrite=True)
        with self.assertRaisesRegex(RuntimeError, "interrupted ingestion"):
            self.ingest(overwrite=True)
        args = self.args()
        args.resume_interrupted = True
        with patch.object(preparation, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(preparation.main(), 0)
        self.assertFalse((self.workspace / "unstructured" / "ingestion_pending.json").exists())
        self.assertEqual([row["itemid"] for row in json.loads((self.workspace / "unstructured" / "cases.json").read_text())],
                         ["001-1", "001-3", "001-2"])

    def test_scaffold_runs_on_fresh_workspace_without_cases_core(self):
        self.batch([{"itemid": "001-1"}])
        self.ingest()
        old_globals = {name: getattr(scaffold, name) for name in ("DATASET_ROOT", "EXTRACTION_ROOT", "UNSTRUCTURED", "CASES_CORE",
            "OUTPUTS", "REPORTS", "CASE_FEATURES_LABELS_JSONL", "CORE_CASE_JSONL", "FACTS_PROCEDURE_INPUTS_JSONL",
            "CLAIM_AWARD_INPUTS_JSONL", "REASONING_INPUTS_JSONL", "SUMMARY_JSON", "SUMMARY_MD", "SANITY_JSON", "SAMPLES_JSON")}
        args = argparse.Namespace(itemids=None, max_cases=None, sync_case_store=False, workspace_root=self.workspace)
        try:
            with patch.object(scaffold, "parse_args", return_value=args), patch.object(scaffold, "build_token_counter", return_value=(len, "synthetic_test_counter")):
                scaffold.main()
            output = json.loads((self.workspace / "extraction" / "outputs" / "cases" / "001-1.json").read_text())
            self.assertIsNone(output["facts_procedure"]["num_applicants"])
            self.assertIsNone(output["core_case"]["num_violations_found"])
        finally:
            for key, value in old_globals.items():
                setattr(scaffold, key, value)


class DownloadTests(unittest.TestCase):
    def test_error_page_not_saved_as_docx(self):
        with self.assertRaises(Exception):
            downloader.validate_document(b"<html>Service unavailable</html>", "docx")

    def test_transient_error_retries_and_hashable_valid_docx_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "001-1.docx"
            with patch.object(downloader.urllib.request, "urlopen", side_effect=[urllib.error.URLError("temporary"), io.BytesIO(docx_bytes())]) as request, patch.object(downloader.time, "sleep"):
                self.assertEqual(downloader.download("https://example.invalid", path, retries=1)[0], "ok")
                self.assertEqual(request.call_count, 2)
            downloader.validate_document(path.read_bytes(), "docx")

    def test_non_retryable_404_does_not_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "001-1.docx"
            path.write_bytes(b"keep")
            error = urllib.error.HTTPError("https://example.invalid", 404, "not found", {}, None)
            with patch.object(downloader.urllib.request, "urlopen", side_effect=error) as request:
                with self.assertRaises(urllib.error.HTTPError):
                    downloader.download("https://example.invalid", path)
                self.assertEqual(request.call_count, 1)
            error.close()
            self.assertEqual(path.read_bytes(), b"keep")

    def test_dry_run_has_no_output_writes_and_failure_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index, out = root / "index.csv", root / "out"
            utils.write_csv(index, [{"itemid": "001-1"}], ["itemid"])
            args = argparse.Namespace(case_index=index, out_dir=out, format="docx", limit=None, sleep=0,
                                      overwrite=False, dry_run=True, retries=0)
            with patch.object(downloader, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(downloader.main(), 0)
            self.assertFalse(out.exists())
            args.dry_run = False
            with patch.object(downloader, "parse_args", return_value=args), patch.object(downloader, "download", side_effect=ValueError("invalid document")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(downloader.main(), 1)


class PlannerTests(unittest.TestCase):
    def test_half_open_window_and_repeatable_manifest(self):
        rows = [{"itemid": "001-1", "judgementdate": "2026-01-01"}, {"itemid": "001-2", "judgementdate": "2026-04-01"}]
        first = planner.build_plan(rows, [], date(2026, 1, 1), date(2026, 4, 1))
        self.assertEqual([row["itemid"] for row in first["selected_rows"]], ["001-1"])
        self.assertEqual(first, planner.build_plan(rows, [], date(2026, 1, 1), date(2026, 4, 1)))
        self.assertEqual(first["manifest"]["source_search_completeness"], "unknown")

    def test_unknown_date_rejected_not_silently_omitted(self):
        with self.assertRaisesRegex(ValueError, "source date"):
            planner.build_plan([{"itemid": "001-1"}], [], date(2026, 1, 1), date(2026, 4, 1))

    def test_known_revision_outside_window_only_with_explicit_option(self):
        rows = [{"itemid": "001-1", "judgementdate": "2020-01-01"}]
        existing = [{"itemid": "001-1"}]
        default = planner.build_plan(rows, existing, date(2026, 1, 1), date(2026, 4, 1))
        self.assertFalse(default["selected_rows"])
        revision = planner.build_plan(rows, existing, date(2026, 1, 1), date(2026, 4, 1), include_known_revisions=True)
        self.assertEqual(revision["manifest"]["records"][0]["action"], "review_metadata_revision")


if __name__ == "__main__":
    unittest.main()

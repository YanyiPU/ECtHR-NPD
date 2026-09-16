"""Synthetic HUDOC search responses only; tests never contact the network."""
from __future__ import annotations

import io
import json
import ssl
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_reconstruction"))
import discover_hudoc_cases as discovery


def record(itemid, **extra):
    return {"itemid": itemid, "languageisocode": "ENG", "doctype": "HEJUD", "kpdate": "2020-01-07T00:00:00",
            "judgementdate": "07/01/2020 00:00:00", "documentcollectionid2": "CASELAW;JUDGMENTS;CHAMBER;ENG", **extra}


def page(rows, total=None):
    return {"resultcount": len(rows) if total is None else total,
            "results": [{"columns": row} for row in rows], "message": None}


def offset(url):
    return int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["start"][0])


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.config = discovery.make_config(date(2020, 1, 1), date(2020, 2, 1), ["CHAMBER"], "ENG", "judgments", 2)

    def run_job(self, fetch, **kwargs):
        return discovery.run_discovery(self.config, self.out, fetch, request_delay=0, **kwargs)

    def test_explicit_scope_and_exclusive_date_query(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(discovery.request_url(self.config, 2)).query)
        self.assertIn('kpdate<"2020-02-01"', query["query"][0])
        self.assertIn('documentcollectionid2="CHAMBER"', query["query"][0])
        self.assertIn("doctype=HEJUD", query["query"][0])
        self.assertEqual(query["start"], ["2"])
        self.assertEqual(query["sort"], ["itemid Ascending"])
        with self.assertRaises(ValueError):
            discovery.make_config(date(2020, 1, 1), date(2020, 2, 1), [], "ENG", "judgments")

    def test_two_pass_pagination_and_deterministic_index(self):
        calls = []
        def fetch(url):
            start = offset(url)
            calls.append(start)
            return page([record("001-1", rank=str(len(calls))), record("001-2")] if start == 0 else [record("001-3")], 3)
        report = self.run_job(fetch)
        self.assertTrue(report["output_ready"])
        self.assertEqual(report["expected_total"], 3)
        self.assertEqual(calls, [0, 2, 0, 2])
        before = (self.out / "hudoc_case_index.csv").read_bytes()
        second = self.run_job(fetch, resume=True)
        self.assertTrue(second["output_ready"])
        self.assertEqual(before, (self.out / "hudoc_case_index.csv").read_bytes())

    def test_midstream_network_failure_resumes_with_prefix_revalidation(self):
        count = 0
        def failing(url):
            nonlocal count
            count += 1
            if count == 2:
                raise urllib.error.URLError("synthetic unavailable")
            return page([record("001-1"), record("001-2")], 3)
        failed = self.run_job(failing)
        self.assertFalse(failed["output_ready"])
        self.assertFalse(failed["restart_required"])
        self.assertEqual(failed["next_start"], 2)
        calls = []
        def repaired(url):
            calls.append(offset(url))
            return page([record("001-1"), record("001-2")] if offset(url) == 0 else [record("001-3")], 3)
        report = self.run_job(repaired, resume=True)
        self.assertTrue(report["output_ready"])
        self.assertEqual(calls, [0, 2, 0, 2])

    def test_changing_total_does_not_publish_partial_index(self):
        report = self.run_job(lambda url: page([record("001-1"), record("001-2")], 3) if offset(url) == 0 else page([record("001-3")], 4))
        self.assertFalse(report["output_ready"])
        self.assertTrue(report["restart_required"])
        self.assertEqual(len((self.out / "hudoc_case_index.csv").read_text().splitlines()), 1)

    def test_duplicate_is_deduplicated_but_not_certified_complete(self):
        report = self.run_job(lambda url: page([record("001-1"), record("001-2")], 3) if offset(url) == 0 else page([record("001-2")], 3))
        self.assertFalse(report["output_ready"])
        self.assertEqual(report["duplicate_count"], 1)
        self.assertEqual(report["unique_records"], 2)

    def test_metadata_drift_with_unchanged_count_fails_verification(self):
        calls = 0
        def fetch(url):
            nonlocal calls
            calls += 1
            return page([record("001-1", importance="1" if calls == 1 else "2")], 1)
        report = self.run_job(fetch)
        self.assertFalse(report["output_ready"])
        self.assertTrue(report["restart_required"])

    def test_language_or_date_scope_mismatch_fails(self):
        for extra in ({"languageisocode": "FRE"}, {"kpdate": "2020-02-01T00:00:00"}, {"judgementdate": "08/01/2020 00:00:00"}):
            with self.subTest(extra=extra):
                report = self.run_job(lambda url: page([record("001-1", **extra)]), restart=True)
                self.assertFalse(report["output_ready"])

    def test_empty_scope_is_two_pass_verified_not_global_coverage(self):
        calls = []
        def fetch(url):
            calls.append(url)
            return page([])
        report = self.run_job(fetch)
        self.assertTrue(report["output_ready"])
        self.assertFalse(report["all_hudoc_coverage"])
        self.assertFalse(report["transactional_snapshot"])
        self.assertEqual(len(calls), 2)

    def test_premature_empty_page_fails(self):
        report = self.run_job(lambda url: page([], 10))
        self.assertFalse(report["output_ready"])
        self.assertIn("before reported total", report["error"])

    def test_changed_resume_scope_rejected_without_writes(self):
        self.run_job(lambda url: page([record("001-1")]))
        before = (self.out / "discovery_manifest.json").read_bytes()
        config = {**self.config, "date_until_exclusive": "2020-03-01"}
        with self.assertRaisesRegex(ValueError, "scope/config"):
            discovery.run_discovery(config, self.out, lambda url: page([]), resume=True)
        self.assertEqual(before, (self.out / "discovery_manifest.json").read_bytes())

    def test_page_budget_is_incomplete_and_resumable(self):
        fetch = lambda url: page([record("001-1"), record("001-2")] if offset(url) == 0 else [record("001-3")], 3)
        report = self.run_job(fetch, max_pages=1)
        self.assertFalse(report["output_ready"])
        self.assertFalse(report["restart_required"])
        self.assertTrue(self.run_job(fetch, resume=True, max_pages=10)["output_ready"])

    def test_tampered_checkpoint_rejected(self):
        self.run_job(lambda url: page([record("001-1")]))
        path = self.out / "discovery_checkpoint.json"
        state = json.loads(path.read_text())
        state["records"][0]["itemid"] = "001-99"
        path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.run_job(lambda url: page([]), resume=True)

    def test_unrelated_existing_index_not_overwritten_even_with_restart(self):
        index = self.out / "hudoc_case_index.csv"
        index.write_text("unrelated user data\n")
        for mode in ({}, {"restart": True}, {"resume": True}):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "nonempty"):
                self.run_job(lambda url: page([]), **mode)
            self.assertEqual(index.read_text(), "unrelated user data\n")
            self.assertFalse((self.out / "discovery_manifest.json").exists())


class HttpTests(unittest.TestCase):
    def response(self):
        response = io.BytesIO(json.dumps(page([])).encode())
        response.geturl = lambda: discovery.ENDPOINT
        return response

    def test_transient_retry_then_success(self):
        with patch.object(discovery.urllib.request, "urlopen", side_effect=[urllib.error.URLError("temporary"), self.response()]) as request, patch.object(discovery.time, "sleep"):
            self.assertEqual(discovery.fetch_json(discovery.ENDPOINT + "?x=1", retries=1)["resultcount"], 0)
            self.assertEqual(request.call_count, 2)

    def test_tls_failure_not_bypassed_or_retried(self):
        error = urllib.error.URLError(ssl.SSLCertVerificationError("missing CA"))
        with patch.object(discovery.urllib.request, "urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, "never disable"):
                discovery.fetch_json(discovery.ENDPOINT + "?x=1")
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()

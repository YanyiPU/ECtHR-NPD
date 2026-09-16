"""Synthetic offline tests for the single-cohort extraction entry points."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from xml.sax.saxutils import escape
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_reconstruction"))
sys.path.insert(0, str(ROOT / "extraction_pipeline" / "code"))
sys.path.insert(0, str(ROOT / "extraction_pipeline"))
import source_utils
import case_store
import run_pipeline_c_backbone as compensation
import build_extraction_layers as scaffold
import holistic_extractor as holistic
import export_reviewed


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_docx(path):
    paragraphs = ["INTRODUCTION", "1. The case concerns detention of the applicant.", "THE FACTS",
        "2. The applicant was detained for two days.", "THE LAW", "APPLICATION OF ARTICLE 41",
        "3. The applicant claimed EUR 2,000 in respect of non-pecuniary damage.",
        "FOR THESE REASONS, THE COURT, UNANIMOUSLY",
        "4. Holds that the respondent State is to pay the applicant EUR 1,000 in respect of non-pecuniary damage."]
    body = "".join(f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>" for text in paragraphs)
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                         f"<w:body>{body}</w:body></w:document>")


class SourceContractTests(unittest.TestCase):
    def test_public_hudoc_case_id_is_source_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            write_csv(path, [{"case_id": "001-12345", "y_amount_eur": "1000"}])
            self.assertEqual(source_utils.read_index(path)[0]["itemid"], "001-12345")
            write_csv(path, [{"case_id": "case_00001"}])
            with self.assertRaisesRegex(ValueError, "HUDOC"):
                source_utils.read_index(path)
            write_csv(path, [{"itemid": "001-12345", "case_id": "001-12346"}])
            with self.assertRaisesRegex(ValueError, "mismatch"):
                source_utils.read_index(path)

    def test_case_store_rejects_path_escape_mismatched_identity_and_pending_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / "cases_by_itemid"
            store.mkdir()
            with self.assertRaisesRegex(ValueError, "unsafe"):
                case_store.case_store_path("../outside", store)
            (store / "001-12345.json").write_text('{"itemid":"001-12346"}')
            with self.assertRaisesRegex(ValueError, "mismatch"):
                case_store.load_case_from_store("001-12345", store)
            (store.parent / "ingestion_pending.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "journal"):
                case_store.load_cases_by_itemid(["001-12345"], store, None)

    def test_missing_validator_never_counts_as_positive_agreement(self):
        for currency in ("EUR", "GBP"):
            result = compensation.layer3_crossval({}, {"awards": {"non_pecuniary": {
                "eur_amount": 100, "original_currency": currency}}})
            self.assertIsNone(result["non_pec_match"])
            self.assertTrue(result["flag_for_review"])
        self.assertTrue(compensation.layer3_crossval({}, {})["flag_for_review"])

    def test_token_counter_is_offline_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            counter, method = scaffold.build_token_counter()
            self.assertEqual(method, "char_div_4_fallback")
            self.assertEqual(counter("abcde"), 2)

    def test_model_prompts_do_not_include_packaging_boilerplate(self):
        for path in (ROOT / "extraction_pipeline/prompts").glob("*.md"):
            self.assertNotIn("Research-code documentation:", path.read_text(), path.name)

    def test_external_workspace_keeps_packaged_prompt_and_schema_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            original = {name: getattr(holistic, name) for name in ("DATASET_ROOT", "STRUCTURED_ROOT", "CASES_CORE_JSON", "OUTPUTS",
                        "CASES_OUTPUT", "RUNS_ROOT", "CASE_STORE_DIR", "UNSTRUCTURED_CASES", "_split_mapping")}
            try:
                holistic.configure_workspace(Path(directory))
                self.assertEqual(holistic.CASE_STORE_DIR, Path(directory).resolve() / "unstructured" / "cases_by_itemid")
                self.assertEqual(holistic.SCHEMA_C, ROOT / "extraction_pipeline/schemas/pipeline_c_backbone.schema.json")
            finally:
                for name, value in original.items():
                    setattr(holistic, name, value)

    def test_local_docx_to_candidate_workflow_without_copied_code_or_api(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            documents = folder / "documents"
            documents.mkdir()
            create_docx(documents / "001-12345.docx")
            index = folder / "public_cases.csv"
            write_csv(index, [{"case_id": "001-12345", "y_amount_eur": "999999_SECRET_TARGET"}])
            workspace = folder / "workspace"
            command = [sys.executable, "-B", str(ROOT / "extraction_pipeline/run_extraction.py"),
                       "--case-index", str(index), "--hudoc-docx-dir", str(documents), "--workspace-root", str(workspace),
                       "--run-name", "synthetic", "--mode", "regex"]
            environment = {k: v for k, v in os.environ.items() if not k.startswith("EXTRACTION_")}
            result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((workspace / "workflow_runs/synthetic.json").read_text())
            self.assertEqual(report["status"], "candidate_extraction_completed")
            self.assertFalse(report["source_index_contains_targets"])
            self.assertNotIn("SECRET_TARGET", (workspace / "unstructured/cases.json").read_text())
            rows = [json.loads(line) for line in (workspace / "extraction/runs/pipeline_c_backbone/synthetic/results_unique.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["cross_validation"]["flag_for_review"])
            self.assertEqual(rows[0]["cross_validation"]["validation_status"], "not_cross_validated_regex_only")
            self.assertFalse((workspace / "extraction/code").exists())


class ReviewedProjectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        schema = json.loads((ROOT / "schema.json").read_text())
        self.case = {key: "unknown" for key in schema["case_level"]}
        self.case.update(case_id="001-12345", split="train", test_view="not_applicable", test_challenging_view="no",
            judgment_date="2020-01-01", judgment_year="2020", respondent_state="Synthetic", country_alpha2="ZZ",
            hudoc_decision_body="CHAMBER", court_formation="Chamber", is_grand_chamber="no", case_importance="low",
            has_separate_opinion="no", represented="yes", hudoc_application_count="1", num_applicants="1",
            num_violations_found="1", has_mixed_outcome="no", violated_articles="3", violated_articles_count="1",
            violation_type="substantive", beneficiary_type="applicant", has_joint_beneficiary="not_found",
            y_amount_eur="1000", y_binary="1", target_status="retained_label")
        self.person = {key: "unknown" for key in schema["applicant_level"]}
        self.person.update(applicant_id="applicant_000001", case_id="001-12345", split="train", applicant_name="Jane Example",
            applicant_type="applicant_source_record", sex="female", age_group="adult", birth_year="1980",
            nationality="French (Jane Example)", nationality_scope="single", beneficiary_type="applicant",
            npd_award_eur="1000", npd_award_status="verified_source_link")
        if "npd_award_scope" in self.person:
            self.person["npd_award_scope"] = "individual"
        self.source = self.root / "001-12345.txt"
        self.source.write_text("Synthetic judgment: EUR 1000 non-pecuniary award to Jane Example.")
        self.review = {"status": "accepted", "adjudicator": "synthetic-reviewer", "source_anchor": "operative paragraph 4",
            "canonical_fields_reviewed": True, "source_document": self.source.name,
            "source_document_sha256": export_reviewed.file_hash(self.source),
            "checks": {key: {"status": "pass", "evidence": "synthetic source anchor"} for key in export_reviewed.CHECKS}}
        self.input = self.root / "reviewed.jsonl"
        self.cohort = self.root / "cohort.csv"
        write_csv(self.cohort, [self.case])

    def save(self):
        self.review["record_sha256"] = export_reviewed.record_hash(self.case, [self.person])
        self.input.write_text(json.dumps({"case": self.case, "applicants": [self.person], "review": self.review}) + "\n")

    def run_export(self):
        return export_reviewed.export(self.input, self.cohort, self.root, self.root / "out")

    def test_export_masks_names_preserves_nationality_and_writes_only_two_csvs(self):
        self.save()
        report = self.run_export()
        self.assertEqual(report["case_rows"], 1)
        self.assertFalse(report["dataset_modified"])
        self.assertEqual({str(p.relative_to(self.root / "out")) for p in (self.root / "out").rglob("*") if p.is_file()},
                         {"data/case_level.csv", "data/applicant_level.csv"})
        text = (self.root / "out/data/applicant_level.csv").read_text()
        self.assertNotIn("Jane Example", text)
        self.assertIn("French ([MASKED])", text)

    def test_unresolved_check_is_not_promoted(self):
        self.review["checks"]["head_separation"]["status"] = "unknown"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            self.run_export()
        self.assertFalse((self.root / "out").exists())

    def test_changed_canonical_record_invalidates_review(self):
        self.save()
        payload = json.loads(self.input.read_text())
        payload["case"]["y_amount_eur"] = "2000"
        self.input.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "bind"):
            self.run_export()

    def test_source_hash_and_cohort_membership_are_required(self):
        self.save()
        self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_export()
        self.case["case_id"] = "001-12346"
        self.person["case_id"] = "001-12346"
        self.save()
        with self.assertRaisesRegex(ValueError, "cohort"):
            self.run_export()

    def test_missing_canonical_field_cannot_be_fabricated(self):
        del self.case["num_applicants"]
        self.save()
        with self.assertRaisesRegex(ValueError, "none are inferred"):
            self.run_export()

    def test_zero_target_requires_source_rationale_even_after_eight_checks(self):
        self.case.update(y_amount_eur="0", y_binary="0", target_status="corrected_amount")
        self.person.update(npd_award_eur="0")
        self.review["correction_rationale"] = "Synthetic source correction."
        self.save()
        with self.assertRaisesRegex(ValueError, "zero target"):
            self.run_export()


if __name__ == "__main__":
    unittest.main()

"""Synthetic, offline exporter contract tests; real extraction schemas."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "extraction_pipeline" / "export_tabular.py"
SPEC = importlib.util.spec_from_file_location("export_tabular", PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def minimal(schema):
    """Build a synthetic required-field fixture, not an extraction result."""
    if "enum" in schema:
        return None if None in schema["enum"] else schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = "null" if "null" in kind else kind[0]
    if kind == "object":
        return {key: minimal(schema["properties"][key]) for key in schema.get("required", [])}
    if kind == "array":
        return [minimal(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        return "fixture"
    if kind in ("integer", "number"):
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    return None


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        schemas = {s: json.loads((exporter.HERE / "schemas" / p).read_text()) for s, p in exporter.SCHEMAS.items()}
        b, c, d, e = [minimal(schemas[s]) for s in "bcde"]
        for payload in (b, c, d, e):
            payload["itemid"] = "source-1"
        self.person = b["facts_procedure"]["applicants"][0]
        self.person.update(applicant_index=1, beneficiary_label="DO_NOT_EXPORT_PERSON_NAME", sex="female", age_group="adult")
        b["facts_procedure"].update(num_applicants=1, is_joint_application=False)
        c["awards"]["non_pecuniary"].update(original_currency="EUR", original_amount=100,
                                             eur_amount=100, granted=True)
        c["awards"].update(bundled_award=False, bundled_award_eur=None)
        self.allocation = {"beneficiary_label": "DO_NOT_EXPORT_PERSON_NAME", "applicant_index": 1,
                           "head": "non_pecuniary", "eur_amount": 100}
        c["award_per_applicant"] = [self.allocation]
        self.merged = {"itemid": "source-1", "judgementdate": "2026-01-01", "core_case": {
            "respondent_country": "Synthetic", "num_application_numbers": 2, "detailed_violations": ["6-1"]},
            "extraction_meta": {"stage_status": {s: "success" for s in "bcde"},
                                "acceptance_status": "candidate_requires_dataset_review"},
            "facts_procedure": b["facts_procedure"], "article_41_extraction": c,
            "legal_analysis": d["legal_analysis"], "reasoning_layer": e["reasoning_layer"],
            "award_per_applicant_fallback": [], "final_awards": {"non_pecuniary_eur": 100, "bundled_award_eur": None}}
        self.payload = {"itemid": "source-1", "status": "success", "result": self.merged,
                        "stage_meta": {s: {"status": "success"} for s in "bcde"}}
        self.input = self.root / "source-1.meta.json"
        self.cross = self.root / "case_crosswalk.csv"
        write_csv(self.cross, [{"itemid": "source-1", "case_id": "stable-case-1"}])
        self.source = self.root / "judgment.txt"
        self.source.write_text("Synthetic operative provision: EUR 100 non-pecuniary award.")
        self.source_hash = exporter.file_hash(self.source)
        self.review = {"itemid": "source-1", "review_status": "accepted", "decision": "include_pure_npd",
            "record_sha256": "", "source_document_path": "judgment.txt", "source_document_sha256": self.source_hash,
            "source_anchor": "operative provision", "adjudicator": "synthetic-reviewer", "heads_status": "pure_npd",
            "currency": "EUR", "source_currency": "EUR", "award_scope": "case_total", "case_total_npd_eur": "100",
            "zero_rationale": "", "allocation_coverage": "partial", "correction_rationale": "",
            "mixed_candidate_resolution": "", "currency_resolution": ""}
        self.case_ledger = self.root / "case_review.csv"
        self.person_ledger = self.root / "person_crosswalk.csv"
        self.alloc_ledger = self.root / "allocation_review.csv"
        self.schemas = schemas

    def save(self, review=True):
        self.input.write_text(json.dumps(self.payload))
        if review:
            self.review["record_sha256"] = exporter.canonical_hash(self.merged)
            write_csv(self.case_ledger, [self.review])

    def person_review(self):
        return {"itemid": "source-1", "applicant_key": exporter.canonical_hash(self.person),
                "applicant_id": "stable-person-1", "review_status": "accepted_identity_and_attributes",
                "source_document_sha256": self.source_hash, "source_anchor": "facts paragraph 1",
                "adjudicator": "synthetic-reviewer"}

    def allocation_review(self, scope="personal", value="100"):
        return {"itemid": "source-1", "allocation_key": exporter.canonical_hash(self.allocation),
                "allocation_id": "stable-allocation-1", "review_status": "accepted", "head": "non_pecuniary",
                "currency": "EUR", "amount_eur": value, "allocation_scope": scope,
                "applicant_id": "stable-person-1" if scope == "personal" else "",
                "source_document_sha256": self.source_hash, "source_anchor": "operative provision",
                "adjudicator": "synthetic-reviewer", "correction_rationale": ""}

    def run_export(self, name="export", reviewed=True, persons=False, allocations=False, split=None):
        self.save(reviewed)
        out = self.root / name
        manifest = exporter.export_records([self.input], self.cross, out,
            self.case_ledger if reviewed else None, self.person_ledger if persons else None,
            self.alloc_ledger if allocations else None, split)
        return out, manifest

    def test_fixture_really_passes_all_current_extraction_validators(self):
        self.assertEqual(exporter.validate_extraction(self.payload, self.merged, self.schemas), [])

    def test_candidates_without_adjudication_never_become_clean(self):
        out, report = self.run_export(reviewed=False)
        self.assertEqual(report["tables"]["candidate/case_level.csv"]["rows"], 1)
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)
        self.assertEqual(read_csv(out / "candidate/applicant_level.csv")[0]["applicant_id"], "")
        self.assertEqual(read_csv(out / "candidate/allocation_level.csv")[0]["allocation_scope"], "unknown")
        for path in out.rglob("*.csv"):
            self.assertNotIn("DO_NOT_EXPORT_PERSON_NAME", path.read_text())
        self.assertFalse(report["predictor_features_exported"])
        self.assertEqual(report["classification"], "internal_restricted_not_a_public_release")

    def test_case_adjudication_alone_does_not_verify_people_or_allocations(self):
        out, report = self.run_export()
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 1)
        self.assertEqual(report["tables"]["clean/applicant_level.csv"]["rows"], 0)
        self.assertEqual(report["tables"]["clean/allocation_level.csv"]["rows"], 0)
        case = read_csv(out / "clean/case_level.csv")[0]
        self.assertEqual(case["split"], "unassigned_pending_policy")
        self.assertEqual(case["y_amount_eur"], "100.00")
        self.assertEqual(case["y_binary"], "1")
        adapter = json.loads((out / "schema_adapter.json").read_text())
        self.assertEqual(adapter["reference_column_count"], 43)
        self.assertEqual(len(adapter["missing_or_unvalidated_columns"]), 39)
        self.assertFalse(adapter["full_case_feature_package_ready"])
        self.assertFalse(adapter["automatic_append_authorized"])

    def test_verified_personal_allocation_links_only_by_explicit_review(self):
        write_csv(self.person_ledger, [self.person_review()])
        write_csv(self.alloc_ledger, [self.allocation_review()])
        self.review["allocation_coverage"] = "complete"
        out, _ = self.run_export(persons=True, allocations=True)
        allocation = read_csv(out / "clean/allocation_level.csv")[0]
        self.assertEqual(allocation["applicant_id"], "stable-person-1")
        self.assertEqual(allocation["amount_eur"], "100.00")
        self.assertEqual(read_csv(out / "clean/case_level.csv")[0]["allocation_export_status"], "complete")

    def test_joint_award_is_one_unlinked_amount_without_equal_split(self):
        self.allocation.update(beneficiary_label="FIRST AND SECOND PERSON", applicant_index=1)
        write_csv(self.alloc_ledger, [self.allocation_review(scope="joint")])
        self.review["allocation_coverage"] = "complete"
        out, _ = self.run_export(allocations=True)
        allocations = read_csv(out / "clean/allocation_level.csv")
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0]["applicant_id"], "")
        self.assertEqual(allocations[0]["amount_eur"], "100.00")
        self.assertEqual(read_csv(out / "clean/applicant_level.csv"), [])

    def test_joint_award_cannot_be_assigned_to_first_applicant(self):
        row = self.allocation_review(scope="joint")
        row["applicant_id"] = "stable-person-1"
        write_csv(self.person_ledger, [self.person_review()])
        write_csv(self.alloc_ledger, [row])
        _, report = self.run_export(persons=True, allocations=True)
        self.assertEqual(report["tables"]["clean/allocation_level.csv"]["rows"], 0)

    def test_unknown_or_mixed_case_is_rejected(self):
        for status in ("unknown", "mixed"):
            self.review["heads_status"] = status
            _, report = self.run_export(name=status)
            self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)

    def test_mixed_model_candidate_requires_source_supported_override(self):
        self.merged["article_41_extraction"]["awards"].update(bundled_award=True, bundled_award_eur=100)
        _, report = self.run_export()
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)
        self.review.update(correction_rationale="Model misclassified an explicitly separate NPD award.",
                           mixed_candidate_resolution="verified_pure_npd_amount")
        _, report = self.run_export(name="resolved")
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 1)

    def test_partial_or_invalid_schema_cannot_promote_even_with_review(self):
        self.payload["status"] = "partial_success"
        _, report = self.run_export(name="partial")
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)
        self.payload["status"] = "success"
        self.merged["facts_procedure"]["unexpected"] = "bad schema"
        _, report = self.run_export(name="invalid")
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)

    def test_missing_amount_is_not_zero_and_zero_needs_source_reason(self):
        self.merged["final_awards"]["non_pecuniary_eur"] = None
        self.review["case_total_npd_eur"] = ""
        _, report = self.run_export(name="missing")
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)
        self.review.update(case_total_npd_eur="0", correction_rationale="Explicit finding of sufficiency in source.")
        _, report = self.run_export(name="zero-no-reason")
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)
        self.review["zero_rationale"] = "finding_sufficient"
        _, report = self.run_export(name="zero-with-reason")
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 1)

    def test_stale_source_or_extraction_hash_blocks_promotion(self):
        self.save()
        self.source.write_text("Changed source")
        out = self.root / "stale"
        report = exporter.export_records([self.input], self.cross, out, self.case_ledger)
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)
        self.source.write_text("Synthetic operative provision: EUR 100 non-pecuniary award.")
        self.merged["final_awards"]["non_pecuniary_eur"] = 200
        self.input.write_text(json.dumps(self.payload))
        report = exporter.export_records([self.input], self.cross, self.root / "stale-extraction", self.case_ledger)
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)

    def test_no_id_crosswalk_means_no_export_and_no_new_random_ids(self):
        write_csv(self.cross, [{"itemid": "different-source", "case_id": "stable-case-1"}])
        self.save()
        with self.assertRaises(ValueError):
            exporter.export_records([self.input], self.cross, self.root / "no-crosswalk")
        self.assertFalse((self.root / "no-crosswalk").exists())

    def test_duplicate_sources_or_case_ids_are_rejected(self):
        self.save()
        with self.assertRaises(ValueError):
            exporter.export_records([self.input, self.input], self.cross, self.root / "duplicates")
        write_csv(self.cross, [{"itemid": "source-1", "case_id": "same"}, {"itemid": "source-2", "case_id": "same"}])
        with self.assertRaises(ValueError):
            exporter.export_records([self.input], self.cross, self.root / "ambiguous-crosswalk")

    def test_duplicate_fallback_allocations_are_not_double_counted(self):
        self.merged["award_per_applicant_fallback"] = [dict(self.allocation)]
        write_csv(self.person_ledger, [self.person_review()])
        write_csv(self.alloc_ledger, [self.allocation_review()])
        out, report = self.run_export(persons=True, allocations=True)
        self.assertEqual(report["tables"]["clean/allocation_level.csv"]["rows"], 0)
        self.assertEqual(read_csv(out / "candidate/allocation_level.csv")[0]["occurrences"], "2")

    def test_person_hash_mismatch_never_links_by_matching_row_index(self):
        row = self.person_review()
        row["applicant_key"] = "0" * 64
        write_csv(self.person_ledger, [row])
        write_csv(self.alloc_ledger, [self.allocation_review()])
        _, report = self.run_export(persons=True, allocations=True)
        self.assertEqual(report["tables"]["clean/applicant_level.csv"]["rows"], 0)
        self.assertEqual(report["tables"]["clean/allocation_level.csv"]["rows"], 0)

    def test_overallocation_blocks_allocations_but_preserves_adjudicated_case_target(self):
        row = self.allocation_review(scope="group", value="120")
        row["correction_rationale"] = "Synthetic conflicting ledger."
        write_csv(self.alloc_ledger, [row])
        out, report = self.run_export(allocations=True)
        self.assertEqual(report["tables"]["clean/allocation_level.csv"]["rows"], 0)
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 1)
        self.assertEqual(read_csv(out / "clean/case_level.csv")[0]["allocation_export_status"], "blocked")

    def test_partial_underallocation_allowed_complete_underallocation_blocked(self):
        row = self.allocation_review(scope="group", value="80")
        row["correction_rationale"] = "Only this explicitly allocated group component is verified."
        write_csv(self.alloc_ledger, [row])
        _, report = self.run_export(name="partial", allocations=True)
        self.assertEqual(report["tables"]["clean/allocation_level.csv"]["rows"], 1)
        self.review["allocation_coverage"] = "complete"
        _, report = self.run_export(name="complete", allocations=True)
        self.assertEqual(report["tables"]["clean/allocation_level.csv"]["rows"], 0)

    def test_currency_is_not_implicitly_converted(self):
        self.merged["article_41_extraction"]["awards"]["non_pecuniary"]["original_currency"] = "USD"
        _, report = self.run_export()
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)

    def test_explicit_split_manifest_only_and_deterministic_outputs(self):
        split = self.root / "splits.csv"
        write_csv(split, [{"case_id": "stable-case-1", "split": "ood", "split_policy_version": "approved-2026Q1"}])
        out1, report1 = self.run_export(name="one", split=split)
        out2, report2 = self.run_export(name="two", split=split)
        self.assertEqual(read_csv(out1 / "clean/case_level.csv")[0]["split"], "ood")
        self.assertEqual(report1, report2)
        self.assertEqual((out1 / "manifest.json").read_bytes(), (out2 / "manifest.json").read_bytes())
        with self.assertRaises(ValueError):
            exporter.export_records([self.input], self.cross, out1)

    def test_invalid_money_is_rejected(self):
        for value in (True, "NaN", "Infinity", "-1", "1.001"):
            with self.assertRaises(ValueError):
                exporter.money(value)

    def test_zero_taxonomy_is_exactly_the_paper_six_categories(self):
        self.assertEqual(exporter.ZERO_RATIONALES, {"no_claim", "unsubstantiated", "rule_60_non_compliance",
            "domestic_award_covers", "applicant_deceased_no_heir", "finding_sufficient"})
        self.review.update(case_total_npd_eur="0", correction_rationale="Explicit zero award.", zero_rationale="waiver")
        _, report = self.run_export()
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 0)

    def test_ledger_paths_cannot_escape_allowed_source_root(self):
        self.save()
        outside = self.root / "outside.txt"
        outside.write_text("Unrelated private fixture")
        allowed = self.root / "allowed"
        allowed.mkdir()
        (allowed / "escape.txt").symlink_to(outside)
        self.review["source_document_sha256"] = exporter.file_hash(outside)
        for relative in ("../outside.txt", str(outside), "escape.txt"):
            with self.subTest(path=relative):
                self.review["source_document_path"] = relative
                reasons = exporter.case_review_errors(self.review, self.case_ledger,
                    self.review["record_sha256"], self.merged, source_root=allowed)
                self.assertIn("source_document_outside_allowed_root", reasons)

    def test_explicit_source_root_allows_only_its_documents(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        (allowed / "judgment.txt").write_bytes(self.source.read_bytes())
        self.save()
        report = exporter.export_records([self.input], self.cross, self.root / "explicit-root", self.case_ledger,
                                          source_root=allowed)
        self.assertEqual(report["tables"]["clean/case_level.csv"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()

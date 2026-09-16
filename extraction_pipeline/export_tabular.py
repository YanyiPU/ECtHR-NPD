#!/usr/bin/env python3
"""Offline, internal-only extraction-to-tabular export with explicit promotion.

Accepts selected holistic *.meta.json / full result JSON records. It never
discovers cases, invents identities, adjudicates judgments, assigns splits,
uploads data, or overwrites an existing output directory. See EXPORT_TABULAR.md.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from openai_compatible_client import schema_errors
from run_pipeline_b_extraction import validate_result as validate_b
from run_pipeline_c_backbone import validate_llm_result as validate_c
from run_pipeline_e_extraction import validate_result as validate_e
from holistic_extractor import validate_legal_analysis_result as validate_d

CASE_FIELDS = ["case_id", "itemid", "record_sha256", "extraction_status", "stage_validation_status",
               "respondent_country", "judgementdate", "num_application_numbers", "num_applicants_candidate",
               "violated_article_codes_json", "npd_eur_candidate", "bundled_eur_candidate",
               "split", "split_policy_version", "promotion_status", "issues_json"]
PERSON_FIELDS = ["case_id", "itemid", "applicant_key", "applicant_id", "source_applicant_index",
                 "birth_year", "sex", "age_group", "nationality", "occurrences", "mapping_status", "issues_json"]
ALLOCATION_FIELDS = ["case_id", "itemid", "allocation_key", "head_candidate", "eur_amount_candidate",
                     "source_applicant_index_candidate", "source_kinds_json", "occurrences",
                     "allocation_scope", "applicant_id", "promotion_status", "issues_json"]
CLEAN_CASE_FIELDS = ["case_id", "y_amount_eur", "y_binary", "currency", "target_scope", "zero_rationale",
                     "allocation_coverage", "allocation_export_status", "split", "split_policy_version", "review_status"]
CLEAN_PERSON_FIELDS = ["case_id", "applicant_id", "birth_year", "sex", "age_group", "nationality", "split", "review_status"]
CLEAN_ALLOCATION_FIELDS = ["case_id", "allocation_id", "applicant_id", "head", "amount_eur", "currency",
                           "allocation_scope", "split", "review_status"]
DECISION_FIELDS = ["case_id", "record_type", "record_key", "decision", "reasons_json"]
SCHEMAS = {"b": "pipeline_b_facts_procedure.schema.json", "c": "pipeline_c_backbone.schema.json",
           "d": "pipeline_d_legal_analysis.schema.json", "e": "pipeline_e_reasoning.schema.json"}
ZERO_RATIONALES = {"no_claim", "unsubstantiated", "rule_60_non_compliance", "domestic_award_covers",
                   "applicant_deceased_no_heir", "finding_sufficient"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def money(value):
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not money")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid money") from exc
    if not amount.is_finite() or amount < 0 or amount != amount.quantize(Decimal("0.01")):
        raise ValueError("Money must be finite, nonnegative and cent-exact")
    return amount


def amount_text(value):
    number = money(value)
    return "" if number is None else format(number, ".2f")


def read_csv(path, required):
    if path is None:
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise ValueError(f"Invalid CSV columns in {Path(path).name}; required: {sorted(required)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV rows in {Path(path).name}")
    return rows


def index_rows(rows, keys, label):
    indexed = {}
    for row in rows:
        key = tuple(row.get(field, "").strip() for field in keys)
        if not all(key) or key in indexed:
            raise ValueError(f"Missing or duplicate {label} key")
        indexed[key] = row
    return indexed


def unique_values(rows, field, label):
    values = [row[field].strip() for row in rows]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{label} IDs must be nonempty and one-to-one")


def load_records(paths):
    records = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Each extraction input must be an object")
        merged = payload.get("result") or (payload.get("layers") or {}).get("merged")
        if merged is None and "extraction_meta" in payload:
            merged = payload
        if not isinstance(merged, dict):
            raise ValueError("Input has no holistic merged result; select a result/meta record")
        itemid = str(merged.get("itemid") or "").strip()
        if not itemid or itemid in records or (payload.get("itemid") and str(payload["itemid"]) != itemid):
            raise ValueError("Missing, duplicate or inconsistent extraction itemid")
        records[itemid] = (payload, merged)
    return records


def validate_extraction(payload, merged, schemas):
    reasons = []
    if payload.get("status") != "success":
        reasons.append("run_not_successful_or_status_missing")
    meta = merged.get("extraction_meta") or {}
    if meta.get("stage_status") != {stage: "success" for stage in SCHEMAS}:
        reasons.append("not_all_stage_statuses_successful")
    if meta.get("acceptance_status") != "candidate_requires_dataset_review":
        reasons.append("missing_current_candidate_acceptance_contract")
    for stage in SCHEMAS:
        if ((payload.get("stage_meta") or {}).get(stage) or {}).get("status") != "success":
            reasons.append(f"{stage}_run_evidence_missing_or_failed")
    itemid = str(merged["itemid"])
    projections = {
        "b": {"itemid": itemid, "facts_procedure": merged.get("facts_procedure")},
        "c": merged.get("article_41_extraction"),
        "d": {"itemid": itemid, "legal_analysis": merged.get("legal_analysis")},
        "e": {"itemid": itemid, "reasoning_layer": merged.get("reasoning_layer")},
    }
    validators = {"b": validate_b, "c": validate_c, "d": validate_d, "e": validate_e}
    for stage, projection in projections.items():
        if not isinstance(projection, dict):
            reasons.append(f"{stage}_structured_result_missing")
            continue
        if str(projection.get("itemid") or "") != itemid:
            reasons.append(f"{stage}_itemid_mismatch")
        # Errors are identifiers only; schema messages can contain source text.
        if validators[stage](projection, schemas[stage]):
            reasons.append(f"{stage}_schema_or_semantic_validation_failed")
    return reasons


def source_review_errors(review, ledger_path, expected_record_hash=None, source_hash=None, source_root=None):
    errors = []
    for field in ("source_anchor", "adjudicator", "source_document_sha256"):
        if not review.get(field, "").strip():
            errors.append(f"missing_{field}")
    declared = review.get("source_document_sha256", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", declared):
        errors.append("invalid_source_document_sha256")
    if expected_record_hash is not None:
        if review.get("record_sha256") != expected_record_hash:
            errors.append("adjudication_does_not_bind_current_extraction")
        source_path = review.get("source_document_path", "").strip()
        allowed_root = Path(source_root).resolve() if source_root else Path(ledger_path).resolve().parent
        path = (allowed_root / source_path).resolve() if source_path else None
        if path is not None and not path.is_relative_to(allowed_root):
            errors.append("source_document_outside_allowed_root")
        elif path is None or not path.is_file():
            errors.append("source_document_missing")
        elif file_hash(path) != declared:
            errors.append("source_document_hash_mismatch")
    elif source_hash != declared:
        errors.append("review_source_differs_from_case_source")
    return errors


def case_review_errors(review, ledger_path, record_hash, merged, source_root=None):
    if review is None:
        return ["case_adjudication_required"]
    errors = source_review_errors(review, ledger_path, expected_record_hash=record_hash, source_root=source_root)
    required_values = {"review_status": "accepted", "decision": "include_pure_npd",
                       "heads_status": "pure_npd", "currency": "EUR", "source_currency": "EUR", "award_scope": "case_total"}
    errors += [f"{field}_not_{expected}" for field, expected in required_values.items() if review.get(field) != expected]
    try:
        target = money(review.get("case_total_npd_eur"))
        if target is None:
            errors.append("explicit_npd_total_required")
        elif target == 0 and review.get("zero_rationale") not in ZERO_RATIONALES:
            errors.append("zero_requires_explicit_source_rationale")
    except ValueError:
        target = None
        errors.append("invalid_adjudicated_npd_amount")
    if review.get("allocation_coverage") not in {"complete", "partial", "none_found", "unknown"}:
        errors.append("explicit_allocation_coverage_required")
    awards = (merged.get("article_41_extraction") or {}).get("awards") or {}
    original_currency = (awards.get("non_pecuniary") or {}).get("original_currency")
    if original_currency not in (None, "EUR") and not (
        review.get("currency_resolution") == "verified_eur_in_source" and review.get("correction_rationale", "").strip()
    ):
        errors.append("non_eur_candidate_requires_verified_eur_source_resolution")
    extracted_mixed = awards.get("bundled_award") is True or awards.get("bundled_award_eur") is not None
    candidate = (merged.get("final_awards") or {}).get("non_pecuniary_eur")
    try:
        changed = money(candidate) != target
    except ValueError:
        changed = True
    if (extracted_mixed or changed) and not review.get("correction_rationale", "").strip():
        errors.append("source_correction_rationale_required")
    if extracted_mixed and review.get("mixed_candidate_resolution") != "verified_pure_npd_amount":
        errors.append("mixed_candidate_requires_explicit_source_resolution")
    return errors


def candidates(values):
    """Collapse indistinguishable hashes for reporting, preserve multiplicity.

    Repeated rows are never eligible for clean allocation/person promotion;
    no numerical total is computed from this raw representation.
    """
    groups = {}
    for source_kind, row in values:
        if not isinstance(row, dict):
            raise ValueError("Candidate applicant/allocation entries must be objects")
        key = canonical_hash(row)
        if key not in groups:
            groups[key] = {"row": row, "sources": [], "occurrences": 0}
        groups[key]["occurrences"] += 1
        groups[key]["sources"].append(source_kind)
    return groups


def export_records(inputs, case_crosswalk, output, case_ledger=None, applicant_crosswalk=None,
                   allocation_ledger=None, split_manifest=None, source_root=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; use an isolated staging directory and controlled apply")
    cross_rows = read_csv(case_crosswalk, {"itemid", "case_id"})
    cross = index_rows(cross_rows, ("itemid",), "case crosswalk")
    unique_values(cross_rows, "case_id", "Case")
    review_rows = read_csv(case_ledger, {"itemid", "review_status", "decision", "record_sha256"})
    reviews = index_rows(review_rows, ("itemid",), "case review")
    person_rows = read_csv(applicant_crosswalk, {"itemid", "applicant_key", "applicant_id", "review_status"})
    people = index_rows(person_rows, ("itemid", "applicant_key"), "applicant crosswalk")
    unique_values(person_rows, "applicant_id", "Applicant")
    alloc_rows = read_csv(allocation_ledger, {"itemid", "allocation_key", "allocation_id", "review_status"})
    allocations = index_rows(alloc_rows, ("itemid", "allocation_key"), "allocation review")
    unique_values(alloc_rows, "allocation_id", "Allocation")
    split_rows = read_csv(split_manifest, {"case_id", "split", "split_policy_version"})
    splits = index_rows(split_rows, ("case_id",), "split")
    for row in split_rows:
        if row["split"] not in {"train", "dev", "test", "ood", "unassigned_pending_policy"} or not row["split_policy_version"].strip():
            raise ValueError("Split manifest must use a supported split and named policy version")
    records = load_records(inputs)
    if not records:
        raise ValueError("No extraction records supplied")
    if any((itemid,) not in cross for itemid in records):
        raise ValueError("Every selected itemid needs an explicit existing case ID crosswalk entry")
    schemas = {stage: json.loads((HERE / "schemas" / name).read_text()) for stage, name in SCHEMAS.items()}
    for schema in schemas.values():
        schema_errors({}, schema)  # dependency/schema preflight, no optional validation
    tables = {"candidate/case_level.csv": [], "candidate/applicant_level.csv": [],
              "candidate/allocation_level.csv": [], "clean/case_level.csv": [],
              "clean/applicant_level.csv": [], "clean/allocation_level.csv": [], "audit/decisions.csv": []}

    def decision(case_id, kind, key, errors):
        tables["audit/decisions.csv"].append({"case_id": case_id, "record_type": kind, "record_key": key,
                                            "decision": "blocked" if errors else "promoted",
                                            "reasons_json": json.dumps(sorted(set(errors)))})

    for itemid, (payload, merged) in sorted(records.items()):
        case_id = cross[(itemid,)]["case_id"].strip()
        record_hash = canonical_hash(merged)
        stage_errors = validate_extraction(payload, merged, schemas)
        review = reviews.get((itemid,))
        errors = stage_errors + case_review_errors(review, case_ledger, record_hash, merged, source_root)
        split_row = splits.get((case_id,), {"split": "unassigned_pending_policy", "split_policy_version": ""})
        split = split_row["split"]
        core, facts = merged.get("core_case") or {}, merged.get("facts_procedure") or {}
        final = merged.get("final_awards") or {}
        tables["candidate/case_level.csv"].append({
            "case_id": case_id, "itemid": itemid, "record_sha256": record_hash,
            "extraction_status": payload.get("status", "unknown"),
            "stage_validation_status": "failed" if stage_errors else "passed",
            "respondent_country": core.get("respondent_country"), "judgementdate": merged.get("judgementdate"),
            "num_application_numbers": core.get("num_application_numbers"), "num_applicants_candidate": facts.get("num_applicants"),
            "violated_article_codes_json": json.dumps(core.get("detailed_violations")),
            "npd_eur_candidate": final.get("non_pecuniary_eur"), "bundled_eur_candidate": final.get("bundled_award_eur"),
            "split": split, "split_policy_version": split_row["split_policy_version"],
            "promotion_status": "blocked" if errors else "promoted", "issues_json": json.dumps(sorted(set(errors)))})
        decision(case_id, "case", record_hash, errors)
        case_clean = not errors
        clean_case = None
        if case_clean:
            clean_case = {"case_id": case_id, "y_amount_eur": amount_text(review["case_total_npd_eur"]),
                "y_binary": int(money(review["case_total_npd_eur"]) > 0),
                "currency": "EUR", "target_scope": "case_total",
                "zero_rationale": review.get("zero_rationale", "") if money(review["case_total_npd_eur"]) == 0 else "",
                "allocation_coverage": review["allocation_coverage"], "split": split,
                "allocation_export_status": "not_exported",
                "split_policy_version": split_row["split_policy_version"], "review_status": "source_adjudicated"}
            tables["clean/case_level.csv"].append(clean_case)
        source_hash = (review or {}).get("source_document_sha256", "").strip().lower()
        clean_person_ids = set()
        for key, group in sorted(candidates([("facts_procedure", p) for p in facts.get("applicants", [])]).items()):
            raw, link = group["row"], people.get((itemid, key))
            reasons = [] if case_clean else ["case_not_promoted"]
            if group["occurrences"] != 1:
                reasons.append("indistinguishable_duplicate_person_candidates")
            if link is None:
                reasons.append("verified_identity_and_attribute_crosswalk_required")
            else:
                if link["review_status"] != "accepted_identity_and_attributes":
                    reasons.append("identity_or_attributes_not_source_verified")
                reasons += source_review_errors(link, applicant_crosswalk, source_hash=source_hash)
            applicant_id = link["applicant_id"].strip() if link and not reasons else ""
            person = {"case_id": case_id, "itemid": itemid, "applicant_key": key, "applicant_id": applicant_id,
                "source_applicant_index": raw.get("applicant_index"), "birth_year": raw.get("birth_year"),
                "sex": raw.get("sex"), "age_group": raw.get("age_group"), "nationality": raw.get("nationality"),
                "occurrences": group["occurrences"], "mapping_status": "unverified" if reasons else "source_verified",
                "issues_json": json.dumps(sorted(set(reasons)))}
            tables["candidate/applicant_level.csv"].append(person)
            decision(case_id, "applicant", key, reasons)
            if not reasons:
                clean_person_ids.add(applicant_id)
                tables["clean/applicant_level.csv"].append({**{k: person[k] for k in CLEAN_PERSON_FIELDS if k in person},
                    "split": split, "review_status": "source_verified_identity_and_attributes"})
        article = merged.get("article_41_extraction") or {}
        raw_allocations = [("primary", r) for r in article.get("award_per_applicant", [])]
        raw_allocations += [("fallback", r) for r in merged.get("award_per_applicant_fallback") or []]
        proposed, allocation_records = [], []
        for key, group in sorted(candidates(raw_allocations).items()):
            raw, checked = group["row"], allocations.get((itemid, key))
            reasons = [] if case_clean else ["case_not_promoted"]
            if group["occurrences"] != 1:
                reasons.append("duplicate_allocation_candidates_require_resolution")
            if checked is None:
                reasons.append("allocation_source_adjudication_required")
            else:
                reasons += source_review_errors(checked, allocation_ledger, source_hash=source_hash)
                if checked["review_status"] != "accepted" or checked.get("head") != "non_pecuniary" or checked.get("currency") != "EUR":
                    reasons.append("allocation_not_accepted_pure_npd_eur")
                scope = checked.get("allocation_scope")
                if scope not in {"personal", "joint", "group", "not_person_linked"}:
                    reasons.append("unknown_allocation_scope")
                person_id = checked.get("applicant_id", "").strip()
                if scope == "personal" and person_id not in clean_person_ids:
                    reasons.append("personal_allocation_requires_verified_person_link")
                if scope != "personal" and person_id:
                    reasons.append("joint_group_unlinked_amount_must_not_link_to_one_person")
                try:
                    value = money(checked.get("amount_eur"))
                    if value is None:
                        reasons.append("explicit_allocation_amount_required")
                    if money(raw.get("eur_amount")) != value or raw.get("head") != checked.get("head"):
                        if not checked.get("correction_rationale", "").strip():
                            reasons.append("allocation_correction_rationale_required")
                except ValueError:
                    reasons.append("invalid_allocation_amount")
            candidate = {"case_id": case_id, "itemid": itemid, "allocation_key": key, "head_candidate": raw.get("head"),
                "eur_amount_candidate": raw.get("eur_amount"), "source_applicant_index_candidate": raw.get("applicant_index"),
                "source_kinds_json": json.dumps(group["sources"]), "occurrences": group["occurrences"],
                "allocation_scope": checked.get("allocation_scope", "unknown") if checked else "unknown",
                "applicant_id": checked.get("applicant_id", "") if checked and not reasons else "",
                "promotion_status": "blocked", "issues_json": ""}
            allocation_records.append((key, candidate, checked, reasons))
            if not reasons:
                proposed.append({"case_id": case_id, "allocation_id": checked["allocation_id"],
                    "applicant_id": checked.get("applicant_id", ""), "head": "non_pecuniary",
                    "amount_eur": amount_text(checked["amount_eur"]), "currency": "EUR",
                    "allocation_scope": checked["allocation_scope"], "split": split, "review_status": "source_adjudicated"})
        aggregate_errors = []
        if case_clean:
            total = sum((money(row["amount_eur"]) for row in proposed), Decimal("0"))
            target = money(review["case_total_npd_eur"])
            coverage = review["allocation_coverage"]
            if total > target:
                aggregate_errors.append("verified_allocation_sum_exceeds_case_npd")
            if coverage == "complete" and (total != target or len(proposed) != len(allocation_records)):
                aggregate_errors.append("complete_allocation_coverage_not_demonstrated")
            if allocation_records and coverage in {"none_found", "unknown"}:
                aggregate_errors.append("case_allocation_coverage_not_adjudicated_for_promotion")
            clean_case["allocation_export_status"] = ("blocked" if aggregate_errors else
                "complete" if coverage == "complete" else "partial" if proposed else "not_exported")
            if aggregate_errors:
                decision(case_id, "allocation_set", record_hash, aggregate_errors)
        for key, candidate, checked, reasons in allocation_records:
            reasons += aggregate_errors
            candidate["promotion_status"] = "blocked" if reasons else "promoted"
            candidate["issues_json"] = json.dumps(sorted(set(reasons)))
            if reasons:
                candidate["applicant_id"] = ""
            tables["candidate/allocation_level.csv"].append(candidate)
            decision(case_id, "allocation", key, reasons)
        if not aggregate_errors:
            tables["clean/allocation_level.csv"].extend(proposed)

    field_map = dict(zip(tables, [CASE_FIELDS, PERSON_FIELDS, ALLOCATION_FIELDS, CLEAN_CASE_FIELDS,
                                CLEAN_PERSON_FIELDS, CLEAN_ALLOCATION_FIELDS, DECISION_FIELDS]))
    reference_path = HERE / "schemas" / "current_case_columns.v20260913.json"
    reference = json.loads(reference_path.read_text())
    mappings = []
    for column in reference["columns"]:
        if column in {"y_amount_eur", "y_binary"}:
            entry = {"column": column, "status": "source_adjudicated_target_available",
                     "table": "clean/case_level.csv", "source_column": column}
        elif column == "split":
            entry = {"column": column, "status": "explicit_manifest_or_unassigned_pending_policy",
                     "table": "clean/case_level.csv", "source_column": "split"}
        elif column == "itemid":
            entry = {"column": column, "status": "private_crosswalk_join_required",
                     "table": "candidate/case_level.csv", "source_column": "itemid", "join_key": "case_id"}
        else:
            entry = {"column": column, "status": "not_reconstructed_or_not_independently_validated",
                     "table": None, "source_column": None}
        mappings.append(entry)
    adapter = {"schema_version": "1", "reference_case_schema": reference["version"],
        "reference_schema_sha256": file_hash(reference_path), "reference_column_count": len(reference["columns"]),
        "batch_kind": "adjudicated_new_batch_not_full_historical_feature_package",
        "full_case_feature_package_ready": False, "automatic_append_authorized": False,
        "stable_id_rule": "Reuse case_crosswalk case_id; private join restores itemid without generating a new ID",
        "column_mapping": mappings,
        "missing_or_unvalidated_columns": [row["column"] for row in mappings if row["table"] is None],
        "allocation_column_mapping": {"amount_eur": "eur_non_pec_amount", "case_id": "itemid via private case crosswalk",
                                       "review_status": "evidence_status via explicit source-adjudicated status adapter"},
        "applicant_column_mapping": {"case_id": "itemid via private case crosswalk", "applicant_id": "applicant_id",
            "birth_year": "birth_year", "sex": "sex", "age_group": "age_group", "nationality": "nationality",
            "split": "split", "review_status": "record_status via explicit source-verified status adapter"},
        "prohibited_shortcuts": ["Do not fabricate missing columns or fill them with zeros",
            "Do not derive demographics from incompletely verified applicant coverage",
            "Do not move Article 41, awards, claims, audit text or identity-linkage fields into predictors"]}
    manifest = {"format_version": "1", "classification": "internal_restricted_not_a_public_release",
        "scientific_status": "source_adjudicated_subset_plus_unaccepted_candidates",
        "predictor_features_exported": False, "paper_benchmark_reproduction": False,
        "zero_rationale_taxonomy": "paper_table12_six_categories",
        "batch_kind": adapter["batch_kind"], "full_case_feature_package_ready": False,
        "inputs": [{"name": Path(path).name, "sha256": file_hash(path)} for path in inputs],
        "control_inputs": [{"name": name, "sha256": file_hash(path)} for name, path in
            (("case_crosswalk", case_crosswalk), ("case_adjudication", case_ledger),
             ("applicant_crosswalk", applicant_crosswalk), ("allocation_adjudication", allocation_ledger),
             ("split_manifest", split_manifest)) if path is not None],
        "tables": {name: {"rows": len(rows), "features": field_map[name]} for name, rows in tables.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extraction-export-", dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        for name, rows in tables.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=field_map[name], extrasaction="raise", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            manifest["tables"][name]["sha256"] = file_hash(path)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (stage / "schema_adapter.json").write_text(json.dumps(adapter, ensure_ascii=False, indent=2) + "\n")
        (stage / "RESTRICTED.txt").write_text(
            "INTERNAL ONLY. Contains source identifiers, quasi-identifiers and private linkage.\n"
            "Clean means source-adjudicated subset, not anonymized or publication-approved.\n"
            "No full judgment text, beneficiary-name columns or prediction features are exported; this does not make the bundle anonymous.\n")
        if output.exists():
            raise ValueError("Output appeared during export; refusing to replace it")
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Explicit selected holistic meta/result JSON; repeat per case")
    parser.add_argument("--case-crosswalk", type=Path, required=True)
    parser.add_argument("--case-adjudication", type=Path)
    parser.add_argument("--applicant-crosswalk", type=Path)
    parser.add_argument("--allocation-adjudication", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--source-root", type=Path, help="Allowed source-document root; defaults to case-adjudication CSV's directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = export_records(args.input, args.case_crosswalk, args.out, args.case_adjudication,
                            args.applicant_crosswalk, args.allocation_adjudication, args.split_manifest, args.source_root)
    print(json.dumps({"classification": report["classification"],
                      "rows": {name: row["rows"] for name, row in report["tables"].items()}}, indent=2))


if __name__ == "__main__":
    main()

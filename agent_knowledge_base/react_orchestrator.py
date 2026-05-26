#!/usr/bin/env python3
"""
Controller-owned ReAct orchestrator for the v3 NPD knowledge base.

The existing orchestrator_v2.py remains the single-prompt baseline. This file
adds a bounded action/observation loop where the model can request only
controller-approved actions. The default dry run builds an auditable ReAct trace
without calling an LLM or inventing a prediction. Use --live to connect an
OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

KB_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = KB_DIR.parent
DATASET_RELEASE_DIR = Path(os.environ.get("ECTHR_NPD_DATASET_RELEASE", str(PACKAGE_ROOT / "dataset_release")))
STRICT_REACT = "strict_react"
AWARD_REDACTED_REACT = "award_redacted_react"
FULL_INFO_AWARD_BLIND_REACT = "full_info_award_blind_react"
CLAIM_BLIND_COURT_OUTCOME_FREE_REACT = "agentic_claim_blind_court_outcome_free_train_only"
CLAIM_AWARE_COURT_OUTCOME_FREE_REACT = "agentic_claim_aware_court_outcome_free_train_only"
COURT_OUTCOME_FREE_REACT_MODES = {
    CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
    CLAIM_AWARE_COURT_OUTCOME_FREE_REACT,
}
REACT_MODES = {
    STRICT_REACT,
    AWARD_REDACTED_REACT,
    FULL_INFO_AWARD_BLIND_REACT,
    CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
    CLAIM_AWARE_COURT_OUTCOME_FREE_REACT,
}
INFERENCE_MODES = {"zero_shot", "few_shot"}
CONTEXT_POLICIES = {"lazy", "eager"}
EXECUTION_MODES = {"react", "compact_final", "react_compact_memory"}
ACTION_PROTOCOL_ID = "react_action_protocol"
DEFAULT_TRAIN_CSV = DATASET_RELEASE_DIR / "data" / "ecthr_npd_cases.csv"
DEFAULT_TRAIN_LABEL_CSV = DATASET_RELEASE_DIR / "model_inputs" / "structured_tree" / "targets" / "train.csv"
REFERENCE_REASONING_PATH = Path(os.environ.get("ECTHR_NPD_REASONING_LAYER_CSV", "your_path/reasoning_layer.csv"))
TARGET_REASONING_PATH = REFERENCE_REASONING_PATH
REFERENCE_APPLICANT_PATH = Path(os.environ.get("ECTHR_NPD_APPLICANT_CSV", "your_path/applicant.csv"))
ARTICLE_DISTRIBUTION_RELATIVE_PATH = Path("modules/empirical/article_award_distribution_train.csv")
COUNTRY_DISTRIBUTION_RELATIVE_PATH = Path("modules/empirical/country_award_distribution_train.csv")
ARTICLE_COUNTRY_DISTRIBUTION_RELATIVE_PATH = Path("modules/empirical/article_country_award_distribution_train.csv")
DEFAULT_API_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen3.5-27b"
MIN_FILTER_POOL = 30

TARGET_QUERY_TEMPLATE = {
    "sources": ["target_extraction_context"],
    "field_contains": [
        "claim_non_pec",
        "claim_head",
        "violated_articles",
        "respondent_country",
        "country_alpha2",
        "judgment_year",
        "judgementdate",
        "violation_type",
        "num_applicants",
        "case_importance",
        "decision_body",
        "gdp_per_capita",
        "exclusion",
        "include_reason",
        "sufficient",
        "dismissed_reason",
    ],
    "max_chars": 20000,
}

CLAIM_BLIND_TARGET_QUERY_TEMPLATE = {
    "sources": [
        "standard_prompting_input",
        "metadata",
        "merits_side_violation_structure",
        "aggregation_structure",
        "external_factors",
    ],
    "field_contains": [
        "violated_articles",
        "respondent_country",
        "respondent_state",
        "country_alpha2",
        "judgment_year",
        "judgementdate",
        "violation_type",
        "violation_subtype",
        "violation_duration",
        "num_violations",
        "num_applicants",
        "application_count",
        "aggregation",
        "case_importance",
        "decision_body",
        "court_formation",
        "gdp",
        "vulnerability",
        "repetitive",
        "detention",
        "delay",
        "severity",
    ],
    "max_chars": 20000,
}

CLAIM_AWARE_TARGET_QUERY_TEMPLATE = {
    "sources": [
        "standard_prompting_input",
        "metadata",
        "financial_request_structure",
        "merits_side_violation_structure",
        "aggregation_structure",
        "external_factors",
    ],
    "field_contains": [
        "claim_non_pec",
        "claim_head",
        "claimed",
        "violated_articles",
        "respondent_country",
        "respondent_state",
        "country_alpha2",
        "judgment_year",
        "judgementdate",
        "violation_type",
        "violation_subtype",
        "violation_duration",
        "num_violations",
        "num_applicants",
        "application_count",
        "aggregation",
        "case_importance",
        "decision_body",
        "court_formation",
        "gdp",
        "vulnerability",
        "repetitive",
        "detention",
        "delay",
        "severity",
    ],
    "max_chars": 20000,
}

REFERENCE_FEATURE_DEFAULT_SOURCES = [
    "split_case_features",
    "reasoning_layer_features",
    "applicant_features",
]

REFERENCE_FEATURE_DEFAULT_FIELD_CONTAINS = [
    "claim_non_pec",
    "claim_head",
    "violation",
    "violation_subtype",
    "reasoning_factor",
    "finding",
    "sufficient",
    "num_applicants",
    "represented",
    "case_importance",
    "decision_body",
    "country",
    "sufficient",
    "dismissed_reason",
    "exclusion",
]

CLAIM_BLIND_REFERENCE_FEATURE_FIELD_CONTAINS = [
    "violation",
    "violation_subtype",
    "reasoning_factor",
    "finding",
    "num_applicants",
    "application_count",
    "represented",
    "case_importance",
    "decision_body",
    "country",
    "gdp",
    "vulnerability",
    "repetitive",
    "detention",
    "delay",
]

CLAIM_AWARE_REFERENCE_FEATURE_FIELD_CONTAINS = [
    "claim_non_pec",
    "claim_head",
    "claimed",
    "violation",
    "violation_subtype",
    "reasoning_factor",
    "finding",
    "num_applicants",
    "application_count",
    "represented",
    "case_importance",
    "decision_body",
    "country",
    "gdp",
    "vulnerability",
    "repetitive",
    "detention",
    "delay",
]

GENERAL_AWARD_CALIBRATION_RUBRIC = """
General award calibration rubric:
- Predict one case-level non-pecuniary EUR amount. Do not use a linear per-applicant multiplier.
- Treat 0 EUR as a valid continuous prediction. Always compare zero/finding-sufficient references against positive references before final_predict.
- Use lazy target access. The recommended target query is sources=["target_extraction_context"] with field_contains for claim_non_pec, claim_head, violated_articles, respondent_country, country_alpha2, judgment_year, violation_type, num_applicants, case_importance, decision_body, and gdp_per_capita.
- In full_info_award_blind_react, claim-side fields can be used as anchors or caps when visible. If a numeric non-pecuniary claim amount is visible, the final award must not exceed that cap. In strict_react, claimed amounts and claim-side Article 41 fields are not available and must not be inferred from silence.
- In relaxed/full-info modes, check target main-table zero-award reason fields such as exclusion_reason_codes, include_reason, satisfaction_sufficient, and dismissed_reason. Treat finding-sufficient/no-award/Rule 60/no-claim signals as strong zero evidence only when they come from structured extraction fields, not generated reasoning summaries.
- Do not use reasoning_layer.award_reasoning_summary as model input or zero-award evidence. It is an LLM-generated extraction summary, not a source-text quote.
- Use resolve_empirical_priors as train-only distributional calibration, then retrieve_train_references for temporally prior train analogues.
- For query_reference_features, use sources=["split_case_features","reasoning_layer_features","applicant_features"] unless there is a specific reason not to.
- Before final_predict, use assess_zero_positive_evidence to state whether the evidence is zero_plausible, positive_plausible, or ambiguous, then use assess_aggregation_pattern to calibrate case-level aggregation scale. The controller automatically runs the leakage gate when final_predict is requested. The final output remains only {"award_eur": number}.
"""

CLAIM_BLIND_COURT_OUTCOME_FREE_RUBRIC = """
Claim-blind court-outcome-free ReAct rubric:
- Predict one case-level non-pecuniary EUR amount. Treat 0 EUR as a valid continuous prediction.
- Use only sanitized facts/procedure/merits-side violation structure, metadata, applicant/application structure, external factors, train-only priors, and train-only reference anchors.
- Do not request or infer target claim amounts, Article 41 text, just satisfaction text, operative clauses, target award outcomes, target zero reasons, satisfaction-sufficient reasoning, dismissal reasoning, label/source/provenance fields, or redaction markers.
- Retrieved references must come only from the train split. Their known train non-pecuniary award anchors are allowed calibration evidence, but claim fields, Article 41 snippets, and label provenance are not.
- Use train priors and train references to calibrate the amount distribution. No separate target zero/positive evidence assessment tool is available in this mode.
- Use assess_aggregation_pattern to calibrate ordinary, small group, large joined, or mass joined case-level scale. Do not mechanically multiply a per-applicant amount.
- Before final_predict, call the required evidence actions at least once. The final output remains only {"award_eur": number}.
"""

CLAIM_AWARE_COURT_OUTCOME_FREE_RUBRIC = """
Claim-aware court-outcome-free ReAct rubric:
- Predict one case-level non-pecuniary EUR amount. Treat 0 EUR as a valid continuous prediction.
- Use sanitized facts/procedure/merits-side violation structure, metadata, applicant/application structure, external factors, explicit target claim/request fields, train-only priors, and train-only reference anchors.
- Target claimed non-pecuniary amounts, claim state, and claim heads are visible and may be used as calibration/cap evidence when present.
- Do not request or infer Article 41 outcome text, just satisfaction outcome text, operative clauses, target award outcomes, target zero reasons, satisfaction-sufficient reasoning, dismissal reasoning, exclusion/include-reason fields, label/source/provenance fields, or redaction markers.
- Retrieved references must come only from the train split. Their known train non-pecuniary award anchors and claim/request fields are allowed calibration evidence, but Article 41 snippets, zero-reason fields, and label provenance are not.
- Use train priors, train references, and visible claim/request fields to calibrate the amount distribution. No separate target zero/positive evidence assessment tool is available in this mode.
- Use assess_aggregation_pattern to calibrate ordinary, small group, large joined, or mass joined case-level scale. Do not mechanically multiply a per-applicant amount.
- Before final_predict, call the required evidence actions at least once. The final output remains only {"award_eur": number}.
"""

ALLOWED_ACTIONS = {
    "inspect_case",
    "query_target_information",
    "search_modules",
    "load_module",
    "load_relevant_modules",
    "resolve_empirical_priors",
    "retrieve_train_references",
    "query_reference_features",
    "assess_zero_positive_evidence",
    "assess_aggregation_pattern",
    "leakage_check",
    "final_predict",
}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_summary": {"type": "string"},
        "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
        "action_input": {"type": "object"},
    },
    "required": ["thought_summary", "action", "action_input"],
    "additionalProperties": False,
}


def allowed_actions_for_state(state: dict[str, Any]) -> list[str]:
    actions = set(ALLOWED_ACTIONS)
    if state.get("react_mode") in COURT_OUTCOME_FREE_REACT_MODES:
        actions.discard("assess_zero_positive_evidence")
    return sorted(actions)


def action_schema_for_state(state: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(ACTION_SCHEMA)
    schema["properties"]["action"]["enum"] = allowed_actions_for_state(state)
    return schema

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "award_eur": {"type": "number", "minimum": 0},
    },
    "required": ["award_eur"],
    "additionalProperties": False,
}

COMPACT_FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "award_eur": {"type": "number", "minimum": 0},
        "rationale_summary": {"type": "string"},
        "zero_positive_decision": {"type": "string"},
        "aggregation_scale_decision": {"type": "string"},
        "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "award_eur",
        "rationale_summary",
        "zero_positive_decision",
        "aggregation_scale_decision",
        "uncertainty",
    ],
    "additionalProperties": False,
}

STRICT_TEXT_REDACTION = "[STRICT_REACT_BLOCKED_TEXT_REDACTED]"
STRICT_BLOCKED_KEY_FRAGMENTS = (
    "award",
    "raw_extractor",
    "safe_non_pec",
    "safe_bundled",
    "safe_total",
    "per_app_non_pec",
    "diff_non_pec",
    "claim",
    "article_41",
    "article41",
)
AWARD_REDACTED_BLOCKED_KEY_FRAGMENTS = (
    "award_eur",
    "award_non_pec",
    "safe_non_pec",
    "raw_extractor_non_pec",
    "per_app_non_pec",
    "diff_non_pec",
    "fx_fill_non_pec",
)
FULL_INFO_AWARD_BLIND_BLOCKED_KEY_FRAGMENTS = (
    "award_eur",
    "award_eur_label_source",
    "award_non_pec",
    "award_pec",
    "award_costs",
    "award_bundled",
    "safe_non_pec",
    "safe_pec",
    "safe_costs",
    "safe_bundled",
    "safe_total",
    "raw_extractor_non_pec",
    "raw_extractor_pec",
    "raw_extractor_costs",
    "raw_extractor_bundled",
    "raw_case_total",
    "per_app_non_pec",
    "per_app_pec",
    "per_app_costs",
    "per_app_bundled",
    "case_total_from_per_app",
    "diff_non_pec",
    "diff_pec",
    "diff_costs",
    "diff_bundled",
    "diff_case_total",
    "cross_non_pec_match",
)
FULL_INFO_AWARD_BLIND_ALLOWED_AWARD_REASON_KEYS = {
    "award_non_pec_satisfaction_sufficient",
    "award_non_pec_dismissed_reason",
}
CLAIM_BLIND_BLOCKED_KEY_FRAGMENTS = (
    "claim",
    "claimed",
    "hint_claim",
    "article_41",
    "article41",
    "just_satisfaction",
    "operative",
    "award_non_pec",
    "safe_non_pec",
    "raw_extractor_non_pec",
    "per_app_non_pec",
    "repair_non_pec",
    "visible_zero_reason",
    "zero_positive_recommendation",
    "zero_positive_assessment",
    "satisfaction_sufficient",
    "dismissed_reason",
    "include_reason",
    "exclusion_tier",
    "exclusion_reason",
    "target_include_reason",
    "target_exclusion",
    "zero_reason",
    "label_source",
    "source_provenance",
    "amount_direct",
    "y_source",
    "award_reasoning",
)
CLAIM_BLIND_BLOCKED_EXACT_KEYS = {
    "award_eur",
    "safe_non_pec_eur",
    "target_include_reason",
    "target_exclusion_tier",
    "include_reason",
    "exclusion_tier",
    "exclusion_reason_codes",
    "y_source",
    "label_source",
    "source_provenance",
    "zero_reason",
    "amount_direct",
}
CLAIM_AWARE_COURT_OUTCOME_FREE_BLOCKED_KEY_FRAGMENTS = (
    "article_41",
    "article41",
    "just_satisfaction",
    "operative",
    "award_non_pec",
    "safe_non_pec",
    "raw_extractor_non_pec",
    "per_app_non_pec",
    "repair_non_pec",
    "visible_zero_reason",
    "zero_positive_recommendation",
    "zero_positive_assessment",
    "satisfaction_sufficient",
    "dismissed_reason",
    "include_reason",
    "exclusion_tier",
    "exclusion_reason",
    "target_include_reason",
    "target_exclusion",
    "zero_reason",
    "label_source",
    "source_provenance",
    "amount_direct",
    "y_source",
    "award_reasoning",
)
CLAIM_AWARE_COURT_OUTCOME_FREE_BLOCKED_EXACT_KEYS = {
    "award_eur",
    "safe_non_pec_eur",
    "target_include_reason",
    "target_exclusion_tier",
    "include_reason",
    "exclusion_tier",
    "exclusion_reason_codes",
    "y_source",
    "label_source",
    "source_provenance",
    "zero_reason",
    "amount_direct",
}
GENERATED_REASONING_SUMMARY_KEYS = {
    "award_reasoning_summary",
}
TARGET_ALWAYS_BLOCKED_KEYS = {
    "all_scl_citations",
}
TARGET_FINAL_AWARD_TEXT_REDACTION = "[TARGET_FINAL_AWARD_TEXT_REDACTED]"
TARGET_AWARD_VALUE_EXEMPT_PATH_TERMS = (
    "claim",
)
TARGET_AWARD_VALUE_SENSITIVE_PATH_TERMS = (
    "combined_input_text",
    "article_41",
    "article41",
    "operative",
    "award",
    "raw_extractor",
    "safe_",
    "per_app",
    "case_total",
)
TARGET_AWARD_CONTEXT_TERMS = (
    "award",
    "awards",
    "awarded",
    "pay",
    "pecuniary",
    "non-pecuniary",
    "non pecuniary",
    "just satisfaction",
    "eur",
    "euro",
)
REFERENCE_RAW_TEXT_KEY_FRAGMENTS = (
    "combined_input_text",
    "procedure_text",
    "facts_text",
    "introduction_text",
    "safe_appendix_text",
    "judgment_text",
    "judgement_text",
    "raw_text",
    "full_text",
    "article_41_text",
    "operative_clause",
)
REFERENCE_FEATURE_BLOCKED_KEY_FRAGMENTS = (
    "award",
    "target",
    "y_amount",
    "y_binary",
    "safe_non_pec",
    "safe_pec",
    "safe_costs",
    "safe_bundled",
    "safe_total",
    "raw_extractor",
    "raw_case_total",
    "per_app",
    "case_total_from_per_app",
    "diff_",
    "cross_non_pec_match",
    "cross_pec_match",
    "cross_costs_match",
    "repair_",
    "label_source",
)

TRAIN_FEATURE_BLOCKED_KEY_FRAGMENTS = (
    "award",
    "target",
    "y_amount",
    "y_binary",
    "safe_non_pec",
    "safe_pec",
    "safe_costs",
    "safe_total",
    "raw_extractor",
    "raw_case_total",
    "per_app",
    "diff_",
    "claim",
    "article_41",
    "article41",
    "operative",
    "label_source",
)
STRICT_TEXT_BLOCK_TERMS = (
    "article 41",
    "just satisfaction",
    "non-pecuniary damage",
    "non pecuniary damage",
    "pecuniary damage",
    "costs and expenses",
    "for these reasons",
    "operative provision",
)
CLAIM_BLIND_TEXT_BLOCK_TERMS = (
    "article 41",
    "just satisfaction",
    "for these reasons",
    "operative provision",
    "operative clauses",
    "costs and expenses",
    "finding of a violation constitutes sufficient satisfaction",
    "constitutes sufficient just satisfaction",
    "sufficient just satisfaction",
    "makes no award",
    "made no award",
    "no award is made",
    "no separate award",
    "dismisses the claim",
    "dismissed the claim",
    "rejects the claim",
    "rejected the claim",
)
MONEY_RE = re.compile(r"(?:€|\beur\b|\beuros?\b|\b\d[\d,\s.\u00a0\u202f]{2,}\b)", re.IGNORECASE)


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_module("npd_v3_orchestrator_v2_runtime", KB_DIR / "orchestrator_v2.py")


def claim_blind_action_protocol_text() -> str:
    return """
Court-outcome-free action protocol:
- inspect_case returns only sanitized target catalogs and route metadata.
- query_target_information may inspect sanitized facts/procedure/merits-side structure, metadata, aggregation structure, external factors, and claim/request fields when the active mode marks target_financial_request_visible=true.
- resolve_empirical_priors returns train-only distributional statistics without label provenance.
- retrieve_train_references returns train-only temporally prior references with sanitized metadata and train award anchors.
- query_reference_features returns sanitized train-reference feature rows only; claim/request reference fields are allowed only in claim-aware mode, while court-outcome, zero-reason, and label-source fields are blocked.
- assess_aggregation_pattern returns target structural aggregation and train-only same-band anchors.
- final_predict accepts one non-negative case-level EUR amount.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_file", required=True)
    parser.add_argument("--case_id", default=None, help="When case_file is CSV/JSONL, select this itemid.")
    parser.add_argument("--react_mode", default=STRICT_REACT, choices=sorted(REACT_MODES))
    parser.add_argument("--inference_mode", default="zero_shot", choices=sorted(INFERENCE_MODES))
    parser.add_argument("--execution_mode", default="react", choices=sorted(EXECUTION_MODES))
    parser.add_argument("--target_context_policy", default="lazy", choices=sorted(CONTEXT_POLICIES))
    parser.add_argument("--reference_context_policy", default="lazy", choices=sorted(CONTEXT_POLICIES))
    parser.add_argument("--kb_dir", default=str(KB_DIR))
    parser.add_argument("--train_csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--train_label_csv", default=str(DEFAULT_TRAIN_LABEL_CSV))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true", help="Build a deterministic trace without model calls.")
    parser.add_argument("--live", action="store_true", help="Call an OpenAI-compatible endpoint for action selection.")
    parser.add_argument("--api_base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key_file", default=None)
    parser.add_argument("--api_key_env", default="EXTRACTION_API_KEY")
    parser.add_argument("--provider_json_schema", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trace_out", default=None)
    parser.add_argument("--prediction_out", default=None)
    parser.add_argument("--compact_packet_file", default=None, help="Use a prebuilt compact packet for compact_final mode.")
    parser.add_argument("--compact_packet_out", default=None, help="Write the compact packet built by compact_final mode.")
    return parser.parse_args()


def load_case(path: Path, case_id: str | None = None) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if case_id is None or str(row.get("itemid")) == case_id:
                    return row
        raise ValueError(f"No matching JSONL row found in {path}")
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if case_id is None or str(row.get("itemid")) == case_id:
                    return row
        raise ValueError(f"No matching CSV row found in {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for row in data:
            if case_id is None or str(row.get("itemid")) == case_id:
                return row
        raise ValueError(f"No matching JSON row found in {path}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a case object or list of case objects")
    return data


def parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        return datetime(int(match.group(0)), 1, 1)
    return None


def parse_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def normalize_articles(value: Any) -> list[str]:
    found: list[str] = []
    if value is None:
        return found
    if isinstance(value, list):
        for item in value:
            for article in normalize_articles(item):
                if article not in found:
                    found.append(article)
        return found
    if isinstance(value, dict):
        for item in value.values():
            for article in normalize_articles(item):
                if article not in found:
                    found.append(article)
        return found
    parts = V2.ARTICLE_TOKEN_RE.split(str(value)) if any(sep in str(value) for sep in ";,/|") else [str(value)]
    for part in parts:
        normalized = V2.normalize_article_token(part)
        if normalized and normalized not in found:
            found.append(normalized)
    return found


def route_state_for_case(case: dict[str, Any], react_mode: str) -> dict[str, Any]:
    route_state = V2.build_route_state(case, V2.AWARD_REDACTED_MODE)
    route_state["react_mode"] = react_mode
    if react_mode == STRICT_REACT:
        route_state["mode"] = STRICT_REACT
        route_state["available_input_types"] = [
            "strict_redacted_standard_input",
            "oracle_articles",
            "safe_metadata",
            "safe_non_claim_non_award_extracted_hints",
            "train_empirical_priors",
        ]
        hints = route_state.get("extracted_hints") or {}
        route_state["extracted_hints"] = {
            key: value
            for key, value in hints.items()
            if not key.lower().startswith("claim") and "award" not in key.lower()
        }
    elif react_mode == FULL_INFO_AWARD_BLIND_REACT:
        route_state["mode"] = FULL_INFO_AWARD_BLIND_REACT
        route_state["available_input_types"] = [
            "award_blind_standard_input",
            "target_extraction_sidecars_by_itemid",
            "article41_claim_layer",
            "article41_reasoning_minus_target_final_awards",
            "metadata",
            "rich_extracted_features",
            "train_empirical_priors",
        ]
    elif react_mode == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        route_state["mode"] = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT
        route_state["available_input_types"] = [
            "redacted_facts_procedure_merits_text",
            "oracle_articles",
            "safe_metadata",
            "merits_side_violation_structure",
            "applicant_application_structure",
            "external_factors",
            "train_only_empirical_priors",
            "train_only_reference_anchors",
        ]
        hints = route_state.get("extracted_hints") or {}
        route_state["extracted_hints"] = {
            key: value
            for key, value in hints.items()
            if not key_is_blocked(key, CLAIM_BLIND_COURT_OUTCOME_FREE_REACT)
        }
        route_state["input_policy_flags"] = {
            "target_financial_request_visible": False,
            "target_court_outcome_visible": False,
            "target_zero_explanation_visible": False,
            "target_label_provenance_visible": False,
            "priors_source_split": "train",
            "retrieval_candidate_split": "train",
        }
    elif react_mode == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        route_state["mode"] = CLAIM_AWARE_COURT_OUTCOME_FREE_REACT
        route_state["available_input_types"] = [
            "redacted_facts_procedure_merits_text",
            "target_claim_request_fields",
            "oracle_articles",
            "safe_metadata",
            "merits_side_violation_structure",
            "applicant_application_structure",
            "external_factors",
            "train_only_empirical_priors",
            "train_only_reference_anchors",
        ]
        hints = route_state.get("extracted_hints") or {}
        route_state["extracted_hints"] = {
            key: value
            for key, value in hints.items()
            if not key_is_blocked(key, CLAIM_AWARE_COURT_OUTCOME_FREE_REACT)
        }
        route_state["input_policy_flags"] = {
            "target_financial_request_visible": True,
            "target_court_outcome_visible": False,
            "target_zero_explanation_visible": False,
            "target_label_provenance_visible": False,
            "priors_source_split": "train",
            "retrieval_candidate_split": "train",
            "leakage_label": "claim-aware relaxed; court-outcome-free and zero-reason-free",
        }
    return route_state


def should_strict_redact_line(line: str) -> bool:
    lowered = line.lower()
    if any(term in lowered for term in STRICT_TEXT_BLOCK_TERMS):
        return True
    has_money = bool(MONEY_RE.search(line))
    if has_money and any(term in lowered for term in ("claim", "claimed", "award", "awarded", "pay")):
        return True
    if "non-pecuniary" in lowered and any(term in lowered for term in ("claim", "award", "damage")):
        return True
    return False


def strict_redact_text(text: Any) -> tuple[str, int]:
    raw = str(text or "")
    if not raw.strip():
        return "", 0
    redacted_lines: list[str] = []
    count = 0
    for line in raw.splitlines():
        if should_strict_redact_line(line):
            redacted_lines.append(STRICT_TEXT_REDACTION)
            count += 1
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines), count


def should_claim_blind_redact_line(line: str) -> bool:
    lowered = line.lower()
    if any(term in lowered for term in CLAIM_BLIND_TEXT_BLOCK_TERMS):
        return True
    if "non-pecuniary" in lowered and any(term in lowered for term in ("claim", "award", "awarded", "damage")):
        return True
    if "non pecuniary" in lowered and any(term in lowered for term in ("claim", "award", "awarded", "damage")):
        return True
    has_money = bool(MONEY_RE.search(line))
    if has_money and any(term in lowered for term in ("claim", "claimed", "award", "awarded", "pay", "damage")):
        return True
    return False


def claim_blind_redact_text(text: Any) -> tuple[str, int]:
    raw = str(text or "")
    if not raw.strip():
        return "", 0
    redacted_lines: list[str] = []
    count = 0
    for line in raw.splitlines():
        if should_claim_blind_redact_line(line):
            redacted_lines.append(STRICT_TEXT_REDACTION)
            count += 1
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines), count


def path_contains_any(path: tuple[str, ...], terms: tuple[str, ...]) -> bool:
    lowered = ".".join(path).lower()
    return any(term in lowered for term in terms)


def line_contains_target_award_value(line: str, target_values: set[str]) -> bool:
    if not target_values:
        return False
    lowered = line.lower()
    if not any(term in lowered for term in TARGET_AWARD_CONTEXT_TERMS):
        return False
    return any(target_value and target_value in line for target_value in target_values)


def redact_target_final_award_text(value: Any, case: dict[str, Any], path: tuple[str, ...] = ()) -> tuple[Any, int]:
    """Redact target final-award snippets from visible source text.

    Claim paths are exempt because claimed amounts are allowed in relaxed modes
    and may coincidentally equal the final award. Structured final-award fields
    are blocked separately by key redaction.
    """
    target_values = target_award_values(case)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        count = 0
        for key, child in value.items():
            child_value, child_count = redact_target_final_award_text(child, case, (*path, str(key)))
            redacted[key] = child_value
            count += child_count
        return redacted, count
    if isinstance(value, list):
        redacted_items: list[Any] = []
        count = 0
        for idx, child in enumerate(value):
            child_value, child_count = redact_target_final_award_text(child, case, (*path, str(idx)))
            redacted_items.append(child_value)
            count += child_count
        return redacted_items, count
    if not isinstance(value, str) or not target_values:
        return value, 0
    if path_contains_any(path, TARGET_AWARD_VALUE_EXEMPT_PATH_TERMS):
        return value, 0
    if not path_contains_any(path, TARGET_AWARD_VALUE_SENSITIVE_PATH_TERMS):
        return value, 0
    redacted_count = 0
    redacted_lines: list[str] = []
    for line in value.splitlines():
        if line_contains_target_award_value(line, target_values):
            redacted_lines.append(TARGET_FINAL_AWARD_TEXT_REDACTION)
            redacted_count += 1
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines), redacted_count


def key_is_blocked(key: Any, react_mode: str) -> bool:
    lowered = str(key or "").strip().lower()
    if not lowered:
        return False
    if lowered in TARGET_ALWAYS_BLOCKED_KEYS:
        return True
    if lowered in GENERATED_REASONING_SUMMARY_KEYS:
        return True
    if react_mode == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        if lowered in CLAIM_BLIND_BLOCKED_EXACT_KEYS:
            return True
        return any(fragment in lowered for fragment in CLAIM_BLIND_BLOCKED_KEY_FRAGMENTS)
    if react_mode == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        if lowered in CLAIM_AWARE_COURT_OUTCOME_FREE_BLOCKED_EXACT_KEYS:
            return True
        return any(fragment in lowered for fragment in CLAIM_AWARE_COURT_OUTCOME_FREE_BLOCKED_KEY_FRAGMENTS)
    if react_mode == FULL_INFO_AWARD_BLIND_REACT and lowered in FULL_INFO_AWARD_BLIND_ALLOWED_AWARD_REASON_KEYS:
        return False
    if react_mode == STRICT_REACT:
        fragments = STRICT_BLOCKED_KEY_FRAGMENTS
    elif react_mode == FULL_INFO_AWARD_BLIND_REACT:
        fragments = FULL_INFO_AWARD_BLIND_BLOCKED_KEY_FRAGMENTS
    else:
        fragments = AWARD_REDACTED_BLOCKED_KEY_FRAGMENTS
    if react_mode in {AWARD_REDACTED_REACT, FULL_INFO_AWARD_BLIND_REACT} and lowered.startswith("claim"):
        return False
    return any(fragment in lowered for fragment in fragments)


def sanitize_for_react_mode(value: Any, react_mode: str, path: tuple[str, ...] = ()) -> tuple[Any, list[str], int]:
    removed_paths: list[str] = []
    text_redactions = 0
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            child_path = (*path, str(key))
            if key_is_blocked(key, react_mode):
                removed_paths.append(".".join(child_path))
                continue
            sanitized_child, child_removed, child_text_redactions = sanitize_for_react_mode(child, react_mode, child_path)
            sanitized[key] = sanitized_child
            removed_paths.extend(child_removed)
            text_redactions += child_text_redactions
        return sanitized, removed_paths, text_redactions
    if isinstance(value, list):
        sanitized_list: list[Any] = []
        for idx, child in enumerate(value):
            sanitized_child, child_removed, child_text_redactions = sanitize_for_react_mode(
                child, react_mode, (*path, str(idx))
            )
            sanitized_list.append(sanitized_child)
            removed_paths.extend(child_removed)
            text_redactions += child_text_redactions
        return sanitized_list, removed_paths, text_redactions
    if isinstance(value, str) and react_mode in {STRICT_REACT, *COURT_OUTCOME_FREE_REACT_MODES}:
        if react_mode in COURT_OUTCOME_FREE_REACT_MODES:
            redacted, count = claim_blind_redact_text(value)
        else:
            redacted, count = strict_redact_text(value)
        return redacted, removed_paths, count
    return value, removed_paths, text_redactions


def build_strict_case_inputs(case: dict[str, Any]) -> dict[str, Any]:
    standard_input, standard_redactions = V2.build_standard_prompting_input(case)
    metadata = V2.extract_safe_metadata(case)
    extracted_hints = V2.extract_safe_extracted_hints(case)
    raw_inputs = {
        "strict_input_policy": {
            "raw_article41_text": "blocked",
            "operative_clauses": "blocked",
            "direct_award_snippets": "blocked",
            "claimed_amounts": "blocked",
            "target_derived_fields": "blocked",
        },
        "standard_prompting_input": standard_input,
        "metadata": metadata,
        "extracted_hints": extracted_hints,
    }
    sanitized, removed_paths, text_redactions = sanitize_for_react_mode(raw_inputs, STRICT_REACT)
    sanitized["target_redaction_report"] = {
        "removed_paths": sorted(set([*standard_redactions, *removed_paths])),
        "strict_text_redaction_marker": STRICT_TEXT_REDACTION,
        "strict_text_redaction_count": text_redactions,
    }
    return sanitized


def extract_external_factors(case: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "gdp_per_capita_current_usd",
        "gdp_constant_2015_usd",
        "gdp_per_capita_log1p",
        "gdp_constant_2015_log1p",
    )
    return {
        key: case.get(key)
        for key in allowed_keys
        if case.get(key) not in (None, "")
    }


def extract_merits_side_violation_structure(case: dict[str, Any]) -> dict[str, Any]:
    facts_procedure = case.get("facts_procedure") if isinstance(case.get("facts_procedure"), dict) else {}
    status = facts_procedure.get("status") if isinstance(facts_procedure.get("status"), dict) else {}
    reasoning_layer = case.get("reasoning_layer") if isinstance(case.get("reasoning_layer"), dict) else {}
    raw = {
        "violated_articles": (
            case.get("violated_articles")
            or case.get("oracle_violated_articles")
            or case.get("violated_articles_text")
        ),
        "num_violations_found": case.get("num_violations_found"),
        "violation_type": case.get("violation_type"),
        "violation_subtype": V2.first_non_empty(case, "violation_subtype") or reasoning_layer.get("violation_subtype"),
        "violation_duration_months": V2.first_non_empty(case, "violation_duration_months"),
        "reasoning_factors": reasoning_layer.get("reasoning_factors") or reasoning_layer.get("reasoning_factor"),
        "harm_severity": reasoning_layer.get("harm_severity") or reasoning_layer.get("severity"),
        "vulnerability": status.get("vulnerability_tags") or status.get("is_vulnerable"),
        "is_repetitive_case": facts_procedure.get("is_repetitive_case"),
        "state_remedial_measures": facts_procedure.get("state_remedial_measures"),
        "is_joint_application": facts_procedure.get("is_joint_application"),
        "source_policy": (
            "facts/procedure/merits-side structure only; financial-request fields, "
            "court-outcome fields, zero-explanation fields, and label provenance fields are blocked"
        ),
    }
    return {key: value for key, value in raw.items() if value not in (None, "", [])}


def extract_financial_request_structure(case: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "claim_non_pec_state",
        "claim_non_pec_original_currency",
        "claim_non_pec_original_amount",
        "claim_non_pec_eur",
        "claim_non_pec_eur_court_stated",
        "claim_head_count",
        "claim_head_non_pec",
        "claim_head_pec",
        "claim_head_costs",
        "claimed_non_pec_eur",
        "claimed_amount_non_pec_eur",
    )
    raw = {
        key: case.get(key)
        for key in allowed_keys
        if case.get(key) not in (None, "")
    }
    raw["source_policy"] = (
        "target claim/request fields are visible by explicit experiment setting; "
        "court outcome, zero-explanation, Article 41 outcome text, and label provenance remain blocked"
    )
    return raw


def build_claim_blind_court_outcome_free_case_inputs(case: dict[str, Any]) -> dict[str, Any]:
    standard_input, standard_redactions = V2.build_standard_prompting_input(case)
    raw_inputs = {
        "input_policy": {
            "name": CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
            "target_financial_request_visible": False,
            "target_court_outcome_visible": False,
            "target_zero_explanation_visible": False,
            "target_label_provenance_visible": False,
            "priors_source_split": "train",
            "retrieval_candidate_split": "train",
        },
        "standard_prompting_input": standard_input,
        "metadata": V2.extract_safe_metadata(case),
        "merits_side_violation_structure": extract_merits_side_violation_structure(case),
        "aggregation_structure": target_aggregation_structure(case),
        "external_factors": extract_external_factors(case),
    }
    sanitized, removed_paths, text_redactions = sanitize_for_react_mode(
        raw_inputs,
        CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
    )
    sanitized, redacted_award_text_count = redact_target_final_award_text(sanitized, case)
    sanitized["target_redaction_report"] = {
        "removed_paths": sorted(set([*standard_redactions, *removed_paths])),
        "text_redaction_marker": STRICT_TEXT_REDACTION,
        "text_redaction_count": text_redactions,
        "input_policy": CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
        "target_final_award_text_redaction_marker": TARGET_FINAL_AWARD_TEXT_REDACTION,
        "target_final_award_text_redaction_count": redacted_award_text_count,
    }
    return sanitized


def build_claim_aware_court_outcome_free_case_inputs(case: dict[str, Any]) -> dict[str, Any]:
    standard_input, standard_redactions = V2.build_standard_prompting_input(case)
    raw_inputs = {
        "input_policy": {
            "name": CLAIM_AWARE_COURT_OUTCOME_FREE_REACT,
            "target_financial_request_visible": True,
            "target_court_outcome_visible": False,
            "target_zero_explanation_visible": False,
            "target_label_provenance_visible": False,
            "priors_source_split": "train",
            "retrieval_candidate_split": "train",
            "leakage_label": "claim-aware relaxed; court-outcome-free and zero-reason-free",
        },
        "standard_prompting_input": standard_input,
        "metadata": V2.extract_safe_metadata(case),
        "financial_request_structure": extract_financial_request_structure(case),
        "merits_side_violation_structure": extract_merits_side_violation_structure(case),
        "aggregation_structure": target_aggregation_structure(case),
        "external_factors": extract_external_factors(case),
    }
    sanitized, removed_paths, text_redactions = sanitize_for_react_mode(
        raw_inputs,
        CLAIM_AWARE_COURT_OUTCOME_FREE_REACT,
    )
    sanitized, redacted_award_text_count = redact_target_final_award_text(sanitized, case)
    sanitized["target_redaction_report"] = {
        "removed_paths": sorted(set([*standard_redactions, *removed_paths])),
        "text_redaction_marker": STRICT_TEXT_REDACTION,
        "text_redaction_count": text_redactions,
        "input_policy": CLAIM_AWARE_COURT_OUTCOME_FREE_REACT,
        "leakage_label": "claim-aware relaxed; court-outcome-free and zero-reason-free",
        "target_final_award_text_redaction_marker": TARGET_FINAL_AWARD_TEXT_REDACTION,
        "target_final_award_text_redaction_count": redacted_award_text_count,
    }
    return sanitized


@lru_cache(maxsize=8)
def load_csv_index(path_text: str) -> dict[str, dict[str, Any]]:
    path = Path(path_text)
    if not path.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            itemid = str(row.get("itemid") or "").strip()
            if itemid:
                index[itemid] = {key: value for key, value in row.items() if value not in (None, "")}
    return index


def add_target_reasoning_sidecar(case_inputs: dict[str, Any], case: dict[str, Any]) -> None:
    itemid = str(case.get("itemid") or "").strip()
    reasoning = load_csv_index(str(TARGET_REASONING_PATH)).get(itemid) if itemid else None
    context = case_inputs.setdefault("target_extraction_context", {})
    if isinstance(context, dict):
        sidecars = context.setdefault("sidecars", {})
        if reasoning:
            sidecars["reasoning_layer"] = {
                key: value
                for key, value in reasoning.items()
                if str(key or "").strip().lower() not in GENERATED_REASONING_SUMMARY_KEYS
            }
        main_zero_reason_fields = {
            key: case.get(key)
            for key in (
                "award_non_pec_satisfaction_sufficient",
                "award_non_pec_dismissed_reason",
                "exclusion_tier",
                "exclusion_reason_codes",
                "include_reason",
                "claim_non_pec_state",
                "claim_non_pec_original_currency",
                "claim_non_pec_original_amount",
                "claim_non_pec_eur_court_stated",
                "claim_head_count",
                "claim_head_non_pec",
            )
            if case.get(key) not in (None, "")
        }
        if main_zero_reason_fields:
            sidecars["main_table_zero_claim_fields"] = main_zero_reason_fields


def build_case_inputs(case: dict[str, Any], react_mode: str) -> dict[str, Any]:
    if react_mode == STRICT_REACT:
        return build_strict_case_inputs(case)
    if react_mode == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        return build_claim_blind_court_outcome_free_case_inputs(case)
    if react_mode == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        return build_claim_aware_court_outcome_free_case_inputs(case)
    case_inputs = V2.build_award_redacted_case_inputs(case)
    add_target_reasoning_sidecar(case_inputs, case)
    effective_mode = FULL_INFO_AWARD_BLIND_REACT if react_mode == FULL_INFO_AWARD_BLIND_REACT else AWARD_REDACTED_REACT
    sanitized, removed_paths, text_redactions = sanitize_for_react_mode(case_inputs, effective_mode)
    redacted_award_text_count = 0
    if react_mode in {AWARD_REDACTED_REACT, FULL_INFO_AWARD_BLIND_REACT}:
        sanitized, redacted_award_text_count = redact_target_final_award_text(sanitized, case)
    if removed_paths or text_redactions:
        report = sanitized.setdefault("target_redaction_report", {})
        report["react_removed_paths"] = sorted(set(removed_paths))
        report["react_text_redaction_count"] = text_redactions
        if react_mode == FULL_INFO_AWARD_BLIND_REACT:
            detailed_standard_paths = report.pop("redacted_standard_input_paths", [])
            detailed_extraction_paths = report.pop("redacted_extraction_paths", [])
            detailed_react_paths = report.pop("react_removed_paths", [])
            report["redacted_standard_input_path_count"] = len(detailed_standard_paths)
            report["redacted_extraction_path_count"] = len(detailed_extraction_paths)
            report["react_removed_path_count"] = len(detailed_react_paths)
            report["award_blind_policy"] = "final_awards_and_per_applicant_awards_blocked; claims_and_rich_non_award_features_allowed"
    if redacted_award_text_count:
        report = sanitized.setdefault("target_redaction_report", {})
        report["target_final_award_text_redaction_marker"] = TARGET_FINAL_AWARD_TEXT_REDACTION
        report["target_final_award_text_redaction_count"] = redacted_award_text_count
    return sanitized


def selected_module_ids(case: dict[str, Any], route_state: dict[str, Any], react_mode: str) -> list[str]:
    selected: list[str] = [ACTION_PROTOCOL_ID]
    if react_mode in {AWARD_REDACTED_REACT, FULL_INFO_AWARD_BLIND_REACT}:
        selected.extend(V2.select_binary_modules(V2.AWARD_REDACTED_MODE))
    elif react_mode in COURT_OUTCOME_FREE_REACT_MODES:
        selected.extend([])
    else:
        selected.extend(
            [
                "output_schema",
                "failure_modes",
                "zero_award_rules",
                "finding_sufficient_guidance",
            ]
        )

    routing = [] if react_mode in COURT_OUTCOME_FREE_REACT_MODES else ["router_policy", "fallback_policy"]
    if react_mode not in COURT_OUTCOME_FREE_REACT_MODES and "6" in route_state.get("violated_articles", []):
        routing.append("art6_limb_router")
    selected.extend(routing)
    selected.extend(V2.select_article_modules(case, route_state))

    deduped: list[str] = []
    for module_id in selected:
        if module_id not in deduped:
            deduped.append(module_id)
    return deduped


def load_module_record(kb_dir: Path, modules: dict[str, dict[str, Any]], module_id: str, react_mode: str) -> dict[str, Any]:
    if module_id == ACTION_PROTOCOL_ID:
        module = modules[module_id]
        if react_mode in COURT_OUTCOME_FREE_REACT_MODES:
            return {
                "module_id": module_id,
                "text": claim_blind_action_protocol_text(),
                "module_type": module.get("module_type"),
                "leakage_tier": (
                    "claim_aware_court_outcome_free"
                    if react_mode == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT
                    else "claim_blind_court_outcome_free"
                ),
            }
        return {
            "module_id": module_id,
            "text": (kb_dir / module["path"]).read_text(encoding="utf-8"),
            "module_type": module.get("module_type"),
            "leakage_tier": module.get("leakage_tier"),
        }
    if react_mode == STRICT_REACT and module_id == "claim_rules":
        raise ValueError("claim_rules is not allowed in strict_react")
    if react_mode in COURT_OUTCOME_FREE_REACT_MODES and module_id in {
        "zero_award_rules",
        "zero_award_classification_rules",
        "finding_sufficient_guidance",
        "mode_contracts",
        "system_policy",
        "binary_gate_policy",
        "output_schema",
    }:
        raise ValueError(f"{module_id} is not allowed in {react_mode}")
    return V2.load_module_record(kb_dir, modules, module_id, V2.AWARD_REDACTED_MODE)


def normalize_distribution_country(value: Any) -> str | None:
    return V2.normalize_respondent_state(value)


def distribution_key(row: dict[str, str], key_fields: list[str]) -> tuple[str, ...] | None:
    key: list[str] = []
    for field in key_fields:
        raw_value = row.get(field)
        if field == "article":
            value = V2.normalize_article_token(raw_value)
        elif field == "country_alpha2":
            value = normalize_distribution_country(raw_value)
        else:
            value = str(raw_value or "").strip()
        if not value:
            return None
        key.append(value)
    return tuple(key)


def distribution_row_to_stats(row: dict[str, str], relative_path: Path) -> dict[str, Any] | None:
    sample_count = parse_int(row.get("sample_count"), default=0)
    if sample_count <= 0:
        return None

    zero_rate = parse_float(row.get("zero_rate"))
    median_anchor = parse_float(row.get("median_positive_or_all") or row.get("median"))
    iqr_anchor = parse_float(row.get("iqr_positive_or_all") or row.get("iqr"))
    p10_anchor = parse_float(row.get("p10_positive_or_all") or row.get("p10"))
    p90_anchor = parse_float(row.get("p90_positive_or_all") or row.get("p90"))
    if None in (zero_rate, median_anchor, iqr_anchor, p10_anchor, p90_anchor):
        return None

    return {
        "sample_count": sample_count,
        "positive_count": parse_int(row.get("positive_count"), default=0),
        "zero_count": parse_int(row.get("zero_count"), default=0),
        "zero_rate": float(zero_rate),
        "median": float(median_anchor),
        "iqr": float(iqr_anchor),
        "p10": float(p10_anchor),
        "p90": float(p90_anchor),
        "median_all": parse_float(row.get("median_all")),
        "p10_all": parse_float(row.get("p10_all")),
        "p90_all": parse_float(row.get("p90_all")),
        "zero_reason_count": parse_int(row.get("zero_reason_count"), default=0),
        "amount_direct_count": parse_int(row.get("amount_direct_count"), default=0),
        "proxy_keep72_count": parse_int(row.get("proxy_keep72_count"), default=0),
        "anchor_basis": "positive_amounts_if_available_else_all",
        "table_source_csv": str(relative_path).replace("\\", "/"),
    }


def load_distribution_table(
    kb_dir: Path,
    relative_path: Path,
    key_fields: list[str],
) -> dict[tuple[str, ...], dict[str, Any]]:
    path = kb_dir / relative_path
    if not path.exists():
        return {}

    table: dict[tuple[str, ...], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = distribution_key(row, key_fields)
            if key is None:
                continue
            stats = distribution_row_to_stats(row, relative_path)
            if stats is not None:
                table[key] = stats
    return table


def weighted_distribution_stats(
    table: dict[tuple[str, ...], dict[str, Any]],
    source_path: Path,
) -> dict[str, Any] | None:
    if not table:
        return None
    total_n = sum(int(stats["sample_count"]) for stats in table.values())
    if total_n <= 0:
        return None

    def weighted_avg(key: str) -> float:
        numerator = sum(float(stats[key]) * int(stats["sample_count"]) for stats in table.values())
        return numerator / total_n

    return {
        "sample_count": total_n,
        "positive_count": sum(int(stats.get("positive_count") or 0) for stats in table.values()),
        "zero_count": sum(int(stats.get("zero_count") or 0) for stats in table.values()),
        "zero_rate": weighted_avg("zero_rate"),
        "median": weighted_avg("median"),
        "iqr": weighted_avg("iqr"),
        "p10": weighted_avg("p10"),
        "p90": weighted_avg("p90"),
        "median_all": weighted_avg("median_all") if all(stats.get("median_all") is not None for stats in table.values()) else None,
        "p10_all": weighted_avg("p10_all") if all(stats.get("p10_all") is not None for stats in table.values()) else None,
        "p90_all": weighted_avg("p90_all") if all(stats.get("p90_all") is not None for stats in table.values()) else None,
        "zero_reason_count": sum(int(stats.get("zero_reason_count") or 0) for stats in table.values()),
        "amount_direct_count": sum(int(stats.get("amount_direct_count") or 0) for stats in table.values()),
        "proxy_keep72_count": sum(int(stats.get("proxy_keep72_count") or 0) for stats in table.values()),
        "anchor_basis": "weighted_mean_of_article_distribution_anchors",
        "aggregation_method": "weighted_mean_of_article_distribution_anchors",
        "table_source_csv": str(source_path).replace("\\", "/"),
    }


def compact_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if stats is None:
        return None
    output: dict[str, Any] = {
        "sample_count": int(stats["sample_count"]),
        "positive_count": int(stats.get("positive_count") or 0),
        "zero_count": int(stats.get("zero_count") or 0),
        "zero_rate": round(float(stats["zero_rate"]), 6),
        "median": float(stats["median"]),
        "iqr": float(stats["iqr"]),
        "p10": float(stats["p10"]),
        "p90": float(stats["p90"]),
        "table_source_csv": stats.get("table_source_csv"),
        "anchor_basis": stats.get("anchor_basis"),
    }
    for key in ("median_all", "p10_all", "p90_all"):
        if stats.get(key) is not None:
            output[key] = float(stats[key])
    for key in ("aggregation_method",):
        if stats.get(key) not in (None, ""):
            output[key] = stats[key]
    return output


def resolve_calibration(kb_dir: Path, route_state: dict[str, Any]) -> dict[str, Any]:
    article_table = load_distribution_table(kb_dir, ARTICLE_DISTRIBUTION_RELATIVE_PATH, ["article"])
    country_table = load_distribution_table(kb_dir, COUNTRY_DISTRIBUTION_RELATIVE_PATH, ["country_alpha2"])
    article_country_table = load_distribution_table(
        kb_dir,
        ARTICLE_COUNTRY_DISTRIBUTION_RELATIVE_PATH,
        ["article", "country_alpha2"],
    )
    global_stats = weighted_distribution_stats(article_table, ARTICLE_DISTRIBUTION_RELATIVE_PATH)

    violated_articles = route_state.get("violated_articles") or []
    respondent_state = normalize_distribution_country(route_state.get("respondent_state"))
    art6_limb = route_state.get("art6_limb")
    country_stats = country_table.get((respondent_state,)) if respondent_state else None

    calibrations: list[dict[str, Any]] = []
    calibration_sources_used: list[str] = []
    weak_support_detected = False

    for raw_article in violated_articles:
        article = V2.normalize_article_token(raw_article) or str(raw_article)
        article_country_stats = (
            article_country_table.get((article, respondent_state))
            if respondent_state
            else None
        )
        article_stats = article_table.get((article,))

        available_buckets: list[tuple[str, dict[str, Any]]] = []
        if article_country_stats:
            available_buckets.append(("article_country", article_country_stats))
        if article == "6" and art6_limb in {"civil", "criminal"} and article_stats:
            available_buckets.append((f"article_limb:{art6_limb}", article_stats))
        if article_stats:
            available_buckets.append(("article", article_stats))
        if country_stats:
            available_buckets.append(("country", country_stats))
        if global_stats:
            available_buckets.append(("global_article_weighted", global_stats))

        selected_bucket = "none"
        selected_stats: dict[str, Any] | None = None
        for bucket_name, stats in available_buckets:
            if int(stats["sample_count"]) >= V2.MIN_EMPIRICAL_SUPPORT_N:
                selected_bucket = bucket_name
                selected_stats = stats
                break

        if selected_stats is None and available_buckets:
            selected_bucket, selected_stats = available_buckets[-1]

        if selected_stats is None:
            weak_support_detected = True
            calibrations.append(
                {
                    "article": article,
                    "respondent_state": respondent_state,
                    "art6_limb": art6_limb if article == "6" else None,
                    "selected_bucket": "none",
                    "selected_sample_count": 0,
                    "bucket_trace": [],
                    "weak_support": True,
                }
            )
            calibration_sources_used.append(f"article={article};country={respondent_state};selected=none;n=0")
            continue

        first_bucket_name = available_buckets[0][0]
        fallback_used = selected_bucket != first_bucket_name
        selected_sample_count = int(selected_stats["sample_count"])
        if fallback_used or selected_sample_count < V2.MIN_EMPIRICAL_SUPPORT_N:
            weak_support_detected = True

        bucket_trace = [
            {
                "bucket": bucket_name,
                "sample_count": int(stats["sample_count"]),
                "zero_rate": round(float(stats["zero_rate"]), 6),
                "table_source_csv": stats.get("table_source_csv"),
            }
            for bucket_name, stats in available_buckets
        ]
        fallback_chain = " > ".join(
            f"{bucket['bucket']}(n={bucket['sample_count']})"
            for bucket in bucket_trace
        )

        calibrations.append(
            {
                "article": article,
                "respondent_state": respondent_state,
                "art6_limb": art6_limb if article == "6" else None,
                "selected_bucket": selected_bucket,
                "selected_sample_count": selected_sample_count,
                "bucket_trace": bucket_trace,
                "fallback_used": fallback_used,
                "weak_support": selected_sample_count < V2.MIN_EMPIRICAL_SUPPORT_N,
                "selected_anchor": compact_stats(selected_stats),
            }
        )
        calibration_sources_used.append(
            f"article={article};country={respondent_state};selected={selected_bucket};"
            f"n={selected_sample_count};fallback_chain={fallback_chain}"
        )

    calibration = {
        "minimum_support_n": V2.MIN_EMPIRICAL_SUPPORT_N,
        "distribution_tables": {
            "article": str(ARTICLE_DISTRIBUTION_RELATIVE_PATH).replace("\\", "/"),
            "country": str(COUNTRY_DISTRIBUTION_RELATIVE_PATH).replace("\\", "/"),
            "article_country": str(ARTICLE_COUNTRY_DISTRIBUTION_RELATIVE_PATH).replace("\\", "/"),
        },
        "selection_policy": "article_country_if_supported_else_article_else_country_else_global",
        "global_available": global_stats is not None,
        "country_context": {
            "country_alpha2": respondent_state,
            "anchor": compact_stats(country_stats),
        },
        "calibration_sources_used": calibration_sources_used,
        "article_calibrations": calibrations,
        "weak_support_detected": weak_support_detected,
    }
    route_state["empirical_calibration"] = calibration
    route_state["calibration_sources_used"] = calibration["calibration_sources_used"]
    route_state["calibration_uncertainty"] = bool(
        calibration.get("weak_support_detected") or route_state.get("art6_limb_ambiguous")
    )
    route_state.setdefault("routing_warnings", [])
    if route_state["calibration_uncertainty"] and "calibration_uncertainty" not in route_state["routing_warnings"]:
        route_state["routing_warnings"].append("calibration_uncertainty")
    return calibration


def train_feature_key_is_blocked(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(fragment in lowered for fragment in TRAIN_FEATURE_BLOCKED_KEY_FRAGMENTS)


def compact_train_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in row.items():
        if train_feature_key_is_blocked(key):
            continue
        if value in (None, ""):
            continue
        compact[str(key)] = value
    return compact


def load_train_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            if str(row.get("split") or "train").strip() != "train":
                continue
            compact = compact_train_feature_row(row)
            if compact:
                rows.append(compact)
        return rows


def load_train_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row.get("itemid") or ""): row for row in csv.DictReader(handle)}


def target_date(case: dict[str, Any]) -> datetime | None:
    return parse_date(
        V2.first_non_empty(case, "judgementdate", "judgment_date", "judgementdate_iso", "judgment_year")
    )


def applicant_band(value: int) -> str:
    if value <= 1:
        return "single"
    if value <= 3:
        return "small_multi"
    if value <= 10:
        return "medium_multi"
    return "large_multi"


def aggregation_count_band(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 20:
        return "6-20"
    if value <= 100:
        return "21-100"
    return "100+"


def split_application_numbers(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    matches = re.findall(r"\b\d{1,6}/\d{2,4}\b", text)
    if matches:
        deduped: list[str] = []
        for match in matches:
            if match not in deduped:
                deduped.append(match)
        return deduped
    return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]


def application_count(row: dict[str, Any]) -> int:
    values = [
        parse_int(row.get("application_count"), default=0),
        len(split_application_numbers(row.get("application_numbers_extracted"))),
        len(split_application_numbers(row.get("appno"))),
    ]
    positive = [value for value in values if value > 0]
    return max(positive) if positive else 1


def structural_applicant_count(row: dict[str, Any]) -> int:
    values = [
        parse_int(row.get(key), default=0)
        for key in (
            "num_applicants_main",
            "bc_reconciliation_num_applicants",
            "num_applicants",
            "num_applicants_proxy",
            "num_rows_applicant_geo_normalized",
            "num_rows_applicant_geo_raw",
        )
    ]
    positive = [value for value in values if value > 0]
    return max(positive) if positive else 1


def applicant_feature_rows(itemid: str) -> list[dict[str, Any]]:
    rows = load_reference_feature_index(str(REFERENCE_APPLICANT_PATH), True).get(itemid) or []
    return rows if isinstance(rows, list) else []


def individualized_applicant_row_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    generic_label_re = re.compile(r"^applicants?\s*\d+$", re.IGNORECASE)
    for row in rows:
        label = str(row.get("beneficiary_label") or "").strip()
        has_specific_label = bool(label and not generic_label_re.match(label))
        has_person_detail = any(str(row.get(key) or "").strip() not in {"", "unknown"} for key in ("birth_year", "sex", "age_group", "nationality"))
        if has_specific_label or has_person_detail:
            count += 1
    return count


def target_aggregation_structure(case: dict[str, Any]) -> dict[str, Any]:
    itemid = str(case.get("itemid") or "")
    applicant_rows = applicant_feature_rows(itemid)
    num_applicants = structural_applicant_count(case)
    app_count = application_count(case)
    applicant_geo_count = max(
        parse_int(case.get("num_rows_applicant_geo_normalized"), default=0),
        parse_int(case.get("num_rows_applicant_geo_raw"), default=0),
        len(applicant_rows),
    )
    appendix_row_count = applicant_geo_count if applicant_geo_count > 0 else None
    individualized_rows = individualized_applicant_row_count(applicant_rows)
    applicant_count_band = aggregation_count_band(num_applicants)
    application_count_band = aggregation_count_band(app_count)

    signal_reasons: list[str] = []
    case_name = str(case.get("case_name_clean") or case.get("case_name") or "")
    if re.search(r"\band others\b", case_name, re.IGNORECASE):
        signal_reasons.append("case_name_contains_and_others")
    if app_count > 1:
        signal_reasons.append("multiple_application_numbers")
    if num_applicants > 1:
        signal_reasons.append("multiple_applicants")
    if appendix_row_count and appendix_row_count > 1:
        signal_reasons.append("multiple_applicant_rows")
    if app_count >= 3 or num_applicants >= 6:
        signal_reasons.append("joined_or_repetitive_scale")

    if num_applicants > 100 or (appendix_row_count or 0) > 100:
        aggregation_class = "mass_joined_case_band"
    elif num_applicants >= 21 or (appendix_row_count or 0) >= 21:
        aggregation_class = "large_joined_case_band"
    elif num_applicants > 1 or app_count > 1:
        aggregation_class = "small_group_band"
    else:
        aggregation_class = "single_case_band"

    return {
        "num_applicants": num_applicants,
        "application_count": app_count,
        "appendix_row_count": appendix_row_count,
        "individualized_applicant_rows": individualized_rows,
        "applicant_count_band": applicant_count_band,
        "application_count_band": application_count_band,
        "joined_repetitive_application_signal": {
            "present": bool(signal_reasons),
            "reasons": signal_reasons,
        },
        "aggregation_class": aggregation_class,
        "source_policy": (
            "target structural metadata and applicant/facts rows only; target final awards, "
            "per-applicant award allocations, and target label-derived fields are not used"
        ),
    }


def recency_score(days: int) -> float:
    if days <= 365:
        return 1.0
    if days <= 365 * 3:
        return 0.85
    if days <= 365 * 5:
        return 0.7
    if days <= 365 * 10:
        return 0.45
    return 0.2


def applicant_similarity(distance: int, target_applicants: int) -> float:
    denominator = max(target_applicants, 1)
    return max(0.0, 1.0 - min(distance / denominator, 1.0))


def build_reference_candidate(
    case: dict[str, Any],
    row: dict[str, Any],
    label: dict[str, Any],
    target_articles: set[str],
    target_country: str,
    target_violation: str,
    target_applicants: int,
    t_date: datetime,
) -> dict[str, Any] | None:
    r_date = parse_date(row.get("judgementdate_iso") or row.get("judgementdate") or row.get("judgment_date"))
    if r_date is None or r_date >= t_date:
        return None

    y_amount = parse_float(label.get("y_amount_eur"))
    if y_amount is None:
        return None

    row_articles = set(normalize_articles(row.get("violated_articles")))
    overlap = len(target_articles & row_articles)
    if overlap <= 0:
        return None

    union = len(target_articles | row_articles) or 1
    row_country = str(row.get("country_alpha2") or row.get("respondent_country") or "").strip().lower()
    row_violation = str(row.get("violation_type") or "").strip().lower()
    row_applicants = parse_int(row.get("num_applicants_proxy") or row.get("num_applicants"), default=1)
    recency_days = (t_date - r_date).days
    exact_articles = row_articles == target_articles
    same_country = row_country == target_country
    same_violation = row_violation == target_violation
    app_distance = abs(row_applicants - target_applicants)
    same_applicant_band = applicant_band(row_applicants) == applicant_band(target_applicants)
    same_formation = str(row.get("decision_body_category") or row.get("doctypebranch") or "").strip().lower() == str(
        V2.first_non_empty(case, "decision_body_category", "doctypebranch") or ""
    ).strip().lower()
    same_importance = str(row.get("case_importance") or "").strip() == str(V2.first_non_empty(case, "case_importance", "importance") or "").strip()
    article_jaccard = overlap / union
    sim_score = (
        3.0 * int(exact_articles)
        + 2.0 * article_jaccard
        + 1.5 * int(same_country)
        + 1.2 * int(same_violation)
        + 0.8 * applicant_similarity(app_distance, target_applicants)
        + 0.6 * int(same_applicant_band)
        + 0.5 * recency_score(recency_days)
        + 0.3 * int(same_formation)
        + 0.2 * int(same_importance)
    )
    return {
        "row": row,
        "label": label,
        "y_amount": y_amount,
        "y_binary": parse_int(label.get("y_binary"), default=1 if y_amount > 0 else 0),
        "y_source": label.get("y_source"),
        "row_articles": row_articles,
        "overlap": overlap,
        "article_jaccard": article_jaccard,
        "exact_articles": exact_articles,
        "same_country": same_country,
        "same_violation": same_violation,
        "row_applicants": row_applicants,
        "app_distance": app_distance,
        "same_applicant_band": same_applicant_band,
        "same_formation": same_formation,
        "same_importance": same_importance,
        "recency_days": recency_days,
        "sim_score": sim_score,
    }


def apply_domain_filters(candidates: list[dict[str, Any]], target_applicants: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filters = [
        ("exact_article_set", lambda c: c["exact_articles"]),
        ("same_country", lambda c: c["same_country"]),
        ("same_violation_type", lambda c: c["same_violation"]),
        ("same_applicant_band", lambda c: c["same_applicant_band"]),
        ("same_formation", lambda c: c["same_formation"]),
        ("same_importance", lambda c: c["same_importance"]),
        ("within_5_years", lambda c: c["recency_days"] <= 365 * 5),
    ]
    active = list(candidates)
    trace: list[dict[str, Any]] = [
        {"filter": "hard_filters", "remaining": len(active), "applied": True}
    ]
    for name, predicate in filters:
        narrowed = [candidate for candidate in active if predicate(candidate)]
        applied = len(narrowed) >= MIN_FILTER_POOL or name in {"exact_article_set", "same_country"}
        if narrowed and applied:
            active = narrowed
        trace.append(
            {
                "filter": name,
                "matched": len(narrowed),
                "remaining": len(active),
                "applied": bool(narrowed and applied),
                "skipped_reason": None if narrowed and applied else "would_over_narrow_or_no_matches",
            }
        )
    return active, trace


def candidate_to_trace(candidate: dict[str, Any], case: dict[str, Any], rank: int) -> dict[str, Any]:
    row = candidate["row"]
    return {
        "rank": rank,
        "itemid": row.get("itemid"),
        "retrieved_judgment_date": row.get("judgementdate_iso") or row.get("judgementdate"),
        "target_judgment_date": V2.first_non_empty(case, "judgementdate", "judgment_date", "judgementdate_iso"),
        "temporal_filter_passed": True,
        "article_overlap_count": candidate["overlap"],
        "article_jaccard": round(float(candidate["article_jaccard"]), 6),
        "exact_article_set": candidate["exact_articles"],
        "same_country": candidate["same_country"],
        "same_violation_type": candidate["same_violation"],
        "applicant_count_distance": candidate["app_distance"],
        "same_applicant_band": candidate["same_applicant_band"],
        "same_formation": candidate["same_formation"],
        "same_importance": candidate["same_importance"],
        "recency_days": candidate["recency_days"],
        "similarity_score": round(float(candidate["sim_score"]), 6),
        "y_binary": candidate["y_binary"],
    }


def reference_feature_key_is_blocked(key: Any, react_mode: str = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT) -> bool:
    lowered = str(key or "").strip().lower()
    if not lowered:
        return True
    if react_mode == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        if lowered in CLAIM_AWARE_COURT_OUTCOME_FREE_BLOCKED_EXACT_KEYS:
            return True
        if any(fragment in lowered for fragment in CLAIM_AWARE_COURT_OUTCOME_FREE_BLOCKED_KEY_FRAGMENTS):
            return True
    else:
        if lowered in CLAIM_BLIND_BLOCKED_EXACT_KEYS:
            return True
        if any(fragment in lowered for fragment in CLAIM_BLIND_BLOCKED_KEY_FRAGMENTS):
            return True
    if any(fragment in lowered for fragment in REFERENCE_RAW_TEXT_KEY_FRAGMENTS):
        return True
    if any(fragment in lowered for fragment in REFERENCE_FEATURE_BLOCKED_KEY_FRAGMENTS):
        return True
    if lowered.startswith("flag_") and any(term in lowered for term in ("award", "per_app", "total")):
        return True
    return False


def value_catalog(value: Any, depth: int = 2) -> dict[str, Any]:
    if isinstance(value, dict):
        keys = [str(key) for key in value.keys()]
        node: dict[str, Any] = {
            "type": "object",
            "field_count": len(keys),
            "fields": keys,
        }
        if depth > 0:
            node["children"] = {
                str(key): value_catalog(child, depth - 1)
                for key, child in value.items()
            }
        return node
    if isinstance(value, list):
        node = {"type": "list", "length": len(value)}
        if value and depth > 0:
            node["item_schema"] = value_catalog(value[0], depth - 1)
        return node
    if isinstance(value, str):
        return {"type": "string", "char_count": len(value)}
    if value is None:
        return {"type": "null"}
    return {"type": type(value).__name__}


def flatten_for_query(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            flattened.update(flatten_for_query(child, (*prefix, str(key))))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for idx, child in enumerate(value):
            flattened.update(flatten_for_query(child, (*prefix, str(idx))))
        return flattened
    return {".".join(prefix): value}


def query_flat_values(
    flattened: dict[str, Any],
    path_prefixes: list[str],
    field_contains: list[str],
    max_chars: int,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    used_chars = 0
    normalized_prefixes = [
        str(prefix).replace("/", ".").strip().strip(".").lower()
        for prefix in path_prefixes
        if str(prefix).strip()
    ]
    normalized_contains = [term.strip().lower() for term in field_contains if str(term).strip()]
    for path, value in sorted(flattened.items()):
        lowered_path = path.lower()
        prefix_match = any(lowered_path == prefix or lowered_path.startswith(prefix + ".") for prefix in normalized_prefixes)
        contains_match = any(term in lowered_path for term in normalized_contains)
        if normalized_prefixes or normalized_contains:
            if not (prefix_match or contains_match):
                continue
        else:
            continue

        value_text = json.dumps(value, ensure_ascii=False)
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(value_text) > remaining and isinstance(value, str):
            selected[path] = value[: max(0, remaining - 20)] + "[TRUNCATED]"
            used_chars = max_chars
            break
        if len(value_text) > remaining:
            continue
        selected[path] = value
        used_chars += len(value_text)
    return selected


def expand_path_prefixes_for_sources(path_prefixes: list[str], requested_sources: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw_prefix in path_prefixes:
        prefix = str(raw_prefix).replace("/", ".").strip().strip(".")
        if not prefix:
            continue
        expanded.append(prefix)
        if any(prefix == source or prefix.startswith(source + ".") for source in requested_sources):
            continue
        for source in requested_sources:
            expanded.append(f"{source}.{prefix}")
    deduped: list[str] = []
    for prefix in expanded:
        if prefix not in deduped:
            deduped.append(prefix)
    return deduped


def normalize_target_query_selectors(
    case_inputs: dict[str, Any],
    sources: list[Any],
    path_prefixes: list[Any],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    allowed_sources = set(case_inputs.keys())
    requested_sources: list[str] = []
    normalized_prefixes = [str(prefix) for prefix in path_prefixes]
    corrections: list[dict[str, str]] = []
    source_aliases = {
        "inline_case_fields": "target_extraction_context",
        "sidecars": "target_extraction_context",
        "sidecar_sources": "target_extraction_context",
        "combined_input_text": "standard_prompting_input",
    }

    for source in sources:
        source_text = str(source)
        if source_text in allowed_sources:
            requested_sources.append(source_text)
            continue
        alias = source_aliases.get(source_text)
        if alias and alias in allowed_sources:
            requested_sources.append(alias)
            corrections.append({"requested_source": source_text, "used_source": alias})
    deduped_sources: list[str] = []
    for source in requested_sources:
        if source not in deduped_sources:
            deduped_sources.append(source)
    return deduped_sources, expand_path_prefixes_for_sources(normalized_prefixes, deduped_sources), corrections


def target_information_catalog(case_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        source: value_catalog(value, depth=2)
        for source, value in case_inputs.items()
    }


def target_query_template_for_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("react_mode") == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        return CLAIM_BLIND_TARGET_QUERY_TEMPLATE
    if state.get("react_mode") == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        return CLAIM_AWARE_TARGET_QUERY_TEMPLATE
    return TARGET_QUERY_TEMPLATE


def reference_feature_field_contains_for_state(state: dict[str, Any]) -> list[str]:
    if state.get("react_mode") == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        return list(CLAIM_BLIND_REFERENCE_FEATURE_FIELD_CONTAINS)
    if state.get("react_mode") == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        return list(CLAIM_AWARE_REFERENCE_FEATURE_FIELD_CONTAINS)
    return list(REFERENCE_FEATURE_DEFAULT_FIELD_CONTAINS)


def route_state_overview(route_state: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "mode",
        "react_mode",
        "violated_articles",
        "respondent_state",
        "judgment_year",
        "court_formation",
        "case_importance",
        "available_input_types",
        "extracted_hints",
        "unsupported_articles",
        "generic_fallback_active",
        "routing_warnings",
        "art6_limb",
        "art6_limb_source",
        "art6_limb_ambiguous",
        "input_policy_flags",
    }
    return {key: route_state.get(key) for key in allowed_keys if key in route_state}


def target_query_observation(state: dict[str, Any], action_input: dict[str, Any]) -> dict[str, Any]:
    sources = action_input.get("sources") or []
    path_prefixes = action_input.get("path_prefixes") or []
    field_contains = action_input.get("field_contains") or []
    if not isinstance(sources, list) or not isinstance(path_prefixes, list) or not isinstance(field_contains, list):
        return {"error": "sources, path_prefixes, and field_contains must be lists when supplied"}

    case_inputs = state["case_inputs"]
    allowed_sources = set(case_inputs.keys())
    requested_sources, normalized_path_prefixes, selector_corrections = normalize_target_query_selectors(
        case_inputs,
        sources,
        path_prefixes,
    )
    if not requested_sources:
        return {
            "query_policy": "lazy_target_context_catalog_first",
            "reason": "no_valid_sources_requested",
            "available_sources": sorted(allowed_sources),
            "target_information_catalog": target_information_catalog(case_inputs),
            "recommended_query_template": target_query_template_for_state(state),
        }

    max_chars = min(max(parse_int(action_input.get("max_chars"), default=20000), 1000), 80000)
    flattened: dict[str, Any] = {}
    for source in requested_sources:
        flattened.update(flatten_for_query(case_inputs[source], (source,)))
    selected = query_flat_values(
        flattened,
        normalized_path_prefixes,
        [str(term) for term in field_contains],
        max_chars,
    )
    if not selected:
        return {
            "query_policy": "lazy_target_context_catalog_first",
            "reason": "no_matching_fields_or_selector_missing",
            "requested_sources": requested_sources,
            "source_catalog": {source: value_catalog(case_inputs[source], depth=2) for source in requested_sources},
            "selector_examples": {
                "path_prefixes": [
                    "standard_prompting_input.combined_input_text",
                    "merits_side_violation_structure.violation_type",
                    "aggregation_structure.num_applicants",
                ],
                "field_contains": reference_feature_field_contains_for_state(state),
            },
            "recommended_query_template": target_query_template_for_state(state),
        }
    return {
        "query_policy": "lazy_target_context_catalog_first",
        "requested_sources": requested_sources,
        "path_prefixes": normalized_path_prefixes,
        "field_contains": field_contains,
        "selector_corrections": selector_corrections,
        "max_chars": max_chars,
        "selected_field_count": len(selected),
        "selected_fields": selected,
    }


def compact_reference_feature_row(
    row: dict[str, Any],
    react_mode: str = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in row.items():
        if reference_feature_key_is_blocked(key, react_mode):
            continue
        if value in (None, ""):
            continue
        compact[str(key)] = value
    return compact


def reference_feature_catalog(
    row: dict[str, Any],
    react_mode: str = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
) -> dict[str, Any]:
    return {
        source: {
            "field_count": len(value) if isinstance(value, dict) else len(value or []),
            "fields": list(value.keys()) if isinstance(value, dict) else value_catalog(value, depth=1),
        }
        for source, value in reference_feature_rows(row, react_mode).items()
    }


@lru_cache(maxsize=8)
def load_reference_feature_index(
    path_text: str,
    multi: bool,
    react_mode: str = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    index: dict[str, Any] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            itemid = str(row.get("itemid") or "").strip()
            if not itemid:
                continue
            compact = compact_reference_feature_row(row, react_mode)
            if not compact:
                continue
            if multi:
                index.setdefault(itemid, []).append(compact)
            else:
                index[itemid] = compact
    return index


def reference_feature_rows(
    row: dict[str, Any],
    react_mode: str = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
) -> dict[str, Any]:
    itemid = str(row.get("itemid") or "").strip()
    features: dict[str, Any] = {
        "split_case_features": compact_reference_feature_row(row, react_mode),
    }
    reasoning = load_reference_feature_index(str(REFERENCE_REASONING_PATH), False, react_mode).get(itemid)
    if reasoning:
        features["reasoning_layer_features"] = reasoning
    applicants = load_reference_feature_index(str(REFERENCE_APPLICANT_PATH), True, react_mode).get(itemid)
    if applicants:
        features["applicant_features"] = applicants
    return features


def candidate_to_reference(
    candidate: dict[str, Any],
    include_feature_rows: bool = False,
    react_mode: str = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
) -> dict[str, Any]:
    row = candidate["row"]
    feature_catalog = reference_feature_catalog(row, react_mode)
    reference = {
        "itemid": row.get("itemid"),
        "case_name_clean": row.get("case_name_clean"),
        "respondent_country": row.get("respondent_country"),
        "country_alpha2": row.get("country_alpha2"),
        "judgementdate": row.get("judgementdate_iso") or row.get("judgementdate"),
        "violated_articles": row.get("violated_articles"),
        "violation_type": row.get("violation_type"),
        "num_applicants_proxy": row.get("num_applicants_proxy"),
        "num_applicants_main": row.get("num_applicants_main"),
        "application_count": application_count(row),
        "applicant_count_band": aggregation_count_band(structural_applicant_count(row)),
        "application_count_band": aggregation_count_band(application_count(row)),
        "reference_non_pec_eur": candidate["y_amount"],
        "reference_y_binary": candidate["y_binary"],
        "reference_feature_policy": (
            "feature_rows_only_no_raw_judgment_text; "
            "court_outcome_zero_reason_and_label_provenance_columns_removed; "
            "reference_non_pec_eur_is_the_only_reference_award_anchor"
        ),
        "reference_feature_sources": {
            source: {"field_count": catalog.get("field_count")}
            for source, catalog in feature_catalog.items()
        },
    }
    if include_feature_rows:
        reference["reference_feature_catalog"] = feature_catalog
        reference["reference_feature_rows"] = reference_feature_rows(row, react_mode)
    return reference


def retrieve_train_references(
    case: dict[str, Any],
    train_rows: list[dict[str, Any]],
    train_labels: dict[str, dict[str, Any]],
    top_k: int,
    include_reference_features: bool = False,
    react_mode: str = CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
) -> dict[str, Any]:
    target_itemid = str(case.get("itemid") or "")
    target_articles = set(normalize_articles(case.get("violated_articles") or case.get("oracle_violated_articles") or case.get("violated_articles_text")))
    target_country = str(V2.first_non_empty(case, "country_alpha2", "respondent_country", "respondent") or "").strip().lower()
    target_violation = str(V2.first_non_empty(case, "violation_type") or "").strip().lower()
    target_applicants = parse_int(V2.first_non_empty(case, "num_applicants", "num_applicants_proxy"), default=1)
    t_date = target_date(case)

    candidates: list[dict[str, Any]] = []
    for row in train_rows:
        if str(row.get("itemid") or "") == target_itemid:
            continue
        if str(row.get("split") or "train") != "train":
            continue
        if t_date is None:
            continue
        label = train_labels.get(str(row.get("itemid") or ""))
        if not label:
            continue
        candidate = build_reference_candidate(
            case,
            row,
            label,
            target_articles,
            target_country,
            target_violation,
            target_applicants,
            t_date,
        )
        if candidate is not None:
            candidates.append(candidate)

    filtered, filter_trace = apply_domain_filters(candidates, target_applicants)
    filtered.sort(key=lambda candidate: candidate["sim_score"], reverse=True)

    positive = [candidate for candidate in filtered if candidate["y_binary"] == 1]
    zero = [candidate for candidate in filtered if candidate["y_binary"] == 0]
    selected_positive = positive[:top_k]
    selected_zero = zero[:top_k]
    balanced_seen: set[str] = set()
    balanced: list[dict[str, Any]] = []
    for pool in (selected_positive, selected_zero, filtered):
        for candidate in pool:
            itemid = str(candidate["row"].get("itemid") or "")
            if itemid in balanced_seen:
                continue
            balanced.append(candidate)
            balanced_seen.add(itemid)
            if len(balanced) >= top_k * 2:
                break
        if len(balanced) >= top_k * 2:
            break

    return {
        "retrieval_policy": "domain_multi_filter_then_similarity",
        "reference_case_payload_policy": (
            "retrieved cases expose tabular/extraction feature rows only; "
            "raw judgment text, court-outcome, zero-reason, and label-derived feature columns are removed; "
            "known train reference_non_pec_eur remains available as the few-shot anchor"
        ),
        "hard_filters": [
            "split=train",
            "retrieved_judgment_date < target_judgment_date",
            "itemid != target_itemid",
            "prediction_values/train label exists",
            "article_overlap > 0",
        ],
        "filter_trace": filter_trace,
        "candidate_counts": {
            "hard_filtered": len(candidates),
            "domain_filtered": len(filtered),
            "positive_available": len(positive),
            "zero_available": len(zero),
        },
        "positive_retrieval_trace": [
            candidate_to_trace(candidate, case, rank)
            for rank, candidate in enumerate(selected_positive, start=1)
        ],
        "zero_retrieval_trace": [
            candidate_to_trace(candidate, case, rank)
            for rank, candidate in enumerate(selected_zero, start=1)
        ],
        "balanced_retrieval_trace": [
            candidate_to_trace(candidate, case, rank)
            for rank, candidate in enumerate(balanced, start=1)
        ],
        "positive_reference_cases": [
            candidate_to_reference(candidate, include_feature_rows=include_reference_features, react_mode=react_mode)
            for candidate in selected_positive
        ],
        "zero_reference_cases": [
            candidate_to_reference(candidate, include_feature_rows=include_reference_features, react_mode=react_mode)
            for candidate in selected_zero
        ],
        "reference_cases": [
            candidate_to_reference(candidate, include_feature_rows=include_reference_features, react_mode=react_mode)
            for candidate in balanced
        ],
    }


def target_award_values(case: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "safe_non_pec_eur",
        "award_eur",
        "award_non_pec_eur_amount",
        "award_non_pec_original_amount",
        "raw_extractor_non_pec_eur",
    ):
        value = V2.first_non_empty(case, key)
        amount = parse_float(value)
        if amount is None or amount < 100:
            continue
        integer = int(amount) if amount.is_integer() else None
        values.add(str(value).strip())
        if integer is not None:
            values.update({str(integer), f"{integer:,}", f"{integer}.0"})
    return {value for value in values if value}


def build_reference_feature_bank(
    train_rows: list[dict[str, Any]],
    retrieval_result: dict[str, Any],
    react_mode: str,
) -> dict[str, dict[str, Any]]:
    row_index = {str(row.get("itemid") or ""): row for row in train_rows}
    itemids: set[str] = set()
    for key in ("reference_cases", "positive_reference_cases", "zero_reference_cases"):
        for reference in retrieval_result.get(key) or []:
            itemid = str(reference.get("itemid") or "")
            if itemid:
                itemids.add(itemid)
    return {
        itemid: reference_feature_rows(row_index[itemid], react_mode)
        for itemid in sorted(itemids)
        if itemid in row_index
    }


def reference_feature_query_observation(state: dict[str, Any], action_input: dict[str, Any]) -> dict[str, Any]:
    if state["inference_mode"] != "few_shot":
        return {"reference_features": {}, "reason": "zero_shot_mode"}

    requested_itemids = action_input.get("itemids") or []
    sources = action_input.get("sources") or []
    path_prefixes = action_input.get("path_prefixes") or []
    field_contains = action_input.get("field_contains") or []
    if not all(isinstance(value, list) for value in (requested_itemids, sources, path_prefixes, field_contains)):
        return {"error": "itemids, sources, path_prefixes, and field_contains must be lists when supplied"}
    if not sources:
        sources = list(REFERENCE_FEATURE_DEFAULT_SOURCES)
    if not path_prefixes and not field_contains:
        field_contains = reference_feature_field_contains_for_state(state)

    bank = state.get("reference_feature_bank") or {}
    allowed_itemids = set(bank.keys())
    itemids = [str(itemid) for itemid in requested_itemids if str(itemid) in allowed_itemids]
    if not itemids:
        return {
            "query_policy": "lazy_reference_feature_query",
            "reason": "no_valid_reference_itemids_requested",
            "available_reference_itemids": sorted(allowed_itemids),
            "reference_feature_catalog": {
                itemid: value_catalog(bank[itemid], depth=2)
                for itemid in sorted(allowed_itemids)
            },
        }

    max_chars = min(max(parse_int(action_input.get("max_chars"), default=20000), 1000), 80000)
    selected_by_itemid: dict[str, Any] = {}
    for itemid in itemids:
        feature_rows = bank[itemid]
        requested_sources = [str(source) for source in sources if str(source) in feature_rows]
        if not requested_sources:
            selected_by_itemid[itemid] = {
                "reason": "no_valid_sources_requested",
                "available_sources": sorted(feature_rows.keys()),
                "feature_catalog": value_catalog(feature_rows, depth=2),
            }
            continue
        normalized_path_prefixes = expand_path_prefixes_for_sources(
            [str(prefix) for prefix in path_prefixes],
            requested_sources,
        )
        flattened: dict[str, Any] = {}
        for source in requested_sources:
            flattened.update(flatten_for_query(feature_rows[source], (source,)))
        selected = query_flat_values(
            flattened,
            normalized_path_prefixes,
            [str(term) for term in field_contains],
            max_chars,
        )
        selected_by_itemid[itemid] = {
            "requested_sources": requested_sources,
            "selected_field_count": len(selected),
            "selected_fields": selected,
        }

    return {
        "query_policy": "lazy_reference_feature_query",
        "itemids": itemids,
        "sources": sources,
        "path_prefixes": path_prefixes,
        "field_contains": field_contains,
        "max_chars_per_itemid": max_chars,
        "reference_features": selected_by_itemid,
    }


def median_float(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2)


def percentile_float(values: list[float], percentile: float) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    rank = (len(clean) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(clean) - 1)
    weight = rank - lower
    return float(clean[lower] * (1 - weight) + clean[upper] * weight)


def compact_award_distribution(values: list[float]) -> dict[str, Any]:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return {
            "n": 0,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "n": len(clean),
        "median": median_float(clean),
        "p75": percentile_float(clean, 0.75),
        "p90": percentile_float(clean, 0.90),
        "max": float(max(clean)),
    }


def aggregation_train_records(
    case: dict[str, Any],
    train_rows: list[dict[str, Any]],
    train_labels: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    target_itemid = str(case.get("itemid") or "")
    t_date = target_date(case)
    records: list[dict[str, Any]] = []
    for row in train_rows:
        itemid = str(row.get("itemid") or "")
        if not itemid or itemid == target_itemid:
            continue
        if str(row.get("split") or "train") != "train":
            continue
        if t_date is not None:
            r_date = parse_date(row.get("judgementdate_iso") or row.get("judgementdate") or row.get("judgment_date"))
            if r_date is None or r_date >= t_date:
                continue
        label = train_labels.get(itemid)
        if not label:
            continue
        amount = parse_float(label.get("y_amount_eur"))
        if amount is None:
            continue
        num_applicants = structural_applicant_count(row)
        app_count = application_count(row)
        records.append(
            {
                "row": row,
                "itemid": itemid,
                "amount": float(amount),
                "positive": float(amount) > 0,
                "num_applicants": num_applicants,
                "application_count": app_count,
                "applicant_count_band": aggregation_count_band(num_applicants),
                "application_count_band": aggregation_count_band(app_count),
                "country_alpha2": str(row.get("country_alpha2") or row.get("respondent_country") or "").strip().lower(),
                "articles": set(normalize_articles(row.get("violated_articles") or row.get("oracle_violated_articles") or row.get("violated_articles_text"))),
                "judgementdate": row.get("judgementdate_iso") or row.get("judgementdate"),
            }
        )
    return records


def aggregation_prior_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    amounts = [record["amount"] for record in records]
    positive_amounts = [record["amount"] for record in records if record["positive"]]
    return {
        "sample_count": len(records),
        "positive_count": len(positive_amounts),
        "zero_count": len(records) - len(positive_amounts),
        "zero_rate": (len(records) - len(positive_amounts)) / len(records) if records else None,
        "all_awards": compact_award_distribution(amounts),
        "positive_awards": compact_award_distribution(positive_amounts),
    }


def build_aggregation_priors(case: dict[str, Any], state: dict[str, Any], target_structure: dict[str, Any]) -> dict[str, Any]:
    train_rows = state.get("train_rows") or []
    train_labels = state.get("train_labels") or {}
    records = aggregation_train_records(case, train_rows, train_labels)
    target_applicant_band = target_structure["applicant_count_band"]
    target_application_band = target_structure["application_count_band"]
    target_country = str(V2.first_non_empty(case, "country_alpha2", "respondent_country", "respondent") or "").strip().lower()
    target_articles = set(normalize_articles(case.get("violated_articles") or case.get("oracle_violated_articles") or case.get("violated_articles_text")))

    groups = {
        "global_same_applicant_band": [
            record for record in records if record["applicant_count_band"] == target_applicant_band
        ],
        "same_applicant_and_application_band": [
            record
            for record in records
            if record["applicant_count_band"] == target_applicant_band
            and record["application_count_band"] == target_application_band
        ],
        "same_country_same_applicant_band": [
            record
            for record in records
            if record["applicant_count_band"] == target_applicant_band
            and target_country
            and record["country_alpha2"] == target_country
        ],
        "article_overlap_same_applicant_band": [
            record
            for record in records
            if record["applicant_count_band"] == target_applicant_band
            and bool(record["articles"] & target_articles)
        ],
        "article_country_same_applicant_band": [
            record
            for record in records
            if record["applicant_count_band"] == target_applicant_band
            and target_country
            and record["country_alpha2"] == target_country
            and bool(record["articles"] & target_articles)
        ],
    }
    prior_stats = {name: aggregation_prior_stats(group) for name, group in groups.items()}
    ordinary_single_case_prior = aggregation_prior_stats(
        [record for record in records if record["applicant_count_band"] == "1"]
    )

    support_minimum = 5
    selected_name = None
    for name in (
        "article_country_same_applicant_band",
        "same_country_same_applicant_band",
        "article_overlap_same_applicant_band",
        "same_applicant_and_application_band",
        "global_same_applicant_band",
    ):
        if prior_stats[name]["sample_count"] >= support_minimum:
            selected_name = name
            break
    if selected_name is None:
        selected_name = "global_same_applicant_band"

    selected_prior = prior_stats[selected_name]
    high_award_anchor = {
        "selected_prior_name": selected_name,
        "support_minimum": support_minimum,
        "selected_prior_positive_p75": selected_prior["positive_awards"]["p75"],
        "selected_prior_positive_p90": selected_prior["positive_awards"]["p90"],
        "ordinary_single_case_positive_p90": ordinary_single_case_prior["positive_awards"]["p90"],
        "anchor_policy": (
            "train-only same applicant-band priors; p75/p90 are computed from "
            "prediction_values/train labels and exclude the target itemid"
        ),
    }
    return {
        "source_policy": (
            "train-only prediction_values/train labels joined to train.csv; target itemid excluded; "
            "rows are also restricted to retrieved/train judgments before the target date when target date is available"
        ),
        "target_band_keys": {
            "applicant_count_band": target_applicant_band,
            "application_count_band": target_application_band,
            "country_alpha2": target_country,
            "violated_articles": sorted(target_articles),
        },
        "priors": prior_stats,
        "ordinary_single_case_prior": ordinary_single_case_prior,
        "high_award_p75_p90_anchor": high_award_anchor,
    }


def same_band_retrieved_references(state: dict[str, Any], target_structure: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    retrieval = state.get("retrieval_result") or {}
    target_applicant_band = target_structure["applicant_count_band"]
    target_application_band = target_structure["application_count_band"]
    case = state["case"]
    target_country = str(V2.first_non_empty(case, "country_alpha2", "respondent_country", "respondent") or "").strip().lower()
    target_articles = set(normalize_articles(case.get("violated_articles") or case.get("oracle_violated_articles") or case.get("violated_articles_text")))
    references = retrieval.get("reference_cases") or []
    same_band: list[dict[str, Any]] = []
    seen_itemids: set[str] = set()
    for reference in references:
        ref_applicant_count = structural_applicant_count(reference)
        ref_application_count = application_count(reference)
        ref_applicant_band = aggregation_count_band(ref_applicant_count)
        ref_application_band = aggregation_count_band(ref_application_count)
        if ref_applicant_band != target_applicant_band:
            continue
        same_band.append(
            {
                "itemid": reference.get("itemid"),
                "case_name_clean": reference.get("case_name_clean"),
                "respondent_country": reference.get("respondent_country"),
                "country_alpha2": reference.get("country_alpha2"),
                "judgementdate": reference.get("judgementdate"),
                "violated_articles": reference.get("violated_articles"),
                "num_applicants": ref_applicant_count,
                "application_count": ref_application_count,
                "applicant_count_band": ref_applicant_band,
                "application_count_band": ref_application_band,
                "same_application_count_band": ref_application_band == target_application_band,
                "reference_non_pec_eur": reference.get("reference_non_pec_eur"),
                "reference_y_binary": reference.get("reference_y_binary"),
                "selection_source": "base_retrieval_result",
            }
        )
        seen_itemids.add(str(reference.get("itemid") or ""))
        if len(same_band) >= limit:
            break

    if len(same_band) < limit:
        records = aggregation_train_records(case, state.get("train_rows") or [], state.get("train_labels") or {})
        supplemental = [
            record
            for record in records
            if record["applicant_count_band"] == target_applicant_band
            and record["itemid"] not in seen_itemids
        ]
        supplemental.sort(
            key=lambda record: (
                int(bool(target_country and record["country_alpha2"] == target_country)),
                int(bool(record["articles"] & target_articles)),
                int(record["application_count_band"] == target_application_band),
                record["amount"],
            ),
            reverse=True,
        )
        for record in supplemental:
            row = record["row"]
            same_band.append(
                {
                    "itemid": record["itemid"],
                    "case_name_clean": row.get("case_name_clean"),
                    "respondent_country": row.get("respondent_country"),
                    "country_alpha2": row.get("country_alpha2"),
                    "judgementdate": record.get("judgementdate"),
                    "violated_articles": row.get("violated_articles"),
                    "num_applicants": record["num_applicants"],
                    "application_count": record["application_count"],
                    "applicant_count_band": record["applicant_count_band"],
                    "application_count_band": record["application_count_band"],
                    "same_application_count_band": record["application_count_band"] == target_application_band,
                    "reference_non_pec_eur": record["amount"],
                    "reference_y_binary": 1 if record["positive"] else 0,
                    "selection_source": "aggregation_same_band_train_retrieval",
                }
            )
            if len(same_band) >= limit:
                break
    return same_band


def assess_aggregation_pattern_observation(state: dict[str, Any]) -> dict[str, Any]:
    case = state["case"]
    target_structure = target_aggregation_structure(case)
    priors = build_aggregation_priors(case, state, target_structure)
    same_band_refs = same_band_retrieved_references(state, target_structure)
    selected_anchor = priors["high_award_p75_p90_anchor"]

    scale_flags: list[str] = []
    if target_structure["aggregation_class"] in {"large_joined_case_band", "mass_joined_case_band"}:
        scale_flags.append("large_or_mass_joined_case_level_sum_possible")
    if target_structure["application_count"] > 1:
        scale_flags.append("multiple_application_numbers")
    if target_structure["joined_repetitive_application_signal"]["present"]:
        scale_flags.append("joined_or_repetitive_signal_present")
    selected_p90 = selected_anchor.get("selected_prior_positive_p90")
    ordinary_p90 = selected_anchor.get("ordinary_single_case_positive_p90")
    if selected_p90 is not None and ordinary_p90 is not None and float(selected_p90) > float(ordinary_p90) * 1.5:
        scale_flags.append("same_band_p90_substantially_above_single_case_p90")

    return {
        "assessment_policy": "controller_owned_aggregation_scale_assessment_before_final_predict",
        "target_structure": target_structure,
        "train_only_applicant_band_priors": priors,
        "same_band_retrieved_references": same_band_refs,
        "scale_flags": scale_flags,
        "high_award_p75_p90_anchor": selected_anchor,
        "final_calibration_instruction": (
            "Use this observation to choose between single_case_band, small_group_band, "
            "large_joined_case_band, and mass_joined_case_band. If the target is large "
            "or mass joined, explicitly compare ordinary priors with same-band p75/p90 "
            "and same-band references; do not cap the case-level prediction at an "
            "ordinary single-case band merely because ordinary references cluster there. "
            "Do not mechanically multiply a per-applicant amount."
        ),
    }


def selected_target_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    case_inputs = state.get("case_inputs") or {}
    if state.get("react_mode") in COURT_OUTCOME_FREE_REACT_MODES:
        flattened: dict[str, Any] = {}
        sources = [
            "metadata",
            "aggregation_structure",
            "merits_side_violation_structure",
            "external_factors",
        ]
        if state.get("react_mode") == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
            sources.insert(1, "financial_request_structure")
        for source in sources:
            if source in case_inputs:
                flattened.update(flatten_for_query(case_inputs[source], (source,)))
        field_contains = (
            CLAIM_AWARE_TARGET_QUERY_TEMPLATE["field_contains"]
            if state.get("react_mode") == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT
            else CLAIM_BLIND_TARGET_QUERY_TEMPLATE["field_contains"]
        )
        return query_flat_values(
            flattened,
            [],
            field_contains,
            20000,
        )
    target_context = case_inputs.get("target_extraction_context") or {}
    flattened = flatten_for_query(target_context, ("target_extraction_context",))
    selected = query_flat_values(
        flattened,
        [],
        [
            "claim_non_pec",
            "claim_head",
            "violated_articles",
            "respondent_country",
            "country_alpha2",
            "judgment_year",
            "judgementdate",
            "violation_type",
            "num_applicants",
            "case_importance",
            "decision_body",
            "gdp_per_capita",
            "exclusion",
            "include_reason",
            "sufficient",
            "dismissed_reason",
        ],
        20000,
    )
    return selected


def first_snapshot_value(snapshot: dict[str, Any], *contains_terms: str) -> Any:
    lowered_terms = [term.lower() for term in contains_terms]
    for path, value in snapshot.items():
        lowered_path = path.lower()
        if all(term in lowered_path for term in lowered_terms) and value not in (None, ""):
            return value
    return None


def target_claim_cap_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    eur_amount = first_snapshot_value(snapshot, "claim_non_pec_eur_court_stated")
    eur_cap = parse_float(eur_amount)
    if eur_cap is not None and eur_cap > 0:
        return {
            "cap_eur": float(eur_cap),
            "source": "claim_non_pec_eur_court_stated",
            "cap_policy": "visible_non_pec_claim_amount_caps_final_award_in_relaxed_modes",
        }
    original_currency = str(first_snapshot_value(snapshot, "claim_non_pec_original_currency") or "").strip().upper()
    original_amount = first_snapshot_value(snapshot, "claim_non_pec_original_amount")
    original_cap = parse_float(original_amount)
    if original_currency in {"EUR", "EURO", "EUROS"} and original_cap is not None and original_cap > 0:
        return {
            "cap_eur": float(original_cap),
            "source": "claim_non_pec_original_amount_eur",
            "cap_policy": "visible_non_pec_claim_amount_caps_final_award_in_relaxed_modes",
        }
    return {
        "cap_eur": None,
        "source": None,
        "cap_policy": "no_numeric_non_pec_claim_cap_visible",
    }


def visible_zero_reason_signals(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    trigger_terms = (
        "no award",
        "finding sufficient",
        "violation sufficient",
        "satisfaction sufficient",
        "sufficient",
        "rule 60",
        "no_claim",
        "no claim",
        "not submitted",
        "structural_zero",
        "q_violation_sufficient",
        "good_zero",
    )
    reason_path_terms = (
        "exclusion",
        "include_reason",
        "satisfaction_sufficient",
        "dismissed_reason",
    )
    for path, value in snapshot.items():
        lowered_path = path.lower()
        if lowered_path.endswith(".redacted") or lowered_path.endswith(".reason"):
            continue
        if not any(term in lowered_path for term in reason_path_terms):
            continue
        value_text = str(value or "").strip()
        lowered_value = value_text.lower()
        path_leaf = lowered_path.rsplit(".", 1)[-1]
        if lowered_value in {"true", "yes"} and path_leaf.endswith("satisfaction_sufficient"):
            signals.append({"path": path, "value": value, "reason": "satisfaction_sufficient_true"})
        elif any(term in lowered_value for term in trigger_terms):
            signals.append({"path": path, "value": value, "reason": "zero_reason_text_match"})
    return signals


def reference_award_summary(reference_cases: list[dict[str, Any]]) -> dict[str, Any]:
    amounts: list[float] = []
    for ref in reference_cases:
        amount = parse_float(ref.get("reference_non_pec_eur"))
        if amount is not None:
            amounts.append(float(amount))
    return {
        "count": len(reference_cases),
        "itemids": [ref.get("itemid") for ref in reference_cases],
        "amounts": amounts,
        "median": median_float(amounts),
        "min": min(amounts) if amounts else None,
        "max": max(amounts) if amounts else None,
    }


def train_tabular_evidence_summary(
    state: dict[str, Any],
    empirical_obs: dict[str, Any] | None = None,
    retrieval_obs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    empirical = empirical_obs or state.get("empirical_calibration") or {}
    retrieval = retrieval_obs or state.get("retrieval_result") or {}
    article_calibrations = empirical.get("article_calibrations") or []
    zero_rates = [
        parse_float(((cal.get("selected_anchor") or cal.get("anchor") or {}).get("zero_rate")))
        for cal in article_calibrations
    ]
    zero_rates = [rate for rate in zero_rates if rate is not None]
    return {
        "evidence_policy": (
            "controller-built sanitized train/target tabular evidence; target financial-request, "
            "court-outcome, zero-explanation, and label-provenance fields are excluded"
        ),
        "train_prior_distribution": {
            "article_calibration_count": len(article_calibrations),
            "max_train_zero_rate": max(zero_rates) if zero_rates else None,
            "median_train_zero_rate": median_float(zero_rates) if zero_rates else None,
            "source_split": "train",
        },
        "train_reference_distribution": {
            "candidate_counts": retrieval.get("candidate_counts"),
            "all_references": reference_award_summary(retrieval.get("reference_cases") or []),
            "positive_award_references": reference_award_summary(retrieval.get("positive_reference_cases") or []),
            "zero_award_references": reference_award_summary(retrieval.get("zero_reference_cases") or []),
            "source_split": "train",
        },
        "model_instruction": (
            "Use this as distributional calibration only. The controller does not provide a separate "
            "zero/positive target assessment in court-outcome-free mode."
        ),
    }


def assess_zero_positive_evidence_observation(state: dict[str, Any]) -> dict[str, Any]:
    retrieval = state.get("retrieval_result") or {}
    positive_refs = retrieval.get("positive_reference_cases") or []
    zero_refs = retrieval.get("zero_reference_cases") or []
    target_snapshot = selected_target_snapshot(state)
    calibration = state.get("empirical_calibration") or {}
    article_calibrations = calibration.get("article_calibrations") or []
    zero_rates = [
        parse_float(((cal.get("selected_anchor") or cal.get("anchor") or {}).get("zero_rate")))
        for cal in article_calibrations
    ]
    zero_rates = [rate for rate in zero_rates if rate is not None]
    max_zero_rate = max(zero_rates) if zero_rates else None
    median_zero_rate = median_float(zero_rates) if zero_rates else None

    if state.get("react_mode") == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        return {
            "assessment_policy": "claim_blind_court_outcome_free_allowed_evidence_summary",
            "target_snapshot_policy": (
                "sanitized merits/metadata/applicant/external fields only; financial-request, "
                "court-outcome, zero-explanation, and label-provenance fields are blocked"
            ),
            "target_snapshot": target_snapshot,
            "train_prior_zero_rates": {
                "max_zero_rate": max_zero_rate,
                "median_zero_rate": median_zero_rate,
                "article_calibration_count": len(article_calibrations),
                "source_split": "train",
            },
            "reference_balance": {
                "candidate_counts": retrieval.get("candidate_counts"),
                "positive": reference_award_summary(positive_refs),
                "zero": reference_award_summary(zero_refs),
                "source_split": "train",
            },
            "evidence_notes": [
                "No target financial-request fields inspected.",
                "No target court-outcome fields inspected.",
                "No target zero-explanation or label-provenance fields inspected.",
                "No controller zero/positive recommendation is supplied in this mode.",
            ],
        }

    claim_state = first_snapshot_value(target_snapshot, "claim_non_pec_state")
    claim_amount = (
        first_snapshot_value(target_snapshot, "claim_non_pec_eur")
        or first_snapshot_value(target_snapshot, "claim_non_pec_original_amount")
    )
    claim_head_count = first_snapshot_value(target_snapshot, "claim_head_count")
    target_applicants = (
        first_snapshot_value(target_snapshot, "num_applicants_main")
        or first_snapshot_value(target_snapshot, "num_applicants_proxy")
        or first_snapshot_value(target_snapshot, "bc_reconciliation_num_applicants")
    )
    claim_cap = target_claim_cap_from_snapshot(target_snapshot)
    zero_reason_signals = visible_zero_reason_signals(target_snapshot)

    zero_indicators: list[str] = []
    positive_indicators: list[str] = []
    claim_state_text = str(claim_state or "").strip().lower()
    claim_head_count_int = parse_int(claim_head_count, default=-1)
    if zero_refs:
        zero_indicators.append("temporally_prior_zero_or_finding_sufficient_references_available")
    if max_zero_rate is not None and max_zero_rate >= 0.20:
        zero_indicators.append("train_prior_zero_rate_at_or_above_20_percent")
    if claim_state_text in {"unclear", "none", "no_claim", "not_claimed", "absent"}:
        zero_indicators.append(f"target_claim_state={claim_state_text}")
    if claim_head_count_int == 0:
        zero_indicators.append("target_claim_head_count_zero")
    if zero_reason_signals:
        zero_indicators.append("visible_target_zero_award_reason_signal")

    if positive_refs:
        positive_indicators.append("temporally_prior_positive_references_available")
    if reference_award_summary(positive_refs).get("median"):
        positive_indicators.append("positive_reference_award_anchor_available")
    if claim_amount not in (None, ""):
        positive_indicators.append("target_claim_amount_or_claim_currency_visible")
    if claim_cap.get("cap_eur") is not None:
        positive_indicators.append("numeric_non_pec_claim_cap_visible")
    if max_zero_rate is not None and max_zero_rate < 0.15:
        positive_indicators.append("train_prior_zero_rate_below_15_percent")

    if zero_reason_signals:
        recommendation = "zero_plausible"
    elif zero_indicators and positive_indicators:
        recommendation = "ambiguous"
    elif zero_indicators:
        recommendation = "zero_plausible"
    elif positive_indicators:
        recommendation = "positive_plausible"
    else:
        recommendation = "insufficient_evidence"

    return {
        "assessment_policy": "compare_zero_refs_positive_refs_and_train_priors_before_final_prediction",
        "target_snapshot_policy": "sanitized_visible_target_fields_only",
        "target_snapshot": target_snapshot,
        "claim_signals": {
            "claim_non_pec_state": claim_state,
            "claim_non_pec_amount_visible": claim_amount,
            "claim_head_count": claim_head_count,
            "num_applicants": target_applicants,
            "non_pec_claim_cap": claim_cap,
        },
        "visible_zero_reason_signals": zero_reason_signals,
        "train_prior_zero_rates": {
            "max_zero_rate": max_zero_rate,
            "median_zero_rate": median_zero_rate,
            "article_calibration_count": len(article_calibrations),
        },
        "reference_balance": {
            "candidate_counts": retrieval.get("candidate_counts"),
            "positive": reference_award_summary(positive_refs),
            "zero": reference_award_summary(zero_refs),
        },
        "zero_indicators": zero_indicators,
        "positive_indicators": positive_indicators,
        "recommendation": recommendation,
        "final_prediction_instruction": (
            "Use this assessment as calibration evidence only. The final action must still output "
            "one continuous award_eur number, including 0 when the zero evidence dominates."
        ),
    }


def audit_visible_state(value: Any, react_mode: str, case: dict[str, Any], path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if key_is_blocked(key, react_mode):
                findings.append({"path": ".".join(child_path), "reason": "blocked_key_visible"})
            findings.extend(audit_visible_state(child, react_mode, case, child_path))
        return findings
    if isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(audit_visible_state(child, react_mode, case, (*path, str(idx))))
        return findings
    if isinstance(value, str):
        lowered = value.lower()
        if react_mode == STRICT_REACT and any(term in lowered for term in ("article 41", "just satisfaction")):
            findings.append({"path": ".".join(path), "reason": "strict_blocked_article41_text_visible"})
        if react_mode in COURT_OUTCOME_FREE_REACT_MODES:
            if any(term in lowered for term in CLAIM_BLIND_TEXT_BLOCK_TERMS):
                findings.append({"path": ".".join(path), "reason": "court_outcome_free_blocked_text_visible"})
            joined_path = ".".join(path).lower()
            blocked_path_terms = ["article_41", "article41", "satisfaction", "dismissed", "include_reason", "exclusion", "y_source", "label_source"]
            if react_mode == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
                blocked_path_terms.extend(["claim", "claimed"])
            if any(term in joined_path for term in blocked_path_terms):
                findings.append({"path": ".".join(path), "reason": "court_outcome_free_blocked_path_text_visible"})
        if not path_contains_any(path, TARGET_AWARD_VALUE_EXEMPT_PATH_TERMS):
            target_values = target_award_values(case)
            for line in value.splitlines():
                if line_contains_target_award_value(line, target_values):
                    findings.append({"path": ".".join(path), "reason": "target_award_value_visible"})
                    break
    return findings


def build_state(
    kb_dir: Path,
    case: dict[str, Any],
    react_mode: str,
    inference_mode: str,
    top_k: int,
    train_csv: Path,
    train_label_csv: Path,
    target_context_policy: str,
    reference_context_policy: str,
) -> dict[str, Any]:
    kb_index = V2.load_kb_index(kb_dir)
    modules = V2.index_modules(kb_index)
    route_state = route_state_for_case(case, react_mode)
    calibration = resolve_calibration(kb_dir, route_state)
    case_inputs = build_case_inputs(case, react_mode)
    module_ids = selected_module_ids(case, route_state, react_mode)
    train_rows = load_train_rows(train_csv) if inference_mode == "few_shot" else []
    train_labels = load_train_labels(train_label_csv) if inference_mode == "few_shot" else {}
    retrieval_result: dict[str, Any] = {
        "retrieval_policy": "zero_shot_no_references",
        "reference_cases": [],
        "positive_reference_cases": [],
        "zero_reference_cases": [],
        "balanced_retrieval_trace": [],
    }
    if inference_mode == "few_shot":
        retrieval_result = retrieve_train_references(
            case,
            train_rows,
            train_labels,
            top_k,
            include_reference_features=reference_context_policy == "eager",
            react_mode=react_mode,
        )
    reference_feature_bank = (
        build_reference_feature_bank(train_rows, retrieval_result, react_mode)
        if inference_mode == "few_shot"
        else {}
    )

    visible_target_state = {
        "route_state": route_state,
        "case_inputs": case_inputs,
    }
    leakage_findings = audit_visible_state(visible_target_state, react_mode, case)
    return {
        "kb_dir": kb_dir,
        "kb_index": kb_index,
        "modules": modules,
        "case": case,
        "itemid": case.get("itemid"),
        "react_mode": react_mode,
        "inference_mode": inference_mode,
        "top_k": top_k,
        "target_context_policy": target_context_policy,
        "reference_context_policy": reference_context_policy,
        "route_state": route_state,
        "case_inputs": case_inputs,
        "selected_module_ids": module_ids,
        "empirical_calibration": calibration,
        "train_rows": train_rows,
        "train_labels": train_labels,
        "retrieval_result": retrieval_result,
        "reference_feature_bank": reference_feature_bank,
        "retrieval_trace": retrieval_result.get("balanced_retrieval_trace", []),
        "reference_cases": retrieval_result.get("reference_cases", []),
        "leakage_audit": {
            "passed": not leakage_findings,
            "findings": leakage_findings,
            "checked_scope": "visible_target_state",
        },
    }


def module_observation(state: dict[str, Any], module_ids: list[str], include_text: bool) -> dict[str, Any]:
    records = []
    for module_id in module_ids:
        record = load_module_record(state["kb_dir"], state["modules"], module_id, state["react_mode"])
        if include_text:
            records.append(record)
        else:
            records.append(
                {
                    "module_id": record["module_id"],
                    "module_type": record.get("module_type"),
                    "leakage_tier": record.get("leakage_tier"),
                    "chars": len(record["text"]),
                }
            )
    return {"modules": records}


def execute_action(state: dict[str, Any], action_object: dict[str, Any], include_module_text: bool = True) -> dict[str, Any]:
    action = action_object.get("action")
    action_input = action_object.get("action_input") or {}
    allowed_actions = allowed_actions_for_state(state)
    if action not in allowed_actions:
        return {
            "error": f"Unsupported action in {state.get('react_mode')}: {action}",
            "allowed_actions": allowed_actions,
        }

    if action == "inspect_case":
        if state.get("target_context_policy") == "lazy":
            return {
                "react_mode": state["react_mode"],
                "inference_mode": state["inference_mode"],
                "target_context_policy": "lazy_catalog_first",
                "route_state_overview": route_state_overview(state["route_state"]),
                "target_information_catalog": target_information_catalog(state["case_inputs"]),
                "recommended_target_query": target_query_template_for_state(state),
                "recommended_reference_feature_query": {
                    "sources": REFERENCE_FEATURE_DEFAULT_SOURCES,
                    "field_contains": reference_feature_field_contains_for_state(state),
                    "max_chars": 20000,
                },
                "next_step": (
                    "Use query_target_information with recommended_target_query. "
                    "Use load_relevant_modules instead of separate search_modules/load_module unless you need a custom module subset."
                ),
            }
        return {
            "react_mode": state["react_mode"],
            "inference_mode": state["inference_mode"],
            "route_state": state["route_state"],
            "case_inputs": state["case_inputs"],
        }
    if action == "query_target_information":
        return target_query_observation(state, action_input)
    if action == "search_modules":
        return {
            "selected_module_ids": state["selected_module_ids"],
            "selection_policy": "deterministic_controller_routing",
        }
    if action == "load_module":
        requested = action_input.get("module_ids") or state["selected_module_ids"]
        if not isinstance(requested, list):
            return {"error": "action_input.module_ids must be a list when supplied"}
        allowed = set(state["selected_module_ids"])
        blocked = [module_id for module_id in requested if module_id not in allowed]
        if blocked:
            return {"error": "requested_modules_not_selected", "blocked_module_ids": blocked}
        return module_observation(state, [str(module_id) for module_id in requested], include_text=include_module_text)
    if action == "load_relevant_modules":
        requested = action_input.get("module_ids")
        if requested is None:
            requested = [module_id for module_id in state["selected_module_ids"] if module_id != ACTION_PROTOCOL_ID]
        if not isinstance(requested, list):
            return {"error": "action_input.module_ids must be a list when supplied"}
        allowed = set(state["selected_module_ids"])
        blocked = [module_id for module_id in requested if module_id not in allowed or module_id == ACTION_PROTOCOL_ID]
        if blocked:
            return {"error": "requested_modules_not_selected_or_protocol_only", "blocked_module_ids": blocked}
        return {
            "selection_policy": "deterministic_controller_routing_loaded_in_one_action",
            **module_observation(state, [str(module_id) for module_id in requested], include_text=include_module_text),
        }
    if action == "resolve_empirical_priors":
        return state["empirical_calibration"]
    if action == "retrieve_train_references":
        if state["inference_mode"] != "few_shot":
            return {"retrieval_trace": [], "reference_cases": [], "reason": "zero_shot_mode"}
        result = dict(state["retrieval_result"])
        if state["react_mode"] in COURT_OUTCOME_FREE_REACT_MODES:
            result["reference_awards_policy"] = (
                "train_only_temporally_prior_reference_awards_allowed; "
                "claim/request reference fields are allowed only in claim-aware mode; "
                "label provenance, court-outcome fields, and zero-explanation fields are not exposed"
            )
        else:
            result["reference_awards_policy"] = (
                "train_only_temporally_prior_reference_awards_allowed; "
                "zero_reason references sourced from prediction_values/train.csv"
            )
        if state.get("reference_context_policy") == "lazy":
            result["next_step"] = (
                "Use query_reference_features with selected reference itemids and field selectors "
                "to retrieve feature-row values only when needed."
            )
        return result
    if action == "query_reference_features":
        return reference_feature_query_observation(state, action_input)
    if action == "assess_zero_positive_evidence":
        return assess_zero_positive_evidence_observation(state)
    if action == "assess_aggregation_pattern":
        return assess_aggregation_pattern_observation(state)
    if action == "leakage_check":
        return state["leakage_audit"]
    if action == "final_predict":
        leakage_audit = state["leakage_audit"]
        if not leakage_audit.get("passed"):
            return {
                "error": "controller_leakage_gate_failed",
                "accepted": False,
                "leakage_audit": leakage_audit,
            }
        award = parse_float(action_input.get("award_eur"))
        if award is None or award < 0:
            return {"error": "final_predict_requires_non_negative_numeric_award_eur"}
        cap = target_claim_cap_from_snapshot(selected_target_snapshot(state))
        uncapped_award = float(award)
        cap_allowed = state["react_mode"] in {
            AWARD_REDACTED_REACT,
            FULL_INFO_AWARD_BLIND_REACT,
            CLAIM_AWARE_COURT_OUTCOME_FREE_REACT,
        }
        cap_value = cap.get("cap_eur") if cap_allowed else None
        cap_applied = cap_value is not None and uncapped_award > float(cap_value)
        final_award = float(cap_value) if cap_applied else uncapped_award
        return {
            "prediction": {"award_eur": final_award},
            "accepted": True,
            "controller_leakage_check": leakage_audit,
            "claim_cap_check": {
                **cap,
                "uncapped_model_award_eur": uncapped_award,
                "cap_applied": cap_applied,
            },
        }
    raise AssertionError(f"Unhandled action: {action}")


def trace_event(step: int, action_object: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step,
        "thought_summary": str(action_object.get("thought_summary") or ""),
        "action": action_object.get("action"),
        "action_input": action_object.get("action_input") or {},
        "observation": observation,
    }


def deterministic_dry_run_trace(state: dict[str, Any]) -> dict[str, Any]:
    planned = [
        {
            "thought_summary": "Inspect the target inputs allowed under the active ReAct mode.",
            "action": "inspect_case",
            "action_input": {},
        },
        {
            "thought_summary": "Query a focused subset of target fields instead of loading all target information.",
            "action": "query_target_information",
            "action_input": target_query_template_for_state(state),
        },
        {
            "thought_summary": "Load the controller-selected legal, routing, output, and policy modules in one action.",
            "action": "load_relevant_modules",
            "action_input": {},
        },
        {
            "thought_summary": "Resolve train-only empirical anchors for amount calibration.",
            "action": "resolve_empirical_priors",
            "action_input": {},
        },
        {
            "thought_summary": "Retrieve temporally valid train references only if this is a few-shot run.",
            "action": "retrieve_train_references",
            "action_input": {"top_k": state["top_k"]},
        },
        {
            "thought_summary": "Query feature rows for selected retrieved references using controller default feature sources.",
            "action": "query_reference_features",
            "action_input": {
                "itemids": [
                    str(ref.get("itemid"))
                    for ref in (state.get("retrieval_result", {}).get("reference_cases") or [])[: min(4, state["top_k"])]
                ],
                "field_contains": reference_feature_field_contains_for_state(state),
                "max_chars": 10000,
            },
        },
        {
            "thought_summary": "Assess whether the target is an ordinary single case or an aggregated joined/multi-applicant case.",
            "action": "assess_aggregation_pattern",
            "action_input": {},
        },
    ]
    if state.get("react_mode") not in COURT_OUTCOME_FREE_REACT_MODES:
        planned.insert(
            -1,
            {
                "thought_summary": "Compare zero/finding-sufficient evidence against positive reference evidence before final prediction.",
                "action": "assess_zero_positive_evidence",
                "action_input": {},
            },
        )
    events = []
    for idx, action_object in enumerate(planned, start=1):
        observation = execute_action(state, action_object, include_module_text=False)
        events.append(trace_event(idx, action_object, observation))

    return {
        "itemid": state["itemid"],
        "react_mode": state["react_mode"],
        "inference_mode": state["inference_mode"],
        "events": events,
        "leakage_audit": state["leakage_audit"],
        "prediction": None,
        "dry_run": True,
        "final_action_template": {
            "thought_summary": "Short calibration summary. The controller will run the leakage gate automatically.",
            "action": "final_predict",
            "action_input": {"award_eur": 0.0},
        },
    }


COMPACT_REFERENCE_KEYS = (
    "itemid",
    "case_name_clean",
    "respondent_country",
    "country_alpha2",
    "judgementdate",
    "judgment_year",
    "violated_articles",
    "violation_type",
    "decision_body_category",
    "case_importance",
    "num_applicants_main",
    "num_applicants_proxy",
    "bc_reconciliation_num_applicants",
    "application_count",
    "reference_non_pec_eur",
    "reference_y_binary",
    "similarity_score",
    "domain_score",
    "retrieval_score",
)


def compact_leakage_label(react_mode: str) -> str:
    if react_mode == STRICT_REACT:
        return "strict_react_compact"
    if react_mode == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        return "claim-blind court-outcome-free train-only compact"
    if react_mode == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        return "claim-aware relaxed court-outcome-free zero-reason-free train-only compact"
    if react_mode == FULL_INFO_AWARD_BLIND_REACT:
        return (
            "full-info award-blind relaxed compact diagnostic; target final awards "
            "and target per-applicant awards blocked, relaxed structured claim/zero "
            "signals may be visible"
        )
    return "award-redacted relaxed compact diagnostic"


def compact_reference_case(reference: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in COMPACT_REFERENCE_KEYS:
        value = reference.get(key)
        if value not in (None, ""):
            compact[key] = value
    if "application_count" not in compact:
        compact["application_count"] = application_count(reference)
    if "num_applicants_main" not in compact and "num_applicants_proxy" not in compact:
        compact["num_applicants_proxy"] = structural_applicant_count(reference)
    return compact


def compact_reference_cases(references: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [compact_reference_case(reference) for reference in references[:limit]]


def observation_without_keys(observation: dict[str, Any], blocked_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in observation.items() if key not in blocked_keys}


def compact_reference_feature_observation(observation: dict[str, Any]) -> dict[str, Any]:
    if not observation:
        return {}
    if "reference_features" in observation:
        return {
            "query_policy": observation.get("query_policy"),
            "itemids": observation.get("itemids"),
            "sources": observation.get("sources"),
            "field_contains": observation.get("field_contains"),
            "reference_features": observation.get("reference_features"),
        }
    return {
        "query_policy": observation.get("query_policy"),
        "reason": observation.get("reason"),
        "available_reference_itemids": observation.get("available_reference_itemids"),
        "catalog_omitted": bool(observation.get("reference_feature_catalog")),
    }


def compact_packet_from_observations(
    state: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    packet_source: str,
    source_trace: str | None = None,
    max_references: int | None = None,
) -> dict[str, Any]:
    max_refs = max_references or max(5, int(state.get("top_k") or 5))
    target_obs = observations.get("query_target_information") or {}
    module_obs = observations.get("load_relevant_modules") or {}
    empirical_obs = observations.get("resolve_empirical_priors") or {}
    retrieval_obs = observations.get("retrieve_train_references") or {}
    ref_feature_obs = observations.get("query_reference_features") or {}
    zero_obs = observations.get("assess_zero_positive_evidence") or {}
    aggregation_obs = observations.get("assess_aggregation_pattern") or {}

    if state["react_mode"] in COURT_OUTCOME_FREE_REACT_MODES:
        target_evidence = {
            "route_state_overview": route_state_overview(state["route_state"]),
            "selected_fields": target_obs.get("selected_fields"),
            "target_query_policy": target_obs.get("query_policy"),
            "tabular_packet_policy": (
                "sanitized target table fields only; claim/request fields are visible only in "
                "claim-aware mode, while court-outcome, zero-explanation, and label-provenance "
                "fields are blocked"
            ),
        }
        train_tabular_section = train_tabular_evidence_summary(state, empirical_obs, retrieval_obs)
        respect_claim_cap = state["react_mode"] == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT
    else:
        target_evidence = {
            "route_state_overview": route_state_overview(state["route_state"]),
            "selected_fields": target_obs.get("selected_fields"),
            "target_snapshot": zero_obs.get("target_snapshot"),
            "claim_signals": zero_obs.get("claim_signals"),
            "visible_zero_reason_signals": zero_obs.get("visible_zero_reason_signals"),
            "target_query_policy": target_obs.get("query_policy"),
        }
        zero_positive_section = observation_without_keys(
            zero_obs,
            {"target_snapshot", "claim_signals", "visible_zero_reason_signals"},
        )
        respect_claim_cap = state["react_mode"] != STRICT_REACT

    packet = {
        "schema_version": "react_compact_packet_v1",
        "itemid": state["itemid"],
        "react_mode": state["react_mode"],
        "inference_mode": state["inference_mode"],
        "top_k": state["top_k"],
        "packet_source": packet_source,
        "source_trace": source_trace,
        "leakage_label": compact_leakage_label(state["react_mode"]),
        "packet_policy": {
            "controller_owned_evidence_packet": True,
            "target_final_awards_excluded": True,
            "target_per_applicant_awards_excluded": True,
            "target_financial_request_fields_excluded": state["react_mode"] == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT,
            "target_court_outcome_fields_excluded": state["react_mode"] in COURT_OUTCOME_FREE_REACT_MODES,
            "target_label_provenance_excluded": True,
            "source_trace_final_predict_excluded": True,
            "old_model_prediction_excluded": True,
            "reference_awards_policy": retrieval_obs.get("reference_awards_policy"),
            "strict_baseline_notice": (
                "Only packets built under strict_react should be reported as strict. "
                "full_info_award_blind_react packets are relaxed diagnostic inputs."
            ),
        },
        "target_evidence": target_evidence,
        "module_summary": {
            "selection_policy": module_obs.get("selection_policy"),
            "modules": module_obs.get("modules"),
        },
        "empirical_priors": {
            "selection_policy": empirical_obs.get("selection_policy"),
            "article_calibrations": empirical_obs.get("article_calibrations"),
            "country_context": empirical_obs.get("country_context"),
            "distribution_tables": empirical_obs.get("distribution_tables"),
            "global_available": empirical_obs.get("global_available"),
            "weak_support_detected": empirical_obs.get("weak_support_detected"),
        },
        "retrieved_references": {
            "retrieval_policy": retrieval_obs.get("retrieval_policy"),
            "hard_filters": retrieval_obs.get("hard_filters"),
            "filter_trace": retrieval_obs.get("filter_trace"),
            "candidate_counts": retrieval_obs.get("candidate_counts"),
            "reference_cases": compact_reference_cases(retrieval_obs.get("reference_cases") or [], max_refs),
            "positive_reference_cases": compact_reference_cases(
                retrieval_obs.get("positive_reference_cases") or [], max_refs
            ),
            "zero_reference_cases": compact_reference_cases(retrieval_obs.get("zero_reference_cases") or [], max_refs),
            "positive_retrieval_trace": retrieval_obs.get("positive_retrieval_trace"),
            "zero_retrieval_trace": retrieval_obs.get("zero_retrieval_trace"),
            "reference_feature_query": compact_reference_feature_observation(ref_feature_obs),
        },
        "aggregation_assessment": aggregation_obs,
        "leakage_audit": state["leakage_audit"],
        "final_prediction_contract": {
            "task": "Predict one case-level non-pecuniary EUR amount.",
            "valid_zero": True,
            "do_not_mechanically_multiply_per_applicant_amounts": True,
            "respect_visible_non_pec_claim_cap_in_relaxed_modes": respect_claim_cap,
            "output_schema": COMPACT_FINAL_SCHEMA,
        },
    }
    if state["react_mode"] in COURT_OUTCOME_FREE_REACT_MODES:
        packet["train_tabular_evidence"] = train_tabular_section
    else:
        packet["zero_positive_assessment"] = zero_positive_section
    return packet


def compact_default_observations(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    references = (state.get("retrieval_result") or {}).get("reference_cases") or []
    reference_ids = [str(ref.get("itemid")) for ref in references[: max(5, int(state.get("top_k") or 5))] if ref.get("itemid")]
    observations = {
        "inspect_case": execute_action(
            state,
            {"action": "inspect_case", "action_input": {}, "thought_summary": ""},
            include_module_text=False,
        ),
        "query_target_information": execute_action(
            state,
            {"action": "query_target_information", "action_input": target_query_template_for_state(state), "thought_summary": ""},
            include_module_text=False,
        ),
        "load_relevant_modules": execute_action(
            state,
            {"action": "load_relevant_modules", "action_input": {}, "thought_summary": ""},
            include_module_text=False,
        ),
        "resolve_empirical_priors": execute_action(
            state,
            {"action": "resolve_empirical_priors", "action_input": {}, "thought_summary": ""},
            include_module_text=False,
        ),
        "retrieve_train_references": execute_action(
            state,
            {"action": "retrieve_train_references", "action_input": {"top_k": state["top_k"]}, "thought_summary": ""},
            include_module_text=False,
        ),
        "query_reference_features": execute_action(
            state,
            {
                "action": "query_reference_features",
                "action_input": {
                    "itemids": reference_ids,
                    "sources": REFERENCE_FEATURE_DEFAULT_SOURCES,
                    "field_contains": reference_feature_field_contains_for_state(state),
                    "max_chars": 16000,
                },
                "thought_summary": "",
            },
            include_module_text=False,
        ),
        "assess_aggregation_pattern": execute_action(
            state,
            {"action": "assess_aggregation_pattern", "action_input": {}, "thought_summary": ""},
            include_module_text=False,
        ),
    }
    if state.get("react_mode") not in COURT_OUTCOME_FREE_REACT_MODES:
        observations["assess_zero_positive_evidence"] = execute_action(
            state,
            {"action": "assess_zero_positive_evidence", "action_input": {}, "thought_summary": ""},
            include_module_text=False,
        )
    return observations


def build_compact_packet_from_state(state: dict[str, Any]) -> dict[str, Any]:
    return compact_packet_from_observations(
        state,
        compact_default_observations(state),
        packet_source="controller_offline_from_case_tables",
    )


def deterministic_compact_packet_trace(state: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "itemid": state["itemid"],
        "react_mode": state["react_mode"],
        "inference_mode": state["inference_mode"],
        "execution_mode": "compact_final",
        "compact_packet": packet,
        "leakage_audit": state["leakage_audit"],
        "prediction": None,
        "dry_run": True,
    }


def load_api_key(api_key_file: str | None, api_key_env: str) -> str:
    env_value = os.environ.get(api_key_env)
    if env_value:
        return env_value.strip()
    if api_key_file:
        text = Path(api_key_file).read_text(encoding="utf-8").strip()
        if text:
            return text.splitlines()[0].strip()
    raise RuntimeError(f"Missing API key. Set {api_key_env} or pass --api_key_file.")


def build_client(
    api_base: str,
    model: str,
    api_key_file: str | None,
    api_key_env: str,
    provider_json_schema: bool,
) -> Any:
    client_path_text = os.environ.get("ECTHR_NPD_OPENAI_COMPATIBLE_CLIENT", "").strip()
    client_path = Path(client_path_text) if client_path_text else None
    if client_path is None:
        raise RuntimeError(
            "Live provider client code is not included in the public release. "
            "Use --dry_run, or set ECTHR_NPD_OPENAI_COMPATIBLE_CLIENT to your own client module path."
        )
    if not client_path.exists():
        raise RuntimeError(
            "Live provider client code is not included in the public release. "
            "Use --dry_run, or set ECTHR_NPD_OPENAI_COMPATIBLE_CLIENT to your own client module path."
        )
    client_mod = load_module(
        "react_openai_compatible_client_runtime",
        client_path,
    )
    return client_mod.OpenAICompatibleClient(
        base_url=api_base,
        api_key=load_api_key(api_key_file, api_key_env),
        model=model,
        use_json_schema=provider_json_schema,
        default_temperature=0.0,
    )


def live_system_prompt(state: dict[str, Any]) -> str:
    protocol = load_module_record(state["kb_dir"], state["modules"], ACTION_PROTOCOL_ID, state["react_mode"])["text"]
    mandatory_actions = required_actions_before_final(state)
    return (
        "You are a bounded ReAct actor for ECtHR Article 41 non-pecuniary damages prediction.\n"
        "Return exactly one JSON Action Object per turn. Do not output markdown.\n"
        "The JSON Action Object has keys thought_summary, action, and action_input.\n"
        "For final_predict, action_input must be exactly {\"award_eur\": <non-negative number>}.\n"
        "Never call final_predict with an empty action_input.\n"
        "Do not reveal private chain-of-thought; use only concise thought_summary.\n"
        "The controller owns all search, retrieval, file access, and leakage checks.\n\n"
        f"Active react_mode: {state['react_mode']}\n"
        f"Active inference_mode: {state['inference_mode']}\n"
        f"Target context policy: {state.get('target_context_policy')}\n"
        f"Reference context policy: {state.get('reference_context_policy')}\n"
        f"Allowed actions: {', '.join(allowed_actions_for_state(state))}\n\n"
        "Information is available through controller actions, but large target and reference payloads are lazy-loaded. "
        "Use inspect_case to see catalogs, then query_target_information or query_reference_features with selectors.\n"
        "Prefer load_relevant_modules over separate search_modules/load_module for the default module set.\n"
            "Before final_predict, you must call these evidence-gathering actions at least once: "
            f"{', '.join(mandatory_actions)}.\n\n"
            "The controller will run leakage_check automatically when final_predict is requested; "
            "calling leakage_check manually is optional for debugging.\n\n"
        f"{calibration_rubric_for_state(state)}\n\n"
        + protocol
    )


def compact_trace_for_model(events: list[dict[str, Any]]) -> str:
    return json.dumps(events, ensure_ascii=False, indent=2)


def calibration_rubric_for_state(state: dict[str, Any]) -> str:
    if state.get("react_mode") == CLAIM_BLIND_COURT_OUTCOME_FREE_REACT:
        return CLAIM_BLIND_COURT_OUTCOME_FREE_RUBRIC
    if state.get("react_mode") == CLAIM_AWARE_COURT_OUTCOME_FREE_REACT:
        return CLAIM_AWARE_COURT_OUTCOME_FREE_RUBRIC
    return GENERAL_AWARD_CALIBRATION_RUBRIC


def observations_from_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for event in events:
        action = event.get("action")
        observation = event.get("observation")
        if not action or action == "final_predict" or not isinstance(observation, dict):
            continue
        observations.setdefault(str(action), observation)
        if action == "load_module" and "load_relevant_modules" not in observations:
            observations["load_relevant_modules"] = observation
    return observations


def compact_observation_for_memory(action: str | None, observation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    if action == "inspect_case":
        catalog = observation.get("target_information_catalog") or {}
        return {
            "react_mode": observation.get("react_mode"),
            "inference_mode": observation.get("inference_mode"),
            "target_context_policy": observation.get("target_context_policy"),
            "route_state_overview": observation.get("route_state_overview"),
            "target_information_catalog_summary": {
                "top_level_keys": sorted(catalog.keys()) if isinstance(catalog, dict) else [],
                "full_catalog_stored_in_controller_memory": bool(catalog),
            },
            "recommended_target_query": observation.get("recommended_target_query"),
            "recommended_reference_feature_query": observation.get("recommended_reference_feature_query"),
            "next_step": observation.get("next_step"),
        }
    if action == "query_target_information":
        return {
            "query_policy": observation.get("query_policy"),
            "requested_sources": observation.get("requested_sources"),
            "selected_field_count": observation.get("selected_field_count"),
            "selected_fields": observation.get("selected_fields"),
            "selector_corrections": observation.get("selector_corrections"),
        }
    if action in {"load_relevant_modules", "load_module"}:
        modules = observation.get("modules") or []
        return {
            "selection_policy": observation.get("selection_policy"),
            "modules": [
                {
                    "module_id": module.get("module_id"),
                    "module_type": module.get("module_type"),
                    "leakage_tier": module.get("leakage_tier"),
                    "chars": module.get("chars") or len(str(module.get("text") or "")),
                }
                for module in modules
                if isinstance(module, dict)
            ],
        }
    if action == "resolve_empirical_priors":
        return {
            "selection_policy": observation.get("selection_policy"),
            "article_calibrations": observation.get("article_calibrations"),
            "country_context": observation.get("country_context"),
            "distribution_tables": observation.get("distribution_tables"),
            "global_available": observation.get("global_available"),
            "weak_support_detected": observation.get("weak_support_detected"),
        }
    if action == "retrieve_train_references":
        top_k = 8
        return {
            "retrieval_policy": observation.get("retrieval_policy"),
            "hard_filters": observation.get("hard_filters"),
            "filter_trace": observation.get("filter_trace"),
            "candidate_counts": observation.get("candidate_counts"),
            "reference_cases": compact_reference_cases(observation.get("reference_cases") or [], top_k),
            "positive_reference_cases": compact_reference_cases(observation.get("positive_reference_cases") or [], top_k),
            "zero_reference_cases": compact_reference_cases(observation.get("zero_reference_cases") or [], top_k),
            "positive_retrieval_trace": observation.get("positive_retrieval_trace"),
            "zero_retrieval_trace": observation.get("zero_retrieval_trace"),
        }
    if action == "query_reference_features":
        return compact_reference_feature_observation(observation)
    if action == "assess_zero_positive_evidence":
        return observation_without_keys(
            observation,
            {"target_snapshot"},
        )
    if action == "assess_aggregation_pattern":
        return observation
    return observation


def react_compact_memory_user_prompt(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    completed_actions: set[str],
    last_event: dict[str, Any] | None,
) -> str:
    missing = missing_required_actions(state, completed_actions)
    memory_index = [
        {
            "step": event.get("step"),
            "action": event.get("action"),
            "observation_keys": sorted((event.get("observation") or {}).keys())
            if isinstance(event.get("observation"), dict)
            else [],
        }
        for event in events
    ]
    prompt_obj: dict[str, Any] = {
        "controller_context_mode": "react_compact_memory",
        "instruction": (
            "Return exactly one next JSON Action Object. The controller stores full observations, "
            "but this prompt only includes compact memory. Use missing_required_actions to choose "
            "the next evidence-gathering action. Use final_predict only when no required actions are missing."
        ),
        "itemid": state["itemid"],
        "react_mode": state["react_mode"],
        "inference_mode": state["inference_mode"],
        "completed_actions": sorted(completed_actions),
        "missing_required_actions": missing,
        "memory_index": memory_index,
        "allowed_actions": allowed_actions_for_state(state),
        "recommended_first_action": "inspect_case",
    }
    if last_event is not None:
        prompt_obj["last_action"] = last_event.get("action")
        prompt_obj["last_observation_compact"] = compact_observation_for_memory(
            str(last_event.get("action") or ""),
            last_event.get("observation") if isinstance(last_event.get("observation"), dict) else {},
        )
    if not missing:
        packet = compact_packet_from_observations(
            state,
            observations_from_events(events),
            packet_source="react_compact_memory_controller_final_context",
        )
        prompt_obj["final_ready_evidence_packet"] = packet
        prompt_obj["final_instruction"] = (
            "You may now call final_predict. If you do, set action_input.award_eur to one "
            "case-level non-pecuniary EUR amount using the final_ready_evidence_packet."
        )
    return json.dumps(prompt_obj, ensure_ascii=False, indent=2)


def chat_json_with_retry(
    client: Any,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            parsed, usage = client.chat_json(messages, schema, schema_name, temperature=temperature)
            return parsed, usage
        except Exception as exc:
            last_error = exc
            if attempt >= 3:
                break
            time.sleep(attempt)
    if last_error:
        raise last_error
    raise RuntimeError("chat_json_with_retry failed without an exception")


def compact_final_system_prompt(state: dict[str, Any]) -> str:
    claim_cap_line = (
        "If a visible non-pecuniary claim cap is present in relaxed modes, the award must not exceed it.\n"
        if state["react_mode"] in {AWARD_REDACTED_REACT, FULL_INFO_AWARD_BLIND_REACT, CLAIM_AWARE_COURT_OUTCOME_FREE_REACT}
        else ""
    )
    return (
        "You are predicting ECtHR Article 41 non-pecuniary damages from a controller-built compact evidence packet.\n"
        "Return exactly one JSON object matching the schema. Do not output markdown.\n"
        "Do not reveal private chain-of-thought; rationale_summary must be concise.\n"
        "Use only the compact packet. Do not infer or reconstruct target final awards or target per-applicant awards.\n"
        "Reference awards and train priors in the packet are allowed calibration evidence.\n"
        f"{claim_cap_line}"
        "Predict one case-level EUR amount; do not mechanically multiply a per-applicant amount.\n\n"
        f"Active react_mode: {state['react_mode']}\n"
        f"Leakage label: {compact_leakage_label(state['react_mode'])}\n\n"
        f"{calibration_rubric_for_state(state)}\n"
    )


def react_compact_memory_system_prompt(state: dict[str, Any]) -> str:
    return (
        "You are a bounded ReAct actor for ECtHR Article 41 non-pecuniary damages prediction.\n"
        "Return exactly one JSON Action Object per turn. Do not output markdown.\n"
        "Do not reveal private chain-of-thought; use only concise thought_summary.\n"
        "The controller owns search, retrieval, file access, memory, and leakage checks.\n"
        "This is react_compact_memory mode: you still choose the next action, but the controller "
        "does not replay full historical observations. Use compact memory, last observation, and "
        "the final evidence packet when it becomes available.\n\n"
        f"Active react_mode: {state['react_mode']}\n"
        f"Active inference_mode: {state['inference_mode']}\n"
        f"Allowed actions: {', '.join(allowed_actions_for_state(state))}\n"
        f"Required before final_predict: {', '.join(required_actions_before_final(state))}\n\n"
        f"{calibration_rubric_for_state(state)}\n"
    )


def run_compact_final_prediction(
    state: dict[str, Any],
    client: Any,
    packet: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": compact_final_system_prompt(state)},
        {
            "role": "user",
            "content": (
                "Compact evidence packet:\n"
                + json.dumps(packet, ensure_ascii=False, indent=2)
                + "\n\nReturn the final prediction JSON now."
            ),
        },
    ]
    model_output, usage = chat_json_with_retry(
        client,
        messages,
        COMPACT_FINAL_SCHEMA,
        "compact_final_prediction",
        temperature,
    )
    final_action = {
        "thought_summary": "Compact final-only prediction from controller evidence packet.",
        "action": "final_predict",
        "action_input": {"award_eur": model_output.get("award_eur")},
    }
    observation = execute_action(state, final_action, include_module_text=False)
    event = trace_event(1, final_action, observation)
    if usage:
        event["usage"] = usage
    return {
        "itemid": state["itemid"],
        "react_mode": state["react_mode"],
        "inference_mode": state["inference_mode"],
        "execution_mode": "compact_final",
        "compact_packet": packet,
        "compact_model_output": model_output,
        "events": [event],
        "leakage_audit": state["leakage_audit"],
        "prediction": observation.get("prediction") if observation.get("accepted") else None,
        "token_usage": {
            "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
            "completion_tokens": int((usage or {}).get("completion_tokens") or 0),
            "total_tokens": int((usage or {}).get("total_tokens") or 0),
        },
        "dry_run": False,
    }


def run_live_trace_compact_memory(
    state: dict[str, Any],
    client: Any,
    max_steps: int,
    temperature: float,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    prediction: dict[str, Any] | None = None
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    completed_actions: set[str] = set()
    system_prompt = react_compact_memory_system_prompt(state)

    for step in range(1, max_steps + 1):
        user_prompt = react_compact_memory_user_prompt(
            state,
            events,
            completed_actions,
            events[-1] if events else None,
        )
        action_object, usage = chat_json_with_retry(
            client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            action_schema_for_state(state),
            "react_action",
            temperature,
        )
        action = action_object.get("action")
        if action == "final_predict":
            missing = missing_required_actions(state, completed_actions)
            if missing:
                observation = {
                    "error": "mandatory_react_actions_missing_before_final_predict",
                    "missing_actions": missing,
                    "compact_memory_policy": "final_predict_blocked_until_required_actions_complete",
                }
            else:
                observation = execute_action(state, action_object, include_module_text=False)
        else:
            observation = execute_action(state, action_object, include_module_text=False)
            if action in allowed_actions_for_state(state) and not observation.get("error"):
                completed_actions.add(str(action))
        event = trace_event(step, action_object, observation)
        if usage:
            event["usage"] = usage
            for key in token_usage:
                token_usage[key] += int(usage.get(key) or 0)
        events.append(event)
        if action == "final_predict" and observation.get("accepted"):
            prediction = observation["prediction"]
            break

    return {
        "itemid": state["itemid"],
        "react_mode": state["react_mode"],
        "inference_mode": state["inference_mode"],
        "execution_mode": "react_compact_memory",
        "events": events,
        "leakage_audit": state["leakage_audit"],
        "prediction": prediction,
        "token_usage": token_usage,
        "dry_run": False,
        "compact_memory_policy": {
            "model_selects_actions": True,
            "full_observations_stored_in_trace": True,
            "prompt_replays_full_history": False,
            "prompt_uses_memory_index_last_observation_and_final_packet": True,
        },
    }


def required_actions_before_final(state: dict[str, Any]) -> list[str]:
    required = ["inspect_case"]
    if state.get("target_context_policy") == "lazy":
        required.append("query_target_information")
    required.extend(
        [
            "load_relevant_modules OR (search_modules AND load_module)",
            "resolve_empirical_priors",
        ]
    )
    if state["inference_mode"] == "few_shot":
        required.append("retrieve_train_references")
        if state.get("react_mode") not in COURT_OUTCOME_FREE_REACT_MODES:
            required.append("assess_zero_positive_evidence")
        required.append("assess_aggregation_pattern")
    return required


def missing_required_actions(state: dict[str, Any], completed_actions: set[str]) -> list[str]:
    missing: list[str] = []
    if "inspect_case" not in completed_actions:
        missing.append("inspect_case")
    if state.get("target_context_policy") == "lazy" and "query_target_information" not in completed_actions:
        missing.append("query_target_information")
    module_loaded = "load_relevant_modules" in completed_actions or (
        "search_modules" in completed_actions and "load_module" in completed_actions
    )
    if not module_loaded:
        missing.append("load_relevant_modules OR (search_modules AND load_module)")
    if "resolve_empirical_priors" not in completed_actions:
        missing.append("resolve_empirical_priors")
    if state["inference_mode"] == "few_shot":
        if "retrieve_train_references" not in completed_actions:
            missing.append("retrieve_train_references")
        if (
            state.get("react_mode") not in COURT_OUTCOME_FREE_REACT_MODES
            and "assess_zero_positive_evidence" not in completed_actions
        ):
            missing.append("assess_zero_positive_evidence")
        if "assess_aggregation_pattern" not in completed_actions:
            missing.append("assess_aggregation_pattern")
    return missing


def run_live_trace(state: dict[str, Any], client: Any, max_steps: int, temperature: float) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    prediction: dict[str, Any] | None = None
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    completed_actions: set[str] = set()
    mandatory_actions = required_actions_before_final(state)
    messages = [
        {"role": "system", "content": live_system_prompt(state)},
        {
            "role": "user",
            "content": (
                "Start the ReAct loop. Request the next controller action. "
                "A good first action is inspect_case."
            ),
        },
    ]

    for step in range(1, max_steps + 1):
        action_object, usage = chat_json_with_retry(
            client,
            messages,
            action_schema_for_state(state),
            "react_action",
            temperature,
        )
        action = action_object.get("action")
        if action == "final_predict":
            missing = missing_required_actions(state, completed_actions)
            if missing:
                observation = {
                    "error": "mandatory_react_actions_missing_before_final_predict",
                    "missing_actions": missing,
                    "required_actions_before_final": mandatory_actions,
                }
            else:
                observation = execute_action(state, action_object, include_module_text=True)
        else:
            observation = execute_action(state, action_object, include_module_text=True)
            if action in allowed_actions_for_state(state) and not observation.get("error"):
                completed_actions.add(str(action))
        event = trace_event(step, action_object, observation)
        if usage:
            event["usage"] = usage
            for key in token_usage:
                token_usage[key] += int(usage.get(key) or 0)
        events.append(event)
        messages.append({"role": "assistant", "content": json.dumps(action_object, ensure_ascii=False)})
        messages.append({"role": "user", "content": "Observation:\n" + json.dumps(observation, ensure_ascii=False, indent=2)})
        if action_object.get("action") == "final_predict" and observation.get("accepted"):
            prediction = observation["prediction"]
            break
        if action == "final_predict" and observation.get("error") == "final_predict_requires_non_negative_numeric_award_eur":
            next_instruction = (
                "Your previous final_predict was invalid because action_input.award_eur was missing "
                "or not a non-negative number. Return exactly one JSON Action Object. If you are ready "
                "to predict, use action=final_predict with action_input exactly "
                "{\"award_eur\": <non-negative number>}. Do not leave action_input empty."
            )
        else:
            next_instruction = (
                "Continue. Return exactly one next JSON Action Object. "
                "Use final_predict only when ready to output the numeric EUR prediction, and then set "
                "action_input exactly to {\"award_eur\": <non-negative number>}."
            )
        messages.append({"role": "user", "content": next_instruction})

    return {
        "itemid": state["itemid"],
        "react_mode": state["react_mode"],
        "inference_mode": state["inference_mode"],
        "events": events,
        "leakage_audit": state["leakage_audit"],
        "prediction": prediction,
        "token_usage": token_usage,
        "dry_run": False,
    }


def write_outputs(result: dict[str, Any], trace_out: str | None, prediction_out: str | None) -> None:
    if trace_out:
        Path(trace_out).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if prediction_out:
        prediction = result.get("prediction")
        if prediction is None:
            return
        Path(prediction_out).parent.mkdir(parents=True, exist_ok=True)
        Path(prediction_out).write_text(json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(result: dict[str, Any]) -> None:
    print("=" * 72)
    execution_mode = result.get("execution_mode") or "react"
    if execution_mode == "compact_final":
        print("KB v3 Compact Final Dry Run" if result.get("dry_run") else "KB v3 Compact Final Live Run")
    elif execution_mode == "react_compact_memory":
        print("KB v3 ReAct Compact Memory Dry Run" if result.get("dry_run") else "KB v3 ReAct Compact Memory Live Run")
    else:
        print("KB v3 ReAct Dry Run" if result.get("dry_run") else "KB v3 ReAct Live Run")
    print("=" * 72)
    print(f"itemid: {result.get('itemid')}")
    print(f"react_mode: {result.get('react_mode')}")
    print(f"inference_mode: {result.get('inference_mode')}")
    print(f"execution_mode: {execution_mode}")
    print(f"events: {len(result.get('events') or [])}")
    if result.get("compact_packet"):
        packet_text = json.dumps(result["compact_packet"], ensure_ascii=False)
        print(f"compact_packet_chars: {len(packet_text)}")
    audit = result.get("leakage_audit") or {}
    print(f"leakage_audit_passed: {audit.get('passed')}")
    if audit.get("findings"):
        print("leakage_findings:")
        print(json.dumps(audit["findings"], ensure_ascii=False, indent=2)[:2000])
    if result.get("prediction") is not None:
        print("prediction:")
        print(json.dumps(result["prediction"], ensure_ascii=False, indent=2))
        if result.get("token_usage"):
            print("token_usage:")
            print(json.dumps(result["token_usage"], ensure_ascii=False, indent=2))
    elif result.get("dry_run"):
        print("prediction: <not generated in dry_run>")


def main() -> None:
    args = parse_args()
    if args.live and args.dry_run:
        raise ValueError("Use either --live or --dry_run, not both.")
    if not args.live:
        args.dry_run = True

    case = load_case(Path(args.case_file), args.case_id)
    state = build_state(
        kb_dir=Path(args.kb_dir),
        case=case,
        react_mode=args.react_mode,
        inference_mode=args.inference_mode,
        top_k=args.top_k,
        train_csv=Path(args.train_csv),
        train_label_csv=Path(args.train_label_csv),
        target_context_policy=args.target_context_policy,
        reference_context_policy=args.reference_context_policy,
    )
    if args.execution_mode == "compact_final":
        if args.compact_packet_file:
            packet = json.loads(Path(args.compact_packet_file).read_text(encoding="utf-8"))
            packet_itemid = str(packet.get("itemid") or "")
            if packet_itemid and packet_itemid != str(state["itemid"]):
                raise ValueError(f"compact_packet_file itemid {packet_itemid} does not match case_id {state['itemid']}")
        else:
            packet = build_compact_packet_from_state(state)
        if args.compact_packet_out:
            Path(args.compact_packet_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.compact_packet_out).write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.live:
            client = build_client(
                args.api_base,
                args.model,
                args.api_key_file,
                args.api_key_env,
                args.provider_json_schema,
            )
            result = run_compact_final_prediction(state, client, packet, args.temperature)
        else:
            result = deterministic_compact_packet_trace(state, packet)
    elif args.execution_mode == "react_compact_memory" and args.live:
        client = build_client(
            args.api_base,
            args.model,
            args.api_key_file,
            args.api_key_env,
            args.provider_json_schema,
        )
        result = run_live_trace_compact_memory(state, client, args.max_steps, args.temperature)
    elif args.live:
        client = build_client(
            args.api_base,
            args.model,
            args.api_key_file,
            args.api_key_env,
            args.provider_json_schema,
        )
        result = run_live_trace(state, client, args.max_steps, args.temperature)
    else:
        result = deterministic_dry_run_trace(state)

    write_outputs(result, args.trace_out, args.prediction_out)
    print_summary(result)


if __name__ == "__main__":
    main()

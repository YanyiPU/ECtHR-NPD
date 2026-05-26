#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from build_extraction_layers import (
    build_profiled_sections,
    build_token_counter,
    flatten_node,
    load_json as load_json_file,
    make_case_record,
    make_core_row,
    merged_text,
    top_level_sections,
)
from case_store import CASE_STORE_DIR, UNSTRUCTURED_CASES, load_cases_by_itemid
from docx_lossless import format_pipeline_c_appendix_text
from openai_compatible_client import ApiCallError, OpenAICompatibleClient
from problem_log import (
    CATEGORY_API,
    CATEGORY_D_SPARSE,
    CATEGORY_EXCEPTION,
    CATEGORY_RECONCILIATION,
    CATEGORY_SCHEMA,
    ProblemLog,
)
from run_pipeline_b_extraction import run_one_case as run_pipeline_b_case
from run_pipeline_b_extraction import _build_deterministic_hints
from pipeline_c_backbone_deterministic import extract_article_41_precedents
from run_pipeline_c_backbone import (
    _build_deterministic_article_41_extraction,
    build_c_input_snapshot,
    run_one_case as run_pipeline_c_case,
)
from run_pipeline_e_extraction import (
    GUIDE as GUIDE_E,
    build_reasoning_validation,
    dedupe_preserve_order,
    normalize_reasoning_result,
    prepare_reasoning_context,
    run_one_case as run_pipeline_e_case,
)


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = EXTRACTION_ROOT.parent
STRUCTURED_ROOT = DATASET_ROOT / "structured"
CASES_CORE_JSON = STRUCTURED_ROOT / "cases_core.json"
OUTPUTS = EXTRACTION_ROOT / "outputs"
CASES_OUTPUT = OUTPUTS / "cases"
RUNS_ROOT = OUTPUTS / "runs" / "holistic"
SCHEMA_DIR = EXTRACTION_ROOT / "schemas"
PROMPT_DIR = EXTRACTION_ROOT / "prompts"

WRITE_LOCK = threading.Lock()
SCHEMA_B = SCHEMA_DIR / "pipeline_b_facts_procedure.schema.json"
SCHEMA_C = SCHEMA_DIR / "pipeline_c_backbone.schema.json"
SCHEMA_D = SCHEMA_DIR / "pipeline_d_legal_analysis.schema.json"
SCHEMA_E = SCHEMA_DIR / "pipeline_e_reasoning.schema.json"
PROMPT_B = PROMPT_DIR / "pipeline_b_system_prompt.md"
PROMPT_C = PROMPT_DIR / "pipeline_c_backbone_system_prompt.md"
PROMPT_D = PROMPT_DIR / "pipeline_d_system_prompt.md"
PROMPT_E = PROMPT_DIR / "pipeline_e_system_prompt.md"

EMPTY_LEGAL_ANALYSIS = {
    "violated_articles_analyzed": [],
    "legal_tests": [],
    "proportionality": {"discussed": False, "steps": [], "result": None},
    "margin_of_appreciation": {"referenced": False, "width": None, "domain": None},
    "nature_of_obligation": {"article": None, "obligation_type": None, "structural_failure": None},
    "subsidiarity": {"discussed": False, "domestic_remedies_exhausted": None, "subsidiarity_analysis_depth": None},
    "precedent_usage": {
        "citation_count": None,
        "grand_chamber_citations": [],
        "distinguished": False,
        "new_principle_established": None,
    },
    "reasoning_quality": {"reasoning_depth": None, "quantitative_reasoning": None},
}

GUIDE_D = {
    "itemid": "string",
    "legal_analysis": {
        "violated_articles_analyzed": ["article numbers as strings"],
        "legal_tests": [
            {
                "article": "string",
                "test_description": "string|null",
                "test_components": ["string"],
                "applied": "boolean",
            }
        ],
        "proportionality": {
            "discussed": "boolean",
            "steps": ["legality", "legitimate_aim", "necessary", "proportionality_strict", "unclear"],
            "result": ["satisfied", "failed", "not_applicable", "unclear", None],
        },
        "margin_of_appreciation": {
            "referenced": "boolean",
            "width": ["broad", "narrow", "very_narrow", "not_applicable", "unclear", None],
            "domain": "string|null",
        },
        "nature_of_obligation": {
            "article": "string|null",
            "obligation_type": ["negative", "positive", "procedural", "both", "unclear", None],
            "structural_failure": "boolean|null",
        },
        "subsidiarity": {
            "discussed": "boolean",
            "domestic_remedies_exhausted": "boolean|null",
            "subsidiarity_analysis_depth": ["extensive", "brief", "none", "unclear", None],
        },
        "precedent_usage": {
            "citation_count": "integer|null",
            "grand_chamber_citations": ["string"],
            "distinguished": "boolean",
            "new_principle_established": "boolean|null",
        },
        "reasoning_quality": {
            "reasoning_depth": ["extensive", "moderate", "minimal", "unclear", None],
            "quantitative_reasoning": "boolean|null",
        },
    },
}

# Function guide (what each def is responsible for)
# - parse_args: Parse CLI arguments for one holistic run.
# - load_json: Read and decode a JSON file from disk.
# - make_run_dir: Create run output directories and return the run path.
# - load_cases_core_lookup: Build an itemid -> core row lookup from cases_core.json.
# - load_completed_ids: Read completed itemids from prior run artifacts for resume mode.
# - prompt_messages_d: Build Pipeline D messages including schema and source evidence.
# - _coerce_string_list: Force unknown list-like values into a clean list[str].
# - normalize_legal_analysis_result: Normalize/clean raw D extraction into contract-compliant fields.
# - validate_legal_analysis_result: Validate legal analysis against schema-required constraints.
# - build_legal_analysis_validation: Build deterministic diagnostics for D output quality checks.
# - merge_legal_retry_feedback: Add targeted retry instructions after D validation failures.
# - run_pipeline_d_case: Execute D extraction with retries, normalization, and validation.
# - _is_legal_analysis_sparse: Detect low-information D outputs that should be downgraded/fallback.
# - _de_combined_system_prompt: Compose a single system prompt for combined D+E calls.
# - _prompt_messages_de_combined: Build messages for combined legal+reasoning extraction.
# - run_pipeline_de_combined_case: Run combined D/E extraction, split outputs, validate, and retry.
# - _build_scaffold: Build deterministic baseline layers before LLM stage execution.
# - _fallback_b_result: Produce safe B fallback using deterministic scaffold facts.
# - _fallback_c_result: Produce safe C fallback preserving deterministic compensation structure.
# - _fallback_d_result: Produce safe D fallback with empty legal-analysis template.
# - _fallback_e_result: Produce safe E fallback with minimal reasoning structure.
# - _usage_or_zero: Normalize usage payload to a stable token accounting shape.
# - _reconciliation_check: Cross-check B/C consistency (applicants, beneficiary mapping, and counts).
# - _record_stage_problem: Append structured stage-level failure diagnostics to ProblemLog.
# - run_one_case: End-to-end orchestrator for one itemid across scaffold + B + C + D/E.
# - _get_group: Infer split/group label for reporting from itemid metadata.
# - write_case_result: Persist pipeline_a metadata, per-pipeline outputs, merged artifact, and per-case log.
# - main: CLI entrypoint that dispatches case-level execution in parallel.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplified JIT extractor: deterministic scaffold + B facts + C compensation + hidden D/E law stage."
    )
    parser.add_argument("--itemids", nargs="+", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("EXTRACTION_CONCURRENCY", "20")))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_run_dir(run_name: str) -> Path:
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CASES_OUTPUT.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_cases_core_lookup() -> dict[str, dict[str, Any]]:
    if not CASES_CORE_JSON.exists():
        return {}
    rows = load_json_file(CASES_CORE_JSON)
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("itemid")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("itemid") or "").strip()
    }


def load_completed_ids(run_dir: Path) -> set[str]:
    seen: set[str] = set()
    for meta_path in run_dir.glob("*.meta.json"):
        try:
            payload = load_json(meta_path)
        except Exception:
            continue
        if payload.get("status") in {"success", "partial_success"}:
            itemid = str(payload.get("itemid") or "").strip()
            if itemid:
                seen.add(itemid)
    return seen


def _digest_text(value: Any) -> dict[str, Any]:
    text = str(value or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if not text:
        return {"present": False, "chars": 0, "paragraphs": 0}
    paragraphs = [part for part in re.split(r"\n{2,}", text) if part.strip()]
    return {
        "present": True,
        "chars": len(text),
        "paragraphs": len(paragraphs),
    }


def _digest_snippets(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {"count": 0, "chars_total": 0}
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            cleaned.append(text)
    return {
        "count": len(cleaned),
        "chars_total": sum(len(text) for text in cleaned),
    }


def _compact_candidate_tree(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, child in value.items():
            compact_child = _compact_candidate_tree(child)
            if compact_child in (None, "", [], {}):
                continue
            compacted[key] = compact_child
        return compacted
    if isinstance(value, list):
        compacted_list = []
        for child in value:
            compact_child = _compact_candidate_tree(child)
            if compact_child in (None, "", [], {}):
                continue
            compacted_list.append(compact_child)
        return compacted_list
    return value


def _build_b_input_digest(row: dict[str, Any]) -> dict[str, Any]:
    evidence = (row.get("facts_procedure") or {}).get("evidence_inputs") or {}
    return {
        "input_contract": [
            "introduction_text",
            "procedure_text",
            "facts_text",
            "appendix_table_text",
        ],
        "field_digests": {
            "introduction_text": _digest_text(evidence.get("introduction_text")),
            "procedure_text": _digest_text(evidence.get("procedure_text")),
            "facts_text": _digest_text(evidence.get("facts_text")),
            "appendix_table_text": _digest_text((row.get("claim_and_award_layer") or {}).get("cross_validation_inputs", {}).get("appendix_table_text")),
        },
    }


def _build_c_input_digest(row: dict[str, Any]) -> dict[str, Any]:
    claim_inputs = (row.get("claim_and_award_layer") or {}).get("cross_validation_inputs") or {}
    return {
        "input_contract": [
            "article_41_or_article_50_text",
            "operative_text",
            "appendix_table_text",
            "claim_narrative_text",
            "scattered_claim_snippets",
        ],
        "field_digests": {
            "article_41_text": _digest_text(claim_inputs.get("article_41_text")),
            "operative_text": _digest_text(claim_inputs.get("operative_text")),
            "appendix_table_text": _digest_text(claim_inputs.get("appendix_table_text")),
            "claim_narrative_text": _digest_text(claim_inputs.get("claim_narrative_text")),
            "scattered_claim_snippets": _digest_snippets(claim_inputs.get("scattered_claim_snippets")),
        },
    }


def _build_de_input_digest(row: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    evidence = (row.get("reasoning_layer") or {}).get("evidence_inputs") or {}
    return {
        "input_contract": [
            "law_and_relevant_law_text",
            "assessment_summary_fallback",
        ],
        "input_mode": prepared.get("input_mode"),
        "field_digests": {
            "law_and_relevant_law_text": _digest_text(prepared.get("law_text_sent")),
            "law_text_original": _digest_text(evidence.get("law_text_excluding_article_41")),
            "relevant_law_text": _digest_text(evidence.get("relevant_law_text")),
            "assessment_summary_text": _digest_text(evidence.get("assessment_summary_text")),
            "summary_intro_text": _digest_text(evidence.get("summary_intro_text")),
        },
        "validation_only_sources": {
            "article_41_text": _digest_text(prepared.get("article_41_text")),
        },
    }


def prompt_messages_d(
    system_prompt: str,
    schema: dict[str, Any],
    row: dict[str, Any],
    prepared: dict[str, Any],
    include_schema_in_payload: bool,
) -> list[dict[str, str]]:
    core = row["core_case"]
    evidence = (row["reasoning_layer"].get("evidence_inputs") or {})
    payload = {
        "case_identification": {
            "itemid": row["itemid"],
            "appno": row["appno"],
            "judgementdate": row["judgementdate"],
            "respondent_country": core.get("respondent_country"),
            "violated_articles": core.get("violated_articles"),
            "detailed_violations": core.get("detailed_violations"),
        },
        "task": {
            "pipeline": "L",
            "goal": "extract structured legal analysis from THE LAW",
            "schema_name": "PipelineDLegalAnalysis",
        },
        "evidence_inputs": {
            "law_and_relevant_law_text": prepared["law_text_sent"],
        },
        "deterministic_hints": {
            "input_mode": prepared["input_mode"],
            "law_text_original_chars": len(prepared["law_text_original"]),
            "law_text_sent_chars": len(prepared["law_text_sent"]),
            "anchor_violation_type": prepared["anchor"].get("violation_type"),
            "anchor_violation_subtype": prepared["anchor"].get("violation_subtype"),
            "violated_articles_hint": core.get("violated_articles"),
        },
        "output_guide": GUIDE_D,
    }
    if include_schema_in_payload:
        payload["output_schema"] = schema
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                out.append(text)
        return out
    return []


def normalize_legal_analysis_result(result: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(result))
    legal_analysis = normalized.get("legal_analysis") or {}
    template = json.loads(json.dumps(EMPTY_LEGAL_ANALYSIS))

    legal_analysis["violated_articles_analyzed"] = dedupe_preserve_order(
        _coerce_string_list(legal_analysis.get("violated_articles_analyzed"))
    )
    if not legal_analysis["violated_articles_analyzed"]:
        legal_analysis["violated_articles_analyzed"] = list((row["core_case"] or {}).get("violated_articles") or [])

    legal_tests = legal_analysis.get("legal_tests")
    legal_analysis["legal_tests"] = legal_tests if isinstance(legal_tests, list) else []

    for key in (
        "proportionality",
        "margin_of_appreciation",
        "nature_of_obligation",
        "subsidiarity",
        "precedent_usage",
        "reasoning_quality",
    ):
        current = legal_analysis.get(key)
        merged = template[key]
        if isinstance(current, dict):
            merged.update(current)
        legal_analysis[key] = merged

    legal_analysis["precedent_usage"]["grand_chamber_citations"] = dedupe_preserve_order(
        _coerce_string_list(legal_analysis["precedent_usage"].get("grand_chamber_citations"))
    )

    normalized["legal_analysis"] = legal_analysis
    return normalized


def validate_legal_analysis_result(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        import jsonschema  # type: ignore

        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(result):
            loc = ".".join(str(x) for x in err.absolute_path) or "<root>"
            errors.append(f"{loc}: {err.message}")
    except Exception:
        pass

    if "itemid" not in result:
        errors.append("missing itemid")
    if "legal_analysis" not in result:
        errors.append("missing legal_analysis")
    return errors


def build_legal_analysis_validation(result: dict[str, Any], prepared: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    hinted = set(_coerce_string_list((row["core_case"] or {}).get("violated_articles")))
    analyzed = set(_coerce_string_list((result.get("legal_analysis") or {}).get("violated_articles_analyzed")))
    legal_analysis = result.get("legal_analysis") or {}
    is_sparse = _is_legal_analysis_sparse(legal_analysis)
    return {
        "input_mode": prepared["input_mode"],
        "law_text_original_chars": len(prepared["law_text_original"]),
        "law_text_sent_chars": len(prepared["law_text_sent"]),
        "violated_articles_hint": sorted(hinted),
        "missing_hinted_articles": sorted(hinted - analyzed),
        "sparse_result": is_sparse,
        "flag_for_review": bool(is_sparse or (hinted - analyzed)),
    }


def merge_legal_retry_feedback(messages: list[dict[str, str]], errors: list[str]) -> list[dict[str, str]]:
    base_messages = messages[:2]
    return base_messages + [{
        "role": "user",
        "content": "Your previous JSON did not validate. Return a full replacement JSON object only. Follow this compact output guide exactly: "
        + json.dumps(GUIDE_D, ensure_ascii=False)
        + " Validation errors: "
        + json.dumps(errors, ensure_ascii=False),
    }]


def run_pipeline_d_case(
    itemid: str,
    row: dict[str, Any],
    client: OpenAICompatibleClient,
    system_prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Single-shot pipeline D execution.

    Returns one of:
    - {status: "success", result, usage, elapsed_seconds}
    - {status: "api_error", api_error, ...}
    - {status: "schema_validation", errors, raw_result, ...}
    """
    start = time.perf_counter()
    prepared = prepare_reasoning_context(row)
    messages = prompt_messages_d(system_prompt, schema, row, prepared, include_schema_in_payload=not client.use_json_schema)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        parsed, usage = client.chat_json(messages=messages, schema=schema, schema_name="pipeline_d_legal_analysis")
    except ApiCallError as exc:
        return {
            "status": "api_error",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": time.perf_counter() - start,
            "usage": usage_total,
            "api_error": exc.to_record(),
        }

    if usage:
        for key in usage_total:
            if isinstance(usage.get(key), int):
                usage_total[key] += usage[key]
    parsed["itemid"] = itemid
    parsed = normalize_legal_analysis_result(parsed, row)
    errors = validate_legal_analysis_result(parsed, schema)
    if errors:
        return {
            "status": "schema_validation",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": time.perf_counter() - start,
            "usage": usage_total,
            "errors": errors,
            "raw_result": parsed,
        }

    parsed["legal_analysis_validation"] = build_legal_analysis_validation(parsed, prepared, row)
    return {
        "status": "success",
        "itemid": itemid,
        "attempts": 1,
        "elapsed_seconds": time.perf_counter() - start,
        "usage": usage_total,
        "result": parsed,
    }


def _is_legal_analysis_sparse(legal_analysis: dict[str, Any]) -> bool:
    """A D result is 'sparse' when none of its main signals fired.

    Used to flag suspect D outputs for re-extraction in a 2nd run, even when
    the schema validation passed.
    """
    if not isinstance(legal_analysis, dict):
        return True
    legal_tests = legal_analysis.get("legal_tests")
    if isinstance(legal_tests, list) and legal_tests:
        return False
    proportionality = legal_analysis.get("proportionality") or {}
    if proportionality.get("discussed"):
        return False
    margin = legal_analysis.get("margin_of_appreciation") or {}
    if margin.get("referenced"):
        return False
    nature = legal_analysis.get("nature_of_obligation") or {}
    if nature.get("obligation_type") not in (None, "unclear"):
        return False
    subsidiarity = legal_analysis.get("subsidiarity") or {}
    if subsidiarity.get("discussed"):
        return False
    precedent = legal_analysis.get("precedent_usage") or {}
    citation_count = precedent.get("citation_count")
    if isinstance(citation_count, int) and citation_count > 0:
        return False
    if precedent.get("grand_chamber_citations"):
        return False
    return True


_DE_COMBINED_BRIDGE = (
    "\n\n---\n\n"
    "You are performing BOTH of the above tasks in a single response. Read the "
    "supplied evidence ONCE and produce one JSON object containing two top-level "
    "fields: `legal_analysis` (per the first instructions) and `reasoning_layer` "
    "(per the second instructions). Also include `itemid` at the top level. Do "
    "not mix the two — keep their fields strictly separate. Do not add any other "
    "top-level keys. Return only the JSON object.\n"
)


def _de_combined_system_prompt(prompt_d: str, prompt_e: str) -> str:
    return prompt_d.rstrip() + "\n\n---\n\n" + prompt_e.rstrip() + _DE_COMBINED_BRIDGE


def _de_combined_response_schema(schema_d: dict[str, Any], schema_e: dict[str, Any]) -> dict[str, Any]:
    """Build a strict root schema for the combined D+E response."""
    legal_analysis = ((schema_d.get("properties") or {}).get("legal_analysis")) or {}
    reasoning_layer = ((schema_e.get("properties") or {}).get("reasoning_layer")) or {}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["itemid", "legal_analysis", "reasoning_layer"],
        "properties": {
            "itemid": {"type": "string"},
            "legal_analysis": legal_analysis,
            "reasoning_layer": reasoning_layer,
        },
    }


def _prompt_messages_de_combined(
    system_prompt: str,
    schema_d: dict[str, Any],
    schema_e: dict[str, Any],
    row: dict[str, Any],
    prepared: dict[str, Any],
    include_schema_in_payload: bool,
) -> list[dict[str, str]]:
    core = row["core_case"]
    evidence = (row["reasoning_layer"].get("evidence_inputs") or {})
    payload = {
        "case_identification": {
            "itemid": row["itemid"],
            "appno": row["appno"],
            "judgementdate": row["judgementdate"],
            "respondent_country": core.get("respondent_country"),
            "violated_articles": core.get("violated_articles"),
            "detailed_violations": core.get("detailed_violations"),
        },
        "task": {
            "pipeline": "DE",
            "goal": "extract legal_analysis (D) AND reasoning_layer (E) in a single JSON",
            "schema_names": ["PipelineDLegalAnalysis", "PipelineEReasoningExtraction"],
        },
        "evidence_inputs": {
            "law_and_relevant_law_text": prepared["law_text_sent"],
        },
        "deterministic_hints": {
            "input_mode": prepared["input_mode"],
            "law_text_original_chars": len(prepared["law_text_original"]),
            "law_text_sent_chars": len(prepared["law_text_sent"]),
            "anchor_violation_type": prepared["anchor"].get("violation_type"),
            "anchor_violation_subtype": prepared["anchor"].get("violation_subtype"),
            "anchor_reasoning_factors": prepared["anchor"].get("reasoning_factors"),
            "violated_articles_hint": core.get("violated_articles"),
        },
        "output_guide": {
            "itemid": "string",
            "legal_analysis": GUIDE_D["legal_analysis"],
            "reasoning_layer": GUIDE_E["reasoning_layer"],
        },
    }
    if include_schema_in_payload:
        payload["output_schemas"] = {
            "legal_analysis": schema_d,
            "reasoning_layer": schema_e,
        }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def run_pipeline_de_combined_case(
    itemid: str,
    row: dict[str, Any],
    client: OpenAICompatibleClient,
    system_prompt_d: str,
    system_prompt_e: str,
    schema_d: dict[str, Any],
    schema_e: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Single LLM call that extracts both D legal_analysis and E reasoning_layer.

    Returns a tuple `(d_payload, e_payload)` mirroring what
    `run_pipeline_d_case` and `run_pipeline_e_case` would each return, so the
    orchestrator can treat the two halves uniformly. The shared LLM `usage` is
    attributed to `d_payload` only — `e_payload.usage` is zero — to avoid
    double-counting in the per-stage breakdown.
    """
    start = time.perf_counter()
    prepared = prepare_reasoning_context(row)
    system_prompt = _de_combined_system_prompt(system_prompt_d, system_prompt_e)
    messages = _prompt_messages_de_combined(
        system_prompt,
        schema_d,
        schema_e,
        row,
        prepared,
        include_schema_in_payload=not client.use_json_schema,
    )

    zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        parsed, usage = client.chat_json(
            messages=messages,
            schema=_de_combined_response_schema(schema_d, schema_e),
            schema_name="pipeline_de_combined",
        )
    except ApiCallError as exc:
        elapsed = time.perf_counter() - start
        api_payload = {
            "status": "api_error",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": elapsed,
            "usage": zero_usage,
            "api_error": exc.to_record(),
        }
        return api_payload, {**api_payload}

    combined_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if usage:
        for key in combined_usage:
            if isinstance(usage.get(key), int):
                combined_usage[key] += usage[key]
    elapsed = time.perf_counter() - start

    # Split into D and E halves and validate each independently against its
    # original schema. Either half can fail without dragging down the other.
    d_half = {"itemid": itemid, "legal_analysis": parsed.get("legal_analysis") or {}}
    e_half = {"itemid": itemid, "reasoning_layer": parsed.get("reasoning_layer") or {}}

    d_normalized = normalize_legal_analysis_result(d_half, row)
    d_errors = validate_legal_analysis_result(d_normalized, schema_d)
    if d_errors:
        d_payload = {
            "status": "schema_validation",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": elapsed,
            "usage": combined_usage,
            "errors": d_errors,
            "raw_result": d_normalized,
        }
    else:
        d_normalized["legal_analysis_validation"] = build_legal_analysis_validation(
            d_normalized, prepared, row
        )
        d_payload = {
            "status": "success",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": elapsed,
            "usage": combined_usage,
            "result": d_normalized,
        }

    e_normalized = normalize_reasoning_result(e_half, prepared)
    # E's standalone validate_result lives inside run_pipeline_e_extraction; we
    # call its public surface via the same path the standalone function uses.
    from run_pipeline_e_extraction import validate_result as _e_validate_result

    e_errors = _e_validate_result(e_normalized, schema_e)
    if e_errors:
        e_payload = {
            "status": "schema_validation",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": elapsed,
            "usage": zero_usage,  # already counted in d_payload
            "errors": e_errors,
            "raw_result": e_normalized,
        }
    else:
        e_normalized["itemid"] = itemid
        e_normalized["reasoning_validation"] = build_reasoning_validation(e_normalized, prepared)
        e_payload = {
            "status": "success",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": elapsed,
            "usage": zero_usage,  # already counted in d_payload
            "result": e_normalized,
        }

    return d_payload, e_payload


def _build_scaffold(source_row: dict[str, Any], core_lookup: dict[str, dict[str, Any]], count_tokens) -> dict[str, Any]:
    itemid = str(source_row.get("itemid") or "")
    section_key, sections = top_level_sections(source_row)
    section_map = {section.get("section_name"): section for section in sections if section.get("section_name")}

    profiled = build_profiled_sections(
        row=source_row,
        sections=sections,
        raw_law_section=section_map.get("law"),
        raw_conclusion_section=section_map.get("conclusion"),
    )
    introduction_text = profiled["introduction_text"]
    procedure_text = profiled["procedure_text"]
    facts_text = profiled["facts_text"]
    summary_intro_text = profiled["summary_intro_text"]
    assessment_summary_text = profiled["assessment_summary_text"]
    relevant_law_text = profiled["relevant_law_text"]
    reasoning_text = profiled["law_text_excluding_article_41"]
    article_41_text = profiled["article_41_text"]
    operative_text = profiled["operative_text"]
    article_41_method = profiled["article_41_method"]

    appendix_table_text = format_pipeline_c_appendix_text(source_row.get("docx_lossless"))
    full_document_text = merged_text([flatten_node(section) for section in sections])
    law_text_full = merged_text([relevant_law_text, reasoning_text, article_41_text])

    facts_input_text = merged_text([introduction_text, procedure_text, facts_text])
    compensation_input_text = merged_text([article_41_text, operative_text, appendix_table_text])
    pipeline_b_tokens = count_tokens(facts_input_text)
    pipeline_c_tokens = count_tokens(compensation_input_text)
    pipeline_d_tokens = count_tokens(reasoning_text)

    existing = core_lookup.get(itemid)
    core_row = make_core_row(source_row, existing, section_key, sections)
    scaffold = make_case_record(
        row=source_row,
        core_row=core_row,
        introduction_text=introduction_text,
        procedure_text=procedure_text,
        facts_text=facts_text,
        summary_intro_text=summary_intro_text,
        assessment_summary_text=assessment_summary_text,
        relevant_law_text=relevant_law_text,
        article_41_text=article_41_text,
        operative_text=operative_text,
        appendix_table_text=appendix_table_text,
        reasoning_text=reasoning_text,
        article_41_method=article_41_method,
        document_profile=profiled["document_profile"],
        pipeline_b_input_mode=profiled["pipeline_b_input_mode"],
        pipeline_e_input_mode=profiled["pipeline_e_input_mode"],
        normalized_top_level_section_names=profiled["normalized_top_level_section_names"],
        section_rows=profiled["section_rows"],
        normalized_text_lengths=profiled["normalized_text_lengths"],
        pipeline_b_tokens=pipeline_b_tokens,
        pipeline_c_tokens=pipeline_c_tokens,
        pipeline_d_tokens=pipeline_d_tokens,
        full_document_text=full_document_text,
        law_text_full=law_text_full,
    )
    scaffold["extraction_meta"]["scaffold_version"] = "jit_simplified_v2"
    scaffold["extraction_meta"]["case_store_path"] = str(CASE_STORE_DIR / f"{itemid}.json")
    return scaffold


def _fallback_b_result(itemid: str, scaffold: dict[str, Any]) -> dict[str, Any]:
    return {"itemid": itemid, "facts_procedure": json.loads(json.dumps(scaffold.get("facts_procedure") or {}))}


def _fallback_c_result(
    itemid: str,
    scaffold: dict[str, Any],
    client: OpenAICompatibleClient,
    prompt_c: str,
    schema_c: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = run_pipeline_c_case(itemid, scaffold, client, prompt_c, schema_c, max_retries=1, regex_only=True)
    return payload["result"], payload.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _fallback_d_result(itemid: str, row: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_reasoning_context(row)
    return {
        "itemid": itemid,
        "legal_analysis": normalize_legal_analysis_result({"legal_analysis": {}}, row)["legal_analysis"],
        "legal_analysis_validation": {
            "input_mode": prepared["input_mode"],
            "law_text_original_chars": len(prepared["law_text_original"]),
            "law_text_sent_chars": len(prepared["law_text_sent"]),
            "violated_articles_hint": sorted(_coerce_string_list((row["core_case"] or {}).get("violated_articles"))),
            "missing_hinted_articles": sorted(_coerce_string_list((row["core_case"] or {}).get("violated_articles"))),
        },
    }


def _fallback_e_result(itemid: str, row: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_reasoning_context(row)
    parsed = normalize_reasoning_result({"reasoning_layer": {}}, prepared)
    parsed["itemid"] = itemid
    parsed["reasoning_validation"] = build_reasoning_validation(parsed, prepared)
    return parsed


def _usage_or_zero(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


_BENEFICIARY_TITLE_RE = re.compile(r"\b(Mr|Ms|Mrs|Miss)\b\.?\s+", re.IGNORECASE)
_BENEFICIARY_SPACE_RE = re.compile(r"\s+")
_GROUP_BENEFICIARY_RE = re.compile(r"\b(each applicant|the applicants?|all applicants?|applicants|jointly|joint applicants?)\b", re.IGNORECASE)
_EXTERNAL_BENEFICIARY_RE = re.compile(
    r"\b(heir|heirs|estate|family|widow|widower|mother|father|son|daughter|parents|next of kin|relative|legal representative|representative)\b",
    re.IGNORECASE,
)
_ORDINAL_INDEX = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
}


def _normalize_beneficiary_label(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = _BENEFICIARY_TITLE_RE.sub("", text).strip()
    cleaned = cleaned.strip(" ,;:.")
    cleaned = _BENEFICIARY_SPACE_RE.sub(" ", cleaned)
    lowered = cleaned.casefold()
    return lowered or None


def _beneficiary_kind(label: Any) -> str:
    normalized = _normalize_beneficiary_label(label)
    if not normalized:
        return "unknown"
    if _GROUP_BENEFICIARY_RE.search(normalized):
        return "group"
    if _EXTERNAL_BENEFICIARY_RE.search(normalized):
        return "external"
    return "applicant"


def _index_from_label(label: Any) -> int | None:
    normalized = _normalize_beneficiary_label(label)
    if not normalized:
        return None
    ordinal_match = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+applicant\b", normalized)
    if ordinal_match:
        return _ORDINAL_INDEX.get(ordinal_match.group(1))
    numeric_match = re.search(r"\bapplicant\s*(\d+)\b", normalized)
    if numeric_match:
        return int(numeric_match.group(1))
    return None


def _build_applicant_label_maps(applicants: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    exact_map: dict[str, int] = {}
    surname_candidates: dict[str, list[int]] = {}
    for fallback_idx, applicant in enumerate(applicants, start=1):
        if not isinstance(applicant, dict):
            continue
        idx = applicant.get("applicant_index")
        if not isinstance(idx, int) or idx < 1:
            idx = fallback_idx
        normalized = _normalize_beneficiary_label(applicant.get("beneficiary_label"))
        if not normalized:
            continue
        exact_map.setdefault(normalized, idx)
        tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
        if tokens:
            surname_candidates.setdefault(tokens[-1], []).append(idx)
    surname_map = {token: ids[0] for token, ids in surname_candidates.items() if len(set(ids)) == 1}
    return exact_map, surname_map


def _resolve_row_index(
    row: dict[str, Any],
    num_applicants: int | None,
    exact_map: dict[str, int],
    surname_map: dict[str, int],
) -> int | None:
    idx = row.get("applicant_index")
    if isinstance(idx, float) and idx.is_integer():
        idx = int(idx)
    if isinstance(idx, int):
        if isinstance(num_applicants, int) and num_applicants > 0 and not (1 <= idx <= num_applicants):
            idx = None
        else:
            return idx
    elif idx is not None:
        idx = None

    label = row.get("beneficiary_label")
    label_index = _index_from_label(label)
    if isinstance(label_index, int):
        if not isinstance(num_applicants, int) or 1 <= label_index <= num_applicants:
            return label_index

    normalized = _normalize_beneficiary_label(label)
    if normalized and normalized in exact_map:
        return exact_map[normalized]
    if normalized:
        tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
        if tokens:
            surname = tokens[-1]
            if surname in surname_map:
                return surname_map[surname]
    return None


def _numeric_amount(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _group_label_indices(label: Any, num_applicants: int | None) -> set[int]:
    normalized = _normalize_beneficiary_label(label)
    if not normalized or not isinstance(num_applicants, int) or num_applicants <= 0:
        return set()
    if re.search(r"\b(all applicants?|the applicants?|applicants jointly|applicants|joint applicants?)\b", normalized):
        return set(range(1, num_applicants + 1))

    indices: set[int] = set()
    for ordinal in re.findall(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth)\b",
        normalized,
    ):
        idx = _ORDINAL_INDEX.get(ordinal)
        if isinstance(idx, int) and 1 <= idx <= num_applicants:
            indices.add(idx)

    if "applicant" in normalized:
        for token in re.findall(r"\b(\d+)\b", normalized):
            idx = int(token)
            if 1 <= idx <= num_applicants:
                indices.add(idx)

    return indices


def _resolve_row_indices(
    row: dict[str, Any],
    num_applicants: int | None,
    exact_map: dict[str, int],
    surname_map: dict[str, int],
) -> set[int]:
    idx = _resolve_row_index(row, num_applicants, exact_map, surname_map)
    if isinstance(idx, int):
        return {idx}
    if _beneficiary_kind(row.get("beneficiary_label")) != "group":
        return set()
    return _group_label_indices(row.get("beneficiary_label"), num_applicants)


def _sum_per_applicant_amount(rows: list[dict[str, Any]]) -> float | None:
    total = 0.0
    seen = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        amount = row.get("eur_amount")
        if isinstance(amount, (int, float)):
            total += float(amount)
            seen = True
    return total if seen else None


def _sum_per_applicant_amounts_by_head(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    totals: dict[str, float] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        head = row.get("head")
        if head not in {"bundled", "non_pecuniary", "pecuniary", "costs"}:
            continue
        amount = _numeric_amount(row.get("eur_amount"))
        if amount is None:
            continue
        totals[head] = totals.get(head, 0.0) + amount
        seen.add(head)
    return {head: totals.get(head) if head in seen else None for head in ("bundled", "non_pecuniary", "pecuniary", "costs")}


def _per_applicant_comparable_diff(final_awards: dict[str, Any], rows: list[dict[str, Any]]) -> float | None:
    row_sums = _sum_per_applicant_amounts_by_head(rows)
    expected_total = 0.0
    observed_total = 0.0
    comparable_seen = False

    for head, award_key in (
        ("bundled", "bundled_award_eur"),
        ("non_pecuniary", "non_pecuniary_eur"),
        ("pecuniary", "pecuniary_eur"),
    ):
        row_sum = row_sums.get(head)
        if row_sum is None:
            continue
        comparable_seen = True
        observed_total += row_sum
        award_total = _numeric_amount(final_awards.get(award_key))
        if award_total is not None:
            expected_total += award_total

    costs_row_sum = row_sums.get("costs")
    if costs_row_sum is not None:
        comparable_seen = True
        observed_total += costs_row_sum
        costs_total = _numeric_amount(final_awards.get("costs_eur"))
        if costs_total is not None:
            expected_total += costs_total

    if not comparable_seen:
        return None
    return abs(expected_total - observed_total)


def _repair_award_rows(rows: Any, b_result: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    num_applicants = ((b_result.get("facts_procedure") or {}).get("num_applicants"))
    applicants = ((b_result.get("facts_procedure") or {}).get("applicants")) or []
    exact_map, surname_map = _build_applicant_label_maps(applicants if isinstance(applicants, list) else [])

    repaired_rows: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    unlabeled_unindexed_positions: list[int] = []

    for row_pos, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, dict):
            continue
        fixed = dict(raw_row)
        label = fixed.get("beneficiary_label")
        if isinstance(label, str):
            label = label.strip() or None
        elif label is not None:
            label = None
        fixed["beneficiary_label"] = label

        idx = _resolve_row_index(fixed, num_applicants if isinstance(num_applicants, int) else None, exact_map, surname_map)
        fixed["applicant_index"] = idx

        if idx is not None:
            used_indices.add(idx)
            if (fixed.get("beneficiary_label") in (None, "")) and 1 <= idx <= len(applicants):
                app = applicants[idx - 1]
                if isinstance(app, dict) and isinstance(app.get("beneficiary_label"), str) and app.get("beneficiary_label"):
                    fixed["beneficiary_label"] = app.get("beneficiary_label")
        elif _beneficiary_kind(label) == "unknown":
            unlabeled_unindexed_positions.append(len(repaired_rows))

        repaired_rows.append(fixed)

    if isinstance(num_applicants, int) and num_applicants > 0 and len(repaired_rows) == num_applicants:
        available = [idx for idx in range(1, num_applicants + 1) if idx not in used_indices]
        if len(unlabeled_unindexed_positions) == len(available):
            for pos, idx in zip(unlabeled_unindexed_positions, available):
                repaired_rows[pos]["applicant_index"] = idx
                if 1 <= idx <= len(applicants):
                    app = applicants[idx - 1]
                    if isinstance(app, dict) and isinstance(app.get("beneficiary_label"), str) and app.get("beneficiary_label"):
                        repaired_rows[pos]["beneficiary_label"] = app.get("beneficiary_label")

    return repaired_rows


def _repair_c_per_applicant(c_result: dict[str, Any], b_result: dict[str, Any]) -> dict[str, Any]:
    article_41 = c_result.get("article_41_extraction")
    if not isinstance(article_41, dict):
        return c_result

    primary_rows = article_41.get("award_per_applicant")
    fallback_rows = c_result.get("award_per_applicant_fallback") or []
    repaired_primary = _repair_award_rows(primary_rows, b_result)
    repaired_fallback = _repair_award_rows(fallback_rows, b_result)

    if repaired_primary:
        article_41["award_per_applicant"] = repaired_primary
    if repaired_fallback:
        c_result["award_per_applicant_fallback"] = repaired_fallback

    final_awards = c_result.get("final_awards") or {}
    total_eur = final_awards.get("total_eur")
    if not isinstance(total_eur, (int, float)):
        return c_result

    primary_diff = _per_applicant_comparable_diff(final_awards, repaired_primary)
    c_result["per_applicant_total_diff_eur"] = primary_diff

    if repaired_primary:
        c_result["per_applicant_source"] = "llm_repaired"
    elif repaired_fallback:
        c_result["per_applicant_source"] = "llm_missing_regex_validator_available"
    else:
        c_result["per_applicant_source"] = "llm_missing"

    return c_result


def _rows_equivalent(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """Return True if two per-applicant row lists describe the same awards.

    Applicant index and EUR amount are the identity here; beneficiary label is
    ignored because it can legitimately differ between LLM-derived and
    regex-derived rows (e.g., 'Applicant 1' vs 'Balčiūnas').
    """
    if not a or not b:
        return False

    def _key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (row.get("applicant_index"), row.get("head"), row.get("eur_amount"))

    return sorted(_key(r) for r in a) == sorted(_key(r) for r in b)


def _reconciliation_check(itemid: str, b_result: dict[str, Any], c_result: dict[str, Any]) -> dict[str, Any]:
    num_applicants = (b_result.get("facts_procedure") or {}).get("num_applicants")
    award_per_applicant = ((c_result.get("article_41_extraction") or {}).get("award_per_applicant") or [])
    fallback_per_applicant = c_result.get("award_per_applicant_fallback") or []
    all_beneficiaries = award_per_applicant if award_per_applicant else fallback_per_applicant

    if not isinstance(num_applicants, int) or num_applicants <= 0:
        status = "unknown"
        detail = f"num_applicants={num_applicants} (invalid)"
    elif not all_beneficiaries:
        status = "unknown"
        detail = "no award_per_applicant found"
    else:
        applicants = ((b_result.get("facts_procedure") or {}).get("applicants") or [])
        exact_map, surname_map = _build_applicant_label_maps(applicants if isinstance(applicants, list) else [])
        mapped_indices: set[int] = set()
        relevant_rows = 0
        for row in all_beneficiaries:
            if not isinstance(row, dict):
                continue
            kind = _beneficiary_kind(row.get("beneficiary_label"))
            if kind == "external":
                continue
            relevant_rows += 1
            mapped_indices.update(_resolve_row_indices(row, num_applicants, exact_map, surname_map))
        if relevant_rows == 0:
            status = "unknown"
            detail = "no applicant-like beneficiary rows found"
            num_beneficiaries = 0
            return {
                "itemid": itemid,
                "status": status,
                "num_applicants": num_applicants,
                "num_beneficiaries": num_beneficiaries,
                "detail": detail,
            }
        num_beneficiaries = len(mapped_indices)
        if num_beneficiaries == num_applicants:
            status = "pass"
            detail = f"mapped applicant count ({num_beneficiaries}) matches applicant count ({num_applicants})"
        elif num_beneficiaries > num_applicants:
            status = "over"
            detail = f"mapped applicant count ({num_beneficiaries}) exceeds applicant count ({num_applicants})"
        else:
            status = "under"
            detail = f"mapped applicant count ({num_beneficiaries}) is less than applicant count ({num_applicants})"

    return {
        "itemid": itemid,
        "status": status,
        "num_applicants": num_applicants,
        "num_beneficiaries": num_beneficiaries if "num_beneficiaries" in locals() else (len(all_beneficiaries) if all_beneficiaries else 0),
        "detail": detail,
    }


def _record_stage_problem(
    problem_log: ProblemLog,
    itemid: str,
    stage: str,
    payload: dict[str, Any],
) -> None:
    """Translate a stage payload's failure status into a problems.jsonl entry."""
    status = payload.get("status")
    if status == "api_error":
        api_error = payload.get("api_error") or {}
        problem_log.record(
            itemid,
            stage,
            CATEGORY_API,
            api_error.get("detail") or "api error",
            kind=api_error.get("kind"),
            http_status=api_error.get("http_status"),
        )
    elif status == "schema_validation":
        problem_log.record(
            itemid,
            stage,
            CATEGORY_SCHEMA,
            payload.get("errors") or "schema validation failed",
        )
    elif status not in {None, "success"}:
        problem_log.record(
            itemid,
            stage,
            CATEGORY_EXCEPTION,
            f"unexpected status: {status}",
        )


def run_one_case(
    itemid: str,
    source_row: dict[str, Any],
    core_lookup: dict[str, dict[str, Any]],
    count_tokens,
    client: OpenAICompatibleClient,
    schema_b: dict[str, Any],
    schema_c: dict[str, Any],
    schema_d: dict[str, Any],
    schema_e: dict[str, Any],
    prompt_b: str,
    prompt_c: str,
    prompt_d: str,
    prompt_e: str,
    problem_log: ProblemLog,
) -> dict[str, Any]:
    start = time.perf_counter()
    scaffold = _build_scaffold(source_row, core_lookup, count_tokens)

    # ---- B: facts/procedure ------------------------------------------------
    b_payload = run_pipeline_b_case(itemid, scaffold, source_row, client, prompt_b, schema_b)
    if b_payload.get("status") == "success":
        b_result = b_payload["result"]
    else:
        _record_stage_problem(problem_log, itemid, "b", b_payload)
        b_result = _fallback_b_result(itemid, scaffold)

    # ---- C: Article 41 / compensation -------------------------------------
    c_payload = run_pipeline_c_case(itemid, scaffold, client, prompt_c, schema_c, regex_only=False)
    
    if c_payload.get("status") in ("success", "schema_validation") and c_payload.get("raw_result"):
        if c_payload.get("status") == "schema_validation":
            _record_stage_problem(problem_log, itemid, "c", c_payload)
            # Schema validation failed, but we have LLM output. Let's compact it anyway.
            from run_pipeline_c_backbone import layer3_crossval, compact_compensation_result
            c_awards_regex = c_payload.get("awards_regex", {})
            crossval = layer3_crossval(c_awards_regex, c_payload["raw_result"])
            c_result = compact_compensation_result(itemid, c_awards_regex, c_payload["raw_result"], crossval, source_row=source_row)
            c_payload["result"] = c_result
        else:
            c_result = c_payload["result"]
            
        c_usage = _usage_or_zero(c_payload)
    else:
        _record_stage_problem(problem_log, itemid, "c", c_payload)
        c_result = {}
        c_usage = _usage_or_zero(c_payload)
    c_result = _repair_c_per_applicant(c_result, b_result)

    # ---- D + E: combined legal analysis + reasoning layer ----------------
    # Single LLM call: D and E share the same law-text context, so sending it
    # twice (~7k tokens of duplication per case) is wasteful. The shared usage
    # is attributed to d_payload.
    d_payload, e_payload = run_pipeline_de_combined_case(
        itemid, scaffold, client, prompt_d, prompt_e, schema_d, schema_e
    )

    if d_payload.get("status") == "success":
        d_result = d_payload["result"]
        if _is_legal_analysis_sparse(d_result.get("legal_analysis") or {}):
            problem_log.record(
                itemid,
                "d",
                CATEGORY_D_SPARSE,
                "legal_analysis returned with no signal (legal_tests, proportionality, "
                "margin_of_appreciation, nature_of_obligation, subsidiarity, precedent_usage all empty)",
            )
    else:
        _record_stage_problem(problem_log, itemid, "d", d_payload)
        d_result = _fallback_d_result(itemid, scaffold)

    if e_payload.get("status") == "success":
        e_result = e_payload["result"]
    else:
        _record_stage_problem(problem_log, itemid, "e", e_payload)
        e_result = _fallback_e_result(itemid, scaffold)

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for payload in (b_payload, c_payload, d_payload, e_payload):
        usage = _usage_or_zero(payload)
        for key in usage_total:
            usage_total[key] += usage[key]

    stage_status = {
        "b": b_payload.get("status"),
        "c": c_payload.get("status"),
        "d": d_payload.get("status"),
        "e": e_payload.get("status"),
    }
    overall_status = "success" if all(status == "success" for status in stage_status.values()) else "partial_success"

    reconciliation = _reconciliation_check(itemid, b_result, c_result)
    if reconciliation["status"] in {"over", "under"}:
        problem_log.record(
            itemid,
            "reconciliation",
            CATEGORY_RECONCILIATION,
            reconciliation["detail"],
            extras={
                "num_applicants": reconciliation["num_applicants"],
                "num_beneficiaries": reconciliation["num_beneficiaries"],
                "reconciliation_status": reconciliation["status"],
            },
        )

    stage_usage = {
        "b": _usage_or_zero(b_payload),
        "c": _usage_or_zero(c_payload),
        "d": _usage_or_zero(d_payload),
        "e": _usage_or_zero(e_payload),
    }

    extraction_meta = {
        **(scaffold.get("extraction_meta") or {}),
        "simplified_run": True,
        "stage_status": stage_status,
        "stage_usage": stage_usage,
    }
    prepared_for_sidecars = prepare_reasoning_context(scaffold)
    b_llm_candidate = _compact_candidate_tree(b_payload.get("result") or b_payload.get("raw_result") or {})
    c_llm_candidate = _compact_candidate_tree(c_payload.get("raw_result") or {})
    d_llm_candidate = _compact_candidate_tree((d_payload.get("result") or d_payload.get("raw_result") or {}).get("legal_analysis") if isinstance(d_payload.get("result") or d_payload.get("raw_result"), dict) else {})
    e_llm_candidate = _compact_candidate_tree((e_payload.get("result") or e_payload.get("raw_result") or {}).get("reasoning_layer") if isinstance(e_payload.get("result") or e_payload.get("raw_result"), dict) else {})

    pipeline_a_metadata_layer = {
        "itemid": itemid,
        "stage": "a",
        "stage_meta": {
            "status": "success",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "source": "deterministic_scaffold",
        },
        "appno": scaffold["appno"],
        "judgementdate": scaffold["judgementdate"],
        "core_case": scaffold["core_case"],
        "extraction_meta": extraction_meta,
    }

    pipeline_b_layer = {
        "itemid": itemid,
        "stage": "b",
        "stage_meta": {
            "status": stage_status["b"],
            "usage": stage_usage["b"],
        },
        "facts_procedure": b_result.get("facts_procedure", {}),
    }
    pipeline_b_llm_layer = {
        "itemid": itemid,
        "stage": "b_llm",
        "stage_meta": {
            "status": stage_status["b"],
            "usage": stage_usage["b"],
            "used_in_combined_output": stage_status["b"] == "success",
        },
        "candidate_compact": b_llm_candidate or {},
        "validation_errors": b_payload.get("errors") or [],
    }
    pipeline_b_regex_layer = {
        "itemid": itemid,
        "stage": "b_regex",
        "stage_meta": {
            "status": "success",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "source": "deterministic_backbone",
        },
        "input_digest": _build_b_input_digest(scaffold),
        "deterministic_hints": _compact_candidate_tree(_build_deterministic_hints(scaffold, source_row)) or {},
    }

    pipeline_c_layer = {
        "itemid": itemid,
        "stage": "c",
        "stage_meta": {
            "status": stage_status["c"],
            "usage": stage_usage["c"],
        },
        "article_41_extraction": c_result.get("article_41_extraction"),
        "award_per_applicant_fallback": c_result.get("award_per_applicant_fallback"),
        "per_applicant_source": c_result.get("per_applicant_source"),
        "per_applicant_total_diff_eur": c_result.get("per_applicant_total_diff_eur"),
        "cross_validation": c_result.get("cross_validation"),
        "final_awards": c_result.get("final_awards"),
    }
    pipeline_c_llm_layer = {
        "itemid": itemid,
        "stage": "c_llm",
        "stage_meta": {
            "status": stage_status["c"],
            "usage": stage_usage["c"],
            "used_in_combined_output": stage_status["c"] == "success",
        },
        "candidate_compact": c_llm_candidate or {},
        "validation_errors": c_payload.get("errors") or [],
    }
    pipeline_c_regex_layer = {
        "itemid": itemid,
        "stage": "c_regex",
        "stage_meta": {
            "status": "success",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "source": "deterministic_regex",
        },
        "input_digest": _build_c_input_digest(scaffold),
        "awards_regex": c_payload.get("awards_regex") or {},
        "article_41_projection_from_regex": _build_deterministic_article_41_extraction(
            itemid,
            scaffold,
            c_payload.get("awards_regex") or {},
        ),
        "article_41_precedents": extract_article_41_precedents(
            (scaffold.get("claim_and_award_layer") or {})
            .get("cross_validation_inputs", {})
            .get("article_41_text")
            or ""
        ),
    }

    pipeline_d_layer = {
        "itemid": itemid,
        "stage": "d",
        "stage_meta": {
            "status": stage_status["d"],
            "usage": stage_usage["d"],
        },
        "legal_analysis": d_result.get("legal_analysis"),
        "legal_analysis_validation": d_result.get("legal_analysis_validation"),
    }
    pipeline_d_llm_layer = {
        "itemid": itemid,
        "stage": "d_llm",
        "stage_meta": {
            "status": stage_status["d"],
            "usage": stage_usage["d"],
            "used_in_combined_output": stage_status["d"] == "success",
        },
        "candidate_compact": d_llm_candidate or {},
        "validation_errors": d_payload.get("errors") or [],
    }
    pipeline_d_regex_layer = {
        "itemid": itemid,
        "stage": "d_regex",
        "stage_meta": {
            "status": "success",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "source": "deterministic_reference",
        },
        "input_digest": _build_de_input_digest(scaffold, prepared_for_sidecars),
        "deterministic_reference": {
            "violated_articles_hint": list((scaffold.get("core_case") or {}).get("violated_articles") or []),
            "anchor_violation_type": prepared_for_sidecars.get("anchor", {}).get("violation_type"),
            "anchor_violation_subtype": list(prepared_for_sidecars.get("anchor", {}).get("violation_subtype") or []),
        },
    }

    pipeline_e_layer = {
        "itemid": itemid,
        "stage": "e",
        "stage_meta": {
            "status": stage_status["e"],
            "usage": stage_usage["e"],
        },
        "reasoning_layer": e_result.get("reasoning_layer"),
        "reasoning_validation": e_result.get("reasoning_validation"),
    }
    pipeline_e_llm_layer = {
        "itemid": itemid,
        "stage": "e_llm",
        "stage_meta": {
            "status": stage_status["e"],
            "usage": stage_usage["e"],
            "used_in_combined_output": stage_status["e"] == "success",
            "usage_attributed_to": "d_combined",
            "llm_invoked": True,
            "note": "E is extracted in the same LLM call as D; tokens are accounted to pipeline_d to avoid double-counting.",
        },
        "candidate_compact": e_llm_candidate or {},
        "validation_errors": e_payload.get("errors") or [],
    }
    pipeline_e_regex_layer = {
        "itemid": itemid,
        "stage": "e_regex",
        "stage_meta": {
            "status": "success",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "source": "deterministic_reference",
        },
        "input_digest": _build_de_input_digest(scaffold, prepared_for_sidecars),
        "deterministic_reference": {
            "anchor_violation_type": prepared_for_sidecars.get("anchor", {}).get("violation_type"),
            "anchor_violation_subtype": list(prepared_for_sidecars.get("anchor", {}).get("violation_subtype") or []),
            "anchor_reasoning_factors": list(prepared_for_sidecars.get("anchor", {}).get("reasoning_factors") or []),
            "award_summary_mode": prepared_for_sidecars.get("award_anchor", {}).get("mode"),
            "award_summary_anchor": prepared_for_sidecars.get("award_anchor", {}).get("summary"),
        },
    }

    log_layer = {
        "itemid": itemid,
        "status": overall_status,
        "elapsed_seconds": time.perf_counter() - start,
        "run_mode": "single",
        "B_C_reconciliation": reconciliation,
        "usage_total": usage_total,
        "stage_status": stage_status,
        "stage_usage": stage_usage,
        "stage_meta": {
            "b": {k: v for k, v in b_payload.items() if k != "result"},
            "c": {k: v for k, v in c_payload.items() if k != "result"},
            "d": {k: v for k, v in d_payload.items() if k != "result"},
            "e": {k: v for k, v in e_payload.items() if k != "result"},
        },
    }

    article_41_text_for_precedents = (
        (scaffold.get("claim_and_award_layer") or {})
        .get("cross_validation_inputs", {})
        .get("article_41_text")
        or ""
    )
    art41_precedents = extract_article_41_precedents(article_41_text_for_precedents)
    core_case_enriched = dict(scaffold["core_case"])
    if art41_precedents:
        existing_scl = list(core_case_enriched.get("all_scl_citations") or [])
        for p in art41_precedents:
            citation_str = p["case_name"]
            if p.get("appno"):
                citation_str = f"{citation_str}, no. {p['appno']}"
            if p.get("date"):
                citation_str = f"{citation_str}, {p['date']}"
            if citation_str not in existing_scl:
                existing_scl.append(citation_str)
        core_case_enriched["all_scl_citations"] = existing_scl
        core_case_enriched["article_41_precedents"] = art41_precedents

    merged = {
        "itemid": itemid,
        "appno": scaffold["appno"],
        "judgementdate": scaffold["judgementdate"],
        "core_case": core_case_enriched,
        "extraction_meta": extraction_meta,
        "B_C_reconciliation": reconciliation,
        "facts_procedure": b_result.get("facts_procedure", {}),
        "article_41_extraction": c_result.get("article_41_extraction"),
        "article_41_precedents": art41_precedents,
        "award_per_applicant_fallback": c_result.get("award_per_applicant_fallback"),
        "per_applicant_source": c_result.get("per_applicant_source"),
        "per_applicant_total_diff_eur": c_result.get("per_applicant_total_diff_eur"),
        "cross_validation": c_result.get("cross_validation"),
        "final_awards": c_result.get("final_awards"),
        "legal_analysis": d_result.get("legal_analysis"),
        "legal_analysis_validation": d_result.get("legal_analysis_validation"),
        "reasoning_layer": e_result.get("reasoning_layer"),
        "reasoning_validation": e_result.get("reasoning_validation"),
    }

    elapsed_seconds = log_layer["elapsed_seconds"]
    return {
        "itemid": itemid,
        "status": overall_status,
        "elapsed_seconds": elapsed_seconds,
        "usage": usage_total,
        "result": merged,
        "layers": {
            "pipeline_a_metadata": pipeline_a_metadata_layer,
            "pipeline_b": pipeline_b_layer,
            "pipeline_b_llm": pipeline_b_llm_layer,
            "pipeline_b_regex": pipeline_b_regex_layer,
            "pipeline_c": pipeline_c_layer,
            "pipeline_c_llm": pipeline_c_llm_layer,
            "pipeline_c_regex": pipeline_c_regex_layer,
            "pipeline_d": pipeline_d_layer,
            "pipeline_d_llm": pipeline_d_llm_layer,
            "pipeline_d_regex": pipeline_d_regex_layer,
            "pipeline_e": pipeline_e_layer,
            "pipeline_e_llm": pipeline_e_llm_layer,
            "pipeline_e_regex": pipeline_e_regex_layer,
            "merged": merged,
            "log": log_layer,
        },
        "stage_meta": log_layer["stage_meta"],
    }



_split_mapping = None

def _get_group(itemid: str) -> str:
    global _split_mapping
    if _split_mapping is None:
        try:
            import json
            from pathlib import Path
            json_path = Path(__file__).resolve().parent.parent.parent / "splits" / "split_membership.json"
            if json_path.exists():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                _split_mapping = {d["itemid"]: d["split"] for d in data}
            else:
                _split_mapping = {}
        except Exception:
            _split_mapping = {}
            
    split_name = _split_mapping.get(itemid, "")
    if split_name == "set1_primary":
        return "set1"
    elif split_name == "set2_postcut_ood":
        return "set2"
    elif split_name == "set3_challenging":
        return "set3"
    else:
        return "others"

def write_case_result(itemid: str, layers: dict[str, dict[str, Any]]) -> None:
    """Write case output layers under outputs/cases/<group>/<itemid>/."""
    group = _get_group(itemid)
    case_dir = CASES_OUTPUT / group / itemid
    with WRITE_LOCK:
        case_dir.mkdir(parents=True, exist_ok=True)

        for legacy_name in ("metadata.json", "llm_extraction.json", "regex_validation.json"):
            legacy_path = case_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        ordered_layers = (
            "pipeline_a_metadata",
            "pipeline_b",
            "pipeline_b_llm",
            "pipeline_b_regex",
            "pipeline_c",
            "pipeline_c_llm",
            "pipeline_c_regex",
            "pipeline_d",
            "pipeline_d_llm",
            "pipeline_d_regex",
            "pipeline_e",
            "pipeline_e_llm",
            "pipeline_e_regex",
            "merged",
            "log",
        )
        for name in ordered_layers:
            payload = layers.get(name)
            if payload is None:
                continue
            (case_dir / f"{name}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    schema_b = load_json(SCHEMA_B)
    schema_c = load_json(SCHEMA_C)
    schema_d = load_json(SCHEMA_D)
    schema_e = load_json(SCHEMA_E)
    prompt_b = PROMPT_B.read_text(encoding="utf-8")
    prompt_c = PROMPT_C.read_text(encoding="utf-8")
    prompt_d = PROMPT_D.read_text(encoding="utf-8")
    prompt_e = PROMPT_E.read_text(encoding="utf-8")
    run_dir = make_run_dir(args.run_name)

    run_metadata = {
        "run_name": args.run_name,
        "itemids": args.itemids,
        "concurrency": args.concurrency,
        "resume": args.resume,
        "case_store_dir": str(CASE_STORE_DIR),
        "unstructured_cases": str(UNSTRUCTURED_CASES),
        "schemas": {
            "b": str(SCHEMA_B),
            "c": str(SCHEMA_C),
            "d": str(SCHEMA_D),
            "e": str(SCHEMA_E),
        },
        "prompts": {
            "b": str(PROMPT_B),
            "c": str(PROMPT_C),
            "d": str(PROMPT_D),
            "e": str(PROMPT_E),
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    already_done = load_completed_ids(run_dir) if args.resume else set()
    todo_ids = [itemid for itemid in args.itemids if itemid not in already_done]

    source_rows = load_cases_by_itemid(todo_ids if todo_ids else args.itemids, fallback_cases_json=UNSTRUCTURED_CASES, backfill_store=True)
    missing = [itemid for itemid in todo_ids if itemid not in source_rows]
    if missing:
        raise RuntimeError(f"{len(missing)} itemids were not found in case store or cases.json")

    core_lookup = load_cases_core_lookup()
    count_tokens, token_method = build_token_counter()
    client = OpenAICompatibleClient.from_env()
    problem_log = ProblemLog(run_dir / "problems.jsonl")
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    success_count = 0
    partial_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_one_case,
                itemid,
                source_rows[itemid],
                core_lookup,
                count_tokens,
                client,
                schema_b,
                schema_c,
                schema_d,
                schema_e,
                prompt_b,
                prompt_c,
                prompt_d,
                prompt_e,
                problem_log,
            ): itemid
            for itemid in todo_ids
        }
        for future in as_completed(futures):
            itemid = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = {"itemid": itemid, "status": "request_failed", "error": str(exc)}
                problem_log.record(itemid, "orchestrator", CATEGORY_EXCEPTION, str(exc))

            usage = payload.get("usage") or {}
            for key in usage_total:
                if isinstance(usage.get(key), int):
                    usage_total[key] += usage[key]

            status = payload.get("status")
            if status in {"success", "partial_success"}:
                if status == "success":
                    success_count += 1
                else:
                    partial_count += 1
                layers = payload.get("layers") or {}
                if not layers and isinstance(payload.get("result"), dict):
                    # safety net: if a worker returns the legacy shape, persist merged-only
                    layers = {"merged": payload["result"]}
                write_case_result(itemid, layers)
            else:
                failure_count += 1

            (run_dir / f"{itemid}.meta.json").write_text(
                json.dumps({k: v for k, v in payload.items() if k != "layers"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    summary = {
        **run_metadata,
        "token_estimation_method": token_method,
        "runtime_seconds": time.perf_counter() - start,
        "successful_cases": success_count,
        "partial_cases": partial_count,
        "failed_cases": failure_count,
        "usage_total": usage_total,
        "completed_cases": success_count + partial_count,
        "todo_cases": len(todo_ids),
        "resumed_cases": len(already_done),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

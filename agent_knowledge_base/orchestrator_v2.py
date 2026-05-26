#!/usr/bin/env python3
"""
Bounded agentic orchestrator for the v3 NPD knowledge base.

This scaffold keeps the v2 module-selection machinery but changes the
experimental contract:

- pure-regression final output
- deterministic module selection
- deterministic empirical-prior bucket resolution from offline train anchors
- target-case non-pecuniary award redaction
- zero-shot versus few-shot prompt assembly
- dry-run inspection

It does not implement a provider-specific LLM call.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


ARTICLE_MAP = {
    "2": "art2",
    "3": "art3",
    "5": "art5",
    "8": "art8",
    "10": "art10",
    "11": "art11",
    "13": "art13",
    "14": "art14",
    "P1-1": "p1a1",
}

SUPPORTED_ARTICLES = set(ARTICLE_MAP) | {"6"}
AWARD_REDACTED_MODE = "award_redacted_full_info"
MODE_ALLOWED_LEAKAGE = {
    AWARD_REDACTED_MODE: {"strict_safe", "article41_structured", "award_redacted_full_info"},
}

MODE_ALIASES: dict[str, str] = {}

EMPIRICAL_TABLE_RELATIVE_PATH = Path("modules/empirical/article_single_violation_stats_train.csv")
MIN_EMPIRICAL_SUPPORT_N = 50

ARTICLE_TOKEN_RE = re.compile(r"[;,/|]+")
WHITESPACE_RE = re.compile(r"\s+")
APPENDIX_REFERENCE_TERMS = ("appended table", "appendix", "annexed table", "annex")
APPENDIX_BLOCKED_KEY_TERMS = (
    "amount awarded",
    "award",
    "eur",
    "euro",
    "pecuniary",
    "costs and expenses",
    "claim amount",
    "compensation",
    "just satisfaction",
    "sum awarded",
)
INPUT_HEADING_ONLY_KEYS = {
    "PROCEDURE",
    "PROCEEDINGS",
    "FACTS",
    "THEFACTS",
    "PROCEDURETHEFACTS",
    "THELAW",
    "PROCEDURETHELAW",
    "THEFACTSTHELAW",
    "THECOURTSASSESSMENT",
    "SUBJECTMATTEROFTHECASE",
}
TARGET_NONPEC_AWARD_REDACTION = "[TARGET_NON_PECUNIARY_AWARD_REDACTED]"
TARGET_NONPEC_AWARD_FIELD_REDACTION = {
    "redacted": True,
    "reason": "target_non_pecuniary_award_label_or_direct_derivative",
}
TARGET_NONPEC_AWARD_KEYS = {
    "award_eur",
    "award_eur_label_source",
    "zero_positive_label",
    "is_zero_award",
    "safe_non_pec_eur",
    "safe_bundled_eur",
    "safe_total_eur",
    "raw_extractor_non_pec_eur",
    "raw_extractor_non_pec_source",
    "raw_extractor_bundled_eur",
    "raw_extractor_bundled_source",
    "raw_case_total_eur",
    "case_total_from_per_app_sum",
    "cross_non_pec_match",
    "label_processing_reason",
    "needs_label_processing",
}
TARGET_NONPEC_AWARD_KEY_PREFIXES = (
    "award_non_pec",
    "fx_fill_non_pec",
    "per_app_non_pec",
    "diff_non_pec",
    "repair_non_pec",
)
EXTRACTION_SIDECAR_RELATIVE_PATHS = {
    "combined_extraction": Path(os.environ.get("ECTHR_NPD_COMBINED_EXTRACTION_CSV", "your_path/combined.csv")),
    "reasoning_layer": Path(os.environ.get("ECTHR_NPD_REASONING_LAYER_CSV", "your_path/reasoning_layer.csv")),
    "facts_procedure_scalars": Path(os.environ.get("ECTHR_NPD_FACTS_PROCEDURE_CSV", "your_path/facts_procedure_scalars.csv")),
    "legal_analysis": Path(os.environ.get("ECTHR_NPD_LEGAL_ANALYSIS_CSV", "your_path/legal_analysis.csv")),
    "per_applicant_alloc_normalized": Path(os.environ.get("ECTHR_NPD_PER_APPLICANT_CSV", "your_path/per_applicant_alloc_normalized.csv")),
}
STANDARD_INPUT_KEYS = (
    "itemid",
    "oracle_violated_articles",
    "violated_articles_text",
    "introduction_text",
    "procedure_text",
    "facts_text",
    "safe_appendix_text",
    "combined_input_text",
    "combined_input_text_with_violated_articles",
)
TARGET_NONPEC_AWARD_TEXT_TERMS = (
    "non-pecuniary",
    "non pecuniary",
    "nonpecuniary",
    "non-pec",
    "non pec",
)
TARGET_AWARD_DISPOSITION_TERMS = (
    "award",
    "awards",
    "awarded",
    "pay",
    "satisfaction sufficient",
    "sufficient just satisfaction",
    "constitutes just satisfaction",
    "constitutes sufficient satisfaction",
    "makes no award",
)


def load_kb_index(kb_dir: Path) -> dict[str, Any]:
    return json.loads((kb_dir / "kb_index.json").read_text(encoding="utf-8"))


def index_modules(kb_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["module_id"]: m for m in kb_index["modules"]}


def load_module_text(kb_dir: Path, modules: dict[str, dict[str, Any]], module_id: str) -> str:
    module = modules[module_id]
    return (kb_dir / module["path"]).read_text(encoding="utf-8")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_respondent_state(value: Any) -> str | None:
    if value in (None, "", []):
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    token = str(value).strip().upper()
    return token or None


@lru_cache(maxsize=1)
def load_article_empirical_stats_table(kb_dir: Path) -> dict[str, dict[str, Any]]:
    path = kb_dir / EMPIRICAL_TABLE_RELATIVE_PATH
    if not path.exists():
        return {}

    table: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            article = normalize_article_token(row.get("article"))
            if not article:
                continue

            sample_count = _to_int(row.get("sample_count"))
            zero_rate = _to_float(row.get("zero_rate"))
            median = _to_float(row.get("median"))
            iqr = _to_float(row.get("iqr"))
            p10 = _to_float(row.get("p10"))
            p90 = _to_float(row.get("p90"))

            if sample_count is None or sample_count <= 0:
                continue
            if None in (zero_rate, median, iqr, p10, p90):
                continue

            table[article] = {
                "sample_count": sample_count,
                "zero_rate": zero_rate,
                "median": median,
                "iqr": iqr,
                "p10": p10,
                "p90": p90,
                "table_source_csv": str(EMPIRICAL_TABLE_RELATIVE_PATH).replace("\\", "/"),
            }
    return table


def build_global_empirical_stats(article_table: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not article_table:
        return None
    total_n = sum(int(stats["sample_count"]) for stats in article_table.values())
    if total_n <= 0:
        return None

    def weighted_avg(key: str) -> float:
        numerator = sum(float(stats[key]) * int(stats["sample_count"]) for stats in article_table.values())
        return numerator / total_n

    return {
        "sample_count": total_n,
        "zero_rate": weighted_avg("zero_rate"),
        "median": weighted_avg("median"),
        "iqr": weighted_avg("iqr"),
        "p10": weighted_avg("p10"),
        "p90": weighted_avg("p90"),
        "aggregation_method": "weighted_mean_of_article_anchors",
        "table_source_csv": str(EMPIRICAL_TABLE_RELATIVE_PATH).replace("\\", "/"),
    }


def resolve_empirical_calibration(kb_dir: Path, route_state: dict[str, Any]) -> dict[str, Any]:
    article_table = load_article_empirical_stats_table(kb_dir)
    global_stats = build_global_empirical_stats(article_table)

    violated_articles = route_state.get("violated_articles") or []
    respondent_state = normalize_respondent_state(route_state.get("respondent_state"))
    art6_limb = route_state.get("art6_limb")

    calibrations: list[dict[str, Any]] = []
    calibration_sources_used: list[str] = []
    weak_support_detected = False

    for raw_article in violated_articles:
        article = normalize_article_token(raw_article) or str(raw_article)
        article_stats = article_table.get(article)

        available_buckets: list[tuple[str, dict[str, Any]]] = []
        if article == "6" and art6_limb in {"civil", "criminal"} and article_stats:
            available_buckets.append((f"article_limb:{art6_limb}", article_stats))
        if article_stats:
            available_buckets.append(("article", article_stats))
        if global_stats:
            available_buckets.append(("global", global_stats))

        selected_bucket = "none"
        selected_stats: dict[str, Any] | None = None
        for bucket_name, stats in available_buckets:
            if int(stats["sample_count"]) >= MIN_EMPIRICAL_SUPPORT_N:
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
            calibration_sources_used.append(f"article={article};selected=none;n=0")
            continue

        first_bucket_name = available_buckets[0][0]
        fallback_used = selected_bucket != first_bucket_name
        if fallback_used or int(selected_stats["sample_count"]) < MIN_EMPIRICAL_SUPPORT_N:
            weak_support_detected = True

        bucket_trace = [
            {
                "bucket": bucket_name,
                "sample_count": int(stats["sample_count"]),
            }
            for bucket_name, stats in available_buckets
        ]

        selected_sample_count = int(selected_stats["sample_count"])
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
                "weak_support": selected_sample_count < MIN_EMPIRICAL_SUPPORT_N,
                "selected_anchor": {
                    "zero_rate": round(float(selected_stats["zero_rate"]), 6),
                    "median": float(selected_stats["median"]),
                    "iqr": float(selected_stats["iqr"]),
                    "p10": float(selected_stats["p10"]),
                    "p90": float(selected_stats["p90"]),
                },
            }
        )
        calibration_sources_used.append(
            f"article={article};selected={selected_bucket};n={selected_sample_count};fallback_chain={fallback_chain}"
        )

    return {
        "minimum_support_n": MIN_EMPIRICAL_SUPPORT_N,
        "global_available": global_stats is not None,
        "calibration_sources_used": calibration_sources_used,
        "article_calibrations": calibrations,
        "weak_support_detected": weak_support_detected,
    }


def validate_mode(kb_index: dict[str, Any], mode: str) -> None:
    allowed = set(kb_index.get("modes") or [])
    effective_mode = MODE_ALIASES.get(mode, mode)
    if effective_mode not in allowed:
        raise ValueError(f"Unsupported KB mode: {mode}")


def validate_module_for_mode(module: dict[str, Any], mode: str) -> None:
    effective_mode = MODE_ALIASES.get(mode, mode)
    allowed_modes = set(module.get("allowed_modes") or [])
    if effective_mode not in allowed_modes:
        raise ValueError(f"Module {module['module_id']} is not allowed in mode {mode}")

    leakage_tier = str(module.get("leakage_tier") or "strict_safe")
    allowed_leakage = MODE_ALLOWED_LEAKAGE.get(effective_mode, {"strict_safe"})
    if leakage_tier not in allowed_leakage:
        raise ValueError(
            f"Module {module['module_id']} has leakage tier {leakage_tier}, not allowed in mode {mode}"
        )


def core_case(case: dict[str, Any]) -> dict[str, Any]:
    value = case.get("core_case")
    return value if isinstance(value, dict) else {}


def normalize_article_token(value: Any) -> str | None:
    token = str(value or "").strip().upper()
    if not token:
        return None
    token = token.replace("ARTICLE ", "").replace("ART. ", "").replace("ART ", "")
    token = token.replace("PROTOCAL", "PROTOCOL")
    token = token.replace("_", "-")
    token = token.replace("P1A1", "P1-1")
    token = token.replace("A1P1", "P1-1")
    token = token.replace("P1-01", "P1-1")
    return token or None


def extend_article_list(found: list[str], raw: Any) -> None:
    if raw is None:
        return
    if isinstance(raw, list):
        for item in raw:
            extend_article_list(found, item)
        return
    if isinstance(raw, dict):
        for value in raw.values():
            extend_article_list(found, value)
        return

    text = str(raw).strip()
    if not text:
        return

    parts = ARTICLE_TOKEN_RE.split(text) if any(sep in text for sep in ";,/|") else [text]
    for part in parts:
        normalized = normalize_article_token(part)
        if normalized and normalized not in found:
            found.append(normalized)


def extract_violated_articles(case: dict[str, Any]) -> list[str]:
    source_case = core_case(case) or case
    found: list[str] = []
    for c in source_case.get("conclusion", []):
        if c.get("type") == "violation":
            base = c.get("base_article") or c.get("article")
            normalized = normalize_article_token(base)
            if normalized and normalized not in found:
                found.append(normalized)
    if not found:
        extend_article_list(found, source_case.get("violated_articles"))
    if not found:
        extend_article_list(found, source_case.get("oracle_violated_articles"))
    if not found:
        extend_article_list(found, source_case.get("violated_articles_text"))
    if not found:
        extend_article_list(found, source_case.get("mentioned_articles"))
    if not found:
        extend_article_list(found, source_case.get("article"))
    return found


def infer_art6_limb(case_like: dict[str, Any]) -> dict[str, Any]:
    hint = str(case_like.get("art6_limb_hint", "")).lower()
    if hint in {"civil", "criminal"}:
        return {
            "limb": hint,
            "source": "explicit_hint",
            "ambiguous": False,
        }

    text = json.dumps(case_like, ensure_ascii=False).lower()
    criminal_markers = ["criminal", "prosecutor", "conviction", "sentence", "defence"]
    civil_markers = ["civil", "compensation", "administrative", "labour", "employment"]
    has_criminal = any(m in text for m in criminal_markers)
    has_civil = any(m in text for m in civil_markers)

    if has_criminal and not has_civil:
        return {
            "limb": "criminal",
            "source": "heuristic_criminal_markers",
            "ambiguous": False,
        }
    if has_civil and not has_criminal:
        return {
            "limb": "civil",
            "source": "heuristic_civil_markers",
            "ambiguous": False,
        }
    return {
        "limb": "ambiguous",
        "source": "heuristic_default",
        "ambiguous": True,
    }


def infer_formation(case: dict[str, Any]) -> str:
    source_case = core_case(case) or case
    raw = str(source_case.get("doctypebranch", "")).upper()
    if "GRAND" in raw:
        return "Grand_Chamber"
    if "COMMITTEE" in raw:
        return "Committee"
    return "Chamber"


def first_non_empty(case: dict[str, Any], *keys: str) -> Any:
    source_case = core_case(case)
    for key in keys:
        value = case.get(key)
        if value not in (None, "", []):
            return value
        if source_case:
            value = source_case.get(key)
            if value not in (None, "", []):
                return value
    return None


def extract_year(case: dict[str, Any]) -> int | None:
    raw = str(first_non_empty(case, "judgementdate", "judgment_date", "judgment_year") or "")
    match = re.search(r"\b(19|20)\d{2}\b", raw)
    if match:
        return int(match.group(0))
    return None


def build_route_state(case: dict[str, Any], mode: str) -> dict[str, Any]:
    violated_articles = extract_violated_articles(case)
    unsupported_articles = [art for art in violated_articles if art not in SUPPORTED_ARTICLES]
    route_state = {
        "mode": mode,
        "violated_articles": violated_articles,
        "respondent_state": normalize_respondent_state(
            first_non_empty(case, "respondent", "respondent_state", "country_alpha2", "respondent_country")
        ),
        "judgment_year": extract_year(case),
        "court_formation": infer_formation(case),
        "case_importance": first_non_empty(case, "importance", "case_importance"),
        "available_input_types": ["oracle_articles"],
        "extracted_hints": {},
        "unsupported_articles": unsupported_articles,
        "generic_fallback_active": bool(unsupported_articles),
        "routing_warnings": [],
    }

    if unsupported_articles:
        route_state["routing_warnings"].append(
            f"generic_article_fallback_used_for={','.join(unsupported_articles)}"
        )

    if mode == AWARD_REDACTED_MODE:
        route_state["available_input_types"].extend(
            [
                "zeroshot_standard_combined_input_text",
                "target_extraction_sidecars_by_itemid",
                "metadata",
                "extracted",
                "article41_claim_layer",
                "article41_reasoning_minus_target_nonpec_award",
                "train_empirical_priors",
            ]
        )
        route_state["extracted_hints"] = {
            "violation_type": first_non_empty(case, "violation_type"),
            "violation_subtype": first_non_empty(case, "violation_subtype"),
            "violation_duration_months": first_non_empty(case, "violation_duration_months"),
            "num_applicants": first_non_empty(case, "num_applicants", "num_applicants_proxy"),
            "claim_type": first_non_empty(case, "claim_non_pec_state", "claim_type"),
            "claim_amount": first_non_empty(case, "claim_non_pec_original_amount", "claim_amount"),
        }

    if "6" in violated_articles:
        art6 = infer_art6_limb(case)
        route_state["art6_limb"] = art6["limb"]
        route_state["art6_limb_source"] = art6["source"]
        route_state["art6_limb_ambiguous"] = art6["ambiguous"]
        if art6["ambiguous"]:
            route_state["art6_limb_candidates"] = ["civil", "criminal"]
            route_state["routing_warnings"].append("art6_limb_ambiguous_dual_routing")

    return route_state


def _normalize_heading_key(text: Any) -> str:
    normalized = str(text or "").strip().upper().replace("’", "'")
    return re.sub(r"[^A-Z0-9]+", "", normalized)


def _is_heading_only_placeholder(text: Any) -> bool:
    key = _normalize_heading_key(text)
    return bool(key) and key in INPUT_HEADING_ONLY_KEYS


def _is_useful_input_text(text: Any) -> bool:
    cleaned = str(text or "").strip()
    return bool(cleaned) and not _is_heading_only_placeholder(cleaned)


def _flatten_content_node(node: dict[str, Any]) -> str:
    lines: list[str] = []

    def _walk(n: dict[str, Any]) -> None:
        content = str(n.get("content") or "").strip()
        if content:
            lines.append(content)
        elements = n.get("elements") or []
        if isinstance(elements, list):
            for child in elements:
                if isinstance(child, dict):
                    _walk(child)

    _walk(node)
    deduped: list[str] = []
    prev = None
    for line in lines:
        if line == prev:
            continue
        deduped.append(line)
        prev = line
    return "\n\n".join(deduped).strip()


def _iter_case_sections(case: dict[str, Any], itemid: str) -> list[dict[str, Any]]:
    content = case.get("content")
    if not isinstance(content, dict) or not content:
        return []

    preferred_keys = [f"{itemid}.docx", itemid] if itemid else []
    for key in preferred_keys:
        sections = content.get(key)
        if isinstance(sections, list):
            return [section for section in sections if isinstance(section, dict)]

    for sections in content.values():
        if isinstance(sections, list):
            return [section for section in sections if isinstance(section, dict)]

    return []


def _extract_section_text_from_case_content(
    case: dict[str, Any],
    itemid: str,
    section_names: set[str],
) -> str:
    lowered_names = {name.lower() for name in section_names}
    for section in _iter_case_sections(case, itemid):
        section_name = str(section.get("section_name") or "").strip().lower()
        if section_name not in lowered_names:
            continue
        text = _flatten_content_node(section)
        if _is_useful_input_text(text):
            return text
        heading = str(section.get("content") or "").strip()
        if _is_useful_input_text(heading):
            return heading
    return ""


def extract_facts_text(case: dict[str, Any]) -> str:
    facts_procedure = case.get("facts_procedure") or {}
    evidence_inputs = facts_procedure.get("evidence_inputs") or {}
    facts_text = str(evidence_inputs.get("facts_text") or case.get("facts_text") or "").strip()
    if _is_useful_input_text(facts_text):
        return facts_text
    facts_text = str(case.get("facts_section_text") or "").strip()
    if _is_useful_input_text(facts_text):
        return facts_text

    itemid = str(case.get("itemid") or "").strip()

    facts_text = _extract_section_text_from_case_content(case, itemid, {"facts"})
    if _is_useful_input_text(facts_text):
        return facts_text

    facts_text = extract_facts_text_from_store(itemid)
    if _is_useful_input_text(facts_text):
        return facts_text
    return ""


def extract_procedure_text(case: dict[str, Any]) -> str:
    facts_procedure = case.get("facts_procedure") or {}
    evidence_inputs = facts_procedure.get("evidence_inputs") or {}
    procedure_text = str(evidence_inputs.get("procedure_text") or case.get("procedure_text") or "").strip()
    if _is_useful_input_text(procedure_text):
        return procedure_text
    itemid = str(case.get("itemid") or "").strip()

    procedure_text = _extract_section_text_from_case_content(case, itemid, {"procedure"})
    if _is_useful_input_text(procedure_text):
        return procedure_text

    procedure_text = extract_procedure_text_from_store(itemid)
    if _is_useful_input_text(procedure_text):
        return procedure_text
    return ""


@lru_cache(maxsize=1)
def _load_raw_facts_helpers() -> tuple[Any, Any, Any] | None:
    extraction_code_dir_text = os.environ.get("ECTHR_NPD_EXTRACTION_CODE_DIR", "").strip()
    if not extraction_code_dir_text:
        return None
    extraction_code_dir = Path(extraction_code_dir_text)
    if not extraction_code_dir.exists():
        return None
    sys.path.insert(0, str(extraction_code_dir))
    from case_store import load_case_from_store  # type: ignore
    from build_extraction_layers import build_profiled_sections, top_level_sections  # type: ignore

    return load_case_from_store, build_profiled_sections, top_level_sections


@lru_cache(maxsize=4096)
def _load_raw_case_from_store(itemid: str) -> dict[str, Any] | None:
    if not itemid:
        return None
    helpers = _load_raw_facts_helpers()
    if not helpers:
        return None
    load_case_from_store, _, _ = helpers
    raw_case = load_case_from_store(itemid)
    return raw_case if isinstance(raw_case, dict) else None


def extract_facts_text_from_store(itemid: str) -> str:
    if not itemid:
        return ""
    helpers = _load_raw_facts_helpers()
    if not helpers:
        return ""
    _, build_profiled_sections, top_level_sections = helpers
    raw_case = _load_raw_case_from_store(itemid)
    if not raw_case:
        return ""
    _, sections = top_level_sections(raw_case)
    profiled = build_profiled_sections(raw_case, sections, None, None)
    facts_text = str(profiled.get("facts_text") or "").strip()
    if _is_useful_input_text(facts_text):
        return facts_text

    summary_intro = str(profiled.get("summary_intro_text") or "").strip()
    assessment = str(profiled.get("assessment_summary_text") or "").strip()
    summary_fallback = "\n\n".join(part for part in [summary_intro, assessment] if part).strip()
    if _is_useful_input_text(summary_fallback):
        return summary_fallback

    intro_text = str(profiled.get("introduction_text") or "").strip()
    if _is_useful_input_text(intro_text):
        return intro_text

    law_fallback = str(profiled.get("law_text_excluding_article_41") or "").strip()
    if _is_useful_input_text(law_fallback):
        return law_fallback

    return ""


def extract_procedure_text_from_store(itemid: str) -> str:
    if not itemid:
        return ""
    helpers = _load_raw_facts_helpers()
    if not helpers:
        return ""
    _, build_profiled_sections, top_level_sections = helpers
    raw_case = _load_raw_case_from_store(itemid)
    if not raw_case:
        return ""
    _, sections = top_level_sections(raw_case)
    profiled = build_profiled_sections(raw_case, sections, None, None)
    return str(profiled.get("procedure_text") or "").strip()


def _clean_appendix_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text)


def _appendix_key_is_safe(key: str) -> bool:
    lowered = key.strip().lower()
    if not lowered:
        return False
    return not any(term in lowered for term in APPENDIX_BLOCKED_KEY_TERMS)


def extract_safe_appendix_text(
    case: dict[str, Any],
    procedure_text: str = "",
    facts_text: str = "",
) -> str:
    itemid = str(case.get("itemid") or "").strip()
    raw_case = _load_raw_case_from_store(itemid)
    if not raw_case:
        return ""

    indicator_text = " ".join(
        part
        for part in [
            procedure_text,
            facts_text,
            json.dumps(raw_case.get("content") or {}, ensure_ascii=False)[:4000],
        ]
        if part
    ).lower()
    if not any(term in indicator_text for term in APPENDIX_REFERENCE_TERMS):
        return ""

    attachments = raw_case.get("attachments")
    if not isinstance(attachments, dict) or not attachments:
        return ""

    appendix_lines: list[str] = []
    for doc_key, tables in attachments.items():
        if not isinstance(tables, dict):
            continue
        for table_name, rows in tables.items():
            if not isinstance(rows, list):
                continue
            for idx, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                safe_pairs: list[str] = []
                for key, value in row.items():
                    if not _appendix_key_is_safe(str(key)):
                        continue
                    cleaned = _clean_appendix_value(value)
                    if cleaned:
                        safe_pairs.append(f"{key}: {cleaned}")
                if safe_pairs:
                    appendix_lines.append(
                        f"{doc_key}/{table_name}/row-{idx}: " + " | ".join(safe_pairs)
                    )

    if not appendix_lines:
        return ""

    return (
        "SAFE APPENDIX FACTS (amount, claim, and award columns removed)\n\n"
        + "\n".join(f"- {line}" for line in appendix_lines)
    )


def extract_procedure_facts_text(case: dict[str, Any]) -> str:
    procedure_text = extract_procedure_text(case)
    facts_text = extract_facts_text(case)
    appendix_safe_text = extract_safe_appendix_text(case, procedure_text, facts_text)
    return "\n\n".join(
        part for part in [procedure_text, facts_text, appendix_safe_text] if part
    ).strip()


def extract_safe_metadata(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "respondent_state": first_non_empty(case, "country_alpha2", "respondent", "respondent_state", "respondent_country"),
        "judgment_year": extract_year(case) or case.get("judgment_year"),
        "court_formation": infer_formation(case),
        "case_importance": first_non_empty(case, "case_importance", "importance"),
    }


def extract_safe_extracted_hints(case: dict[str, Any]) -> dict[str, Any]:
    facts_procedure = case.get("facts_procedure") or {}
    status = facts_procedure.get("status") or {}
    reasoning_layer = case.get("reasoning_layer") or {}
    return {
        "num_applicants": first_non_empty(case, "num_applicants", "num_applicants_proxy", "bc_reconciliation_num_applicants"),
        "is_joint_application": facts_procedure.get("is_joint_application"),
        "victim_relationship": facts_procedure.get("is_indirect_victim"),
        "vulnerability": status.get("vulnerability_tags") or status.get("is_vulnerable"),
        "is_repetitive_case": facts_procedure.get("is_repetitive_case"),
        "domestic_award_prior": facts_procedure.get("domestic_award_prior"),
        "domestic_award_prior_eur": facts_procedure.get("domestic_award_prior_eur"),
        "state_remedial_measures": facts_procedure.get("state_remedial_measures"),
        "violation_type": case.get("violation_type"),
        "violation_subtype": first_non_empty(case, "violation_subtype") or reasoning_layer.get("violation_subtype"),
        "violation_duration_months": first_non_empty(case, "violation_duration_months"),
    }


def _is_target_nonpec_award_key(key: Any, path: tuple[str, ...] = ()) -> bool:
    lowered = str(key or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("claim_non_pec") or lowered.startswith("claim_"):
        return False
    if lowered in TARGET_NONPEC_AWARD_KEYS:
        return True
    if any(lowered.startswith(prefix) for prefix in TARGET_NONPEC_AWARD_KEY_PREFIXES):
        return True
    if "safe_non_pec" in lowered:
        return True
    if "raw_extractor_non_pec" in lowered:
        return True
    if "non_pec" in lowered and any(term in lowered for term in ("award", "label", "match", "source")):
        return True
    if "nonpec" in lowered and any(term in lowered for term in ("award", "label", "match", "source")):
        return True
    if lowered in {"eur_non_pec_sum", "non_pecuniary_eur", "non_pec_eur"}:
        return True
    if lowered.startswith("safe_") and lowered.endswith("_eur"):
        return True

    joined_path = ".".join(path).lower()
    if "awards.non_pecuniary" in joined_path and lowered in {
        "granted",
        "satisfaction_sufficient",
        "original_currency",
        "original_amount",
        "eur_amount",
        "dismissed_reason",
    }:
        return True
    return False


def sanitize_target_award_fields(value: Any, path: tuple[str, ...] = ()) -> tuple[Any, list[str]]:
    redacted_paths: list[str] = []
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        row_head = str(value.get("head") or "").strip().lower()
        for key, child in value.items():
            child_path = (*path, str(key))
            lowered_key = str(key or "").strip().lower()
            if _is_target_nonpec_award_key(key, path):
                sanitized[key] = TARGET_NONPEC_AWARD_FIELD_REDACTION
                redacted_paths.append(".".join(child_path))
                continue
            if row_head == "non_pecuniary" and lowered_key in {
                "eur_amount",
                "original_amount",
                "amount",
                "granted",
                "satisfaction_sufficient",
                "dismissed_reason",
            }:
                sanitized[key] = TARGET_NONPEC_AWARD_FIELD_REDACTION
                redacted_paths.append(".".join(child_path))
                continue
            sanitized_child, child_redactions = sanitize_target_award_fields(child, child_path)
            sanitized[key] = sanitized_child
            redacted_paths.extend(child_redactions)
        return sanitized, redacted_paths
    if isinstance(value, list):
        sanitized_list: list[Any] = []
        for idx, child in enumerate(value):
            sanitized_child, child_redactions = sanitize_target_award_fields(child, (*path, str(idx)))
            sanitized_list.append(sanitized_child)
            redacted_paths.extend(child_redactions)
        return sanitized_list, redacted_paths
    return value, redacted_paths


def _target_nonpec_award_values(case: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def add(raw: Any) -> None:
        if raw in (None, "", []):
            return
        text = str(raw).strip()
        if text and text not in values:
            values.append(text)

    for key in (
        "award_eur",
        "safe_non_pec_eur",
        "award_non_pec_eur_amount",
        "award_non_pec_original_amount",
        "raw_extractor_non_pec_eur",
    ):
        add(first_non_empty(case, key))
    return values


def _amount_variants(value: str) -> set[str]:
    text = str(value or "").strip()
    variants = {text} if text else set()
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return variants

    if number.is_integer():
        integer = int(number)
        plain = str(integer)
        comma = f"{integer:,}"
        space = comma.replace(",", " ")
        nbsp = comma.replace(",", "\u00a0")
        thin_space = comma.replace(",", "\u202f")
        dot = comma.replace(",", ".")
        variants.update({plain, comma, space, nbsp, thin_space, dot, f"{plain}.0"})
    else:
        variants.add(f"{number:.2f}")
    return {variant for variant in variants if variant}


def _text_contains_amount_variant(text: str, variants: set[str]) -> bool:
    normalized_text = text.replace("\u00a0", " ").replace("\u202f", " ")
    normalized_variants = {variant.replace("\u00a0", " ").replace("\u202f", " ") for variant in variants}
    return any(variant and variant in normalized_text for variant in normalized_variants)


def _should_redact_nonpec_award_line(text: str, variants: set[str]) -> bool:
    lowered = text.lower()
    has_nonpec_context = any(term in lowered for term in TARGET_NONPEC_AWARD_TEXT_TERMS)
    has_disposition_context = any(term in lowered for term in TARGET_AWARD_DISPOSITION_TERMS)
    has_amount = _text_contains_amount_variant(text, variants)
    if has_amount and (has_nonpec_context or has_disposition_context):
        if "claim" in lowered and not has_disposition_context:
            return False
        return True
    if has_nonpec_context and (
        "sufficient just satisfaction" in lowered
        or "finding of a violation" in lowered and "just satisfaction" in lowered
        or "makes no award" in lowered
    ):
        return True
    return False


def redact_target_nonpec_award_text(text: Any, case: dict[str, Any]) -> tuple[str, int]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return "", 0

    variants: set[str] = set()
    for value in _target_nonpec_award_values(case):
        variants.update(_amount_variants(value))

    redaction_count = 0
    redacted_lines: list[str] = []
    for line in raw_text.splitlines() or [raw_text]:
        if _should_redact_nonpec_award_line(line, variants):
            redacted_lines.append(TARGET_NONPEC_AWARD_REDACTION)
            redaction_count += 1
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines), redaction_count


def _flatten_award_redacted_content_node(node: dict[str, Any], case: dict[str, Any]) -> tuple[str, int]:
    lines: list[str] = []
    redaction_count = 0

    def _walk(n: dict[str, Any]) -> None:
        nonlocal redaction_count
        content = str(n.get("content") or "").strip()
        if content:
            redacted, count = redact_target_nonpec_award_text(content, case)
            if redacted:
                lines.append(redacted)
            redaction_count += count
        elements = n.get("elements") or []
        if isinstance(elements, list):
            for child in elements:
                if isinstance(child, dict):
                    _walk(child)

    _walk(node)
    deduped: list[str] = []
    prev = None
    for line in lines:
        if line == prev:
            continue
        deduped.append(line)
        prev = line
    return "\n\n".join(deduped).strip(), redaction_count


def extract_award_redacted_full_text(case: dict[str, Any]) -> str:
    itemid = str(case.get("itemid") or "").strip()
    raw_case = _load_raw_case_from_store(itemid) or case
    sections = _iter_case_sections(raw_case, itemid)
    section_blocks: list[str] = []
    redaction_count = 0

    for section in sections:
        section_name = str(section.get("section_name") or "").strip() or "section"
        text, count = _flatten_award_redacted_content_node(section, case)
        if text:
            section_blocks.append(f"## {section_name.upper()}\n{text}")
        redaction_count += count

    if not section_blocks:
        text = extract_procedure_facts_text(case)
        redacted, redaction_count = redact_target_nonpec_award_text(text, case)
        section_blocks.append(redacted)

    header = (
        "TARGET JUDGMENT TEXT WITH TARGET NON-PECUNIARY AWARD REDACTED\n"
        f"Redacted target non-pecuniary award disposition spans: {redaction_count}\n"
    )
    return header + "\n\n".join(part for part in section_blocks if part).strip()


def build_award_redacted_case_inputs(case: dict[str, Any]) -> dict[str, Any]:
    standard_input, standard_redactions = build_standard_prompting_input(case)
    extraction_context, extraction_redactions = build_target_extraction_context(case)
    return {
        "award_redaction_policy": {
            "target_non_pecuniary_award": "redacted",
            "base_input_matches_zeroshot_prompting_standard_input": True,
            "target_extractions_knowledge_and_reference_materials": "allowed_after_target_nonpec_award_redaction",
            "few_shot_reference_non_pecuniary_awards": "allowed_only_for_temporally_prior_train_cases",
        },
        "standard_prompting_input": standard_input,
        "target_extraction_context": extraction_context,
        "target_redaction_report": {
            "redacted_standard_input_paths": sorted(set(standard_redactions)),
            "redacted_extraction_paths": sorted(set(extraction_redactions)),
            "redaction_marker": TARGET_NONPEC_AWARD_REDACTION,
        },
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_standard_prompting_input(case: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    standard = {key: case.get(key) for key in STANDARD_INPUT_KEYS if key in case}
    if "combined_input_text" not in standard:
        introduction = str(first_non_empty(case, "introduction_text") or "").strip()
        procedure = str(first_non_empty(case, "procedure_text") or "").strip()
        facts = str(first_non_empty(case, "facts_text") or "").strip()
        appendix = str(first_non_empty(case, "safe_appendix_text") or "").strip()
        combined = "\n\n".join(part for part in [introduction, procedure, facts, appendix] if part).strip()
        if not combined:
            combined = extract_procedure_facts_text(case)
        standard["combined_input_text"] = combined
    if "oracle_violated_articles" not in standard:
        standard["oracle_violated_articles"] = extract_violated_articles(case)
    sanitized, redactions = sanitize_target_award_fields(standard)
    return sanitized, [f"standard_prompting_input.{path}" for path in redactions]


@lru_cache(maxsize=16)
def _load_sidecar_index(relative_path_text: str, multi: bool) -> dict[str, Any]:
    path = Path(relative_path_text)
    if not path.exists():
        return {}
    index: dict[str, Any] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            itemid = str(row.get("itemid") or "").strip()
            if not itemid:
                continue
            if multi:
                index.setdefault(itemid, []).append(row)
            else:
                index[itemid] = row
    return index


def load_extraction_sidecars(itemid: str) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, relative_path in EXTRACTION_SIDECAR_RELATIVE_PATHS.items():
        multi = name == "per_applicant_alloc_normalized"
        index = _load_sidecar_index(str(relative_path), multi)
        value = index.get(itemid, [] if multi else {})
        if value:
            loaded[name] = value
    return loaded


def build_target_extraction_context(case: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    itemid = str(case.get("itemid") or "").strip()
    context: dict[str, Any] = {
        "lookup_key": {"itemid": itemid},
        "sidecar_sources": {
            name: str(path).replace("\\", "/")
            for name, path in EXTRACTION_SIDECAR_RELATIVE_PATHS.items()
        },
        "sidecars": load_extraction_sidecars(itemid),
    }
    if "sidecars" not in context or not context["sidecars"]:
        context["inline_case_fields"] = {
            key: value
            for key, value in case.items()
            if key not in STANDARD_INPUT_KEYS
        }
    sanitized, redactions = sanitize_target_award_fields(context)
    return sanitized, [f"target_extraction_context.{path}" for path in redactions]


def build_case_inputs(case: dict[str, Any], route_state: dict[str, Any]) -> dict[str, Any]:
    mode = route_state["mode"]
    case_inputs: dict[str, Any] = {
        "itemid": case.get("itemid"),
        "oracle_violated_articles": route_state["violated_articles"],
    }

    if mode == AWARD_REDACTED_MODE:
        case_inputs.update(build_award_redacted_case_inputs(case))
        return case_inputs

    raise ValueError(f"Unsupported v3 mode: {mode}")


def render_prompt_text(case_inputs: dict[str, Any], payload: dict[str, Any]) -> str:
    sections = [
        "# SYSTEM POLICY",
        "\n\n".join(item["text"] for item in payload["binary_modules"] if item["module_id"] == "system_policy"),
        "# MODE CONTRACTS",
        "\n\n".join(item["text"] for item in payload["binary_modules"] if item["module_id"] == "mode_contracts"),
        "# ROUTE STATE",
        json.dumps(payload["route_state"], ensure_ascii=False, indent=2),
        "# CASE INPUTS",
        json.dumps(case_inputs, ensure_ascii=False, indent=2),
        "# REGRESSION AND ZERO-AWARD GUIDANCE MODULES",
        "\n\n".join(
            f"## {item['module_id']}\n{item['text']}"
            for item in payload["binary_modules"]
            if item["module_id"] not in {"system_policy", "mode_contracts"}
        ),
        "# ROUTING MODULES",
        "\n\n".join(f"## {item['module_id']}\n{item['text']}" for item in payload["routing_modules"]),
        "# ARTICLE CONSULT MODULES",
        "\n\n".join(f"## {item['module_id']}\n{item['text']}" for item in payload["consult_modules"]),
    ]
    
    prompt_tail = """
# FINAL INSTRUCTION
Now, please carefully read the above policies, mode contracts, route state, and input facts.
Perform your step-by-step reasoning applying the provided module guidance.
The target case base input is the same standard input used in the earlier zeroshot prompting experiment. Target extraction sidecars are attached by itemid after target non-pecuniary award redaction. Do not infer any redacted target non-pecuniary award value from redaction markers.
You MUST output your final prediction as a valid JSON object matching the `Final Prediction Object` schema defined in the SYSTEM POLICY / OUTPUT SCHEMA.
Do not output anything outside the JSON block. Your output should begin with `{` and end with `}`.
"""
    
    return "\n\n".join(section for section in sections if section.strip()) + prompt_tail

def select_binary_modules(mode: str) -> list[str]:
    selected = [
        "system_policy",
        "mode_contracts",
        "failure_modes",
        "zero_award_rules",
        "finding_sufficient_guidance",
        "output_schema",
    ]
    if mode == AWARD_REDACTED_MODE:
        selected.append("claim_rules")
    return selected


def select_article_modules(case: dict[str, Any], route_state: dict[str, Any]) -> list[str]:
    selected: list[str] = []

    def add_unique(module_id: str) -> None:
        if module_id not in selected:
            selected.append(module_id)

    for art in route_state["violated_articles"]:
        if art == "6":
            limb = route_state.get("art6_limb") or infer_art6_limb(case)["limb"]
            if limb == "criminal":
                add_unique("art6_criminal")
            elif limb == "civil":
                add_unique("art6_civil")
            else:
                add_unique("art6_civil")
                add_unique("art6_criminal")
        else:
            add_unique(ARTICLE_MAP.get(art, "generic_article_fallback"))
    if len(route_state["violated_articles"]) > 1 or any(a in {"13", "14"} for a in route_state["violated_articles"]):
        add_unique("cross_article_synthesis")
    return selected


def load_module_record(
    kb_dir: Path,
    modules: dict[str, dict[str, Any]],
    module_id: str,
    mode: str,
) -> dict[str, Any]:
    module = modules[module_id]
    validate_module_for_mode(module, mode)
    return {
        "module_id": module_id,
        "text": load_module_text(kb_dir, modules, module_id),
        "module_type": module.get("module_type"),
        "leakage_tier": module.get("leakage_tier"),
    }


def assemble_payload(kb_dir: Path, case: dict[str, Any], mode: str) -> dict[str, Any]:
    kb_index = load_kb_index(kb_dir)
    validate_mode(kb_index, mode)
    modules = index_modules(kb_index)
    route_state = build_route_state(case, mode)
    binary_module_ids = select_binary_modules(mode)
    consult_module_ids = select_article_modules(case, route_state)
    routing_module_ids = ["router_policy", "fallback_policy"]
    if "6" in route_state["violated_articles"]:
        routing_module_ids.append("art6_limb_router")

    empirical_calibration = resolve_empirical_calibration(kb_dir, route_state)
    route_state["empirical_calibration"] = empirical_calibration
    route_state["calibration_sources_used"] = empirical_calibration["calibration_sources_used"]

    if empirical_calibration["weak_support_detected"] or route_state.get("art6_limb_ambiguous"):
        route_state["calibration_uncertainty"] = True
        route_state["routing_warnings"].append("calibration_uncertainty")
    else:
        route_state["calibration_uncertainty"] = False

    case_inputs = build_case_inputs(case, route_state)
    payload = {
        "route_state": route_state,
        "case_inputs": case_inputs,
        "binary_modules": [
            load_module_record(kb_dir, modules, mid, mode)
            for mid in binary_module_ids
        ],
        "routing_modules": [
            load_module_record(kb_dir, modules, mid, mode)
            for mid in routing_module_ids
        ],
        "consult_modules": [
            load_module_record(kb_dir, modules, mid, mode)
            for mid in consult_module_ids
        ],
    }
    payload["prompt_text"] = render_prompt_text(case_inputs, payload)
    return payload


def print_summary(payload: dict[str, Any]) -> None:
    print("=" * 72)
    print("KB v3 Award-Redacted Agentic Dry Run")
    print("=" * 72)
    print(json.dumps(payload["route_state"], indent=2, ensure_ascii=False))
    print("-" * 72)
    print("Case inputs:")
    print(json.dumps(payload["case_inputs"], indent=2, ensure_ascii=False)[:1200])
    print("-" * 72)
    print("Regression guidance modules:")
    for item in payload["binary_modules"]:
        print(f"  - {item['module_id']}")
    print("Routing modules:")
    for item in payload["routing_modules"]:
        print(f"  - {item['module_id']}")
    print("Consult modules:")
    for item in payload["consult_modules"]:
        print(f"  - {item['module_id']}")
    print("-" * 72)
    print(f"Prompt chars: {len(payload['prompt_text'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_file", required=True)
    parser.add_argument("--mode", default=AWARD_REDACTED_MODE, choices=[AWARD_REDACTED_MODE])
    parser.add_argument("--kb_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prompt_out", default=None, help="Optional path to write rendered prompt text.")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    case_path = Path(args.case_file)
    if case_path.suffix == ".jsonl":
        with case_path.open("r", encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), "")
        if not first_line:
            raise ValueError(f"No JSONL rows found in {case_path}")
        case = json.loads(first_line)
    else:
        case = json.loads(case_path.read_text(encoding="utf-8"))
    if isinstance(case, list):
        case = case[0]

    payload = assemble_payload(kb_dir, case, args.mode)
    if args.prompt_out:
        Path(args.prompt_out).write_text(payload["prompt_text"], encoding="utf-8")
    if args.dry_run:
        print_summary(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

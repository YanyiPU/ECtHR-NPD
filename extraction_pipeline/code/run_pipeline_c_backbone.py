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

from openai_compatible_client import ApiCallError, OpenAICompatibleClient
from pipeline_c_backbone_deterministic import layer1_deterministic
from shared_compensation_evidence import KNOWN_CURRENCY_CODES, clean_text


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = EXTRACTION_ROOT.parent
INPUT_JSONL = EXTRACTION_ROOT / "outputs" / "case_features_labels.jsonl"
SPLITS_DIR = DATASET_ROOT / "splits"
PROMPT_PATH = EXTRACTION_ROOT / "prompts" / "pipeline_c_backbone_system_prompt.md"
SCHEMA_PATH = EXTRACTION_ROOT / "schemas" / "pipeline_c_backbone.schema.json"
RUNS_ROOT = EXTRACTION_ROOT / "runs" / "pipeline_c_backbone"

DEFAULT_SPLITS = ["set1_primary", "set2_postcut_ood", "set3_challenging"]
WRITE_LOCK = threading.Lock()
TOLERANCE_PER_HEAD = {
    "non_pecuniary": 1.0,
    "pecuniary": 1.0,
    "costs": 0.01,
    "bundled": 1.0,
}
LEGACY_CURRENCY_CODES = (
    "LTL",
    "ITL",
    "FRF",
    "DEM",
    "ESP",
    "BEF",
    "NLG",
    "ATS",
    "PTE",
    "GRD",
    "SIT",
    "SKK",
    "EEK",
    "CYP",
    "MTL",
    "TRL",
    "ROL",
    "RUR",
)
CURRENCY_NAME_TO_CODE = {
    "italian lire": "ITL",
    "italian lira": "ITL",
    "french francs": "FRF",
    "french franc": "FRF",
    "deutsche marks": "DEM",
    "german marks": "DEM",
    "german mark": "DEM",
    "pounds sterling": "GBP",
    "us dollars": "USD",
    "us dollar": "USD",
    "lithuanian litai": "LTL",
    "lithuanian litas": "LTL",
    "turkish liras": "TRY",
    "turkish lira": "TRY",
    "turkish pounds": "TRL",
    "turkish pound": "TRL",
}
ALL_CURRENCY_CODES = tuple(dict.fromkeys((*KNOWN_CURRENCY_CODES, *LEGACY_CURRENCY_CODES)))
GENERIC_CURRENCY_NAME_RE = r"(?:%s)" % "|".join(sorted((re.escape(name) for name in CURRENCY_NAME_TO_CODE), key=len, reverse=True))
GENERIC_CURRENCY_TOKEN_RE = r"(?:euros?|eur|" + "|".join(ALL_CURRENCY_CODES) + r"|" + GENERIC_CURRENCY_NAME_RE + r")"
AMOUNT_TOKEN_RE = r"[0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.\d+)?|[0-9]+(?:\.\d+)?"
CLAIM_WITH_AMOUNT_RE = re.compile(r"\b(claim(?:ed|s)?|request(?:ed|s)?|sought)\b", re.IGNORECASE)
AWARD_WITH_AMOUNT_RE = re.compile(r"\b(award(?:ed|s)?|granted)\b", re.IGNORECASE)
CLAIM_HEAD_HINTS = {
    "non_pecuniary": re.compile(r"\b(non-pecuniary|non pecuniary|moral damage)\b", re.IGNORECASE),
    "pecuniary": re.compile(
        r"\b(pecuniary|material damage|loss of earnings?|loss of income|lost income|loss of opportunities?|loss of opportunity|lost profits?)\b",
        re.IGNORECASE,
    ),
    "costs": re.compile(r"\b(costs?(?: and expenses)?|expenses)\b", re.IGNORECASE),
}
LEAVE_TO_COURT_RE = re.compile(
    r"\bleft (?:the amount|the matter|the sum|the issue) to the Court'?s discretion\b"
    r"|\bleft it to the Court to determine\b",
    re.IGNORECASE,
)
EXPLICIT_CLAIM_PATTERNS = {
    "non_pecuniary": re.compile(
        rf"\b(?:claim(?:ed|s)?|request(?:ed|s)?|sought)\b[^.\n]{{0,80}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"(?:[^.\n()]{{0,80}}?\((?:approximately|approx\.?|about)?\s*(?P<eur_currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<eur_amount>{AMOUNT_TOKEN_RE})\))?"
        rf"[^.\n]{{0,240}}?\b(?:in respect of|for|as compensation for)\b[^.\n]{{0,160}}?\b(non-pecuniary|non pecuniary|moral damage)\b",
        re.IGNORECASE,
    ),
    "pecuniary": re.compile(
        rf"\b(?:claim(?:ed|s)?|request(?:ed|s)?|sought)\b[^.\n]{{0,80}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"(?:[^.\n()]{{0,80}}?\((?:approximately|approx\.?|about)?\s*(?P<eur_currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<eur_amount>{AMOUNT_TOKEN_RE})\))?"
        rf"[^.\n]{{0,240}}?\b(?:in respect of|for|as compensation for)\b[^.\n]{{0,160}}?\b(pecuniary|material damage|loss of earnings?|loss of income|lost income|loss of opportunities?|loss of opportunity|lost profits?)\b",
        re.IGNORECASE,
    ),
    "costs": re.compile(
        rf"\b(?:claim(?:ed|s)?|request(?:ed|s)?|sought)\b[^.\n]{{0,80}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"(?:[^.\n()]{{0,80}}?\((?:approximately|approx\.?|about)?\s*(?P<eur_currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<eur_amount>{AMOUNT_TOKEN_RE})\))?"
        rf"[^.\n]{{0,240}}?\b(?:in respect of|for|as compensation for)\b[^.\n]{{0,160}}?\b(costs?(?: and expenses)?|expenses)\b",
        re.IGNORECASE,
    ),
}
EXPLICIT_AWARD_PAREN_PATTERNS = {
    "non_pecuniary": re.compile(
        rf"\b(?:award(?:ed|s)?|grant(?:ed|s)?)\b[^.\n]{{0,120}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"[^.\n()]{{0,80}}?\((?:approximately|approx\.?|about)?\s*(?P<eur_currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<eur_amount>{AMOUNT_TOKEN_RE})\)"
        rf"[^.\n]{{0,260}}?\b(?:in respect of|for|by way of)\b[^.\n]{{0,180}}?\b(non-pecuniary|non pecuniary|moral damage)\b",
        re.IGNORECASE,
    ),
    "pecuniary": re.compile(
        rf"\b(?:award(?:ed|s)?|grant(?:ed|s)?)\b[^.\n]{{0,120}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"[^.\n()]{{0,80}}?\((?:approximately|approx\.?|about)?\s*(?P<eur_currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<eur_amount>{AMOUNT_TOKEN_RE})\)"
        rf"[^.\n]{{0,260}}?\b(?:in respect of|for|by way of)\b[^.\n]{{0,180}}?\b(pecuniary|material damage|loss of earnings?|loss of income|lost income|loss of opportunities?|loss of opportunity|lost profits?)\b",
        re.IGNORECASE,
    ),
    "costs": re.compile(
        rf"\b(?:award(?:ed|s)?|grant(?:ed|s)?)\b[^.\n]{{0,120}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"[^.\n()]{{0,80}}?\((?:approximately|approx\.?|about)?\s*(?P<eur_currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<eur_amount>{AMOUNT_TOKEN_RE})\)"
        rf"[^.\n]{{0,260}}?\b(?:in respect of|for|by way of)\b[^.\n]{{0,180}}?\b(costs?(?: and expenses)?|expenses)\b",
        re.IGNORECASE,
    ),
}
EXPLICIT_AWARD_DIRECT_PATTERNS = {
    "non_pecuniary": re.compile(
        rf"\b(?:award(?:ed|s)?|grant(?:ed|s)?)\b[^.\n]{{0,120}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"[^.\n()]{{0,260}}?\b(?:in respect of|for|by way of)\b[^.\n]{{0,180}}?\b(non-pecuniary|non pecuniary|moral damage)\b",
        re.IGNORECASE,
    ),
    "pecuniary": re.compile(
        rf"\b(?:award(?:ed|s)?|grant(?:ed|s)?)\b[^.\n]{{0,120}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"[^.\n()]{{0,260}}?\b(?:in respect of|for|by way of)\b[^.\n]{{0,180}}?\b(pecuniary|material damage|loss of earnings?|loss of income|lost income|loss of opportunities?|loss of opportunity|lost profits?)\b",
        re.IGNORECASE,
    ),
    "costs": re.compile(
        rf"\b(?:award(?:ed|s)?|grant(?:ed|s)?)\b[^.\n]{{0,120}}?(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>{AMOUNT_TOKEN_RE})"
        rf"[^.\n()]{{0,260}}?\b(?:in respect of|for|by way of)\b[^.\n]{{0,180}}?\b(costs?(?: and expenses)?|expenses)\b",
        re.IGNORECASE,
    ),
}
_ORDINAL_WORD_TO_INDEX = {
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
_ORDINAL_WORD_RE = r"(?:%s)" % "|".join(_ORDINAL_WORD_TO_INDEX.keys())
_CARDINAL_WORD_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_CARDINAL_WORD_RE = r"(?:%s)" % "|".join(_CARDINAL_WORD_TO_INT.keys())
_JOINT_APPLICANT_LABEL_RE = rf"(?P<label>{_ORDINAL_WORD_RE}(?:\s*,\s*{_ORDINAL_WORD_RE})*(?:\s*(?:,)?\s*and\s+{_ORDINAL_WORD_RE})?)\s+applicants?\s+jointly"
_JOINT_AWARD_AMOUNT_BEFORE_LABEL_RE = re.compile(
    rf"(?:EUR|euros?)\s*(?P<amount>{AMOUNT_TOKEN_RE})[^.\n]{{0,160}}?\bto\s+(?:the\s+)?{_JOINT_APPLICANT_LABEL_RE}",
    re.IGNORECASE,
)
_JOINT_AWARD_LABEL_BEFORE_AMOUNT_RE = re.compile(
    rf"\bto\s+(?:the\s+)?{_JOINT_APPLICANT_LABEL_RE}[^.\n]{{0,160}}?(?:EUR|euros?)\s*(?P<amount>{AMOUNT_TOKEN_RE})",
    re.IGNORECASE,
)
_EUR_PAREN_AMOUNT_RE = re.compile(
    rf"\(\s*(?:approximately|approx\.?|about)?\s*EUR\s*(?P<amount>{AMOUNT_TOKEN_RE})\s*\)"
    rf"|\(\s*(?:approximately|approx\.?|about)?\s*(?P<amount_alt>{AMOUNT_TOKEN_RE})\s*euros?\s*\(EUR\)\s*\)",
    re.IGNORECASE,
)
_EUR_DIRECT_AMOUNT_RE = re.compile(rf"\bEUR\s*(?P<amount>{AMOUNT_TOKEN_RE})\b", re.IGNORECASE)
_CLAIM_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
GUIDE = {
    "itemid": "string",
    "article_41_applied": "boolean|null",
    "claims": {
        "bundled_claim": "boolean|null",
        "bundled_original_currency": "3-letter code|null",
        "bundled_original_amount": "number|null",
        "non_pecuniary": {"state": ["explicit_amount", "leave_to_court", "no_claim", "unclear", None], "original_currency": "3-letter code|null", "original_amount": "number|null", "eur_approx_court_stated": "number|null"},
        "pecuniary": {"state": ["explicit_amount", "leave_to_court", "no_claim", "unclear", None], "original_currency": "3-letter code|null", "original_amount": "number|null", "eur_approx_court_stated": "number|null"},
        "costs": {"state": ["explicit_amount", "leave_to_court", "no_claim", "unclear", None], "original_currency": "3-letter code|null", "original_amount": "number|null", "eur_approx_court_stated": "number|null", "scope": ["domestic", "strasbourg", "both", "unspecified", None]},
    },
    "awards": {
        "bundled_award": "boolean|null",
        "bundled_award_eur": "number|null",
        "non_pecuniary": {"granted": "boolean|null", "satisfaction_sufficient": "boolean|null", "original_currency": "3-letter code|null", "original_amount": "number|null", "eur_amount": "court-stated EUR equiv if non-EUR currency, else same as original_amount; null if no EUR stated", "dismissed_reason": ["no_claim", "unsubstantiated", "domestic_award_covers", "rule_60_non_compliance", "untimely", "applicant_deceased_no_heir", None]},
        "pecuniary": {"granted": "boolean|null", "satisfaction_sufficient": "boolean|null", "original_currency": "3-letter code|null", "original_amount": "number|null", "eur_amount": "court-stated EUR equiv if non-EUR currency, else same as original_amount; null if no EUR stated", "no_causal_link": "boolean|null", "dismissed_reason": ["no_claim", "unsubstantiated", "domestic_award_covers", "rule_60_non_compliance", "untimely", "applicant_deceased_no_heir", None]},
        "costs": {"granted": "boolean|null", "original_currency": "3-letter code|null", "original_amount": "number|null", "eur_amount": "court-stated EUR equiv if non-EUR currency, else same as original_amount; null if no EUR stated", "legal_aid_deduction_eur": "number|null", "net_eur": "number|null", "dismissed_reason": ["rule_60_non_compliance", "not_for_convention_purpose", "unsubstantiated", "no_claim", None]},
    },
    "award_per_applicant": [{"beneficiary_label": "string|null (or use applicant_index)", "applicant_index": "integer|null (or use beneficiary_label)", "head": ["pecuniary", "non_pecuniary", "costs", "bundled", None], "eur_amount": "number|null"}],
    "reasoning": {"government_sufficiency_argument": "boolean|null", "government_counter_offer_eur": "number|null", "retrial_recommended": "boolean|null", "award_reason": "string|null", "costs_reason": "string|null"},
}

# Function guide (what each def is responsible for)
# - _explicit_bundled_claim_signalled: Detect explicit bundled-claim phrasing in Article 41 text.
# - _text_has_claim_language: Detect language indicating applicant claims were actually made.
# - _text_has_award_only_language: Detect award-only phrasing that may imply no explicit claim detail.
# - _numeric_amount: Safely coerce numeric-like values to float or None.
# - _clear_claim_head: Reset one claim head to an "unclear" empty state.
# - _normalize_violation_type: Canonicalize violation_type labels to allowed enum values.
# - _sanitize_claims_against_award_context: Reconcile claims section with source text/award context.
# - _normalize_llm_compensation: Normalize Pipeline C LLM payload to schema-safe structure.
# - parse_args: Parse CLI options for Pipeline C execution.
# - load_json: Load JSON content from disk.
# - load_jsonl_by_itemid: Read JSONL and index rows by itemid.
# - load_split_ids: Resolve split case ids from JSON or CSV files.
# - dedupe_preserve_order: De-duplicate ids while preserving order.
# - make_run_dir: Create run directory for artifacts.
# - _amounts_match: Compare numeric amounts with head-specific tolerance.
# - layer3_crossval: Build deterministic cross-validation between regex layer and LLM layer.
# - build_final_awards: Merge regex, LLM, and cross-validation into final awards block.
# - prompt_messages: Build prompt payload for Pipeline C extraction.
# - _coerce_value: Coerce field values to expected primitive/list shapes.
# - _coerce_llm_types: Apply schema-aware coercion pass to LLM compensation output.
# - validate_llm_result: Validate normalized C output against schema and semantic invariants.
# - merge_retry_feedback: Add retry instructions after validation failures.
# - load_existing_results: Load existing successful case ids for resume mode.
# - write_jsonl_line: Thread-safe append to run JSONL.
# - write_case_file: Persist per-case Pipeline C output.
# - compact_compensation_result: Build compact case-level compensation artifact.
# - run_one_case: Execute one Pipeline C extraction with retries and validation.
# - write_per_split_results: Write split-level result files from successful rows.
# - main: CLI entrypoint orchestrating Pipeline C runs.


def _explicit_bundled_claim_signalled(article_41_text: str) -> bool:
    text = (article_41_text or "").lower()
    patterns = [
        r"claimed .* in respect of pecuniary and non-pecuniary damage and costs",
        r"claimed .* in respect of pecuniary and non-pecuniary damage",
        r"claimed .* jointly in respect of .*costs",
        r"claimed a global sum",
        r"claimed a lump sum",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _text_has_claim_language(text: str) -> bool:
    patterns = [
        r"\bapplicants?\s+claimed\b",
        r"\bapplicants?\s+requested\b",
        r"\bapplicants?\s+sought\b",
        r"\bclaimed\s+[^.;:\n]{0,120}\b(?:eur|usd|gbp|azn|rub)\b",
        r"\brequested\s+[^.;:\n]{0,120}\b(?:eur|usd|gbp|azn|rub)\b",
        r"\bsought\s+[^.;:\n]{0,120}\b(?:eur|usd|gbp|azn|rub)\b",
        r"\bleft the amount to the court'?s discretion\b",
        r"\bleft the matter to the court'?s discretion\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _text_has_award_only_language(text: str) -> bool:
    patterns = [
        r"\bconsiders it reasonable to award\b",
        r"\bconsidered it reasonable to award\b",
        r"\bthe court awards?\b",
        r"\baward the sum indicated in the appended table\b",
        r"\baward the sums indicated in the appended table\b",
        r"\bthe following amounts\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _numeric_amount(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clear_claim_head(head_obj: dict[str, Any]) -> dict[str, Any]:
    cleared = dict(head_obj)
    cleared["state"] = "unclear"
    cleared["original_currency"] = None
    cleared["original_amount"] = None
    cleared["eur_approx_court_stated"] = None
    return cleared


def _parse_amount_text(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(" ", "").replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_currency_token(token: Any) -> str | None:
    if token is None:
        return None
    text = str(token).strip()
    lowered = text.casefold()
    if lowered in {"eur", "euro", "euros"}:
        return "EUR"
    mapped_name = CURRENCY_NAME_TO_CODE.get(lowered)
    if mapped_name is not None:
        return mapped_name
    upper = text.upper()
    if upper in ALL_CURRENCY_CODES:
        return upper
    return None


def build_c_input_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    cross_inputs = (row.get("claim_and_award_layer") or {}).get("cross_validation_inputs") or {}
    return {
        "article_41_text": cross_inputs.get("article_41_text") or "",
        "operative_text": cross_inputs.get("operative_text") or "",
        "appendix_table_text": cross_inputs.get("appendix_table_text") or "",
        "claim_narrative_text": cross_inputs.get("claim_narrative_text") or "",
        "scattered_claim_snippets": cross_inputs.get("scattered_claim_snippets") or [],
    }


def _extract_article_41_sections(article_41_text: str) -> dict[str, str]:
    text = article_41_text or ""
    matches = [
        ("pecuniary", re.search(r"^\s*A\.\s*Pecuniary damage\b", text, re.IGNORECASE | re.MULTILINE)),
        ("non_pecuniary", re.search(r"^\s*B\.\s*Non-pecuniary damage\b", text, re.IGNORECASE | re.MULTILINE)),
        ("costs", re.search(r"^\s*C\.\s*Costs and expenses\b", text, re.IGNORECASE | re.MULTILINE)),
    ]
    spans: list[tuple[str, int, int]] = []
    for idx, (head, match) in enumerate(matches):
        if match is None:
            continue
        start = match.end()
        end = len(text)
        for _, next_match in matches[idx + 1:]:
            if next_match is not None:
                end = next_match.start()
                break
        spans.append((head, start, end))
    return {head: text[start:end] for head, start, end in spans}


def _parse_joint_label_indices(label: str) -> list[int]:
    indices: list[int] = []
    for token in re.findall(_ORDINAL_WORD_RE, label.lower()):
        idx = _ORDINAL_WORD_TO_INDEX.get(token)
        if idx is not None and idx not in indices:
            indices.append(idx)
    return indices


def _extract_joint_award_hints(article_41_text: str) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for head, section_text in _extract_article_41_sections(article_41_text).items():
        for pattern in (_JOINT_AWARD_AMOUNT_BEFORE_LABEL_RE, _JOINT_AWARD_LABEL_BEFORE_AMOUNT_RE):
            for match in pattern.finditer(section_text):
                amount = _parse_amount_text(match.group("amount"))
                label = str(match.group("label") or "").strip()
                indices = _parse_joint_label_indices(label)
                if amount is None or len(indices) < 2:
                    continue
                hint = {
                    "head": head,
                    "eur_amount": amount,
                    "applicant_indices": indices,
                    "beneficiary_label": f"{label} applicants jointly",
                }
                if hint not in hints:
                    hints.append(hint)
    return hints


def _claim_rank(state: Any) -> int:
    return {
        "explicit_amount": 4,
        "leave_to_court": 3,
        "no_claim": 2,
        "unclear": 1,
        None: 0,
    }.get(state, 0)


def _merge_claim_head(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if _claim_rank(candidate.get("state")) > _claim_rank(existing.get("state")):
        return candidate
    merged = dict(existing)
    if existing.get("state") == candidate.get("state"):
        for field in ("original_currency", "original_amount", "eur_approx_court_stated", "scope"):
            if merged.get(field) is None and candidate.get(field) is not None:
                merged[field] = candidate.get(field)
    return merged


def _award_head_score(head: dict[str, Any]) -> tuple[int, int, int]:
    return (
        1 if head.get("granted") else 0,
        1 if head.get("original_currency") not in (None, "EUR") else 0,
        1 if head.get("original_amount") is not None else 0,
        1 if head.get("eur_amount") is not None else 0,
    )


def _merge_award_head(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if _award_head_score(candidate) > _award_head_score(existing):
        return candidate
    merged = dict(existing)
    if _award_head_score(candidate) == _award_head_score(existing):
        for field in ("original_currency", "original_amount", "eur_amount", "dismissed_reason", "legal_aid_deduction_eur", "net_eur", "no_causal_link"):
            if merged.get(field) is None and candidate.get(field) is not None:
                merged[field] = candidate.get(field)
    return merged


def _explicit_claim_head(currency_token: Any, amount_token: Any, eur_amount_token: Any, scope: str | None = None) -> dict[str, Any]:
    original_currency = _normalize_currency_token(currency_token)
    original_amount = _parse_amount_text(amount_token)
    eur_approx = _parse_amount_text(eur_amount_token)
    if original_currency == "EUR" and original_amount is not None and eur_approx is None:
        eur_approx = original_amount
    head = {
        "state": "explicit_amount",
        "original_currency": original_currency,
        "original_amount": original_amount,
        "eur_approx_court_stated": eur_approx,
    }
    if scope is not None:
        head["scope"] = scope
    return head


def _sum_claim_head_candidates(candidates: list[dict[str, Any]], scope: str | None = None) -> dict[str, Any] | None:
    if not candidates:
        return None

    eur_values = [value for value in (_numeric_amount(candidate.get("eur_approx_court_stated")) for candidate in candidates) if value is not None]
    eur_total = round(sum(eur_values), 6) if eur_values else None

    original_currencies = {
        str(candidate.get("original_currency"))
        for candidate in candidates
        if candidate.get("original_currency") not in (None, "")
    }
    original_amounts = [_numeric_amount(candidate.get("original_amount")) for candidate in candidates]
    can_sum_original = len(original_currencies) == 1 and all(value is not None for value in original_amounts)
    original_currency = next(iter(original_currencies)) if len(original_currencies) == 1 else None
    original_total = round(sum(value for value in original_amounts if value is not None), 6) if can_sum_original else None

    head = {
        "state": "explicit_amount",
        "original_currency": original_currency,
        "original_amount": original_total,
        "eur_approx_court_stated": eur_total,
    }
    if original_currency == "EUR" and original_total is not None and head["eur_approx_court_stated"] is None:
        head["eur_approx_court_stated"] = original_total
    if scope is not None:
        head["scope"] = scope
    return head


def _leave_to_court_claim_head(scope: str | None = None) -> dict[str, Any]:
    head = _empty_claim_head("leave_to_court")
    if scope is not None:
        head["scope"] = scope
    return head


def _claim_num_applicants(source_row: dict[str, Any] | None) -> int | None:
    if not isinstance(source_row, dict):
        return None
    candidates = [
        ((source_row.get("facts_procedure") or {}).get("num_applicants")),
        ((source_row.get("core_case") or {}).get("num_applicants_proxy")),
        source_row.get("n_applicants"),
    ]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


def _parse_count_token(token: str | None) -> int | None:
    if token is None:
        return None
    cleaned = str(token).strip().lower()
    if not cleaned:
        return None
    if cleaned.isdigit():
        return int(cleaned)
    return _CARDINAL_WORD_TO_INT.get(cleaned)


def _claim_recipient_multiplier(sentence: str, match_start: int, match_end: int, num_applicants: int | None) -> int:
    left = sentence[max(0, match_start - 160):match_start]
    right = sentence[match_end:min(len(sentence), match_end + 80)]
    local_context = f"{left} {right}".lower()

    if "each" not in local_context:
        return 1

    ordinal_hits = re.findall(_ORDINAL_WORD_RE, left.lower())
    if ordinal_hits:
        return len(dict.fromkeys(ordinal_hits))

    count_match = re.search(rf"\b(?:other\s+)?(?P<count>\d+|{_CARDINAL_WORD_RE})\s+applicants?\b", left, re.IGNORECASE)
    if count_match:
        count = _parse_count_token(count_match.group("count"))
        if isinstance(count, int) and count > 0:
            return count

    if re.search(r"\bthe applicants?\b", left, re.IGNORECASE) and isinstance(num_applicants, int) and num_applicants > 0:
        return num_applicants

    return 1


def _extract_eur_claim_candidates_from_section(head: str, section_text: str, num_applicants: int | None, scope: str | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not section_text:
        return candidates

    seen_signatures: set[tuple[str, int, float]] = set()
    sentences = [sentence.strip() for sentence in _CLAIM_SENTENCE_SPLIT_RE.split(clean_text(section_text)) if sentence.strip()]
    for sentence_index, sentence in enumerate(sentences):
        if not CLAIM_WITH_AMOUNT_RE.search(sentence):
            continue

        occupied_spans: list[tuple[int, int]] = []
        for match in _EUR_PAREN_AMOUNT_RE.finditer(sentence):
            amount = _parse_amount_text(match.group("amount") or match.group("amount_alt"))
            if amount is None:
                continue
            multiplier = _claim_recipient_multiplier(sentence, match.start(), match.end(), num_applicants)
            signature = (head, sentence_index, float(match.start()))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            occupied_spans.append(match.span())
            candidate = {
                "state": "explicit_amount",
                "original_currency": None,
                "original_amount": None,
                "eur_approx_court_stated": round(amount * multiplier, 6),
            }
            if scope is not None:
                candidate["scope"] = scope
            candidates.append(candidate)

        for match in _EUR_DIRECT_AMOUNT_RE.finditer(sentence):
            if any(start <= match.start() < end for start, end in occupied_spans):
                continue
            amount = _parse_amount_text(match.group("amount"))
            if amount is None:
                continue
            multiplier = _claim_recipient_multiplier(sentence, match.start(), match.end(), num_applicants)
            signature = (head, sentence_index, float(match.start()))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            candidate = {
                "state": "explicit_amount",
                "original_currency": "EUR",
                "original_amount": round(amount * multiplier, 6),
                "eur_approx_court_stated": round(amount * multiplier, 6),
            }
            if scope is not None:
                candidate["scope"] = scope
            candidates.append(candidate)

    return candidates


def _aggregate_explicit_claim_head_from_section(head: str, section_text: str, num_applicants: int | None, scope: str | None = None) -> dict[str, Any] | None:
    eur_candidates = _extract_eur_claim_candidates_from_section(head, section_text, num_applicants, scope=scope)
    return _sum_claim_head_candidates(eur_candidates, scope=scope)


def _explicit_award_head(
    currency_token: Any,
    amount_token: Any,
    eur_amount_token: Any,
    *,
    head: str,
    awards_regex: dict[str, Any],
    satisfaction_sufficient: bool = False,
) -> dict[str, Any]:
    original_currency = _normalize_currency_token(currency_token)
    original_amount = _parse_amount_text(amount_token)
    eur_amount = _parse_amount_text(eur_amount_token)
    if original_currency == "EUR" and original_amount is not None and eur_amount is None:
        eur_amount = original_amount
    base = {
        "granted": True,
        "satisfaction_sufficient": satisfaction_sufficient if head in {"non_pecuniary", "pecuniary"} else False,
        "original_currency": original_currency,
        "original_amount": original_amount,
        "eur_amount": eur_amount,
        "dismissed_reason": None,
    }
    if head == "pecuniary":
        base["no_causal_link"] = None
    if head == "costs":
        base["legal_aid_deduction_eur"] = awards_regex.get("legal_aid_deduction_eur")
        base["net_eur"] = awards_regex.get("net_costs_eur")
    return base


def _extract_deterministic_claims(source_row: dict[str, Any] | None, article_41_applied: Any) -> dict[str, Any]:
    claim_state = "unclear" if article_41_applied else None
    claims = {
        "bundled_claim": None,
        "bundled_original_currency": None,
        "bundled_original_amount": None,
        "non_pecuniary": _empty_claim_head(claim_state),
        "pecuniary": _empty_claim_head(claim_state),
        "costs": _empty_costs_claim_head(claim_state),
    }
    if not article_41_applied or not isinstance(source_row, dict):
        return claims

    snapshot = build_c_input_snapshot(source_row)
    article_41_text = snapshot.get("article_41_text") or ""
    num_applicants = _claim_num_applicants(source_row)
    section_map = _extract_article_41_sections(article_41_text)

    for head in ("non_pecuniary", "pecuniary", "costs"):
        scope = "strasbourg" if head == "costs" else None
        aggregated = _aggregate_explicit_claim_head_from_section(head, section_map.get(head) or "", num_applicants, scope=scope)
        if aggregated is not None:
            claims[head] = _merge_claim_head(claims[head], aggregated)

    text_blocks: list[str] = [article_41_text]
    if not all((claims.get(head) or {}).get("state") == "explicit_amount" for head in ("non_pecuniary", "pecuniary", "costs")):
        text_blocks.append(snapshot.get("claim_narrative_text") or "")
        for snippet in snapshot.get("scattered_claim_snippets") or []:
            if isinstance(snippet, dict):
                text_blocks.append(str(snippet.get("snippet") or ""))

    for text in text_blocks:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        for head, pattern in EXPLICIT_CLAIM_PATTERNS.items():
            for match in pattern.finditer(cleaned):
                scope = "strasbourg" if head == "costs" else None
                candidate = _explicit_claim_head(
                    match.group("currency"),
                    match.group("amount"),
                    match.group("eur_amount"),
                    scope=scope,
                )
                claims[head] = _merge_claim_head(claims[head], candidate)

        if CLAIM_WITH_AMOUNT_RE.search(cleaned) and LEAVE_TO_COURT_RE.search(cleaned):
            for head, head_re in CLAIM_HEAD_HINTS.items():
                if not head_re.search(cleaned):
                    continue
                scope = "strasbourg" if head == "costs" else None
                claims[head] = _merge_claim_head(claims[head], _leave_to_court_claim_head(scope=scope))

    for snippet in snapshot.get("scattered_claim_snippets") or []:
        if not isinstance(snippet, dict):
            continue
        head = snippet.get("head")
        if head not in {"non_pecuniary", "pecuniary", "costs"}:
            continue
        scope = "strasbourg" if head == "costs" else None
        candidate = _explicit_claim_head(
            snippet.get("currency_code"),
            snippet.get("amount"),
            None,
            scope=scope,
        )
        claims[head] = _merge_claim_head(claims[head], candidate)

    return claims


def _extract_deterministic_awards(
    source_row: dict[str, Any] | None,
    awards_regex: dict[str, Any],
    article_41_applied: Any,
) -> dict[str, Any]:
    satisfaction_sufficient = bool(awards_regex.get("satisfaction_sufficient"))
    awards = {
        "bundled_award": True if isinstance(awards_regex.get("bundled_award_eur"), (int, float)) else None,
        "bundled_award_eur": float(awards_regex["bundled_award_eur"]) if isinstance(awards_regex.get("bundled_award_eur"), (int, float)) else None,
        "non_pecuniary": _deterministic_award_head(awards_regex.get("non_pecuniary_eur"), satisfaction_sufficient=satisfaction_sufficient),
        "pecuniary": _deterministic_pec_award_head(awards_regex.get("pecuniary_eur")),
        "costs": _deterministic_costs_award_head(awards_regex.get("costs_eur"), awards_regex),
    }
    if not article_41_applied or not isinstance(source_row, dict):
        return awards

    snapshot = build_c_input_snapshot(source_row)
    text_blocks: list[str] = [
        snapshot.get("article_41_text") or "",
        snapshot.get("operative_text") or "",
    ]
    for text in text_blocks:
        cleaned = clean_text(text)
        if not cleaned or not AWARD_WITH_AMOUNT_RE.search(cleaned):
            continue
        for head, pattern in EXPLICIT_AWARD_PAREN_PATTERNS.items():
            for match in pattern.finditer(cleaned):
                candidate = _explicit_award_head(
                    match.group("currency"),
                    match.group("amount"),
                    match.group("eur_amount"),
                    head=head,
                    awards_regex=awards_regex,
                    satisfaction_sufficient=satisfaction_sufficient if head == "non_pecuniary" else False,
                )
                awards[head] = _merge_award_head(awards[head], candidate)
        for head, pattern in EXPLICIT_AWARD_DIRECT_PATTERNS.items():
            for match in pattern.finditer(cleaned):
                candidate = _explicit_award_head(
                    match.group("currency"),
                    match.group("amount"),
                    None,
                    head=head,
                    awards_regex=awards_regex,
                    satisfaction_sufficient=satisfaction_sufficient if head == "non_pecuniary" else False,
                )
                awards[head] = _merge_award_head(awards[head], candidate)
    return awards


def _normalize_violation_type(value: Any) -> Any:
    allowed = {"substantive", "procedural", "both", "unknown", None}
    if isinstance(value, list):
        cleaned = [item for item in value if isinstance(item, str) and item in allowed]
        cleaned_unique = []
        for item in cleaned:
            if item not in cleaned_unique:
                cleaned_unique.append(item)
        if "both" in cleaned_unique:
            return "both"
        if "substantive" in cleaned_unique and "procedural" in cleaned_unique:
            return "both"
        if len(cleaned_unique) == 1:
            return cleaned_unique[0]
        if len(cleaned_unique) > 1:
            return None
        if not cleaned_unique:
            return None
    if value in allowed:
        return value
    return value


def _repair_claims_from_source(parsed: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(parsed))
    claims = normalized.get("claims") or {}
    deterministic_claims = _extract_deterministic_claims(row, normalized.get("article_41_applied"))

    for head in ("non_pecuniary", "pecuniary", "costs"):
        head_obj = claims.get(head) or {}
        deterministic_head = deterministic_claims.get(head) or {}
        state = head_obj.get("state")
        deterministic_state = deterministic_head.get("state")

        if state in {None, "unclear"} and deterministic_state in {"explicit_amount", "leave_to_court", "no_claim"}:
            claims[head] = deterministic_head
            continue

        if state == "explicit_amount":
            merged = dict(head_obj)
            for field in ("original_currency", "original_amount", "eur_approx_court_stated", "scope"):
                if merged.get(field) is None and deterministic_head.get(field) is not None:
                    merged[field] = deterministic_head.get(field)
            claims[head] = merged

    normalized["claims"] = claims
    return normalized


def _comparable_claim_amount_eur(head_obj: dict[str, Any]) -> float | None:
    eur_amount = _numeric_amount(head_obj.get("eur_approx_court_stated"))
    if eur_amount is not None:
        return eur_amount
    if head_obj.get("original_currency") == "EUR":
        return _numeric_amount(head_obj.get("original_amount"))
    return None


def _award_amount_for_claim_validation(head: str, award_obj: dict[str, Any]) -> float | None:
    eur_amount = _numeric_amount(award_obj.get("eur_amount"))
    if eur_amount is not None:
        return eur_amount
    if head in {"non_pecuniary", "pecuniary"} and award_obj.get("satisfaction_sufficient") is True:
        return 0.0
    return None


def _validate_claim_award_head(head: str, claims: dict[str, Any], awards: dict[str, Any]) -> dict[str, Any]:
    claim_obj = claims.get(head) or {}
    award_obj = awards.get(head) or {}
    claim_state = claim_obj.get("state")
    award_amount = _award_amount_for_claim_validation(head, award_obj)
    claim_amount = _comparable_claim_amount_eur(claim_obj)
    tolerance = TOLERANCE_PER_HEAD.get(head, 1.0)

    if award_amount is None:
        return {
            "status": "no_award",
            "flag_for_review": False,
            "claim_state": claim_state,
            "claim_amount_eur": claim_amount,
            "award_amount_eur": None,
            "note": "No awarded amount to validate against the claim.",
        }

    if claim_state == "no_claim":
        if abs(award_amount) <= tolerance:
            return {
                "status": "pass_no_claim_zero_award",
                "flag_for_review": False,
                "claim_state": claim_state,
                "claim_amount_eur": 0.0,
                "award_amount_eur": award_amount,
                "note": "no_claim is compatible only with zero award.",
            }
        return {
            "status": "fail_no_claim_positive_award",
            "flag_for_review": True,
            "claim_state": claim_state,
            "claim_amount_eur": 0.0,
            "award_amount_eur": award_amount,
            "note": "Award is positive even though claim state is no_claim.",
        }

    if claim_state == "leave_to_court":
        return {
            "status": "pass_leave_to_court",
            "flag_for_review": False,
            "claim_state": claim_state,
            "claim_amount_eur": None,
            "award_amount_eur": award_amount,
            "note": "leave_to_court permits a definite awarded amount.",
        }

    if claim_state == "explicit_amount":
        if claim_amount is None:
            return {
                "status": "review_explicit_amount_uncomparable",
                "flag_for_review": True,
                "claim_state": claim_state,
                "claim_amount_eur": None,
                "award_amount_eur": award_amount,
                "note": "Claim is explicit but no comparable EUR total is available.",
            }
        if award_amount <= claim_amount + tolerance:
            return {
                "status": "pass_award_leq_claim",
                "flag_for_review": False,
                "claim_state": claim_state,
                "claim_amount_eur": claim_amount,
                "award_amount_eur": award_amount,
                "note": "Award does not exceed the claimed amount.",
            }
        return {
            "status": "fail_award_exceeds_claim",
            "flag_for_review": True,
            "claim_state": claim_state,
            "claim_amount_eur": claim_amount,
            "award_amount_eur": award_amount,
            "note": "Award exceeds the explicit claimed amount.",
        }

    return {
        "status": "review_unclear_claim_with_award",
        "flag_for_review": True,
        "claim_state": claim_state,
        "claim_amount_eur": claim_amount,
        "award_amount_eur": award_amount,
        "note": "Award exists but the claim state is unclear.",
    }


def _validate_claim_award_contract(llm: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(llm, dict):
        return {"flag_for_review": False, "heads": {}}

    claims = llm.get("claims") or {}
    awards = llm.get("awards") or {}
    head_results = {
        head: _validate_claim_award_head(head, claims, awards)
        for head in ("non_pecuniary", "pecuniary", "costs")
    }
    return {
        "flag_for_review": any(payload.get("flag_for_review") for payload in head_results.values()),
        "heads": head_results,
    }


def _sanitize_claims_against_award_context(parsed: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(parsed))
    claims = normalized.get("claims") or {}
    awards = normalized.get("awards") or {}
    reasoning = normalized.get("reasoning") or {}
    cross_inputs = (row.get("claim_and_award_layer") or {}).get("cross_validation_inputs") or {}
    article_41_text = cross_inputs.get("article_41_text") or ""
    operative_text = cross_inputs.get("operative_text") or ""
    claim_narrative_text = cross_inputs.get("claim_narrative_text") or ""
    context_text = " ".join(
        [
            article_41_text,
            operative_text,
            claim_narrative_text,
            str(reasoning.get("award_reason") or ""),
            str(reasoning.get("costs_reason") or ""),
        ]
    )

    if _text_has_claim_language(context_text) or not _text_has_award_only_language(context_text):
        return normalized

    normalized["claims"] = claims
    return normalized


def _repair_award_head_totals_from_rows(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(parsed))
    awards = normalized.get("awards") or {}
    award_rows = normalized.get("award_per_applicant")
    if not isinstance(award_rows, list):
        return normalized

    row_sums: dict[str, float] = {}
    row_counts: dict[str, int] = {}
    row_values: dict[str, list[float]] = {}
    for raw_row in award_rows:
        if not isinstance(raw_row, dict):
            continue
        head = raw_row.get("head")
        if head not in {"non_pecuniary", "pecuniary", "costs"}:
            continue
        amount = _numeric_amount(raw_row.get("eur_amount"))
        if amount is None:
            continue
        row_sums[head] = row_sums.get(head, 0.0) + amount
        row_counts[head] = row_counts.get(head, 0) + 1
        row_values.setdefault(head, []).append(amount)

    for head in ("non_pecuniary", "pecuniary", "costs"):
        row_sum = row_sums.get(head)
        if row_sum is None:
            continue
        head_obj = awards.get(head) or {}
        head_total = _numeric_amount(head_obj.get("eur_amount"))
        row_count = row_counts.get(head, 0)
        if head_total is None:
            head_obj["eur_amount"] = row_sum
            if head_obj.get("original_currency") == "EUR":
                head_obj["original_amount"] = row_sum
            awards[head] = head_obj
            continue
        tolerance = TOLERANCE_PER_HEAD.get(head, 1.0)
        if abs(head_total - row_sum) <= tolerance:
            continue
        # When rows contain multiple allocations but the head total matches only
        # one row, prefer the aggregated row sum as the true head-level amount.
        if row_count > 1 or any(abs(head_total - value) <= tolerance for value in row_values.get(head, [])):
            head_obj["eur_amount"] = row_sum
            if head_obj.get("original_currency") == "EUR":
                head_obj["original_amount"] = row_sum
            awards[head] = head_obj

    normalized["awards"] = awards
    return normalized


def _collapse_joint_award_rows_from_source_text(parsed: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(parsed))
    award_rows = normalized.get("award_per_applicant")
    if not isinstance(award_rows, list) or not award_rows:
        return normalized

    article_41_text = build_c_input_snapshot(row).get("article_41_text") or ""
    hints = _extract_joint_award_hints(article_41_text)
    if not hints:
        return normalized

    rows = list(award_rows)
    for hint in hints:
        head = hint["head"]
        amount = hint["eur_amount"]
        indices = set(hint["applicant_indices"])
        tolerance = TOLERANCE_PER_HEAD.get(head, 1.0)
        matched_positions: list[int] = []
        matched_indices: list[int] = []
        existing_joint = False

        for pos, raw_row in enumerate(rows):
            if not isinstance(raw_row, dict):
                continue
            if raw_row.get("head") != head:
                continue
            label = str(raw_row.get("beneficiary_label") or "").lower()
            if "jointly" in label and _parse_amount_text(raw_row.get("eur_amount")) is not None:
                row_amount = _parse_amount_text(raw_row.get("eur_amount"))
                if row_amount is not None and abs(row_amount - amount) <= tolerance:
                    existing_joint = True
            idx = raw_row.get("applicant_index")
            row_amount = _parse_amount_text(raw_row.get("eur_amount"))
            if not isinstance(idx, int) or idx not in indices or row_amount is None:
                continue
            if abs(row_amount - amount) > tolerance:
                continue
            matched_positions.append(pos)
            matched_indices.append(idx)

        if existing_joint:
            continue
        if set(matched_indices) != indices:
            continue

        insert_at = min(matched_positions)
        for pos in sorted(matched_positions, reverse=True):
            del rows[pos]
        rows.insert(
            insert_at,
            {
                "beneficiary_label": hint["beneficiary_label"],
                "applicant_index": None,
                "head": head,
                "eur_amount": amount,
            },
        )

    normalized["award_per_applicant"] = rows
    return normalized


def _normalize_llm_compensation(parsed: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(parsed))
    normalized = _repair_claims_from_source(normalized, row)
    claims = normalized.get("claims") or {}
    awards = normalized.get("awards") or {}
    award_rows = normalized.get("award_per_applicant")
    article_41_text = ((row.get("claim_and_award_layer") or {}).get("cross_validation_inputs") or {}).get("article_41_text") or ""

    if claims.get("bundled_claim") is True and not _explicit_bundled_claim_signalled(article_41_text):
        claims["bundled_claim"] = None
        claims["bundled_original_currency"] = None
        claims["bundled_original_amount"] = None

    for head in ("non_pecuniary", "pecuniary", "costs"):
        head_obj = claims.get(head) or {}
        if head_obj.get("state") == "explicit_amount":
            if head_obj.get("original_amount") is None and head_obj.get("eur_approx_court_stated") is None:
                head_obj["state"] = "unclear"
        if head_obj.get("original_currency") == "EUR" and head_obj.get("original_amount") is not None and head_obj.get("eur_approx_court_stated") is None:
            head_obj["eur_approx_court_stated"] = head_obj.get("original_amount")
        claims[head] = head_obj

    normalized["claims"] = claims
    for head in ("non_pecuniary", "pecuniary", "costs"):
        head_obj = awards.get(head) or {}
        if head_obj.get("original_currency") == "EUR" and head_obj.get("original_amount") is not None and head_obj.get("eur_amount") is None:
            head_obj["eur_amount"] = head_obj.get("original_amount")
        awards[head] = head_obj
    normalized["awards"] = awards
    if isinstance(award_rows, list):
        normalized_rows: list[dict[str, Any]] = []
        for raw_row in award_rows:
            if not isinstance(raw_row, dict):
                continue
            row_copy = dict(raw_row)
            label = row_copy.get("beneficiary_label")
            if isinstance(label, str):
                row_copy["beneficiary_label"] = label.strip() or None
            elif label is not None:
                row_copy["beneficiary_label"] = None
            idx = row_copy.get("applicant_index")
            if isinstance(idx, int) and idx >= 1 and not row_copy.get("beneficiary_label"):
                row_copy["beneficiary_label"] = f"Applicant {idx}"
            normalized_rows.append(row_copy)
        normalized["award_per_applicant"] = normalized_rows
    normalized = _collapse_joint_award_rows_from_source_text(normalized, row)
    normalized = _repair_award_head_totals_from_rows(normalized)
    return _sanitize_claims_against_award_context(normalized, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run run5-style backbone Pipeline C extraction.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--itemids", nargs="+", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("EXTRACTION_CONCURRENCY", "8")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("EXTRACTION_MAX_RETRIES", "3")))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regex-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_by_itemid(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["itemid"]] = row
    return rows


def load_split_ids(split_name: str) -> list[str]:
    json_path = SPLITS_DIR / f"{split_name}_ids.json"
    if json_path.exists():
        data = load_json(json_path)
        if isinstance(data, list):
            return [str(x) for x in data]
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find split definition for {split_name}")
    with csv_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        itemid_idx = header.index("itemid")
        return [line.rstrip("\n").split(",")[itemid_idx] for line in handle if line.strip()]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def make_run_dir(run_name: str | None) -> Path:
    dirname = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cases").mkdir(exist_ok=True)
    return run_dir


def _amounts_match(a: Any, b: Any, head: str = "non_pecuniary") -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    tolerance = TOLERANCE_PER_HEAD.get(head, 1.0)
    return abs(float(a) - float(b)) <= tolerance


def layer3_crossval(regex: dict[str, Any], llm: dict[str, Any] | None) -> dict[str, Any]:
    llm_awards = (llm or {}).get("awards", {})
    claim_award_validation = _validate_claim_award_contract(llm)

    def llm_eur(head: str) -> Any:
        return (llm_awards.get(head) or {}).get("eur_amount")

    def llm_is_non_eur(head: str) -> bool:
        """True when LLM says the award currency is non-EUR (pre-2002 ITL/FRF/GBP cases etc.)."""
        currency = (llm_awards.get(head) or {}).get("original_currency")
        return currency is not None and currency != "EUR"

    def crossval_head(head: str, regex_key: str) -> tuple[bool | None, str]:
        """Return (match, status).

        Regex is a validator only. If regex misses a head while the LLM extracts
        one, report validator_missing instead of a mismatch. Review should only
        be triggered by actual contradictions, not validator absence.
        """
        regex_val = regex.get(regex_key)
        llm_val = llm_eur(head)
        if regex_val is None:
            if llm_val is None:
                return True, "both_missing"
            if llm_is_non_eur(head):
                # Regex can't reliably recover inline EUR conversions for legacy
                # non-EUR awards; do not treat that validator gap as a mismatch.
                return True, "llm_only_non_eur"
            return None, "validator_missing"
        if llm_val is None:
            return False, "llm_missing"
        match = _amounts_match(regex_val, llm_val, head)
        return match, "match" if match else "mismatch"

    non_pec_match, non_pec_status = crossval_head("non_pecuniary", "non_pecuniary_eur")
    pec_match, pec_status = crossval_head("pecuniary", "pecuniary_eur")
    costs_match, costs_status = crossval_head("costs", "costs_eur")
    notes = {
        key: value
        for key, value in {
            "non_pecuniary": non_pec_status,
            "pecuniary": pec_status,
            "costs": costs_status,
        }.items()
        if value not in {"match", "both_missing"}
    }
    return {
        "non_pec_match": non_pec_match,
        "non_pec_status": non_pec_status,
        "pecuniary_match": pec_match,
        "pecuniary_status": pec_status,
        "costs_match": costs_match,
        "costs_status": costs_status,
        "claim_award_validation": claim_award_validation.get("heads") or {},
        "flag_for_review": any(value is False for value in (non_pec_match, pec_match, costs_match)) or bool(claim_award_validation.get("flag_for_review")),
        **({"notes": notes} if notes else {}),
    }


def build_final_awards(regex: dict[str, Any], llm: dict[str, Any] | None, crossval: dict[str, Any]) -> dict[str, Any]:
    llm_awards = (llm or {}).get("awards", {})
    per_applicant = (llm or {}).get("award_per_applicant") or []

    def per_applicant_sum(head: str) -> Any:
        total = 0.0
        seen = False
        for row in per_applicant:
            if not isinstance(row, dict):
                continue
            if row.get("head") != head:
                continue
            amount = row.get("eur_amount")
            if isinstance(amount, (int, float)):
                total += float(amount)
                seen = True
        return total if seen else None

    def resolve_eur(head: str) -> tuple[Any, str]:
        llm_val = (llm_awards.get(head) or {}).get("eur_amount")
        per_applicant_val = per_applicant_sum(head)
        if llm_val is not None:
            return llm_val, "llm"
        if per_applicant_val is not None:
            return per_applicant_val, "llm_per_applicant_sum"
        return None, "not_awarded"

    non_pec_eur, non_pec_src = resolve_eur("non_pecuniary")
    pec_eur, pec_src = resolve_eur("pecuniary")
    costs_eur, costs_src = resolve_eur("costs")
    
    bundled_award_eur = llm_awards.get("bundled_award_eur")
    bundled_src = "llm" if bundled_award_eur is not None else None
    if bundled_award_eur is None:
        bundled_award_eur = per_applicant_sum("bundled")
        if bundled_award_eur is not None:
            bundled_src = "llm_per_applicant_sum"
    if bundled_src is None:
        bundled_src = "not_awarded"

    satisfaction_sufficient = (llm_awards.get("non_pecuniary") or {}).get("satisfaction_sufficient", False)
    if satisfaction_sufficient and non_pec_eur is None:
        non_pec_eur = 0.0
        non_pec_src = "llm_satisfaction_sufficient"

    numeric_parts = [x for x in (non_pec_eur, pec_eur, costs_eur) if isinstance(x, (int, float))]
    total_eur = sum(numeric_parts) if numeric_parts else bundled_award_eur

    return {
        "non_pecuniary_eur": non_pec_eur,
        "non_pecuniary_source": non_pec_src,
        "pecuniary_eur": pec_eur,
        "pecuniary_source": pec_src,
        "costs_eur": costs_eur,
        "costs_source": costs_src,
        "bundled_award_eur": bundled_award_eur,
        "bundled_award_source": bundled_src,
        "total_eur": total_eur,
        "legal_aid_deduction_eur": (llm_awards.get("costs") or {}).get("legal_aid_deduction_eur"),
        "net_costs_eur": (llm_awards.get("costs") or {}).get("net_eur"),
        "satisfaction_sufficient": satisfaction_sufficient,
        "voting_pattern": regex.get("voting_pattern"),
        "payment_deadline_days": regex.get("payment_deadline_days"),
        "default_interest_formula": regex.get("default_interest_formula"),
        "flag_for_review": crossval.get("flag_for_review", False),
    }


def prompt_messages(system_prompt: str, schema: dict[str, Any], row: dict[str, Any], include_schema_in_payload: bool) -> list[dict[str, str]]:
    core = row["core_case"]
    claim_inputs = row["claim_and_award_layer"]["cross_validation_inputs"]
    payload = {
        "case_identification": {
            "itemid": row["itemid"],
            "appno": row["appno"],
            "judgementdate": row["judgementdate"],
            "respondent_country": core.get("respondent_country"),
            "violated_articles": core.get("violated_articles"),
            "decision_body_category": core.get("decision_body_category"),
        },
        "input_scope": [
            "article_41_or_article_50_text",
            "operative_clauses",
            "appendix_table_text_if_needed",
            "claim_narrative_text_if_needed",
            "scattered_claim_snippets_if_needed",
        ],
        "evidence_inputs": {
            "article_41_text": claim_inputs.get("article_41_text") or "",
            "operative_text": claim_inputs.get("operative_text") or "",
            "appendix_table_text": claim_inputs.get("appendix_table_text") or "",
            "claim_narrative_text": claim_inputs.get("claim_narrative_text") or "",
            "scattered_claim_snippets": claim_inputs.get("scattered_claim_snippets") or [],
        },
        "output_guide": GUIDE,
    }
    if include_schema_in_payload:
        payload["output_schema"] = schema
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


_BOOL_TRUE_STRS = {"true", "1", "yes", "on"}
_BOOL_FALSE_STRS = {"false", "0", "no", "off"}

_ENUM_NORM = {
    "substantive": "substantive",
    "procedural": "procedural",
    "both": "both",
    "unknown": "unknown",
    "explicit_amount": "explicit_amount",
    "leave_to_court": "leave_to_court",
    "no_claim": "no_claim",
    "unclear": "unclear",
    "domestic": "domestic",
    "strasbourg": "strasbourg",
    "both": "both",
    "unspecified": "unspecified",
    "pecuniary": "pecuniary",
    "non_pecuniary": "non_pecuniary",
    "costs": "costs",
    "bundled": "bundled",
    "rule_60_non_compliance": "rule_60_non_compliance",
    "not_for_convention_purpose": "not_for_convention_purpose",
    "unsubstantiated": "unsubstantiated",
    "no_claim": "no_claim",
    "domestic_award_covers": "domestic_award_covers",
    "untimely": "untimely",
    "applicant_deceased_no_heir": "applicant_deceased_no_heir",
}


def _coerce_value(value: Any, field_path: str) -> Any:
    if value is None:
        return None
    if field_path.endswith(".state") or field_path.endswith(".scope") or field_path.endswith(".head") or field_path.endswith(".dismissed_reason"):
        if isinstance(value, str):
            normalized = _ENUM_NORM.get(value.lower())
            if normalized is not None:
                return normalized
    if field_path.endswith(".granted") or field_path.endswith(".satisfaction_sufficient") or field_path.endswith(".no_causal_link") or field_path.endswith(".bundled_claim") or field_path.endswith(".article_41_applied"):
        if isinstance(value, str):
            if value.lower() in _BOOL_TRUE_STRS:
                return True
            if value.lower() in _BOOL_FALSE_STRS:
                return False
    if field_path.endswith(".applicant_index"):
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return value


def _coerce_llm_types(result: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(result))
    def coerce_dict(obj: dict[str, Any], prefix: str) -> dict[str, Any]:
        for key, value in list(obj.items()):
            field_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                obj[key] = coerce_dict(value, field_path)
            elif isinstance(value, list):
                obj[key] = [
                    coerce_dict(item, field_path) if isinstance(item, dict)
                    else _coerce_value(item, field_path)
                    for item in value
                ]
            else:
                obj[key] = _coerce_value(value, field_path)
        return obj
    return coerce_dict(result, "")


def validate_llm_result(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        import jsonschema  # type: ignore
        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(result):
            loc = ".".join(str(x) for x in err.absolute_path) or "<root>"
            errors.append(f"{loc}: {err.message}")
    except Exception:
        pass
    for field in ("itemid", "article_41_applied", "claims", "awards", "award_per_applicant", "reasoning"):
        if field not in result:
            errors.append(f"missing required top-level field: {field}")
    if errors:
        return errors

    def check_keys(obj: Any, required: tuple[str, ...], label: str) -> None:
        if not isinstance(obj, dict):
            errors.append(f"{label}: expected object")
            return
        for key in required:
            if key not in obj:
                errors.append(f"{label}.{key}: missing")

    def check_nullable_bool(value: Any, label: str) -> None:
        if value is not None and not isinstance(value, bool):
            errors.append(f"{label}: expected boolean|null")

    def check_nullable_number(value: Any, label: str) -> None:
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{label}: expected number|null")

    def check_nullable_string(value: Any, label: str) -> None:
        if value is not None and not isinstance(value, str):
            errors.append(f"{label}: expected string|null")

    check_nullable_bool(result.get("article_41_applied"), "article_41_applied")

    claims = result.get("claims")
    check_keys(claims, ("bundled_claim", "bundled_original_currency", "bundled_original_amount", "non_pecuniary", "pecuniary", "costs"), "claims")
    if isinstance(claims, dict):
        check_nullable_bool(claims.get("bundled_claim"), "claims.bundled_claim")
        check_nullable_string(claims.get("bundled_original_currency"), "claims.bundled_original_currency")
        check_nullable_number(claims.get("bundled_original_amount"), "claims.bundled_original_amount")
        for head in ("non_pecuniary", "pecuniary"):
            head_obj = claims.get(head)
            check_keys(head_obj, ("state", "original_currency", "original_amount", "eur_approx_court_stated"), f"claims.{head}")
            if isinstance(head_obj, dict):
                if head_obj.get("state") not in ("explicit_amount", "leave_to_court", "no_claim", "unclear", None):
                    errors.append(f"claims.{head}.state: invalid enum")
                check_nullable_string(head_obj.get("original_currency"), f"claims.{head}.original_currency")
                check_nullable_number(head_obj.get("original_amount"), f"claims.{head}.original_amount")
                check_nullable_number(head_obj.get("eur_approx_court_stated"), f"claims.{head}.eur_approx_court_stated")
                if head_obj.get("original_currency") == "EUR" and head_obj.get("original_amount") is not None and head_obj.get("eur_approx_court_stated") is None:
                    errors.append(f"claims.{head}.eur_approx_court_stated: required when original_currency=EUR and original_amount is present")
        costs = claims.get("costs")
        check_keys(costs, ("state", "original_currency", "original_amount", "eur_approx_court_stated", "scope"), "claims.costs")
        if isinstance(costs, dict):
            if costs.get("state") not in ("explicit_amount", "leave_to_court", "no_claim", "unclear", None):
                errors.append("claims.costs.state: invalid enum")
            if costs.get("scope") not in ("domestic", "strasbourg", "both", "unspecified", None):
                errors.append("claims.costs.scope: invalid enum")
            check_nullable_string(costs.get("original_currency"), "claims.costs.original_currency")
            check_nullable_number(costs.get("original_amount"), "claims.costs.original_amount")
            check_nullable_number(costs.get("eur_approx_court_stated"), "claims.costs.eur_approx_court_stated")
            if costs.get("original_currency") == "EUR" and costs.get("original_amount") is not None and costs.get("eur_approx_court_stated") is None:
                errors.append("claims.costs.eur_approx_court_stated: required when original_currency=EUR and original_amount is present")

    awards = result.get("awards")
    check_keys(awards, ("bundled_award", "bundled_award_eur", "non_pecuniary", "pecuniary", "costs"), "awards")
    if isinstance(awards, dict):
        check_nullable_bool(awards.get("bundled_award"), "awards.bundled_award")
        check_nullable_number(awards.get("bundled_award_eur"), "awards.bundled_award_eur")
        _pec_non_pec_dismissed_valid = ("no_claim", "unsubstantiated", "domestic_award_covers", "rule_60_non_compliance", "untimely", "applicant_deceased_no_heir", "no_causal_link", None)
        _costs_dismissed_valid = ("rule_60_non_compliance", "not_for_convention_purpose", "unsubstantiated", "no_claim", None)
        
        non_pec = awards.get("non_pecuniary")
        check_keys(non_pec, ("granted", "satisfaction_sufficient", "original_currency", "original_amount", "eur_amount", "dismissed_reason"), "awards.non_pecuniary")
        if isinstance(non_pec, dict):
            check_nullable_bool(non_pec.get("granted"), "awards.non_pecuniary.granted")
            check_nullable_bool(non_pec.get("satisfaction_sufficient"), "awards.non_pecuniary.satisfaction_sufficient")
            check_nullable_string(non_pec.get("original_currency"), "awards.non_pecuniary.original_currency")
            check_nullable_number(non_pec.get("original_amount"), "awards.non_pecuniary.original_amount")
            check_nullable_number(non_pec.get("eur_amount"), "awards.non_pecuniary.eur_amount")
            if non_pec.get("original_currency") == "EUR" and non_pec.get("original_amount") is not None and non_pec.get("eur_amount") is None:
                errors.append("awards.non_pecuniary.eur_amount: required when original_currency=EUR and original_amount is present")
            if non_pec.get("dismissed_reason") not in _pec_non_pec_dismissed_valid:
                errors.append("awards.non_pecuniary.dismissed_reason: invalid enum")
        
        pec = awards.get("pecuniary")
        check_keys(pec, ("granted", "satisfaction_sufficient", "original_currency", "original_amount", "eur_amount", "no_causal_link", "dismissed_reason"), "awards.pecuniary")
        if isinstance(pec, dict):
            check_nullable_bool(pec.get("granted"), "awards.pecuniary.granted")
            check_nullable_bool(pec.get("satisfaction_sufficient"), "awards.pecuniary.satisfaction_sufficient")
            check_nullable_string(pec.get("original_currency"), "awards.pecuniary.original_currency")
            check_nullable_number(pec.get("original_amount"), "awards.pecuniary.original_amount")
            check_nullable_number(pec.get("eur_amount"), "awards.pecuniary.eur_amount")
            if pec.get("original_currency") == "EUR" and pec.get("original_amount") is not None and pec.get("eur_amount") is None:
                errors.append("awards.pecuniary.eur_amount: required when original_currency=EUR and original_amount is present")
            check_nullable_bool(pec.get("no_causal_link"), "awards.pecuniary.no_causal_link")
            if pec.get("dismissed_reason") not in _pec_non_pec_dismissed_valid:
                errors.append("awards.pecuniary.dismissed_reason: invalid enum")
                
        costs = awards.get("costs")
        check_keys(costs, ("granted", "original_currency", "original_amount", "eur_amount", "legal_aid_deduction_eur", "net_eur", "dismissed_reason"), "awards.costs")
        if isinstance(costs, dict):
            check_nullable_bool(costs.get("granted"), "awards.costs.granted")
            check_nullable_string(costs.get("original_currency"), "awards.costs.original_currency")
            check_nullable_number(costs.get("original_amount"), "awards.costs.original_amount")
            check_nullable_number(costs.get("eur_amount"), "awards.costs.eur_amount")
            if costs.get("original_currency") == "EUR" and costs.get("original_amount") is not None and costs.get("eur_amount") is None:
                errors.append("awards.costs.eur_amount: required when original_currency=EUR and original_amount is present")
            check_nullable_number(costs.get("legal_aid_deduction_eur"), "awards.costs.legal_aid_deduction_eur")
            check_nullable_number(costs.get("net_eur"), "awards.costs.net_eur")
            if costs.get("dismissed_reason") not in _costs_dismissed_valid:
                errors.append("awards.costs.dismissed_reason: invalid enum")

    award_per_applicant = result.get("award_per_applicant")
    if not isinstance(award_per_applicant, list):
        errors.append("award_per_applicant: expected array")
    else:
        for i, row in enumerate(award_per_applicant):
            check_keys(row, ("beneficiary_label", "applicant_index", "head", "eur_amount"), f"award_per_applicant[{i}]")
            if isinstance(row, dict):
                check_nullable_string(row.get("beneficiary_label"), f"award_per_applicant[{i}].beneficiary_label")
                idx = row.get("applicant_index")
                if idx is not None:
                    if not isinstance(idx, int):
                        errors.append(f"award_per_applicant[{i}].applicant_index: expected integer|null")
                    elif idx < 1:
                        errors.append(f"award_per_applicant[{i}].applicant_index: must be >= 1")
                if row.get("beneficiary_label") in (None, "") and idx is None:
                    errors.append(f"award_per_applicant[{i}]: beneficiary_label or applicant_index required")
                if row.get("head") not in ("pecuniary", "non_pecuniary", "costs", "bundled", None):
                    errors.append(f"award_per_applicant[{i}].head: invalid enum")
                check_nullable_number(row.get("eur_amount"), f"award_per_applicant[{i}].eur_amount")

    reasoning = result.get("reasoning")
    check_keys(reasoning, ("government_sufficiency_argument", "government_counter_offer_eur", "retrial_recommended", "award_reason", "costs_reason"), "reasoning")
    if isinstance(reasoning, dict):
        check_nullable_bool(reasoning.get("government_sufficiency_argument"), "reasoning.government_sufficiency_argument")
        check_nullable_number(reasoning.get("government_counter_offer_eur"), "reasoning.government_counter_offer_eur")
        check_nullable_bool(reasoning.get("retrial_recommended"), "reasoning.retrial_recommended")
        check_nullable_string(reasoning.get("award_reason"), "reasoning.award_reason")
        check_nullable_string(reasoning.get("costs_reason"), "reasoning.costs_reason")

    return errors


def merge_retry_feedback(messages: list[dict[str, str]], errors: list[str]) -> list[dict[str, str]]:
    base_messages = messages[:2]
    return base_messages + [{
        "role": "user",
        "content": "Your previous JSON did not validate. Return a full replacement JSON object only. Follow this compact output guide exactly: "
        + json.dumps(GUIDE, ensure_ascii=False)
        + " Validation errors: "
        + json.dumps(errors, ensure_ascii=False),
    }]


def load_existing_results(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(json.loads(line)["itemid"])
    return seen


def write_jsonl_line(path: Path, row: dict[str, Any]) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_case_file(case_dir: Path, itemid: str, payload: dict[str, Any]) -> None:
    with WRITE_LOCK:
        (case_dir / f"{itemid}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_claim_head(state: str | None) -> dict[str, Any]:
    return {
        "state": state,
        "original_currency": None,
        "original_amount": None,
        "eur_approx_court_stated": None,
    }


def _empty_costs_claim_head(state: str | None) -> dict[str, Any]:
    return {
        **_empty_claim_head(state),
        "scope": None,
    }


def _deterministic_award_head(amount: Any, satisfaction_sufficient: bool = False) -> dict[str, Any]:
    if isinstance(amount, (int, float)):
        amount = float(amount)
        return {
            "granted": True,
            "satisfaction_sufficient": False,
            "original_currency": "EUR",
            "original_amount": amount,
            "eur_amount": amount,
            "dismissed_reason": None,
        }
    return {
        "granted": False,
        "satisfaction_sufficient": satisfaction_sufficient,
        "original_currency": None,
        "original_amount": None,
        "eur_amount": None,
        "dismissed_reason": None,
    }


def _deterministic_pec_award_head(amount: Any) -> dict[str, Any]:
    base = _deterministic_award_head(amount, satisfaction_sufficient=False)
    base["no_causal_link"] = None
    return base


def _deterministic_costs_award_head(amount: Any, awards_regex: dict[str, Any]) -> dict[str, Any]:
    if isinstance(amount, (int, float)):
        amount = float(amount)
        return {
            "granted": True,
            "original_currency": "EUR",
            "original_amount": amount,
            "eur_amount": amount,
            "legal_aid_deduction_eur": awards_regex.get("legal_aid_deduction_eur"),
            "net_eur": awards_regex.get("net_costs_eur"),
            "dismissed_reason": None,
        }
    return {
        "granted": False,
        "original_currency": None,
        "original_amount": None,
        "eur_amount": None,
        "legal_aid_deduction_eur": awards_regex.get("legal_aid_deduction_eur"),
        "net_eur": awards_regex.get("net_costs_eur"),
        "dismissed_reason": None,
    }


def _build_deterministic_article_41_extraction(itemid: str, source_row: dict[str, Any] | None, awards_regex: dict[str, Any]) -> dict[str, Any]:
    claim_and_award_layer = (source_row or {}).get("claim_and_award_layer") or {}
    article_41_applied = claim_and_award_layer.get("article_41_applied")
    satisfaction_sufficient = bool(awards_regex.get("satisfaction_sufficient"))
    bundled_amount = awards_regex.get("bundled_award_eur")
    notes = awards_regex.get("deterministic_notes") or []
    award_reason = "; ".join(str(note).strip() for note in notes if str(note).strip()) or None
    claims = _extract_deterministic_claims(source_row, article_41_applied)
    awards = _extract_deterministic_awards(source_row, awards_regex, article_41_applied)
    if awards_regex.get("non_pecuniary_eur") is None and not satisfaction_sufficient:
        awards["non_pecuniary"] = _deterministic_award_head(None, satisfaction_sufficient=False)
    if awards_regex.get("pecuniary_eur") is None:
        awards["pecuniary"] = _deterministic_pec_award_head(None)
    if awards_regex.get("costs_eur") is None:
        awards["costs"] = _deterministic_costs_award_head(None, awards_regex)

    return {
        "itemid": itemid,
        "article_41_applied": article_41_applied,
        "claims": claims,
        "awards": {
            "bundled_award": True if isinstance(bundled_amount, (int, float)) else awards.get("bundled_award"),
            "bundled_award_eur": float(bundled_amount) if isinstance(bundled_amount, (int, float)) else awards.get("bundled_award_eur"),
            "non_pecuniary": awards.get("non_pecuniary") or _deterministic_award_head(None, satisfaction_sufficient=satisfaction_sufficient),
            "pecuniary": awards.get("pecuniary") or _deterministic_pec_award_head(None),
            "costs": awards.get("costs") or _deterministic_costs_award_head(None, awards_regex),
        },
        "award_per_applicant": awards_regex.get("award_per_applicant") or [],
        "reasoning": {
            "government_sufficiency_argument": None,
            "government_counter_offer_eur": None,
            "retrial_recommended": None,
            "award_reason": award_reason,
            "costs_reason": None,
        },
    }


def compact_compensation_result(
    itemid: str,
    awards_regex: dict[str, Any],
    article_41_extraction: dict[str, Any] | None,
    cross_validation: dict[str, Any],
    source_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = article_41_extraction or _build_deterministic_article_41_extraction(itemid, source_row, awards_regex)
    fallback_per_applicant = []
    if not (parsed.get("award_per_applicant") or []):
        fallback_per_applicant = awards_regex.get("award_per_applicant") or []
    return {
        "itemid": itemid,
        "article_41_extraction": parsed,
        "award_per_applicant_fallback": fallback_per_applicant,
        "cross_validation": cross_validation,
        "final_awards": build_final_awards(awards_regex, parsed, cross_validation),
    }


def run_one_case(
    itemid: str,
    row: dict[str, Any],
    client: OpenAICompatibleClient,
    system_prompt: str,
    schema: dict[str, Any],
    max_retries: int = 1,  # kept for backward-compat with the standalone CLI; ignored by design
    regex_only: bool = False,
) -> dict[str, Any]:
    """Single-shot pipeline C execution.

    Returns one of:
    - {status: "success", result, usage, elapsed_seconds}            (LLM path or regex_only path)
    - {status: "api_error", api_error, awards_regex, ...}            (caller can fall back to regex-only)
    - {status: "schema_validation", errors, raw_result, awards_regex, ...}
    """
    start = time.perf_counter()
    claim_inputs = row["claim_and_award_layer"]["cross_validation_inputs"]
    operative_text = claim_inputs.get("operative_text") or ""
    appendix_table_text = claim_inputs.get("appendix_table_text") or ""
    article_41_text = claim_inputs.get("article_41_text") or ""
    conclusion_header = operative_text[:500]
    num_applicants = ((row.get("facts_procedure") or {}).get("num_applicants")) or ((row.get("core_case") or {}).get("num_applicants_proxy"))
    input_snapshot = build_c_input_snapshot(row)
    awards_regex = layer1_deterministic(
        operative_text,
        appendix_table_text,
        conclusion_header,
        article_41_text,
        num_applicants if isinstance(num_applicants, int) else None,
    )

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if regex_only:
        crossval = {"non_pec_match": True, "pecuniary_match": True, "costs_match": True, "flag_for_review": False}
        return {
            "status": "success",
            "itemid": itemid,
            "attempts": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "usage": usage_total,
            "input_snapshot": input_snapshot,
            "awards_regex": awards_regex,
            "result": compact_compensation_result(itemid, awards_regex, None, crossval, source_row=row),
        }

    messages = prompt_messages(system_prompt, schema, row, include_schema_in_payload=not client.use_json_schema)
    try:
        parsed, usage = client.chat_json(messages=messages, schema=schema, schema_name="pipeline_c_backbone")
    except ApiCallError as exc:
        return {
            "status": "api_error",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": time.perf_counter() - start,
            "usage": usage_total,
            "input_snapshot": input_snapshot,
            "api_error": exc.to_record(),
            "awards_regex": awards_regex,
        }

    if usage:
        for key in usage_total:
            if isinstance(usage.get(key), int):
                usage_total[key] += usage[key]
    parsed["itemid"] = itemid
    parsed = _normalize_llm_compensation(parsed, row)
    parsed = _coerce_llm_types(parsed)
    errors = validate_llm_result(parsed, schema)
    if errors:
        return {
            "status": "schema_validation",
            "itemid": itemid,
            "attempts": 1,
            "elapsed_seconds": time.perf_counter() - start,
            "usage": usage_total,
            "input_snapshot": input_snapshot,
            "errors": errors,
            "raw_result": parsed,
            "awards_regex": awards_regex,
        }

    crossval = layer3_crossval(awards_regex, parsed)
    result = compact_compensation_result(itemid, awards_regex, parsed, crossval, source_row=row)
    return {
        "status": "success",
        "itemid": itemid,
        "attempts": 1,
        "elapsed_seconds": time.perf_counter() - start,
        "usage": usage_total,
        "input_snapshot": input_snapshot,
        "raw_result": parsed,
        "awards_regex": awards_regex,
        "result": result,
    }


def write_per_split_results(run_dir: Path, split_to_ids: dict[str, list[str]], success_rows: dict[str, dict[str, Any]]) -> None:
    by_split_dir = run_dir / "by_split"
    by_split_dir.mkdir(exist_ok=True)
    for split_name, ids in split_to_ids.items():
        path = by_split_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for itemid in ids:
                row = success_rows.get(itemid)
                if row is not None:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = load_json(SCHEMA_PATH)
    run_dir = make_run_dir(args.run_name)

    split_to_ids = {split: load_split_ids(split) for split in args.splits}
    if args.itemids:
        unique_ids = dedupe_preserve_order([str(x) for x in args.itemids])
        split_to_ids["manual_itemids"] = unique_ids
    else:
        unique_ids = dedupe_preserve_order([itemid for split in args.splits for itemid in split_to_ids[split]])
    if args.max_cases is not None:
        unique_ids = unique_ids[: args.max_cases]
        chosen = set(unique_ids)
        split_to_ids = {k: [x for x in v if x in chosen] for k, v in split_to_ids.items()}

    rows_by_itemid = load_jsonl_by_itemid(INPUT_JSONL)
    missing = [itemid for itemid in unique_ids if itemid not in rows_by_itemid]
    if missing:
        raise RuntimeError(f"{len(missing)} itemids were not found in {INPUT_JSONL}")

    unique_results_path = run_dir / "results_unique.jsonl"
    meta_results_path = run_dir / "results_meta.jsonl"
    failures_path = run_dir / "failures.jsonl"
    run_metadata_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "summary.json"
    case_dir = run_dir / "cases"

    already_done = load_existing_results(unique_results_path) if args.resume else set()
    todo_ids = [itemid for itemid in unique_ids if itemid not in already_done]

    run_metadata = {
        "input_jsonl": str(INPUT_JSONL),
        "splits": args.splits,
        "split_counts_requested": {k: len(v) for k, v in split_to_ids.items()},
        "unique_cases_requested": len(unique_ids),
        "unique_cases_to_run": len(todo_ids),
        "resume": args.resume,
        "dry_run": args.dry_run,
        "regex_only": args.regex_only,
        "schema": str(SCHEMA_PATH),
        "prompt": str(PROMPT_PATH),
    }
    run_metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(run_metadata, ensure_ascii=False, indent=2))
        return

    client = OpenAICompatibleClient.from_env()
    start = time.perf_counter()
    success_rows: dict[str, dict[str, Any]] = {}
    failure_rows: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_one_case,
                itemid,
                rows_by_itemid[itemid],
                client,
                system_prompt,
                schema,
                args.max_retries,
                args.regex_only,
            ): itemid
            for itemid in todo_ids
        }
        for future in as_completed(futures):
            itemid = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = {"status": "request_failed", "itemid": itemid, "errors": [str(exc)]}

            usage = payload.get("usage") or {}
            for key in usage_total:
                if isinstance(usage.get(key), int):
                    usage_total[key] += usage[key]

            if payload["status"] == "success":
                result = payload["result"]
                success_rows[itemid] = result
                write_jsonl_line(unique_results_path, result)
                write_jsonl_line(
                    meta_results_path,
                    {
                        "itemid": itemid,
                        "attempts": payload.get("attempts"),
                        "elapsed_seconds": payload.get("elapsed_seconds"),
                        "usage": payload.get("usage"),
                    },
                )
                write_case_file(case_dir, itemid, result)
            else:
                failure_rows.append(payload)
                write_jsonl_line(failures_path, payload)

    if args.resume and unique_results_path.exists():
        with unique_results_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                success_rows[row["itemid"]] = row

    write_per_split_results(run_dir, split_to_ids, success_rows)

    summary = {
        **run_metadata,
        "runtime_seconds": time.perf_counter() - start,
        "successful_cases": len(success_rows),
        "failed_cases": len(failure_rows),
        "usage_total": usage_total,
        "results_unique_jsonl": str(unique_results_path),
        "results_meta_jsonl": str(meta_results_path),
        "failures_jsonl": str(failures_path),
        "by_split_dir": str(run_dir / "by_split"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

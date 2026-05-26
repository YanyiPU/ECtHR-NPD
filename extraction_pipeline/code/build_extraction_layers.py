#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from case_store import iter_cases_from_cases_json, load_cases_by_itemid, write_case_to_store
from docx_lossless import format_pipeline_c_appendix_text
from shared_compensation_evidence import build_shared_compensation_evidence, extract_narrative_claim_rows
from universal_evidence_scanners import extract_identity_evidence, extract_reasoning_evidence


DATASET_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
UNSTRUCTURED = DATASET_ROOT / "unstructured" / "cases.json"
CASES_CORE = DATASET_ROOT / "structured" / "cases_core.json"
OUTPUTS = EXTRACTION_ROOT / "outputs"
REPORTS = EXTRACTION_ROOT / "reports"

CASE_FEATURES_LABELS_JSONL = OUTPUTS / "case_features_labels.jsonl"
CORE_CASE_JSONL = OUTPUTS / "core_case.jsonl"
FACTS_PROCEDURE_INPUTS_JSONL = OUTPUTS / "facts_procedure_inputs.jsonl"
CLAIM_AWARD_INPUTS_JSONL = OUTPUTS / "claim_award_inputs.jsonl"
REASONING_INPUTS_JSONL = OUTPUTS / "reasoning_inputs.jsonl"
SUMMARY_JSON = REPORTS / "extraction_layer_summary.json"
SUMMARY_MD = REPORTS / "EXTRACTION_LAYER_SUMMARY.md"
SANITY_JSON = REPORTS / "extraction_sanity_checks.json"
SAMPLES_JSON = REPORTS / "extraction_samples.json"

APPLICATION_RE = re.compile(r"\b\d{1,6}/\d{2}\b")
ARTICLE_41_RE = re.compile(r"\b(?:APPLICATION OF )?ARTICLE\s+(?:41|50)\b|\bJUST SATISFACTION\b", re.IGNORECASE)
COMPENSATION_HEADING_RE = re.compile(
    r"\b(?:APPLICATION OF )?ARTICLE\s+(?:41|50)\b|\bJUST SATISFACTION\b",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic split extraction scaffold layers.")
    parser.add_argument("--itemids", nargs="+", default=None, help="Only rebuild these itemids.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit processed cases.")
    parser.add_argument("--sync-case-store", action="store_true", help="Also refresh per-itemid raw case files while reading.")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_jsonl_by_itemid(path: Path, rows: list[dict[str, Any]]) -> None:
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                existing[row["itemid"]] = row
    for row in rows:
        existing[row["itemid"]] = row
    ordered = [existing[itemid] for itemid in sorted(existing)]
    write_jsonl(path, ordered)


def extract_judgment_year(value: Any) -> str | None:
    text = str(value or "")
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None


def str_norm(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def party_from_list(row: dict[str, Any], index: int) -> str:
    parties = row.get("parties")
    if isinstance(parties, list) and len(parties) > index:
        return str_norm(parties[index])
    return ""


def case_name_from_docname(docname: Any) -> str:
    text = str_norm(docname)
    if not text:
        return ""
    match = re.match(r"^CASE OF\s+(.+)$", text, flags=re.IGNORECASE)
    return str_norm(match.group(1) if match else text)


def applicant_from_docname(docname: Any) -> str:
    case_name = case_name_from_docname(docname)
    match = re.match(r"^(.+?)\s+v\.\s+.+$", case_name, flags=re.IGNORECASE)
    return str_norm(match.group(1) if match else "")


def respondent_from_docname(docname: Any) -> str:
    case_name = case_name_from_docname(docname)
    match = re.match(r"^.+?\s+v\.\s+(.+)$", case_name, flags=re.IGNORECASE)
    return str_norm(match.group(1) if match else "")


def normalize_article(article: Any) -> str | None:
    if not article:
        return None
    normalized = str(article).strip().upper()
    return normalized or None


def extract_mentioned_articles(row: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for article in row.get("article") or []:
        normalized = normalize_article(article)
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen


def extract_detailed_violations(row: dict[str, Any]) -> list[dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    for entry in row.get("conclusion") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "violation":
            continue
        detailed.append(
            {
                "article": normalize_article(entry.get("article")),
                "base_article": normalize_article(entry.get("base_article") or entry.get("article")),
                "element": entry.get("element"),
                "type": entry.get("type"),
            }
        )
    return detailed


def extract_violated_articles(row: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for entry in extract_detailed_violations(row):
        article = entry.get("base_article") or entry.get("article")
        if article and article not in seen:
            seen.append(article)
    return seen


def has_mixed_outcome(row: dict[str, Any]) -> bool:
    has_violation = False
    has_no_violation = False
    for entry in row.get("conclusion") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "violation":
            has_violation = True
        if entry.get("type") == "no-violation":
            has_no_violation = True
    return has_violation and has_no_violation


def decision_body_category(row: dict[str, Any]) -> str:
    branch = str(row.get("doctypebranch") or "").upper()
    body_name = str(row.get("originatingbody_name") or "").upper()
    if "GRAND CHAMBER" in branch or "GRAND CHAMBER" in body_name:
        return "Grand Chamber"
    if "SINGLE JUDGE" in branch or "SINGLE JUDGE" in body_name:
        return "Single Judge"
    if "COMMITTEE" in branch or "COMMITTEE" in body_name:
        return "Committee"
    if "CHAMBER" in branch or "SECTION" in body_name or body_name == "CHAMBER":
        return "Chamber"
    return "Other"


def split_semicolon(value: Any) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def extract_application_numbers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        seen: list[str] = []
        for item in value:
            for number in extract_application_numbers(item):
                if number not in seen:
                    seen.append(number)
        return seen
    if isinstance(value, dict):
        return extract_application_numbers(list(value.values()))
    text = str(value)
    seen: list[str] = []
    for match in APPLICATION_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def merged_text(parts: list[str]) -> str:
    return clean_text("\n\n".join(part for part in parts if part))


def flatten_node(node: dict[str, Any]) -> str:
    def _walk(n: dict[str, Any]) -> list[str]:
        res = []
        content = n.get("content")
        if content:
            res.append(str(content))
        for child in n.get("elements") or []:
            if isinstance(child, dict):
                res.extend(_walk(child))
        return res
    return clean_text("\n\n".join(_walk(node)))


def top_level_sections(row: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    content = row.get("content")
    if not isinstance(content, dict) or not content:
        return None, []
    content_key = next(iter(content))
    sections = content[content_key]
    if not isinstance(sections, list):
        return content_key, []
    return content_key, [section for section in sections if isinstance(section, dict)]


def first_section_by_name(sections: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for section in sections:
        if section.get("section_name") == name:
            return section
    return None


def heading_norm(text: str) -> str:
    normalized = clean_text(text).upper()
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("“", '"').replace("”", '"')
    return normalized


def canonical_section_name(section_name: Any, heading_text: Any) -> str:
    raw_name = clean_text(section_name).lower()
    heading = heading_norm(str(heading_text or ""))

    if "FOR THESE REASONS" in heading:
        return "operative"
    if "THE COURT'S ASSESSMENT" in heading:
        return "assessment_summary"
    if (
        "APPLICATION OF ARTICLE 41" in heading
        or "APPLICATION OF ARTICLE 50" in heading
        or "JUST SATISFACTION" in heading
    ):
        return "article_41"
    if "SUBJECT MATTER OF THE CASE" in heading:
        return "summary_intro"
    if heading in {"PROCEDURE", "PROCEEDINGS", "PROCEEDINGS BEFORE THE COURT"}:
        return "procedure"
    if heading == "INTRODUCTION":
        return "introduction"
    if "THE FACTS" in heading or "THE CIRCUMSTANCES OF THE CASE" in heading:
        return "facts"
    if (
        "RELEVANT LEGAL FRAMEWORK" in heading
        or "RELEVANT DOMESTIC LAW" in heading
        or "RELEVANT LEGAL FRAMEWORK AND PRACTICE" in heading
        or "RELEVANT COUNCIL OF EUROPE INSTRUMENTS" in heading
    ):
        return "relevant_law"
    if (
        heading == "THE LAW"
        or heading == "MERITS"
        or "ALLEGED VIOLATION OF ARTICLE" in heading
        or heading.startswith("III. MERITS")
    ):
        return "law"
    if heading.startswith("TABLE-") or heading.startswith("TABLE "):
        return "appendix"
    if "OPINION" in heading:
        return "opinion"

    if raw_name == "conclusion":
        return "conclusion_misc"
    if raw_name in {
        "introduction",
        "procedure",
        "facts",
        "relevant_law",
        "law",
        "submission",
        "appendix",
        "opinion",
    }:
        return raw_name
    return raw_name or "unclassified"


def recursive_article_41_candidates(node: dict[str, Any], depth: int = 0) -> list[tuple[int, dict[str, Any]]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    content = clean_text(node.get("content"))
    if is_compensation_heading(content):
        candidates.append((depth, node))
    for child in node.get("elements") or []:
        if isinstance(child, dict):
            candidates.extend(recursive_article_41_candidates(child, depth + 1))
    return candidates


def is_compensation_heading(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if not COMPENSATION_HEADING_RE.search(text):
        return False
    words = text.split()
    if len(words) > 24:
        return False
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.55


def extract_article_41_and_reasoning(law_section: dict[str, Any] | None) -> tuple[str, str, str | None]:
    if not law_section:
        return "", "", None

    law_heading = clean_text(law_section.get("content"))
    immediate_children = [child for child in law_section.get("elements") or [] if isinstance(child, dict)]
    article_41_children = [child for child in immediate_children if is_compensation_heading(str(child.get("content") or ""))]
    if article_41_children:
        article_41_text = clean_text("\n\n".join(flatten_node(child) for child in article_41_children))
        remaining_parts = [law_heading] if law_heading else []
        for child in immediate_children:
            if child not in article_41_children:
                child_text = flatten_node(child)
                if child_text:
                    remaining_parts.append(child_text)
        reasoning_text = clean_text("\n\n".join(remaining_parts))
        return article_41_text, reasoning_text, "immediate_heading_match"

    law_text = flatten_node(law_section)
    candidates = recursive_article_41_candidates(law_section)
    if candidates:
        candidates.sort(key=lambda item: item[0])
        article_41_text = flatten_node(candidates[0][1])
        if article_41_text and article_41_text in law_text:
            reasoning_text = clean_text(law_text.replace(article_41_text, "", 1))
        else:
            match = ARTICLE_41_RE.search(law_text)
            reasoning_text = clean_text(law_text[: match.start()]) if match else law_text
            article_41_text = clean_text(law_text[match.start() :]) if match else article_41_text
        return clean_text(article_41_text), clean_text(reasoning_text), "recursive_heading_match"

    match = COMPENSATION_HEADING_RE.search(law_text)
    if match:
        return clean_text(law_text[match.start() :]), clean_text(law_text[: match.start()]), "text_fallback_match"

    return "", clean_text(law_text), None


def build_profiled_sections(
    row: dict[str, Any],
    sections: list[dict[str, Any]],
    raw_law_section: dict[str, Any] | None,
    raw_conclusion_section: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_sections: dict[str, list[str]] = defaultdict(list)
    section_rows: list[dict[str, Any]] = []

    for idx, section in enumerate(sections):
        heading = clean_text(section.get("content"))
        canonical = canonical_section_name(section.get("section_name"), heading)
        text = flatten_node(section)
        if text:
            normalized_sections[canonical].append(text)
        section_rows.append(
            {
                "index": idx,
                "raw_section_name": section.get("section_name"),
                "heading": heading[:160],
                "canonical_section_name": canonical,
                "text_length": len(text),
            }
        )

    introduction_text = merged_text(normalized_sections["introduction"])
    procedure_text = merged_text(normalized_sections["procedure"])
    facts_text = merged_text(normalized_sections["facts"])
    summary_intro_text = merged_text(normalized_sections["summary_intro"])
    assessment_summary_text = merged_text(normalized_sections["assessment_summary"])
    relevant_law_text = merged_text(normalized_sections["relevant_law"])
    law_text = merged_text(normalized_sections["law"])
    article_41_text = merged_text(normalized_sections["article_41"])
    operative_text = merged_text(normalized_sections["operative"])
    conclusion_misc_text = merged_text(normalized_sections["conclusion_misc"])

    article_41_method: str | None = None
    if article_41_text:
        article_41_method = "canonical_article_41_section"
        reasoning_text = merged_text([relevant_law_text, law_text])
    else:
        fallback_article_41_text, law_reasoning_text, article_41_method = extract_article_41_and_reasoning(raw_law_section)
        article_41_text = fallback_article_41_text
        reasoning_text = merged_text([relevant_law_text, law_reasoning_text or law_text])

    if not operative_text:
        operative_text = flatten_node(raw_conclusion_section) if raw_conclusion_section else conclusion_misc_text

    narrative_text = merged_text([introduction_text, procedure_text, facts_text])
    summary_text = merged_text([summary_intro_text, assessment_summary_text])

    if narrative_text:
        document_profile = "narrative_rich"
        pipeline_b_input_mode = "narrative_sections"
        routed_intro = introduction_text
        routed_proc = procedure_text
        routed_facts = facts_text
    elif summary_text:
        document_profile = "summary_mode"
        pipeline_b_input_mode = "summary_intro_plus_assessment"
        routed_intro = summary_intro_text
        routed_proc = ""
        routed_facts = assessment_summary_text
    elif reasoning_text or article_41_text:
        document_profile = "law_only"
        pipeline_b_input_mode = "law_fallback"
        routed_intro = summary_intro_text
        routed_proc = ""
        routed_facts = merged_text([assessment_summary_text, reasoning_text])
    elif operative_text or conclusion_misc_text:
        document_profile = "conclusion_only"
        pipeline_b_input_mode = "skip"
        routed_intro = ""
        routed_proc = ""
        routed_facts = ""
    else:
        document_profile = "unknown"
        pipeline_b_input_mode = "skip"
        routed_intro = ""
        routed_proc = ""
        routed_facts = ""

    if reasoning_text:
        pipeline_e_input_mode = "law_and_relevant_law"
        routed_reasoning = reasoning_text
    elif assessment_summary_text:
        pipeline_e_input_mode = "assessment_summary_fallback"
        routed_reasoning = assessment_summary_text
    elif summary_intro_text:
        pipeline_e_input_mode = "summary_intro_fallback"
        routed_reasoning = summary_intro_text
    else:
        pipeline_e_input_mode = "empty"
        routed_reasoning = ""

    normalized_names_present = sorted(
        name for name, parts in normalized_sections.items() if merged_text(parts)
    )

    return {
        "introduction_text": routed_intro,
        "procedure_text": routed_proc,
        "facts_text": routed_facts,
        "summary_intro_text": summary_intro_text,
        "assessment_summary_text": assessment_summary_text,
        "relevant_law_text": relevant_law_text,
        "law_text_excluding_article_41": routed_reasoning,
        "article_41_text": article_41_text,
        "operative_text": operative_text,
        "article_41_method": article_41_method,
        "document_profile": document_profile,
        "pipeline_b_input_mode": pipeline_b_input_mode,
        "pipeline_e_input_mode": pipeline_e_input_mode,
        "normalized_top_level_section_names": normalized_names_present,
        "section_rows": section_rows,
        "normalized_text_lengths": {
            "introduction": len(introduction_text),
            "procedure": len(procedure_text),
            "facts": len(facts_text),
            "summary_intro": len(summary_intro_text),
            "assessment_summary": len(assessment_summary_text),
            "relevant_law": len(relevant_law_text),
            "law": len(law_text),
            "article_41": len(article_41_text),
            "operative": len(operative_text),
            "conclusion_misc": len(conclusion_misc_text),
        },
    }


def build_token_counter():
    try:
        import tiktoken  # type: ignore

        encoder = tiktoken.get_encoding("cl100k_base")

        def count_tokens(text: str) -> int:
            return len(encoder.encode(text))

        return count_tokens, "tiktoken/cl100k_base"
    except Exception:
        def count_tokens(text: str) -> int:
            return math.ceil(len(text) / 4)

        return count_tokens, "char_div_4_fallback"


def sample_excerpt(text: str, limit: int = 280) -> str:
    text = clean_text(text)
    return text[:limit]


def empty_claim_head() -> dict[str, Any]:
    return {
        "claim_state": None,
        "dismissed": None,
        "claimed_eur": None,
        "claimed_local_currency_amount": None,
        "claimed_local_currency_code": None,
        "court_converted_eur_used": None,
        "per_applicant_claims": [],
    }


def empty_award_head() -> dict[str, Any]:
    return {
        "dismissed": None,
        "awarded_eur": None,
        "awarded_local_currency_amount": None,
        "awarded_local_currency_code": None,
        "court_converted_eur_used": None,
        "per_applicant_awards": [],
        "reason": None,
    }


def make_case_record(
    row: dict[str, Any],
    core_row: dict[str, Any],
    introduction_text: str,
    procedure_text: str,
    facts_text: str,
    summary_intro_text: str,
    assessment_summary_text: str,
    relevant_law_text: str,
    article_41_text: str,
    operative_text: str,
    appendix_table_text: str,
    reasoning_text: str,
    article_41_method: str | None,
    document_profile: str,
    pipeline_b_input_mode: str,
    pipeline_e_input_mode: str,
    normalized_top_level_section_names: list[str],
    section_rows: list[dict[str, Any]],
    normalized_text_lengths: dict[str, int],
    pipeline_b_tokens: int,
    pipeline_c_tokens: int,
    pipeline_d_tokens: int,
    full_document_text: str,
    law_text_full: str,
) -> dict[str, Any]:
    claim_narrative_text = merged_text(
        [
            summary_intro_text,
            assessment_summary_text,
            introduction_text,
            procedure_text,
            facts_text,
        ]
    )
    article_41_applied = bool(article_41_text)
    shared_compensation_evidence = build_shared_compensation_evidence(
        article_41_text=article_41_text,
        operative_text=operative_text,
        appendix_table_text=appendix_table_text,
    )
    scattered_claim_snippets = extract_narrative_claim_rows(text=full_document_text)
    scattered_identity_snippets = extract_identity_evidence(text=full_document_text)
    scattered_reasoning_snippets = extract_reasoning_evidence(text=full_document_text)

    return {
        "itemid": row.get("itemid"),
        "appno": row.get("appno"),
        "judgementdate": row.get("judgementdate"),
        "core_case": core_row,
        "shared_evidence": {
            "compensation_evidence": shared_compensation_evidence,
            "scattered_claim_snippets": scattered_claim_snippets,
            "scattered_identity_snippets": scattered_identity_snippets,
            "scattered_reasoning_snippets": scattered_reasoning_snippets,
        },
        "facts_procedure": {
            "num_applicants": core_row["num_applicants_proxy"],
            "applicants": [],
            "status": {
                "is_vulnerable": None,
                "is_represented": core_row["represented"],
            },
            "is_indirect_victim": None,
            "is_repetitive_case": None,
            "is_pilot_judgment": None,
            "partial_admissibility": None,
            "complaints_summary": [],
            "legal_aid_from_coe": None,
            "domestic_award_prior": None,
            "domestic_award_prior_eur": None,
            "state_remedial_measures": None,
            "evidence_inputs": {
                "introduction_text": introduction_text,
                "procedure_text": procedure_text,
                "facts_text": facts_text,
                "summary_intro_text": summary_intro_text,
                "assessment_summary_text": assessment_summary_text,
                "pipeline_b_input_mode": pipeline_b_input_mode,
                "scattered_identity_snippets": scattered_identity_snippets,
            },
        },
        "claim_and_award_layer": {
            "article_41_applied": article_41_applied,
            "claim_state": None,
            "claims": {
                "pecuniary": empty_claim_head(),
                "non_pecuniary": empty_claim_head(),
                "costs_expenses": empty_claim_head(),
            },
            "is_seeking": {
                "is_just_satisfaction": article_41_applied if article_41_text else None,
            },
            "bundled_claim": None,
            "cross_validation_inputs": {
                "article_41_text": article_41_text,
                "operative_text": operative_text,
                "appendix_table_text": appendix_table_text,
                "claim_narrative_text": claim_narrative_text,
                "article_41_extraction_method": article_41_method,
                "scattered_claim_snippets": scattered_claim_snippets,
            },
        },
        "award_layer": {
            "awards": {
                "pecuniary": empty_award_head(),
                "non_pecuniary": empty_award_head(),
                "costs_expenses": empty_award_head(),
            },
            "award": {
                "is_awarded": None,
                "is_partial": None,
                "reasoning_mode": None,
                "no_award_reason": None,
            },
            "award_is_joint": None,
            "award_allocation_by_beneficiary": [],
            "legal_aid_deduction_eur": None,
            "reserved_article_41": None,
            "cross_validation_inputs": {
                "article_41_text": article_41_text,
                "operative_text": operative_text,
                "appendix_table_text": appendix_table_text,
            },
        },
        "reasoning_layer": {
            "violation_type": None,
            "violation_subtype": None,
            "violation_duration_months": None,
            "reasoning": {
                "reasoning_factors": [],
            },
            "government_sufficiency_argument": None,
            "government_counter_offer": None,
            "retrial_recommended": None,
            "evidence_inputs": {
                "law_text": law_text_full,
                "law_text_excluding_article_41": reasoning_text,
                "summary_intro_text": summary_intro_text,
                "assessment_summary_text": assessment_summary_text,
                "relevant_law_text": relevant_law_text,
                "pipeline_e_input_mode": pipeline_e_input_mode,
                "scattered_reasoning_snippets": scattered_reasoning_snippets,
            },
        },
        "legal_analysis": {
            "violated_articles_analyzed": [],
            "legal_tests": [],
            "proportionality": {
                "discussed": None,
                "steps": [],
                "result": None,
            },
            "margin_of_appreciation": {
                "referenced": None,
                "width": None,
                "domain": None,
            },
            "nature_of_obligation": {
                "article": None,
                "obligation_type": None,
                "structural_failure": None,
            },
            "subsidiarity": {
                "discussed": None,
                "domestic_remedies_exhausted": None,
                "subsidiarity_analysis_depth": None,
            },
            "precedent_usage": {
                "citation_count": None,
                "grand_chamber_citations": [],
                "distinguished": None,
                "new_principle_established": None,
            },
            "reasoning_quality": {
                "reasoning_depth": None,
                "quantitative_reasoning": None,
            },
        },
        "extraction_meta": {
            "source_dataset": str(UNSTRUCTURED),
            "pipeline_b_input_tokens_estimate": pipeline_b_tokens,
            "pipeline_c_input_tokens_estimate": pipeline_c_tokens,
            "pipeline_d_input_tokens_estimate": pipeline_d_tokens,
            "article_41_extraction_method": article_41_method,
            "document_profile": document_profile,
            "raw_top_level_section_names": core_row["top_level_section_names"],
            "normalized_top_level_section_names": normalized_top_level_section_names,
            "pipeline_b_input_mode": pipeline_b_input_mode,
            "pipeline_e_input_mode": pipeline_e_input_mode,
            "normalized_text_lengths": normalized_text_lengths,
            "section_rows": section_rows,
        },
    }


def make_core_row(row: dict[str, Any], existing: dict[str, Any] | None, section_key: str | None, sections: list[dict[str, Any]]) -> dict[str, Any]:
    violated = extract_violated_articles(row)
    mentioned = extract_mentioned_articles(row)
    detailed = extract_detailed_violations(row)
    app_numbers = extract_application_numbers(row.get("appno") or row.get("extractedappno"))
    country_name = existing.get("country_name") if existing else None
    country_alpha2 = existing.get("country_alpha2") if existing else None
    judgment_year = (
        (str(existing.get("year")).strip() if existing and existing.get("year") is not None else None)
        or extract_judgment_year(row.get("kpdate"))
        or extract_judgment_year(row.get("judgementdate"))
    )
    return {
        "itemid": row.get("itemid"),
        "appno": row.get("appno"),
        "docname": row.get("docname"),
        "case_name_clean": case_name_from_docname(row.get("docname")),
        "party_applicant": party_from_list(row, 0) or applicant_from_docname(row.get("docname")),
        "party_respondent": party_from_list(row, 1) or respondent_from_docname(row.get("docname")) or row.get("respondent"),
        "ecli": row.get("ecli"),
        "judgementdate": row.get("judgementdate"),
        "judgment_year": judgment_year,
        "case_importance": row.get("importance"),
        "doctypebranch": row.get("doctypebranch"),
        "decision_body_category": decision_body_category(row),
        "is_grand_chamber": decision_body_category(row) == "Grand Chamber",
        "has_separate_opinion": str(row.get("separateopinion") or "").upper() == "TRUE",
        "respondent_country": country_name or row.get("respondent"),
        "country_alpha2": country_alpha2,
        "originatingbody_name": row.get("originatingbody_name"),
        "all_scl_citations": row.get("scl") or [],
        "article": row.get("article") or [],
        "conclusion": row.get("conclusion") or [],
        "violated_articles": violated,
        "detailed_violations": detailed,
        "num_violations_found": len(detailed),
        "has_mixed_outcome": has_mixed_outcome(row),
        "represented": bool([x for x in (row.get("representedby") or []) if str(x).strip() and str(x).strip().upper() != "N/A"]),
        "voting_pattern": None,
        "article_41_precedents": [],
        "mentioned_articles": mentioned,
        "num_applicants_proxy": max(1, len(app_numbers)) if app_numbers else 1,
        "application_numbers_extracted": app_numbers,
        "content_source_key": section_key,
        "top_level_section_names": [section.get("section_name") for section in sections if section.get("section_name")],
        "conclusion_count": len(row.get("conclusion") or []),
    }


def make_markdown(summary: dict[str, Any], sanity_checks: list[dict[str, Any]]) -> str:
    lines = [
        "# Extraction Layer Summary",
        "",
        f"- Input dataset: `{summary['input_dataset']}`",
        f"- Cases processed: **{summary['cases_processed']}**",
        f"- Runtime seconds: **{summary['runtime_seconds']:.3f}**",
        f"- Token estimation method: **{summary['token_estimation_method']}**",
        "",
        "## Outputs",
        "",
        f"- [case_features_labels.jsonl]({CASE_FEATURES_LABELS_JSONL})",
        f"- [core_case.jsonl]({CORE_CASE_JSONL})",
        f"- [facts_procedure_inputs.jsonl]({FACTS_PROCEDURE_INPUTS_JSONL})",
        f"- [claim_award_inputs.jsonl]({CLAIM_AWARD_INPUTS_JSONL})",
        f"- [reasoning_inputs.jsonl]({REASONING_INPUTS_JSONL})",
        "",
        "## Token Consumption",
        "",
        f"- Pipeline B total input tokens: **{summary['tokens']['pipeline_b_total']}**",
        f"- Pipeline C total input tokens: **{summary['tokens']['pipeline_c_total']}**",
        f"- Pipeline D total input tokens: **{summary['tokens']['pipeline_d_total']}**",
        f"- Combined total input tokens: **{summary['tokens']['combined_total']}**",
        f"- Average tokens per case:",
        f"  - B: **{summary['tokens']['pipeline_b_avg']:.2f}**",
        f"  - C: **{summary['tokens']['pipeline_c_avg']:.2f}**",
        f"  - D: **{summary['tokens']['pipeline_d_avg']:.2f}**",
        "",
        "## Five Sanity Checks",
        "",
    ]
    for check in sanity_checks:
        lines.append(f"### {check['name']}")
        lines.append(f"- Status: **{check['status']}**")
        for metric, value in check["metrics"].items():
            lines.append(f"- {metric}: **{value}**")
        if check.get("note"):
            lines.append(f"- Note: {check['note']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    count_tokens, token_method = build_token_counter()

    cases_core = {row["itemid"]: row for row in load_json(CASES_CORE)}
    selected_itemids = [str(x) for x in args.itemids] if args.itemids else None
    if selected_itemids and args.max_cases is not None:
        selected_itemids = selected_itemids[: args.max_cases]
    if selected_itemids:
        rows_by_itemid = load_cases_by_itemid(selected_itemids, fallback_cases_json=UNSTRUCTURED, backfill_store=True)
        missing = [itemid for itemid in selected_itemids if itemid not in rows_by_itemid]
        if missing:
            raise RuntimeError(f"{len(missing)} itemids could not be loaded from case store or cases.json")
        cases_iterable = (rows_by_itemid[itemid] for itemid in selected_itemids)
    else:
        cases_iterable = iter_cases_from_cases_json(UNSTRUCTURED)
    targeted_refresh = bool(selected_itemids)

    core_rows: list[dict[str, Any]] = []
    facts_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    reasoning_rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    top_level_presence = Counter()
    normalized_top_level_presence = Counter()
    document_profile_counts = Counter()
    content_coverage = 0
    paragraphs_coverage = 0
    compensation_raw_marker = 0
    compensation_extracted = 0
    conclusion_present = 0
    operative_extracted = 0
    appendix_table_extracted = 0

    violation_match = 0
    conclusion_count_match = 0
    decision_body_match = 0
    mentioned_match = 0
    total_cases = 0

    tokens_b_total = 0
    tokens_c_total = 0
    tokens_d_total = 0

    for row in cases_iterable:
        if not selected_itemids and args.max_cases is not None and total_cases >= args.max_cases:
            break
        total_cases += 1
        if args.sync_case_store:
            write_case_to_store(row, force=False)
        itemid = row["itemid"]
        existing = cases_core.get(itemid)
        section_key, sections = top_level_sections(row)
        if sections:
            content_coverage += 1
        if isinstance(row.get("paragraphs"), list) and row.get("paragraphs"):
            paragraphs_coverage += 1

        section_map = {section.get("section_name"): section for section in sections if section.get("section_name")}
        for name in section_map:
            top_level_presence[name] += 1

        profiled = build_profiled_sections(
            row=row,
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
        article_41_method = profiled["article_41_method"]
        operative_text = profiled["operative_text"]
        law_text_full = merged_text([relevant_law_text, reasoning_text, article_41_text])
        appendix_table_text = format_pipeline_c_appendix_text(row.get("docx_lossless"))
        for name in profiled["normalized_top_level_section_names"]:
            normalized_top_level_presence[name] += 1
        document_profile_counts[profiled["document_profile"]] += 1

        if COMPENSATION_HEADING_RE.search(law_text_full):
            compensation_raw_marker += 1
        if article_41_text:
            compensation_extracted += 1
        if "conclusion" in section_map:
            conclusion_present += 1
        if operative_text:
            operative_extracted += 1
        if appendix_table_text:
            appendix_table_extracted += 1

        core_row = make_core_row(row, existing, section_key, sections)
        core_rows.append(core_row)

        facts_input_text = clean_text("\n\n".join(part for part in [introduction_text, procedure_text, facts_text] if part))
        pipeline_b_tokens = count_tokens(facts_input_text)
        pipeline_c_text = clean_text(
            "\n\n".join(part for part in [article_41_text, operative_text, appendix_table_text] if part)
        )
        pipeline_c_tokens = count_tokens(pipeline_c_text)
        pipeline_d_tokens = count_tokens(reasoning_text)

        tokens_b_total += pipeline_b_tokens
        tokens_c_total += pipeline_c_tokens
        tokens_d_total += pipeline_d_tokens

        full_document_text = merged_text([flatten_node(section) for section in sections])

        case_records.append(
            make_case_record(
                row=row,
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
        )

        facts_rows.append(
            {
                "itemid": itemid,
                "appno": row.get("appno"),
                "judgementdate": row.get("judgementdate"),
                "num_applicants_proxy": core_row["num_applicants_proxy"],
                "procedure_text": procedure_text,
                "facts_text": facts_text,
                "introduction_text": introduction_text,
                "summary_intro_text": summary_intro_text,
                "assessment_summary_text": assessment_summary_text,
                "pipeline_b_input_mode": profiled["pipeline_b_input_mode"],
                "input_tokens_estimate": pipeline_b_tokens,
            }
        )

        claim_rows.append(
            {
                "itemid": itemid,
                "appno": row.get("appno"),
                "judgementdate": row.get("judgementdate"),
                "violated_articles": core_row["violated_articles"],
                "num_violations_found": core_row["num_violations_found"],
                "article_41_text": article_41_text,
                "operative_text": operative_text,
                "appendix_table_text": appendix_table_text,
                "claim_narrative_text": merged_text(
                    [
                        summary_intro_text,
                        assessment_summary_text,
                        introduction_text,
                        procedure_text,
                        facts_text,
                    ]
                ),
                "article_41_extraction_method": article_41_method,
                "compensation_section_label": "article_41_or_50" if article_41_text else None,
                "metadata_conclusion_text": row.get("__conclusion"),
                "input_tokens_estimate": pipeline_c_tokens,
            }
        )

        reasoning_rows.append(
            {
                "itemid": itemid,
                "appno": row.get("appno"),
                "judgementdate": row.get("judgementdate"),
                "violated_articles": core_row["violated_articles"],
                "law_text_excluding_article_41": reasoning_text,
                "summary_intro_text": summary_intro_text,
                "assessment_summary_text": assessment_summary_text,
                "relevant_law_text": relevant_law_text,
                "pipeline_e_input_mode": profiled["pipeline_e_input_mode"],
                "article_41_removed": bool(article_41_text),
                "article_41_extraction_method": article_41_method,
                "input_tokens_estimate": pipeline_d_tokens,
            }
        )

        if existing:
            if split_semicolon(existing.get("violated_articles")) == core_row["violated_articles"]:
                violation_match += 1
            if int(existing.get("conclusion_count") or 0) == core_row["conclusion_count"]:
                conclusion_count_match += 1
            if str(existing.get("decision_body_category") or "") == core_row["decision_body_category"]:
                decision_body_match += 1
            if split_semicolon(existing.get("mentioned_articles")) == core_row["mentioned_articles"]:
                mentioned_match += 1

        if len(sample_rows) < 5:
            sample_rows.append(
                {
                    "itemid": itemid,
                    "docname": row.get("docname"),
                    "violated_articles": core_row["violated_articles"],
                    "article_41_excerpt": sample_excerpt(article_41_text),
                    "operative_excerpt": sample_excerpt(operative_text),
                    "appendix_table_excerpt": sample_excerpt(appendix_table_text),
                    "reasoning_excerpt": sample_excerpt(reasoning_text),
                    "document_profile": profiled["document_profile"],
                    "pipeline_b_input_mode": profiled["pipeline_b_input_mode"],
                    "pipeline_e_input_mode": profiled["pipeline_e_input_mode"],
                }
            )

    if targeted_refresh:
        merge_jsonl_by_itemid(CASE_FEATURES_LABELS_JSONL, case_records)
        merge_jsonl_by_itemid(CORE_CASE_JSONL, core_rows)
        merge_jsonl_by_itemid(FACTS_PROCEDURE_INPUTS_JSONL, facts_rows)
        merge_jsonl_by_itemid(CLAIM_AWARD_INPUTS_JSONL, claim_rows)
        merge_jsonl_by_itemid(REASONING_INPUTS_JSONL, reasoning_rows)
    else:
        write_jsonl(CASE_FEATURES_LABELS_JSONL, case_records)
        write_jsonl(CORE_CASE_JSONL, core_rows)
        write_jsonl(FACTS_PROCEDURE_INPUTS_JSONL, facts_rows)
        write_jsonl(CLAIM_AWARD_INPUTS_JSONL, claim_rows)
        write_jsonl(REASONING_INPUTS_JSONL, reasoning_rows)

    # Per-case JSON files for fast individual loading
    per_case_dir = OUTPUTS / "cases"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    for rec in case_records:
        itemid = rec["itemid"]
        (per_case_dir / f"{itemid}.json").write_text(
            json.dumps(rec, ensure_ascii=False), encoding="utf-8"
        )

    if total_cases == 0:
        raise RuntimeError("No cases were processed")
    if targeted_refresh:
        refresh_summary = {
            "mode": "targeted_refresh",
            "itemids": selected_itemids,
            "cases_processed": total_cases,
            "jsonl_outputs_updated": [
                str(CASE_FEATURES_LABELS_JSONL),
                str(CORE_CASE_JSONL),
                str(FACTS_PROCEDURE_INPUTS_JSONL),
                str(CLAIM_AWARD_INPUTS_JSONL),
                str(REASONING_INPUTS_JSONL),
            ],
            "per_case_dir": str(per_case_dir),
            "runtime_seconds": time.perf_counter() - start,
        }
        print(json.dumps(refresh_summary, ensure_ascii=False, indent=2))
        return

    sanity_checks = [
        {
            "name": "Content And Paragraph Coverage",
            "status": "PASS" if content_coverage == total_cases and paragraphs_coverage == total_cases else "WARN",
            "metrics": {
                "content_coverage": f"{content_coverage}/{total_cases}",
                "paragraphs_coverage": f"{paragraphs_coverage}/{total_cases}",
            },
            "note": "Current ECHR-NPD canonical input already contains nested content and paragraphs for all cases.",
        },
        {
            "name": "Top-Level Section Coverage",
            "status": "PASS" if top_level_presence["facts"] == total_cases and top_level_presence["law"] == total_cases and top_level_presence["conclusion"] == total_cases else "WARN",
            "metrics": {
                "procedure_present": f"{top_level_presence['procedure']}/{total_cases}",
                "facts_present": f"{top_level_presence['facts']}/{total_cases}",
                "law_present": f"{top_level_presence['law']}/{total_cases}",
                "conclusion_present": f"{top_level_presence['conclusion']}/{total_cases}",
            },
            "note": "Procedure is not universal in the oldest judgments, but facts/law/conclusion are the key extraction anchors.",
        },
        {
            "name": "Article 41 Extraction Coverage",
            "name": "Compensation Section Extraction Coverage",
            "status": "PASS" if compensation_extracted == compensation_raw_marker else "WARN",
            "metrics": {
                "law_sections_with_compensation_heading_marker": compensation_raw_marker,
                "compensation_section_text_extracted": compensation_extracted,
                "extraction_rate": f"{(compensation_extracted / compensation_raw_marker * 100):.2f}%" if compensation_raw_marker else "n/a",
            },
            "note": "Checks that recursive section extraction recovers compensation sections when the law section explicitly contains an Article 41 / Article 50 / Just Satisfaction heading.",
        },
        {
            "name": "Operative Section Extraction Coverage",
            "status": "PASS" if operative_extracted == conclusion_present else "WARN",
            "metrics": {
                "cases_with_conclusion_section": conclusion_present,
                "operative_text_extracted": operative_extracted,
                "extraction_rate": f"{(operative_extracted / conclusion_present * 100):.2f}%" if conclusion_present else "n/a",
            },
            "note": "Uses the top-level conclusion section as the canonical operative-clause source for Pipeline C inputs.",
        },
        {
            "name": "DOCX Table Augmentation Coverage",
            "status": "PASS" if appendix_table_extracted > 0 else "WARN",
            "metrics": {
                "cases_with_appendix_table_text": appendix_table_extracted,
                "share_of_cases": f"{(appendix_table_extracted / total_cases * 100):.2f}%",
            },
            "note": "Counts cases where source docx table blocks were recovered and appended to Pipeline C evidence inputs.",
        },
        {
            "name": "Core-Case Consistency Vs Existing cases_core",
            "status": "PASS"
            if violation_match == total_cases and conclusion_count_match == total_cases and decision_body_match == total_cases and mentioned_match == total_cases
            else "WARN",
            "metrics": {
                "violated_articles_exact_match": f"{violation_match}/{total_cases}",
                "conclusion_count_exact_match": f"{conclusion_count_match}/{total_cases}",
                "decision_body_exact_match": f"{decision_body_match}/{total_cases}",
                "mentioned_articles_exact_match": f"{mentioned_match}/{total_cases}",
            },
            "note": "Confirms that the new deterministic backbone agrees with the current canonical flat table on the main metadata and violation fields.",
        },
    ]

    runtime_seconds = time.perf_counter() - start
    summary = {
        "input_dataset": str(UNSTRUCTURED),
        "cases_processed": total_cases,
        "runtime_seconds": runtime_seconds,
        "token_estimation_method": token_method,
        "tokens": {
            "pipeline_b_total": tokens_b_total,
            "pipeline_c_total": tokens_c_total,
            "pipeline_d_total": tokens_d_total,
            "combined_total": tokens_b_total + tokens_c_total + tokens_d_total,
            "pipeline_b_avg": tokens_b_total / total_cases,
            "pipeline_c_avg": tokens_c_total / total_cases,
            "pipeline_d_avg": tokens_d_total / total_cases,
        },
        "top_level_section_presence": dict(top_level_presence),
        "normalized_top_level_section_presence": dict(normalized_top_level_presence),
        "document_profile_counts": dict(document_profile_counts),
        "article_41": {
            "raw_marker_count": compensation_raw_marker,
            "extracted_count": compensation_extracted,
        },
        "operative": {
            "conclusion_present": conclusion_present,
            "extracted_count": operative_extracted,
        },
        "docx_tables": {
            "cases_with_appendix_table_text": appendix_table_extracted,
        },
        "outputs": {
            "case_features_labels_jsonl": str(CASE_FEATURES_LABELS_JSONL),
            "core_case_jsonl": str(CORE_CASE_JSONL),
            "facts_procedure_inputs_jsonl": str(FACTS_PROCEDURE_INPUTS_JSONL),
            "claim_award_inputs_jsonl": str(CLAIM_AWARD_INPUTS_JSONL),
            "reasoning_inputs_jsonl": str(REASONING_INPUTS_JSONL),
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    SANITY_JSON.write_text(json.dumps(sanity_checks, ensure_ascii=False, indent=2), encoding="utf-8")
    SAMPLES_JSON.write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(make_markdown(summary, sanity_checks), encoding="utf-8")


if __name__ == "__main__":
    main()

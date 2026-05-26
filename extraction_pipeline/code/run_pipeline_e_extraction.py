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


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = EXTRACTION_ROOT.parent
INPUT_JSONL = EXTRACTION_ROOT / "outputs" / "case_features_labels.jsonl"
SPLITS_DIR = DATASET_ROOT / "splits"
PROMPT_PATH = EXTRACTION_ROOT / "prompts" / "pipeline_e_system_prompt.md"
SCHEMA_PATH = EXTRACTION_ROOT / "schemas" / "pipeline_e_reasoning.schema.json"
RUNS_ROOT = EXTRACTION_ROOT / "runs" / "pipeline_e"
MAX_REASONING_CHARS = int(os.environ.get("PIPELINE_E_MAX_REASONING_CHARS", "24000"))
REASONING_KEYWORD_RE = re.compile(
    r"\b(court (?:considers|notes|observes|reiterates|finds|concludes)|"
    r"procedural obligation|procedural limb|effective investigation|"
    r"constitut(?:e|ed|es) torture|amount(?:ed)? to torture|"
    r"constitut(?:e|ed|es) inhuman treatment|amount(?:ed)? to inhuman treatment|"
    r"constitut(?:e|ed|es) degrading treatment|amount(?:ed)? to degrading treatment|"
    r"finding of a violation constitutes in itself sufficient just satisfaction|"
    r"equitable basis|severity|duration|vulnerab|precedent)\b",
    re.IGNORECASE,
)
ARTICLE3_TORTURE_RE = re.compile(r"\b(constitut(?:e|ed|es) torture|amount(?:ed)? to torture)\b", re.IGNORECASE)
ARTICLE3_INHUMAN_RE = re.compile(r"\b(constitut(?:e|ed|es) inhuman treatment|amount(?:ed)? to inhuman treatment)\b", re.IGNORECASE)
ARTICLE3_DEGRADING_RE = re.compile(r"\b(constitut(?:e|ed|es) degrading treatment|amount(?:ed)? to degrading treatment)\b", re.IGNORECASE)
PROCEDURAL_RE = re.compile(r"\b(procedural obligation|procedural limb|effective investigation|failure to investigate|lack of an effective investigation)\b", re.IGNORECASE)
SUBSTANTIVE_RE = re.compile(r"\b(substantive limb|torture|inhuman treatment|degrading treatment)\b", re.IGNORECASE)
FINDING_SUFFICIENT_RE = re.compile(r"\bfinding of a violation constitutes in itself sufficient just satisfaction\b", re.IGNORECASE)
EQUITABLE_BASIS_RE = re.compile(r"\bequitable basis\b", re.IGNORECASE)
# Article 5 (liberty & security) sub-limb signals
ARTICLE5_LENGTH_DETENTION_RE = re.compile(
    r"\b(length of (?:the )?(?:pre-?trial )?detention|excessive(?:ly)? long detention|unreasonable length of detention|"
    r"detention (?:was|has been) excessive|special diligence)\b",
    re.IGNORECASE,
)
ARTICLE5_UNLAWFUL_DETENTION_RE = re.compile(
    r"\b(unlawful detention|arbitrary detention|detention.{0,40}(?:unlawful|arbitrary)|not(?:hing)? in accordance with (?:a )?procedure prescribed by law)\b",
    re.IGNORECASE,
)
ARTICLE5_HABEAS_RE = re.compile(
    r"\b(speedy review|habeas corpus|review of lawfulness of detention|decide speedily)\b",
    re.IGNORECASE,
)
ARTICLE5_COMPENSATION_RE = re.compile(r"\benforceable right to compensation\b", re.IGNORECASE)
ARTICLE5_GROUNDS_RE = re.compile(
    r"\b(relevant and sufficient reasons|relevant and sufficient grounds|continued detention)\b",
    re.IGNORECASE,
)
# Article 6 sub-limb signals
ARTICLE6_LENGTH_PROCEEDINGS_RE = re.compile(
    r"\b(length of (?:the )?proceedings|reasonable time|excessive length of proceedings)\b",
    re.IGNORECASE,
)
ARTICLE6_FAIR_TRIAL_RE = re.compile(r"\b(fair hearing|fairness of (?:the )?proceedings|equality of arms|adversarial)\b", re.IGNORECASE)
ARTICLE6_WITNESS_RE = re.compile(r"\b(examine witnesses|question witnesses|absent witness)\b", re.IGNORECASE)
ARTICLE6_ACCESS_COURT_RE = re.compile(r"\b(access to (?:a )?court|right of access to court)\b", re.IGNORECASE)
# Article 8 sub-limb signals
ARTICLE8_PRIVATE_LIFE_RE = re.compile(r"\b(private life|personal autonomy|physical integrity)\b", re.IGNORECASE)
ARTICLE8_FAMILY_LIFE_RE = re.compile(r"\bfamily life\b", re.IGNORECASE)
ARTICLE8_HOME_RE = re.compile(r"\b(search of (?:the )?home|right to respect for (?:the )?home)\b", re.IGNORECASE)
ARTICLE8_CORRESPONDENCE_RE = re.compile(r"\b(correspondence|secret surveillance|interception of communications)\b", re.IGNORECASE)
# Reasoning-factor signals beyond equitable_basis
SEVERITY_RE = re.compile(r"\b(severity|seriousness of (?:the )?violation|particularly serious|gravity of)\b", re.IGNORECASE)
DURATION_FACTOR_RE = re.compile(
    r"\b(lengthy (?:detention|proceedings)|prolonged|length of (?:the )?(?:detention|proceedings)|"
    r"continued detention|took over \d+ years|a total of \d+ years)\b",
    re.IGNORECASE,
)
VULNERABILITY_RE = re.compile(
    r"\b(vulnerab|minor applicant|child applicant|detained applicant|applicant[’'s]*\s+(?:vulnerab|minority|disability)|"
    r"suffering and frustration)\b",
    re.IGNORECASE,
)
PRECEDENT_ANCHOR_RE = re.compile(
    r"\b(see,?\s+(?:mutatis mutandis,?\s+)?[A-ZŠŽČŁÖÜÆØÅÄÉÈÊÍÓÑÀÂÇÙÛ][A-Za-zŠšŽžČčŁłÖöÜüÆæØøÅåÄäÉéÈèÊêÍíÓóÑñÀàÂâÇçÙùÛû'’\-\. ]{1,60}\s+v\.)",
    re.IGNORECASE,
)
CLAIM_CEILING_RE = re.compile(
    r"\b(award(?:s|ed)?\s+less than (?:the )?claimed|exceed(?:s|ed) the (?:amount|sum) claimed|"
    r"within the (?:amount|sum) claimed|below the claim)\b",
    re.IGNORECASE,
)
NO_AWARD_RE = re.compile(r"\b(?:makes?|made)\s+no\s+award(?: under Article 41)?\b|\bno award under Article 41\b", re.IGNORECASE)
RULE_60_RE = re.compile(r"\brule\s+60\b", re.IGNORECASE)
SUMMARY_NO_AWARD_RE = re.compile(r"\bno award\b|\bmade no award\b|\brule 60\b", re.IGNORECASE)
SUMMARY_SUFFICIENT_RE = re.compile(r"\bsufficient just satisfaction\b|\bfinding of (?:a )?violation.*sufficient\b", re.IGNORECASE)

DEFAULT_SPLITS = ["set1_primary", "set2_postcut_ood", "set3_challenging"]
WRITE_LOCK = threading.Lock()
GUIDE = {
    "itemid": "string",
    "reasoning_layer": {
        "violation_type": ["substantive", "procedural", "both", "other", None],
        "violation_subtype": ["short controlled labels"],
        "violation_duration_months": "number|null",
        "reasoning": {
            "reasoning_factors": [
                "severity_of_harm",
                "duration_of_violation",
                "applicant_vulnerability",
                "claim_ceiling",
                "equitable_basis",
                "finding_violation_sufficient",
                "precedent_anchor",
            ],
            "award_reasoning_summary": "string|null",
        },
    },
}

# Function guide (what each def is responsible for)
# - parse_args: Parse CLI options for Pipeline E reasoning extraction.
# - load_json: Load JSON file content.
# - load_jsonl_by_itemid: Build itemid-indexed row map from JSONL.
# - load_split_ids: Resolve split ids from JSON or CSV split definitions.
# - dedupe_preserve_order: Remove duplicates while keeping input order.
# - coerce_string_list: Coerce unknown list-like values into clean list[str].
# - clean_text: Normalize text safely before reasoning-span extraction.
# - split_paragraphs: Split text into compact paragraph units.
# - head_tail_truncate: Keep head/tail evidence when text exceeds char budget.
# - locate_reasoning_text: Locate and trim the strongest law/reasoning section candidate.
# - build_reasoning_source_text: Build final reasoning source text from structured evidence.
# - build_reasoning_anchor: Compute deterministic reasoning anchors/signals from case text.
# - build_award_summary_anchor: Compute deterministic award-summary anchor from Article 41 text.
# - prepare_reasoning_context: Build context package consumed by prompt + normalization.
# - make_run_dir: Create run directory for Pipeline E outputs.
# - prompt_messages: Build Pipeline E prompt payload (schema + context + evidence).
# - validate_result: Validate raw E output against reasoning schema.
# - normalize_reasoning_result: Normalize E output into stable contract-compliant fields.
# - build_reasoning_validation: Build deterministic validation diagnostics for reasoning layer.
# - merge_retry_feedback: Append targeted retry instructions after validation errors.
# - load_existing_results: Load completed itemids to support resume mode.
# - write_jsonl_line: Append one JSON row safely with thread lock.
# - write_case_file: Persist per-case reasoning extraction artifact.
# - run_one_case: Execute one Pipeline E extraction with retry/validation loop.
# - write_per_split_results: Emit split-level JSONL files for successful rows.
# - main: CLI entrypoint orchestrating Pipeline E execution.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-based Pipeline E extraction on selected ECHR-NPD cases.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--itemids", nargs="+", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("EXTRACTION_CONCURRENCY", "8")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("EXTRACTION_MAX_RETRIES", "3")))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        coerced: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                coerced.append(text)
        return coerced
    return []


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    return [clean_text(part) for part in re.split(r"\n{2,}", text or "") if clean_text(part)]


def head_tail_truncate(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return f"{text[:keep].rstrip()}\n\n[TRUNCATED]\n\n{text[-keep:].lstrip()}"


def locate_reasoning_text(law_text: str, max_chars: int) -> tuple[str, str]:
    law_text = clean_text(law_text)
    if not law_text:
        return "", "empty"
    paragraphs = split_paragraphs(law_text)
    if not paragraphs:
        return head_tail_truncate(law_text, max_chars), "fallback_head_tail"

    keep_indices: set[int] = set()
    for idx, para in enumerate(paragraphs):
        if REASONING_KEYWORD_RE.search(para):
            keep_indices.add(idx)
            if idx > 0:
                keep_indices.add(idx - 1)
            if idx + 1 < len(paragraphs):
                keep_indices.add(idx + 1)

    if keep_indices:
        selected = "\n\n".join(paragraphs[idx] for idx in sorted(keep_indices))
        return head_tail_truncate(selected, max_chars), "keyword_locator"

    return head_tail_truncate(law_text, max_chars), "fallback_head_tail"


def build_reasoning_source_text(evidence: dict[str, Any]) -> tuple[str, str]:
    input_mode = evidence.get("pipeline_e_input_mode") or "empty"
    law_text = evidence.get("law_text_excluding_article_41") or ""
    relevant_law_text = evidence.get("relevant_law_text") or ""
    assessment_summary_text = evidence.get("assessment_summary_text") or ""
    summary_intro_text = evidence.get("summary_intro_text") or ""

    if input_mode == "law_and_relevant_law":
        combined = clean_text("\n\n".join(part for part in [relevant_law_text, law_text] if clean_text(part)))
        located_text, located_mode = locate_reasoning_text(combined, MAX_REASONING_CHARS)
        return located_text, located_mode
    if input_mode == "assessment_summary_fallback":
        return clean_text(assessment_summary_text), "assessment_summary_fallback"
    if input_mode == "summary_intro_fallback":
        return clean_text("\n\n".join(part for part in [summary_intro_text, assessment_summary_text] if clean_text(part))), "summary_intro_fallback"
    return "", "empty"


def build_reasoning_anchor(row: dict[str, Any], law_text: str, article_41_text: str) -> dict[str, Any]:
    violated = set((row.get("core_case") or {}).get("violated_articles") or [])
    detailed_violations = (row.get("core_case") or {}).get("detailed_violations") or []
    detailed_articles = {str(d.get("article") or "") for d in detailed_violations if isinstance(d, dict)}
    text = clean_text(law_text)
    subtypes: list[str] = []
    has_procedural = bool(PROCEDURAL_RE.search(text))
    has_substantive = bool(SUBSTANTIVE_RE.search(text))

    if "3" in violated:
        if ARTICLE3_TORTURE_RE.search(text):
            subtypes.append("article3_torture")
        if ARTICLE3_INHUMAN_RE.search(text):
            subtypes.append("article3_inhuman_treatment")
        if ARTICLE3_DEGRADING_RE.search(text):
            subtypes.append("article3_degrading_treatment")
        if has_procedural:
            subtypes.append("article3_procedural")
    if "2" in violated and has_procedural:
        subtypes.append("article2_procedural")

    if "5" in violated:
        # If detailed_articles explicitly names 5-X sub-articles, trust them as
        # authoritative and suppress law-text keyword fallback for other sub-articles
        # (law text often discusses multiple sub-articles' principles without finding violation).
        has_explicit_5_sub = any(a.startswith("5-") for a in detailed_articles)
        if "5-3" in detailed_articles or (not has_explicit_5_sub and (ARTICLE5_LENGTH_DETENTION_RE.search(text) or ARTICLE5_GROUNDS_RE.search(text))):
            subtypes.append("article5_3_excessive_pre_trial_detention")
        if "5-1" in detailed_articles or (not has_explicit_5_sub and ARTICLE5_UNLAWFUL_DETENTION_RE.search(text)):
            subtypes.append("article5_1_unlawful_or_arbitrary_detention")
        if "5-4" in detailed_articles or (not has_explicit_5_sub and ARTICLE5_HABEAS_RE.search(text)):
            subtypes.append("article5_4_review_of_lawfulness")
        if "5-5" in detailed_articles or (not has_explicit_5_sub and ARTICLE5_COMPENSATION_RE.search(text)):
            subtypes.append("article5_5_compensation")

    if "6" in violated:
        if "6-1" in detailed_articles and ARTICLE6_LENGTH_PROCEEDINGS_RE.search(text):
            subtypes.append("article6_1_length_of_proceedings")
        if "6-1" in detailed_articles and ARTICLE6_ACCESS_COURT_RE.search(text):
            subtypes.append("article6_1_access_to_court")
        if "6-1" in detailed_articles and ARTICLE6_FAIR_TRIAL_RE.search(text):
            subtypes.append("article6_1_fair_trial")
        if ("6-3" in detailed_articles or "6-3-d" in detailed_articles) and ARTICLE6_WITNESS_RE.search(text):
            subtypes.append("article6_3_witness_examination")

    if "8" in violated:
        if ARTICLE8_FAMILY_LIFE_RE.search(text):
            subtypes.append("article8_family_life")
        if ARTICLE8_PRIVATE_LIFE_RE.search(text):
            subtypes.append("article8_private_life")
        if ARTICLE8_HOME_RE.search(text):
            subtypes.append("article8_home")
        if ARTICLE8_CORRESPONDENCE_RE.search(text):
            subtypes.append("article8_correspondence")

    violation_type = None
    if has_procedural and has_substantive:
        violation_type = "both"
    elif has_procedural:
        violation_type = "procedural"
    elif has_substantive:
        violation_type = "substantive"
    # Art 5/6/8 violations without Art 2/3 substantive/procedural keywords are treated as substantive by default
    if violation_type is None and (violated & {"5", "6", "8", "10", "11"}):
        violation_type = "substantive"

    reasoning_factors: list[str] = []
    article_41_text_clean = clean_text(article_41_text)
    combined_reasoning_text = article_41_text_clean + "\n\n" + text
    if FINDING_SUFFICIENT_RE.search(article_41_text_clean):
        reasoning_factors.append("finding_violation_sufficient")
    if EQUITABLE_BASIS_RE.search(article_41_text_clean):
        reasoning_factors.append("equitable_basis")
    if SEVERITY_RE.search(combined_reasoning_text):
        reasoning_factors.append("severity_of_harm")
    if DURATION_FACTOR_RE.search(combined_reasoning_text):
        reasoning_factors.append("duration_of_violation")
    if VULNERABILITY_RE.search(combined_reasoning_text):
        reasoning_factors.append("applicant_vulnerability")
    if PRECEDENT_ANCHOR_RE.search(article_41_text_clean):
        reasoning_factors.append("precedent_anchor")
    if CLAIM_CEILING_RE.search(combined_reasoning_text):
        reasoning_factors.append("claim_ceiling")

    return {
        "violation_type": violation_type,
        "violation_subtype": dedupe_preserve_order(subtypes),
        "reasoning_factors": dedupe_preserve_order(reasoning_factors),
    }


def _format_money(amount: float) -> str:
    if float(amount).is_integer():
        return format(int(amount), ",d")
    return format(amount, ",.2f").rstrip("0").rstrip(".")


def _head_label(head: str) -> str:
    return {
        "non_pecuniary": "non-pecuniary damage",
        "pecuniary": "pecuniary damage",
        "costs": "costs and expenses",
        "bundled": "just satisfaction",
    }.get(head, head.replace("_", " "))


def _summary_rows_for_head(rows: list[dict[str, Any]], head: str) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and row.get("head") == head and isinstance(row.get("eur_amount"), (int, float))]


def _format_explicit_award_summary(det: dict[str, Any], num_applicants: int | None) -> str | None:
    rows = det.get("award_per_applicant") or []
    parts: list[str] = []

    for head, key in (
        ("non_pecuniary", "non_pecuniary_eur"),
        ("pecuniary", "pecuniary_eur"),
        ("costs", "costs_eur"),
    ):
        amount = det.get(key)
        if not isinstance(amount, (int, float)):
            continue
        head_rows = _summary_rows_for_head(rows, head)
        if isinstance(num_applicants, int) and num_applicants > 1 and len(head_rows) == num_applicants:
            per_values = {float(row["eur_amount"]) for row in head_rows}
            if len(per_values) == 1:
                per_amount = next(iter(per_values))
                parts.append(f"EUR {_format_money(per_amount)} to each applicant for {_head_label(head)}")
                continue
        parts.append(f"EUR {_format_money(float(amount))} for {_head_label(head)}")

    bundled_amount = det.get("bundled_award_eur")
    if not parts and isinstance(bundled_amount, (int, float)):
        parts.append(f"a bundled EUR {_format_money(float(bundled_amount))} award")

    if not parts:
        return None

    if len(parts) == 1:
        return f"The Court awarded {parts[0]}."
    if len(parts) == 2:
        return f"The Court awarded {parts[0]} and {parts[1]}."
    return f"The Court awarded {', '.join(parts[:-1])}, and {parts[-1]}."


def build_award_summary_anchor(row: dict[str, Any], article_41_text: str) -> dict[str, Any]:
    article_41_text = clean_text(article_41_text)
    if not article_41_text:
        return {"mode": "none", "summary": None}

    if NO_AWARD_RE.search(article_41_text):
        if RULE_60_RE.search(article_41_text):
            return {
                "mode": "no_award_rule60",
                "summary": "The Court made no award under Article 41 because the applicant failed to submit quantified claims with supporting documents within the time limit set by the Court, as required by Rule 60.",
            }
        return {
            "mode": "no_award",
            "summary": "The Court made no award under Article 41.",
        }

    num_applicants = ((row.get("facts_procedure") or {}).get("num_applicants")) or ((row.get("core_case") or {}).get("num_applicants_proxy"))
    det = layer1_deterministic(
        "",
        "",
        "",
        article_41_text,
        num_applicants if isinstance(num_applicants, int) else None,
    )
    if det.get("satisfaction_sufficient"):
        return {
            "mode": "satisfaction_sufficient",
            "summary": "The Court held that the finding of a violation constituted in itself sufficient just satisfaction.",
        }

    award_summary = _format_explicit_award_summary(det, num_applicants if isinstance(num_applicants, int) else None)
    if award_summary:
        return {
            "mode": "explicit_award",
            "summary": award_summary,
        }

    return {"mode": "none", "summary": None}


def prepare_reasoning_context(row: dict[str, Any]) -> dict[str, Any]:
    evidence = (row["reasoning_layer"].get("evidence_inputs") or {})
    law_text = evidence.get("law_text_excluding_article_41") or ""
    article_41_text = (row["claim_and_award_layer"].get("cross_validation_inputs") or {}).get("article_41_text") or ""
    scattered_reasoning_snippets = evidence.get("scattered_reasoning_snippets") or []
    routed_text, input_mode = build_reasoning_source_text(evidence)
    anchor = build_reasoning_anchor(row, law_text, article_41_text)
    award_anchor = build_award_summary_anchor(row, article_41_text)
    return {
        "law_text_original": clean_text(law_text),
        "law_text_sent": routed_text,
        "article_41_text": clean_text(article_41_text),
        "scattered_reasoning_snippets": scattered_reasoning_snippets,
        "input_mode": input_mode,
        "anchor": anchor,
        "award_anchor": award_anchor,
    }


def make_run_dir(run_name: str | None) -> Path:
    dirname = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cases").mkdir(exist_ok=True)
    return run_dir


def prompt_messages(system_prompt: str, schema: dict[str, Any], row: dict[str, Any], prepared: dict[str, Any], include_schema_in_payload: bool) -> list[dict[str, str]]:
    core = row["core_case"]
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
            "pipeline": "E",
            "goal": "extract structured legal reasoning fields",
            "schema_name": "PipelineEReasoningExtraction"
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
            "award_summary_mode": prepared["award_anchor"].get("mode"),
            "award_summary_anchor": prepared["award_anchor"].get("summary"),
        },
        "output_guide": GUIDE,
    }
    if include_schema_in_payload:
        payload["output_schema"] = schema
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def validate_result(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
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
    if "reasoning_layer" not in result:
        errors.append("missing reasoning_layer")
    return errors


def normalize_reasoning_result(result: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(result))
    layer = normalized.get("reasoning_layer") or {}
    anchor = prepared["anchor"]
    award_anchor = prepared.get("award_anchor") or {}
    input_mode = prepared["input_mode"]

    anchor_vtype = anchor.get("violation_type")
    if anchor_vtype and layer.get("violation_type") in (None, "other"):
        layer["violation_type"] = anchor_vtype

    current_subtypes = coerce_string_list(layer.get("violation_subtype"))
    if not current_subtypes and anchor.get("violation_subtype"):
        layer["violation_subtype"] = coerce_string_list(anchor["violation_subtype"])
    else:
        layer["violation_subtype"] = dedupe_preserve_order([*current_subtypes, *(anchor.get("violation_subtype") or [])])

    if "violation_duration_months" not in layer:
        layer["violation_duration_months"] = None

    reasoning = layer.get("reasoning") or {}
    current_factors = reasoning.get("reasoning_factors") or []
    reasoning["reasoning_factors"] = dedupe_preserve_order([*current_factors, *(anchor.get("reasoning_factors") or [])])
    current_summary = clean_text(reasoning.get("award_reasoning_summary"))
    anchor_mode = award_anchor.get("mode")
    anchor_summary = clean_text(award_anchor.get("summary"))
    if anchor_summary:
        if anchor_mode in {"no_award_rule60", "no_award", "satisfaction_sufficient"}:
            reasoning["award_reasoning_summary"] = anchor_summary
        elif anchor_mode == "explicit_award":
            contradicts_anchor = (
                not current_summary
                or bool(SUMMARY_NO_AWARD_RE.search(current_summary))
                or bool(SUMMARY_SUFFICIENT_RE.search(current_summary))
            )
            if contradicts_anchor:
                reasoning["award_reasoning_summary"] = anchor_summary
    layer["reasoning"] = reasoning

    if input_mode == "empty" and not anchor_vtype and not (anchor.get("violation_subtype") or []):
        layer["violation_type"] = None
        layer["violation_subtype"] = []
        layer["violation_duration_months"] = None
        layer["reasoning"] = {
            "reasoning_factors": dedupe_preserve_order(anchor.get("reasoning_factors") or []),
            "award_reasoning_summary": anchor_summary or None,
        }

    normalized["reasoning_layer"] = layer
    return normalized


def build_reasoning_validation(result: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    layer = result.get("reasoning_layer") or {}
    anchor = prepared["anchor"]
    award_anchor = prepared.get("award_anchor") or {}
    current_vtype = layer.get("violation_type")
    anchor_vtype = anchor.get("violation_type")
    current_subtypes = set(coerce_string_list(layer.get("violation_subtype")))
    anchor_subtypes = set(anchor.get("violation_subtype") or [])
    summary = clean_text(((layer.get("reasoning") or {}).get("award_reasoning_summary")))
    award_anchor_mode = award_anchor.get("mode")
    award_summary_anchor = clean_text(award_anchor.get("summary"))
    summary_conflicts_with_anchor = False
    if award_anchor_mode in {"no_award_rule60", "no_award"} and summary:
        summary_conflicts_with_anchor = not SUMMARY_NO_AWARD_RE.search(summary)
    elif award_anchor_mode == "satisfaction_sufficient" and summary:
        summary_conflicts_with_anchor = not SUMMARY_SUFFICIENT_RE.search(summary)
    elif award_anchor_mode == "explicit_award" and summary:
        summary_conflicts_with_anchor = bool(SUMMARY_NO_AWARD_RE.search(summary) or SUMMARY_SUFFICIENT_RE.search(summary))
    return {
        "input_mode": prepared["input_mode"],
        "law_text_original_chars": len(prepared["law_text_original"]),
        "law_text_sent_chars": len(prepared["law_text_sent"]),
        "anchor_violation_type": anchor_vtype,
        "anchor_violation_subtype": sorted(anchor_subtypes),
        "anchor_violation_type_mismatch": bool(anchor_vtype and current_vtype and anchor_vtype != current_vtype),
        "anchor_violation_subtype_missing": sorted(anchor_subtypes - current_subtypes),
        "award_summary_anchor_mode": award_anchor_mode,
        "award_summary_anchor": award_summary_anchor or None,
        "award_summary_conflicts_with_anchor": summary_conflicts_with_anchor,
    }


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


def run_one_case(
    itemid: str,
    row: dict[str, Any],
    client: OpenAICompatibleClient,
    system_prompt: str,
    schema: dict[str, Any],
    max_retries: int = 1,  # kept for backward-compat with the standalone CLI; ignored by design
) -> dict[str, Any]:
    """Single-shot pipeline E execution.

    Returns one of:
    - {status: "success", result, usage, elapsed_seconds}
    - {status: "api_error", api_error, ...}
    - {status: "schema_validation", errors, raw_result, ...}
    """
    start = time.perf_counter()
    prepared = prepare_reasoning_context(row)
    messages = prompt_messages(system_prompt, schema, row, prepared, include_schema_in_payload=not client.use_json_schema)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        parsed, usage = client.chat_json(messages=messages, schema=schema, schema_name="pipeline_e_reasoning")
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
    parsed = normalize_reasoning_result(parsed, prepared)
    errors = validate_result(parsed, schema)
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

    parsed["itemid"] = itemid
    parsed["reasoning_validation"] = build_reasoning_validation(parsed, prepared)
    return {
        "status": "success",
        "itemid": itemid,
        "attempts": 1,
        "elapsed_seconds": time.perf_counter() - start,
        "usage": usage_total,
        "result": parsed,
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

    PER_CASE_DIR = EXTRACTION_ROOT / "outputs" / "cases"
    if args.itemids and PER_CASE_DIR.exists():
        rows_by_itemid = {}
        for itemid in unique_ids:
            case_file = PER_CASE_DIR / f"{itemid}.json"
            if case_file.exists():
                rows_by_itemid[itemid] = load_json(case_file)
        missing_per_case = [itemid for itemid in unique_ids if itemid not in rows_by_itemid]
        if missing_per_case:
            print(f"[INFO] {len(missing_per_case)} cases not in per-case dir, falling back to full JSONL...")
            rows_by_itemid = load_jsonl_by_itemid(INPUT_JSONL)
    else:
        rows_by_itemid = load_jsonl_by_itemid(INPUT_JSONL)
    missing = [itemid for itemid in unique_ids if itemid not in rows_by_itemid]
    if missing:
        raise RuntimeError(f"{len(missing)} itemids were not found")

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

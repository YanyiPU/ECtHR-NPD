from __future__ import annotations

import re
from typing import Any


EUR_PREFIX_RE = re.compile(r"\bEUR\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?)\b", re.IGNORECASE)
EUR_SUFFIX_RE = re.compile(r"\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?)\s*(?:euros?|eur)\b", re.IGNORECASE)
KNOWN_CURRENCY_CODES = (
    "EUR",
    "USD",
    "GBP",
    "AZN",
    "RUB",
    "TRY",
    "UAH",
    "PLN",
    "HUF",
    "RON",
    "CZK",
    "SEK",
    "NOK",
    "DKK",
    "CHF",
    "BGN",
    "RSD",
    "HRK",
    "BAM",
    "MKD",
    "ALL",
    "AMD",
    "GEL",
    "MDL",
    "ISK",
)
GENERIC_CURRENCY_TOKEN_RE = r"(?:euros?|eur|" + "|".join(KNOWN_CURRENCY_CODES) + r")"
GENERIC_PREFIX_RE = re.compile(
    rf"\b(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\s*(?P<amount>[0-9]{{1,3}}(?:[,\s][0-9]{{3}})*(?:\.\d+)?|[0-9]+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
GENERIC_SUFFIX_RE = re.compile(
    rf"\b(?P<amount>[0-9]{{1,3}}(?:[,\s][0-9]{{3}})*(?:\.\d+)?|[0-9]+(?:\.\d+)?)\s*(?P<currency>{GENERIC_CURRENCY_TOKEN_RE})\b",
    re.IGNORECASE,
)
TITLE_NAME_RE = re.compile(
    r"\b(Mr|Ms|Mrs|Miss)\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+)*)"
)
ROLE_LABEL_RE = re.compile(
    r"\b(the\s+(?:first|second|third|fourth|fifth)\s+applicant|the\s+applicant(?:'s\s+mother)?|the\s+applicants(?:'?\s+mother)?|the\s+applicant’s\s+mother)\b",
    re.IGNORECASE,
)
CLAIM_KEYWORD_RE = re.compile(r"\b(claim(?:ed|s)?|requested|seeks?|sought)\b", re.IGNORECASE)
JUST_SATISFACTION_RE = re.compile(r"\b(just satisfaction|article\s+41|article\s+50)\b", re.IGNORECASE)
STRASBOURG_CLAIM_CONTEXT_RE = re.compile(
    r"\b(before|to)\s+the\s+Court\b|\bunder\s+Article\s+(?:41|50)\b|\bjust satisfaction\b",
    re.IGNORECASE,
)
IMPLICIT_CLAIM_CONTEXT_RE = re.compile(
    r"\b(in respect of|for)\b.*\b(non-pecuniary|non pecuniary|pecuniary|costs?|expenses|damage)\b"
    r"|\bleft the amount to the Court'?s discretion\b",
    re.IGNORECASE,
)
DOMESTIC_CLAIM_CONTEXT_RE = re.compile(
    r"\b(domestic|district court|regional court|city court|supreme court|court of appeal|civil claim|"
    r"civil action|administrative court|criminal court|before the domestic courts?|first-instance court|"
    r"cassation court|prosecutor|investigator)\b",
    re.IGNORECASE,
)
NON_PEC_TERMS_RE = re.compile(r"\b(non-pecuniary|non pecuniary|moral damage)\b", re.IGNORECASE)
PEC_TERMS_RE = re.compile(r"\bpecuniary\b", re.IGNORECASE)
COSTS_TERMS_RE = re.compile(r"\b(costs?(?: and expenses)?|expenses)\b", re.IGNORECASE)
GENERIC_DAMAGE_TERMS_RE = re.compile(r"\b(damage|damages|compensation)\b", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _to_amount(match_text: str) -> float | None:
    try:
        return float(match_text.replace(",", ""))
    except ValueError:
        return None


def extract_eur_amounts(text: str) -> list[float]:
    amounts: list[float] = []
    seen: set[float] = set()
    for pattern in (EUR_PREFIX_RE, EUR_SUFFIX_RE):
        for match in pattern.finditer(text or ""):
            value = _to_amount(match.group(1))
            if value is None or value in seen:
                continue
            seen.add(value)
            amounts.append(value)
    return amounts


def normalize_currency(token: str | None) -> str | None:
    if not token:
        return None
    text = clean_text(token).casefold()
    if text in {"eur", "euro", "euros"}:
        return "EUR"
    if text.upper() in KNOWN_CURRENCY_CODES:
        return text.upper()
    return None


def extract_currency_amount_mentions(text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for pattern in (GENERIC_PREFIX_RE, GENERIC_SUFFIX_RE):
        for match in pattern.finditer(text or ""):
            amount = _to_amount(match.group("amount").replace(" ", "").replace(",", ""))
            currency_code = normalize_currency(match.group("currency"))
            if amount is None or currency_code is None:
                continue
            span = match.span()
            if span in seen:
                continue
            seen.add(span)
            mentions.append(
                {
                    "amount": amount,
                    "currency_code": currency_code,
                    "start": span[0],
                    "end": span[1],
                    "matched_text": clean_text(match.group(0)),
                }
            )
    mentions.sort(key=lambda item: item["start"])
    return mentions


def split_paragraphs(text: str) -> list[str]:
    return [clean_text(part) for part in re.split(r"\n{2,}", text or "") if clean_text(part)]


def classify_head(snippet: str) -> str | None:
    lower = snippet.lower()
    if "costs and expenses" in lower or "costs" in lower:
        return "costs"
    if "non-pecuniary" in lower or "nonpecuniary" in lower:
        return "non_pecuniary"
    if "pecuniary" in lower:
        return "pecuniary"
    return None


def classify_kind(snippet: str) -> str | None:
    lower = snippet.lower()
    if any(token in lower for token in ("claimed", "claim", "requested", "sought")):
        return "claim"
    if any(token in lower for token in ("the court awards", "awards", "to pay", "holds", "sum awarded")):
        return "award"
    return None


def extract_beneficiary_labels(snippet: str) -> list[str]:
    labels: list[str] = []
    seen_norm: set[str] = set()
    for match in ROLE_LABEL_RE.finditer(snippet):
        label = clean_text(match.group(1))
        norm = label.casefold()
        if label and norm not in seen_norm:
            seen_norm.add(norm)
            labels.append(label)
    for title, name in TITLE_NAME_RE.findall(snippet):
        label = clean_text(f"{title} {name}")
        norm = label.casefold()
        if label and norm not in seen_norm:
            seen_norm.add(norm)
            labels.append(label)
    return labels


def trim_snippet(text: str, limit: int = 260) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    keep = max(60, limit // 2)
    return f"{text[:keep].rstrip()} [TRUNCATED] {text[-keep:].lstrip()}"


def _detected_claim_heads(snippet: str) -> list[str]:
    heads: list[str] = []
    if NON_PEC_TERMS_RE.search(snippet):
        heads.append("non_pecuniary")
    if PEC_TERMS_RE.search(snippet):
        heads.append("pecuniary")
    if COSTS_TERMS_RE.search(snippet):
        heads.append("costs")
    return heads


def extract_narrative_claim_rows(text: str, source: str = "narrative", max_rows: int = 16) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str | None, float, str]] = set()
    relaxed_source = source in {
        "summary_intro_text",
        "assessment_summary_text",
        "introduction_text",
        "procedure_text",
        "facts_text",
    }
    text = clean_text(text)
    for clause in CLAUSE_SPLIT_RE.split(text or ""):
        clause = clause.strip()
        if not clause:
            continue
        if not CLAIM_KEYWORD_RE.search(clause):
            continue
        heads = _detected_claim_heads(clause)
        has_generic_damage = bool(GENERIC_DAMAGE_TERMS_RE.search(clause))
        if not (heads or JUST_SATISFACTION_RE.search(clause) or has_generic_damage):
            continue
        explicit_strasbourg_context = bool(STRASBOURG_CLAIM_CONTEXT_RE.search(clause))
        implicit_relaxed_context = bool(relaxed_source and IMPLICIT_CLAIM_CONTEXT_RE.search(clause))
        if not (explicit_strasbourg_context or implicit_relaxed_context):
            continue
        if DOMESTIC_CLAIM_CONTEXT_RE.search(clause) and not explicit_strasbourg_context:
            continue
        mentions = extract_currency_amount_mentions(clause)
        if not mentions:
            continue
        if len(heads) == 1:
            for mention in mentions:
                key = (heads[0], mention["amount"], mention["currency_code"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "source": source,
                        "head": heads[0],
                        "amount": mention["amount"],
                        "currency_code": mention["currency_code"],
                        "snippet": trim_snippet(clause),
                    }
                )
        elif len(mentions) == 1:
            head = "bundled" if (heads or has_generic_damage) else None
            key = (head, mentions[0]["amount"], mentions[0]["currency_code"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": source,
                    "head": head,
                    "amount": mentions[0]["amount"],
                    "currency_code": mentions[0]["currency_code"],
                    "snippet": trim_snippet(clause),
                }
            )
        if len(rows) >= max_rows:
            break
    return rows[:max_rows]


def build_shared_compensation_evidence(
    article_41_text: str,
    operative_text: str,
    appendix_table_text: str,
    max_entries_per_kind: int = 12,
) -> dict[str, Any]:
    sections = [
        ("article_41", article_41_text),
        ("operative", operative_text),
        ("appendix", appendix_table_text),
    ]
    claim_rows: list[dict[str, Any]] = []
    award_rows: list[dict[str, Any]] = []
    beneficiary_candidates: list[str] = []
    seen_beneficiaries: set[str] = set()

    for source, raw_text in sections:
        for paragraph in split_paragraphs(raw_text):
            amounts = extract_eur_amounts(paragraph)
            if not amounts:
                continue
            kind = classify_kind(paragraph)
            head_guess = classify_head(paragraph)
            labels = extract_beneficiary_labels(paragraph)
            joint_flag = True if "jointly" in paragraph.lower() else None
            if "each" in paragraph.lower() and joint_flag is None:
                joint_flag = False
            for label in labels:
                norm = label.casefold()
                if norm not in seen_beneficiaries:
                    seen_beneficiaries.add(norm)
                    beneficiary_candidates.append(label)
            target = claim_rows if kind == "claim" else award_rows if kind == "award" else None
            if target is None:
                continue
            for amount_eur in amounts:
                target.append(
                    {
                        "source": source,
                        "head_guess": head_guess,
                        "amount_eur": amount_eur,
                        "beneficiary_labels": labels,
                        "is_joint": joint_flag,
                        "snippet": trim_snippet(paragraph),
                    }
                )

    return {
        "beneficiary_label_candidates": beneficiary_candidates[:16],
        "claim_amount_snippets": claim_rows[:max_entries_per_kind],
        "award_amount_snippets": award_rows[:max_entries_per_kind],
    }

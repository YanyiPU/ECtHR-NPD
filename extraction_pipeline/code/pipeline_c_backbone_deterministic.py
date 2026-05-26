from __future__ import annotations

import re
from typing import Any


_EUR_AMOUNT = r"EUR\s+([\d,]+(?:\.\d+)?)"
NON_PEC_REGEX = re.compile(rf"{_EUR_AMOUNT}[^;.]*?(?:in respect of|for)\s+non-?pecuniary\s+damage", re.IGNORECASE)
PEC_REGEX = re.compile(rf"{_EUR_AMOUNT}[^;.]*?(?:in respect of|for)\s+pecuniary\s+damage", re.IGNORECASE)
COSTS_REGEX = re.compile(rf"{_EUR_AMOUNT}[^;.]*?(?:in respect of|for|by way of)\s+costs(?:\s+and\s+expenses)?", re.IGNORECASE)
LEGAL_AID_DEDUCTION_REGEX = re.compile(r"less\s+EUR\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)

# Non-EUR awards with inline court-stated EUR equivalent, e.g.:
# "3,000,000 ITL (approximately EUR 1,549) in respect of non-pecuniary damage"
# "GBP 5,000 (EUR 7,250) by way of costs and expenses"
_EUR_PAREN = r"\((?:approximately\s+)?EUR\s+([\d,]+(?:\.\d+)?)\)"
NON_PEC_PAREN_REGEX = re.compile(
    rf"{_EUR_PAREN}[^;.]*?(?:in respect of|for)\s+non-?pecuniary\s+damage", re.IGNORECASE
)
PEC_PAREN_REGEX = re.compile(
    rf"{_EUR_PAREN}[^;.]*?(?:in respect of|for)\s+pecuniary\s+damage", re.IGNORECASE
)
COSTS_PAREN_REGEX = re.compile(
    rf"{_EUR_PAREN}[^;.]*?(?:in respect of|for|by way of)\s+costs(?:\s+and\s+expenses)?", re.IGNORECASE
)
AWARD_EACH_APPLICANT_EUR_RE = re.compile(r"awards?\s+each applicant\s+EUR\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)
AWARD_THE_APPLICANT_EUR_RE = re.compile(r"awards?\s+the applicant\s+EUR\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)
AWARD_EACH_APPLICANT_EUR_PAREN_RE = re.compile(
    rf"awards?\s+each applicant[^()]*{_EUR_PAREN}",
    re.IGNORECASE,
)
AWARD_THE_APPLICANT_EUR_PAREN_RE = re.compile(
    rf"awards?\s+the applicant[^()]*{_EUR_PAREN}",
    re.IGNORECASE,
)
AWARD_UNDER_THIS_HEAD_EUR_RE = re.compile(r"awards?\s+(?:the applicant\s+)?EUR\s+([\d,]+(?:\.\d+)?)\s+under this head", re.IGNORECASE)
DEADLINE_REGEX = re.compile(r"within\s+(three|3)\s+months", re.IGNORECASE)
INTEREST_REGEX = re.compile(r"(marginal lending rate of the European Central Bank[^;.]*)", re.IGNORECASE)
SATISFACTION_REGEX = re.compile(r"finding of (?:a )?violation constitutes.*?sufficient.*?just satisfaction", re.IGNORECASE | re.DOTALL)
VOTING_REGEX = re.compile(r"COURT[,\s]+(UNANIMOUSLY|by\s+(\d+)\s+votes?\s+to\s+(\d+))", re.IGNORECASE)

APPNO_LINE_RE = re.compile(r"(?m)^\s*\|?\s*(\d{1,6}/\d{2})\s*$")
NAME_YEAR_RE = re.compile(r"(?m)^\s*([^\n|]{3,}?)\s*\n\s*((?:19|20)\d{2})\s*$")
HONORIFIC_NAME_RE = re.compile(r"\b(Mr|Ms|Mrs|Miss)\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+)*)")
TO_EACH_BLOCK_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s+to each:\s*", re.IGNORECASE)
TO_SINGLE_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s+to\s+((?:Mr|Ms|Mrs|Miss)\s+[^\n;|]+)", re.IGNORECASE)
JOINTLY_TO_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s+jointly\s+to\s+([\s\S]+)", re.IGNORECASE)
AMOUNT_ONLY_RE = re.compile(r"([\d,]+(?:\.\d+)?)")

# Precedent citation: "Stašaitis v. Lithuania, no. 47679/99, § 96, 21 March 2002"
# Case name tokens include European accented letters; we keep the name span greedy-bounded.
PRECEDENT_CITATION_RE = re.compile(
    r"(?P<case>[A-ZŠŽČŁÖÜÆØÅÄÉÈÊÍÓÑÀÂÇÙÛ][A-Za-zŠšŽžČčŁłÖöÜüÆæØøÅåÄäÉéÈèÊêÍíÓóÑñÀàÂâÇçÙùÛû'’\-\. ]{1,80}? v\.\s+[A-Za-zÀ-ÿ\-\. ]{2,40}?)"
    r"\s*,\s*nos?\.\s*(?P<appno>\d{1,6}/\d{2}(?:\s*(?:,|and)\s*\d{1,6}/\d{2})*)"
    r"(?P<tail>[^.\n]{0,150})",
    re.UNICODE,
)
_DATE_IN_TAIL_RE = re.compile(
    r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2})",
    re.UNICODE,
)
GRAND_CHAMBER_HINT_RE = re.compile(r"\[GC\]|Grand Chamber", re.IGNORECASE)
APPENDIX_BUNDLED_AWARD_HEADER_RE = re.compile(
    r"amount awarded for pecuniary and non-pecuniary damage and costs(?: and expenses)? per applicant",
    re.IGNORECASE,
)
APPENDIX_PEC_AWARD_HEADER_RE = re.compile(
    r"amount awarded for pecuniary damage",
    re.IGNORECASE,
)
APPENDIX_NON_PEC_AWARD_HEADER_RE = re.compile(
    r"amount awarded for non-?pecuniary damage",
    re.IGNORECASE,
)
APPENDIX_COSTS_AWARD_HEADER_RE = re.compile(
    r"amount awarded .* costs(?: and expenses)?",
    re.IGNORECASE,
)
FOLLOWING_AMOUNTS_RE = re.compile(r"following amounts", re.IGNORECASE)
STOP_OPERATIVE_RE = re.compile(
    r"(?:that from the expiry|dismisses the remainder|done in english|default period plus three percentage points)",
    re.IGNORECASE,
)


def _parse_eur(s: str) -> float:
    if not s:
        return 0.0
    val = s.replace(",", "").strip()
    if not val or val == ".":
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_name(text: str) -> str:
    text = re.sub(r"\b(Mr|Ms|Mrs|Miss)\b\.?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]+", " ", text)
    return " ".join(text.lower().split())


def _title_case_name(text: str) -> str:
    tokens = []
    for part in text.split():
        if part.isupper() and len(part) > 1:
            tokens.append(part.title())
        else:
            tokens.append(part)
    return " ".join(tokens)


def _split_appno_blocks(text: str) -> list[str]:
    matches = list(APPNO_LINE_RE.finditer(text))
    if not matches:
        return [text] if text.strip() else []
    blocks: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        blocks.append(text[start:end].strip())
    return [block for block in blocks if block]


def _split_columns(block: str) -> list[str]:
    return [col for col in (_clean_text(part) for part in block.split("|")) if col]


def _extract_honorific_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for title, raw_name in HONORIFIC_NAME_RE.findall(text):
        sex = "male" if title.lower() == "mr" else "female"
        normalized = _normalize_name(raw_name)
        if normalized:
            mapping[normalized] = sex
            surname = normalized.split()[-1]
            mapping.setdefault(surname, sex)
    return mapping


def _extract_name_year_pairs(text: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for raw_name, year in NAME_YEAR_RE.findall(text):
        name = _title_case_name(_clean_text(raw_name))
        if not name:
            continue
        try:
            birth_year = int(year)
        except ValueError:
            continue
        pairs.append((name, birth_year))
    return pairs


def _extract_names_from_text(text: str) -> list[str]:
    names = []
    for _, raw_name in HONORIFIC_NAME_RE.findall(text):
        cleaned = _title_case_name(_clean_text(raw_name))
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return names


def _parse_amount_only(text: str) -> float | None:
    match = AMOUNT_ONLY_RE.search(text)
    if not match:
        return None
    return _parse_eur(match.group(1))


def _split_paragraphs(text: str) -> list[str]:
    return [part for part in re.split(r"\n{2,}", _clean_text(text)) if part]


def _article41_head(paragraph: str) -> str | None:
    lowered = paragraph.casefold()
    if "non-pecuniary damage" in lowered or "nonpecuniary damage" in lowered:
        return "non_pecuniary"
    if "pecuniary damage" in lowered:
        return "pecuniary"
    if "costs and expenses" in lowered or re.search(r"\bcosts\b", lowered):
        return "costs"
    return None


def _expand_each_applicant_rows(head: str, amount: float, num_applicants: int | None) -> list[dict[str, Any]]:
    if not isinstance(num_applicants, int) or num_applicants <= 0:
        return []
    return [
        {
            "beneficiary_label": f"Applicant {idx}",
            "applicant_index": idx,
            "head": head,
            "eur_amount": amount,
        }
        for idx in range(1, num_applicants + 1)
    ]


def _extract_article41_awards(article_41_text: str, num_applicants: int | None) -> tuple[dict[str, float | None], list[dict[str, Any]], list[str]]:
    totals: dict[str, float | None] = {
        "non_pecuniary_eur": None,
        "pecuniary_eur": None,
        "costs_eur": None,
    }
    entries: list[dict[str, Any]] = []
    notes: list[str] = []

    for paragraph in _split_paragraphs(article_41_text):
        head = _article41_head(paragraph)
        if head is None:
            continue

        amount = None
        expand_each = False

        match = AWARD_EACH_APPLICANT_EUR_RE.search(paragraph) or AWARD_EACH_APPLICANT_EUR_PAREN_RE.search(paragraph)
        if match:
            amount = _parse_eur(match.group(1))
            expand_each = True
        else:
            match = AWARD_THE_APPLICANT_EUR_RE.search(paragraph) or AWARD_THE_APPLICANT_EUR_PAREN_RE.search(paragraph)
            if match:
                amount = _parse_eur(match.group(1))
            else:
                match = AWARD_UNDER_THIS_HEAD_EUR_RE.search(paragraph)
                if match:
                    amount = _parse_eur(match.group(1))

        if amount is None:
            continue

        total_key = f"{head}_eur"
        if expand_each and isinstance(num_applicants, int) and num_applicants > 0:
            totals[total_key] = amount * num_applicants
            entries.extend(_expand_each_applicant_rows(head, amount, num_applicants))
            notes.append(f"Recovered Article 41 paragraph award for {head} using 'each applicant EUR X'.")
        else:
            totals[total_key] = amount
            if isinstance(num_applicants, int) and num_applicants == 1:
                entries.extend(_expand_each_applicant_rows(head, amount, 1))
            notes.append(f"Recovered Article 41 paragraph award for {head}.")

    return totals, entries, notes


def _parse_joint_cell(cell_text: str, head: str) -> tuple[float | None, list[dict[str, Any]]]:
    text = _clean_text(cell_text)
    match = JOINTLY_TO_RE.search(text)
    if not match:
        amount = _parse_amount_only(text)
        if amount is None:
            return None, []
        return amount, []
    amount = _parse_eur(match.group(1))
    beneficiaries = _title_case_name(_clean_text(match.group(2)))
    return amount, [{"beneficiary_label": beneficiaries, "head": head, "eur_amount": amount}]


def _parse_non_pec_cell(cell_text: str) -> tuple[float | None, list[dict[str, Any]]]:
    text = _clean_text(cell_text)
    per_applicant: list[dict[str, Any]] = []
    total = 0.0
    matches = list(TO_EACH_BLOCK_RE.finditer(text))
    if matches:
        for idx, match in enumerate(matches):
            amount = _parse_eur(match.group(1))
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            names = _extract_names_from_text(text[start:end])
            if names:
                for name in names:
                    per_applicant.append({"beneficiary_label": name, "head": "non_pecuniary", "eur_amount": amount})
                total += amount * len(names)
            else:
                total += amount
    else:
        for amount_str, raw_name in TO_SINGLE_RE.findall(text):
            amount = _parse_eur(amount_str)
            name = _title_case_name(_clean_text(raw_name))
            per_applicant.append({"beneficiary_label": name, "head": "non_pecuniary", "eur_amount": amount})
            total += amount
    if per_applicant:
        return total, per_applicant
    amount = _parse_amount_only(text)
    return amount, []


def _parse_bundled_appendix(table_text: str) -> tuple[float | None, list[dict[str, Any]]]:
    total = 0.0
    seen = False
    entries: list[dict[str, Any]] = []
    for block in _split_appno_blocks(table_text):
        columns = _split_columns(block)
        if not columns:
            continue
        amount = _parse_amount_only(columns[-1])
        if amount is None:
            continue
        seen = True
        total += amount
        name_pairs = _extract_name_year_pairs(block)
        label = name_pairs[0][0] if name_pairs else None
        entries.append({"beneficiary_label": label, "head": "bundled", "eur_amount": amount})
    return (total if seen else None), entries


def _parse_separate_appendix(table_text: str, has_costs_column: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = {
        "non_pecuniary_eur": None,
        "pecuniary_eur": None,
        "costs_eur": None,
    }
    running = {
        "non_pecuniary_eur": 0.0,
        "pecuniary_eur": 0.0,
        "costs_eur": 0.0,
    }
    seen = {
        "non_pecuniary_eur": False,
        "pecuniary_eur": False,
        "costs_eur": False,
    }
    entries: list[dict[str, Any]] = []

    for block in _split_appno_blocks(table_text):
        columns = _split_columns(block)
        if len(columns) < 2:
            continue
        award_columns = columns[-3:] if has_costs_column and len(columns) >= 3 else columns[-2:]
        if len(award_columns) == 2:
            pec_cell, non_pec_cell = award_columns
            costs_cell = None
        else:
            pec_cell, non_pec_cell, costs_cell = award_columns

        pec_amount, pec_entries = _parse_joint_cell(pec_cell, "pecuniary")
        if pec_amount is not None:
            running["pecuniary_eur"] += pec_amount
            seen["pecuniary_eur"] = True
            entries.extend(pec_entries)

        non_pec_amount, non_pec_entries = _parse_non_pec_cell(non_pec_cell)
        if non_pec_amount is not None:
            running["non_pecuniary_eur"] += non_pec_amount
            seen["non_pecuniary_eur"] = True
            entries.extend(non_pec_entries)

        if costs_cell:
            costs_amount = _parse_amount_only(costs_cell)
            if costs_amount is not None:
                running["costs_eur"] += costs_amount
                seen["costs_eur"] = True

    for key in totals:
        if seen[key]:
            totals[key] = running[key]
    return totals, entries


def _extract_following_amounts_block(operative_text: str) -> str:
    lines = [line.strip() for line in operative_text.splitlines()]
    start_idx = None
    for idx, line in enumerate(lines):
        if FOLLOWING_AMOUNTS_RE.search(line):
            start_idx = idx + 1
            break
    if start_idx is None:
        return ""
    captured: list[str] = []
    for line in lines[start_idx:]:
        if not line:
            continue
        if STOP_OPERATIVE_RE.search(line):
            break
        captured.append(line)
    return "\n".join(captured).strip()


def extract_article_41_precedents(article_41_text: str) -> list[dict[str, Any]]:
    """Pull structured precedent citations out of the Article 41 paragraph block.

    Each returned entry has `case_name`, `appno`, `date` (if captured), and
    `grand_chamber` (bool), preserving first-seen order for reproducibility.
    """
    text = _clean_text(article_41_text)
    if not text:
        return []
    seen: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    for match in PRECEDENT_CITATION_RE.finditer(text):
        case_name = (match.group("case") or "").strip(" ,;:.")
        appno = (match.group("appno") or "").strip()
        tail = match.group("tail") or ""
        date_match = _DATE_IN_TAIL_RE.search(tail)
        date = date_match.group(1).strip() if date_match else None
        if not case_name or not appno:
            continue
        key = (_normalize_name(case_name), appno)
        if key in seen:
            continue
        seen.add(key)
        # Look 80 chars back from the match start to detect Grand Chamber tag
        start = max(0, match.start() - 80)
        window = text[start : match.end() + 40]
        results.append(
            {
                "case_name": case_name,
                "appno": appno,
                "date": date,
                "grand_chamber": bool(GRAND_CHAMBER_HINT_RE.search(window)),
            }
        )
    return results


def _parse_following_amounts_block(block: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not block:
        return {"non_pecuniary_eur": None, "pecuniary_eur": None, "costs_eur": None}, []
    totals = {
        "non_pecuniary_eur": None,
        "pecuniary_eur": None,
        "costs_eur": None,
    }
    entries: list[dict[str, Any]] = []

    non_pec_values = [_parse_eur(m.group(1)) for m in NON_PEC_REGEX.finditer(block)]
    pec_values = [_parse_eur(m.group(1)) for m in PEC_REGEX.finditer(block)]
    costs_values = [_parse_eur(m.group(1)) for m in COSTS_REGEX.finditer(block)]
    if non_pec_values:
        totals["non_pecuniary_eur"] = sum(non_pec_values)
    if pec_values:
        totals["pecuniary_eur"] = sum(pec_values)
    if costs_values:
        totals["costs_eur"] = sum(costs_values)
    return totals, entries


def layer1_deterministic(
    operative_text: str,
    appendix_table_text: str,
    conclusion_header: str = "",
    article_41_text: str = "",
    num_applicants: int | None = None,
) -> dict[str, Any]:
    non_pec_m = NON_PEC_REGEX.search(operative_text)
    pec_m = PEC_REGEX.search(operative_text)
    costs_m = COSTS_REGEX.search(operative_text)
    deduction_m = LEGAL_AID_DEDUCTION_REGEX.search(operative_text)
    satisfaction_m = SATISFACTION_REGEX.search(operative_text)

    non_pec_eur = _parse_eur(non_pec_m.group(1)) if non_pec_m else None
    pec_eur = _parse_eur(pec_m.group(1)) if pec_m else None
    costs_eur = _parse_eur(costs_m.group(1)) if costs_m else None
    deduction_eur = _parse_eur(deduction_m.group(1)) if deduction_m else None

    # Fallback: non-EUR awards with inline court-stated EUR equivalent
    # e.g. "3,000,000 ITL (approximately EUR 1,549) in respect of non-pecuniary damage"
    if non_pec_eur is None:
        m = NON_PEC_PAREN_REGEX.search(operative_text)
        if m:
            non_pec_eur = _parse_eur(m.group(1))
    if pec_eur is None:
        m = PEC_PAREN_REGEX.search(operative_text)
        if m:
            pec_eur = _parse_eur(m.group(1))
    if costs_eur is None:
        m = COSTS_PAREN_REGEX.search(operative_text)
        if m:
            costs_eur = _parse_eur(m.group(1))

    satisfaction_sufficient = bool(satisfaction_m)
    if satisfaction_sufficient and non_pec_eur is None:
        non_pec_eur = 0.0

    deadline_m = DEADLINE_REGEX.search(operative_text)
    deadline_days = 90 if deadline_m else None

    interest_m = INTEREST_REGEX.search(operative_text)
    interest_formula = interest_m.group(1).strip() if interest_m else None

    voting_m = VOTING_REGEX.search(conclusion_header or operative_text)
    if voting_m:
        if "unanimously" in voting_m.group(1).lower():
            voting_pattern = "unanimously"
        else:
            voting_pattern = f"{voting_m.group(2)} votes to {voting_m.group(3)}"
    else:
        voting_pattern = "unknown"

    award_per_applicant: list[dict[str, Any]] = []
    source_non_pec = "regex" if non_pec_eur is not None else None
    source_pec = "regex" if pec_eur is not None else None
    source_costs = "regex" if costs_eur is not None else None
    bundled_award_eur = None
    bundled_source = None
    notes: list[str] = []

    following_block = _extract_following_amounts_block(operative_text)
    if following_block:
        following_totals, following_entries = _parse_following_amounts_block(following_block)
        award_per_applicant.extend(following_entries)
        if non_pec_eur is None and following_totals["non_pecuniary_eur"] is not None:
            non_pec_eur = following_totals["non_pecuniary_eur"]
            source_non_pec = "following_amounts"
        if pec_eur is None and following_totals["pecuniary_eur"] is not None:
            pec_eur = following_totals["pecuniary_eur"]
            source_pec = "following_amounts"
        if costs_eur is None and following_totals["costs_eur"] is not None:
            costs_eur = following_totals["costs_eur"]
            source_costs = "following_amounts"
        if any(value is not None for value in following_totals.values()):
            notes.append("Recovered award totals from 'following amounts' operative block.")

    table_text = _clean_text(appendix_table_text)
    if table_text:
        if APPENDIX_BUNDLED_AWARD_HEADER_RE.search(table_text):
            bundled_award_eur, appendix_entries = _parse_bundled_appendix(table_text)
            if bundled_award_eur is not None:
                bundled_source = "appendix_table_bundled"
                award_per_applicant.extend(appendix_entries)
                notes.append("Recovered bundled award total from appended table.")
        elif APPENDIX_PEC_AWARD_HEADER_RE.search(table_text) and APPENDIX_NON_PEC_AWARD_HEADER_RE.search(table_text):
            has_costs_column = bool(APPENDIX_COSTS_AWARD_HEADER_RE.search(table_text))
            appendix_totals, appendix_entries = _parse_separate_appendix(table_text, has_costs_column)
            award_per_applicant.extend(appendix_entries)
            if non_pec_eur is None and appendix_totals["non_pecuniary_eur"] is not None:
                non_pec_eur = appendix_totals["non_pecuniary_eur"]
                source_non_pec = "appendix_table"
            if pec_eur is None and appendix_totals["pecuniary_eur"] is not None:
                pec_eur = appendix_totals["pecuniary_eur"]
                source_pec = "appendix_table"
            if costs_eur is None and appendix_totals["costs_eur"] is not None:
                costs_eur = appendix_totals["costs_eur"]
                source_costs = "appendix_table"
            if any(value is not None for value in appendix_totals.values()):
                notes.append("Recovered head-specific award totals from appended table.")

    article41_totals, article41_entries, article41_notes = _extract_article41_awards(article_41_text, num_applicants)
    if non_pec_eur is None and article41_totals["non_pecuniary_eur"] is not None:
        non_pec_eur = article41_totals["non_pecuniary_eur"]
        source_non_pec = "article_41_text"
    if pec_eur is None and article41_totals["pecuniary_eur"] is not None:
        pec_eur = article41_totals["pecuniary_eur"]
        source_pec = "article_41_text"
    if costs_eur is None and article41_totals["costs_eur"] is not None:
        costs_eur = article41_totals["costs_eur"]
        source_costs = "article_41_text"
    if article41_entries:
        award_per_applicant.extend(article41_entries)
    notes.extend(article41_notes)

    all_sources = [source_non_pec, source_pec, source_costs, bundled_source]
    has_high = any(s == "regex" for s in all_sources if s is not None)
    has_medium = any(s in ("following_amounts", "appendix_table", "appendix_table_bundled") for s in all_sources if s is not None)
    has_low = bool(award_per_applicant) and not (has_high or has_medium)
    if has_high:
        quality = "high"
    elif has_medium:
        quality = "medium"
    elif has_low:
        quality = "low"
    else:
        quality = "none"
        notes.append("No deterministic signal found for any award head.")

    return {
        "non_pecuniary_eur": non_pec_eur,
        "pecuniary_eur": pec_eur,
        "costs_eur": costs_eur,
        "bundled_award_eur": bundled_award_eur,
        "legal_aid_deduction_eur": deduction_eur,
        "net_costs_eur": (costs_eur - deduction_eur) if (costs_eur is not None and deduction_eur is not None) else costs_eur,
        "satisfaction_sufficient": satisfaction_sufficient,
        "payment_deadline_days": deadline_days,
        "default_interest_formula": interest_formula,
        "voting_pattern": voting_pattern,
        "award_per_applicant": award_per_applicant,
        "deterministic_source_non_pecuniary": source_non_pec,
        "deterministic_source_pecuniary": source_pec,
        "deterministic_source_costs": source_costs,
        "deterministic_source_bundled": bundled_source,
        "deterministic_notes": notes,
        "deterministic_match_quality": quality,
    }

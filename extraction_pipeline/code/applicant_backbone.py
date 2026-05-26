from __future__ import annotations

import re
from typing import Any


APPNO_LINE_RE = re.compile(r"(?m)^\s*\|?\s*(\d{1,6}/\d{2})\s*$")
NAME_YEAR_RE = re.compile(r"(?m)^\s*([^\n|]{3,}?)\s*\n\s*((?:19|20)\d{2})\s*$")
TITLE_NAME_RE = re.compile(r"\b(Mr|Ms|Mrs|Miss)\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`’.-]+)*)")
SINGLE_NATIONAL_RE = re.compile(
    r"by\s+an?\s+([A-Za-z-]+)\s+national,\s+(Mr|Ms|Mrs|Miss)\s+([A-Z][^,(]+)",
    re.IGNORECASE,
)
PLURAL_NATIONAL_RE = re.compile(
    r"by\s+(?:\w+\s+)?([A-Za-z-]+)\s+nationals?\b",
    re.IGNORECASE,
)
EXPLICIT_APPLICANT_COUNT_RE = re.compile(r"\((\d+)\s+applicants?\)", re.IGNORECASE)
ORDINAL_INDEX_RE = re.compile(
    r"\b(?:(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+applicant|applicant\s+(\d+))\b",
    re.IGNORECASE,
)
ORDINAL_MAP = {
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
}


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_name(text: str) -> str:
    text = re.sub(r"\b(Mr|Ms|Mrs|Miss)\b\.?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]+", " ", text)
    return " ".join(text.lower().split())


def title_case_name(text: str) -> str:
    tokens: list[str] = []
    for part in text.split():
        if part.isupper() and len(part) > 1:
            tokens.append(part.title())
        else:
            tokens.append(part)
    return " ".join(tokens)


def judgment_year(value: Any) -> int | None:
    text = clean_text(value)
    match = re.search(r"((?:19|20)\d{2})", text)
    return int(match.group(1)) if match else None


def age_group_from_birth_year(birth_year: int | None, judgment_year_value: int | None) -> str:
    if birth_year is None or judgment_year_value is None:
        return "unknown"
    age = judgment_year_value - birth_year
    if age < 0:
        return "unknown"
    if age < 13:
        return "child"
    if age < 18:
        return "adolescent"
    if age >= 65:
        return "elderly"
    return "adult"


def extract_name_year_pairs(text: str) -> list[tuple[str, int]]:
    columns = [col.strip() for col in text.split("|") if col.strip()]
    candidate_cells: list[str] = []
    for col in columns:
        lowered = col.lower()
        if "applicant’s name" in lowered or "applicant's name" in lowered:
            continue
        if "application no." in lowered or "date of introduction" in lowered:
            continue
        if "amount awarded" in lowered or "substance of the complaint" in lowered or "factual information" in lowered:
            continue
        if not re.search(r"\b(?:19|20)\d{2}\b", col):
            continue
        if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", col):
            continue
        candidate_cells.append(col)

    pairs: list[tuple[str, int]] = []
    for cell in candidate_cells or [text]:
        for raw_name, year in NAME_YEAR_RE.findall(cell):
            name = title_case_name(clean_text(raw_name))
            if not name:
                continue
            try:
                birth_year = int(year)
            except ValueError:
                continue
            pairs.append((name, birth_year))

    deduped: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return deduped


def extract_title_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for title, raw_name in TITLE_NAME_RE.findall(text):
        sex = "male" if title.lower() == "mr" else "female"
        full = normalize_name(raw_name)
        if full:
            mapping[full] = sex
            mapping.setdefault(full.split()[-1], sex)
    return mapping


def docname_names(docname: str) -> list[str]:
    match = re.search(r"CASE OF (.+?) v\.", docname or "", re.IGNORECASE)
    if not match:
        return []
    applicant_side = match.group(1).strip()
    if " AND OTHERS" in applicant_side.upper():
        return []
    parts = [part.strip() for part in re.split(r"\s+AND\s+", applicant_side, flags=re.IGNORECASE) if part.strip()]
    return [title_case_name(part) for part in parts]


def extract_uniform_nationality(procedure_text: str) -> str | None:
    single = SINGLE_NATIONAL_RE.search(procedure_text)
    if single:
        return single.group(1).title()
    plural = PLURAL_NATIONAL_RE.search(procedure_text)
    if plural:
        return plural.group(1).title()
    return None


def extract_num_applicants(
    num_applicants_proxy: Any,
    source_row: dict[str, Any],
    appendix_text: str,
    procedure_text: str,
) -> tuple[int, list[str]]:
    notes: list[str] = []
    match = EXPLICIT_APPLICANT_COUNT_RE.search(appendix_text) or EXPLICIT_APPLICANT_COUNT_RE.search(procedure_text)
    if match:
        count = int(match.group(1))
        notes.append("num_applicants from explicit '(N applicants)' marker")
        return count, notes

    applicant_pairs = extract_name_year_pairs(appendix_text)
    if applicant_pairs:
        notes.append("num_applicants from parsed applicant rows")
        return len(applicant_pairs), notes

    source_n = source_row.get("n_applicants")
    if isinstance(source_n, int) and source_n > 0:
        notes.append("num_applicants from source metadata n_applicants")
        return source_n, notes
    if isinstance(source_n, str) and source_n.isdigit():
        notes.append("num_applicants from source metadata n_applicants")
        return int(source_n), notes

    names = docname_names(str(source_row.get("docname") or ""))
    if names:
        notes.append("num_applicants from docname applicant side")
        return len(names), notes

    if isinstance(num_applicants_proxy, int) and num_applicants_proxy > 0:
        notes.append("num_applicants from existing proxy")
        return num_applicants_proxy, notes
    if isinstance(num_applicants_proxy, str) and num_applicants_proxy.isdigit():
        notes.append("num_applicants from existing proxy")
        return int(num_applicants_proxy), notes

    notes.append("num_applicants defaulted to 1")
    return 1, notes


def build_applicants(
    num_applicants: int,
    source_row: dict[str, Any],
    appendix_text: str,
    procedure_text: str,
    judgment_year_value: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    title_map = extract_title_map("\n".join([appendix_text, procedure_text]))
    applicant_pairs = extract_name_year_pairs(appendix_text)
    applicants: list[dict[str, Any]] = []
    if applicant_pairs:
        notes.append("applicant identities from appendix table rows")
        for idx, (name, birth_year) in enumerate(applicant_pairs, start=1):
            normalized = normalize_name(name)
            parts = normalized.split()
            sex = title_map.get(normalized) or (title_map.get(parts[-1]) if parts else None) or "unknown"
            applicants.append(
                {
                    "applicant_index": idx,
                    "beneficiary_label": name,
                    "birth_year": birth_year,
                    "sex": sex,
                    "age_group": age_group_from_birth_year(birth_year, judgment_year_value),
                    "nationality": None,
                }
            )
    elif SINGLE_NATIONAL_RE.search(procedure_text):
        match = SINGLE_NATIONAL_RE.search(procedure_text)
        assert match is not None
        nationality = match.group(1).title()
        sex = "male" if match.group(2).lower() == "mr" else "female"
        name = title_case_name(clean_text(match.group(3)))
        birth_match = re.search(r"born(?:\s+in)?\s+((?:19|20)\d{2})", procedure_text, re.IGNORECASE)
        birth_year = int(birth_match.group(1)) if birth_match else None
        notes.append("single applicant identity from formulaic procedure opening")
        applicants.append(
            {
                "applicant_index": 1,
                "beneficiary_label": name,
                "birth_year": birth_year,
                "sex": sex,
                "age_group": age_group_from_birth_year(birth_year, judgment_year_value),
                "nationality": nationality,
            }
        )
    else:
        names = docname_names(str(source_row.get("docname") or ""))
        if names:
            notes.append("applicant labels from docname")
            for idx, name in enumerate(names, start=1):
                applicants.append(
                    {
                        "applicant_index": idx,
                        "beneficiary_label": name,
                        "birth_year": None,
                        "sex": title_map.get(normalize_name(name), "unknown"),
                        "age_group": "unknown",
                        "nationality": None,
                    }
                )

    uniform_nationality = extract_uniform_nationality(procedure_text)
    if uniform_nationality:
        for applicant in applicants:
            if applicant["nationality"] is None:
                applicant["nationality"] = uniform_nationality
        notes.append("nationality applied from procedure opening")

    if len(applicants) < num_applicants:
        notes.append("applicant list padded to match num_applicants")
        for idx in range(len(applicants) + 1, num_applicants + 1):
            applicants.append(
                {
                    "applicant_index": idx,
                    "beneficiary_label": None,
                    "birth_year": None,
                    "sex": "unknown",
                    "age_group": "unknown",
                    "nationality": uniform_nationality,
                }
            )
    elif len(applicants) > num_applicants:
        applicants = applicants[:num_applicants]
        notes.append("applicant list truncated to num_applicants")

    return applicants, notes


def parse_ordinal_index(label: str | None) -> int | None:
    if not label:
        return None
    match = ORDINAL_INDEX_RE.search(label)
    if not match:
        return None
    if match.group(1):
        return ORDINAL_MAP.get(match.group(1).lower())
    if match.group(2):
        return int(match.group(2))
    return None

from __future__ import annotations

import re
from typing import Any

# Pipeline B target identifying keywords
IDENTITY_KEYWORDS_RE = re.compile(
    r"\b(born\s+in|national\s+of|citizen\s+of|represented\s+by|living\s+in|residing\s+in"
    r"|applicant\s+company|registered\s+in|the\s+applicants?)\b",
    re.IGNORECASE,
)

# Pipeline E target logic keywords
REASONING_KEYWORDS_RE = re.compile(
    r"\b(quashed|acknowledged|remedy|remedies|held\s+that|dismissed|rejected|upheld|found\s+that"
    r"|violation|fundamental\s+flaw|reopened)\b",
    re.IGNORECASE,
)

# Splitting logic reusing similar rules
CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")

def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def trim_snippet(text: str, limit: int = 300) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    keep = max(60, limit // 2)
    return f"{text[:keep].rstrip()} [TRUNCATED] {text[-keep:].lstrip()}"

def extract_identity_evidence(text: str, source: str = "full_document", max_rows: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    text = clean_text(text)
    for clause in CLAUSE_SPLIT_RE.split(text or ""):
        clause = clause.strip()
        if not clause:
            continue
        if IDENTITY_KEYWORDS_RE.search(clause):
            # only keep non-trivial sentences
            if len(clause.split()) > 4:
                snippet = trim_snippet(clause)
                if snippet not in seen:
                    seen.add(snippet)
                    rows.append({
                        "source": source,
                        "type": "identity_mention",
                        "snippet": snippet
                    })
        if len(rows) >= max_rows:
            break
    return rows

def extract_reasoning_evidence(text: str, source: str = "full_document", max_rows: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    text = clean_text(text)
    for clause in CLAUSE_SPLIT_RE.split(text or ""):
        clause = clause.strip()
        if not clause:
            continue
        if REASONING_KEYWORDS_RE.search(clause):
            # narrow down to domestic context for reasoning logic (the hardest part is domestic remedies)
            if re.search(r"\b(domestic court|supreme court|authorities|state|government|appeal)\b", clause, re.IGNORECASE):
                snippet = trim_snippet(clause)
                if snippet not in seen:
                    seen.add(snippet)
                    rows.append({
                        "source": source,
                        "type": "reasoning_logic",
                        "snippet": snippet
                    })
        if len(rows) >= max_rows:
            break
    return rows

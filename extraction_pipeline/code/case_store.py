#!/usr/bin/env python3
from __future__ import annotations

import json
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any, Iterable

from docx_lossless import format_pipeline_c_appendix_text


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = EXTRACTION_ROOT.parent
UNSTRUCTURED_CASES = DATASET_ROOT / "unstructured" / "cases.json"
CASE_STORE_DIR = DATASET_ROOT / "unstructured" / "cases_by_itemid"


def case_store_path(itemid: str, case_store_dir: Path = CASE_STORE_DIR) -> Path:
    return case_store_dir / f"{itemid}.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_case_row(row: dict[str, Any], *, force_refresh_derived: bool = False) -> dict[str, Any]:
    """Attach lightweight derived fields used repeatedly by extraction."""
    if not isinstance(row, dict):
        return row
    enriched = dict(row)
    if force_refresh_derived or "appendix_table_text" not in enriched:
        enriched["appendix_table_text"] = format_pipeline_c_appendix_text(enriched.get("docx_lossless"))
    return enriched


def load_case_from_store(itemid: str, case_store_dir: Path = CASE_STORE_DIR) -> dict[str, Any] | None:
    path = case_store_path(str(itemid), case_store_dir=case_store_dir)
    if not path.exists():
        return None
    row = load_json(path)
    return enrich_case_row(row) if isinstance(row, dict) else None


def write_case_to_store(
    row: dict[str, Any],
    case_store_dir: Path = CASE_STORE_DIR,
    force: bool = False,
    *,
    force_refresh_derived: bool = False,
) -> Path:
    itemid = str(row.get("itemid") or "").strip()
    if not itemid:
        raise ValueError("case row is missing itemid")
    case_store_dir.mkdir(parents=True, exist_ok=True)
    path = case_store_path(itemid, case_store_dir=case_store_dir)
    if path.exists() and not force:
        return path
    payload = enrich_case_row(row, force_refresh_derived=force_refresh_derived)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def iter_cases_from_cases_json(
    path: Path = UNSTRUCTURED_CASES,
    wanted_itemids: Iterable[str] | None = None,
    chunk_size: int = 1024 * 1024,
) -> Iterable[dict[str, Any]]:
    decoder = JSONDecoder()
    wanted = {str(x) for x in wanted_itemids} if wanted_itemids else None

    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        eof = False

        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            idx = 0
            progressed = False

            while True:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1

                if not started:
                    if idx >= len(buffer):
                        break
                    if buffer[idx] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    started = True
                    progressed = True
                    idx += 1
                    continue

                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx < len(buffer) and buffer[idx] == ",":
                    progressed = True
                    idx += 1
                    continue

                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx < len(buffer) and buffer[idx] == "]":
                    return
                if idx >= len(buffer):
                    break

                try:
                    row, end = decoder.raw_decode(buffer, idx)
                except JSONDecodeError:
                    break

                if not isinstance(row, dict):
                    raise ValueError("cases.json contains a non-object entry")

                progressed = True
                idx = end

                itemid = str(row.get("itemid") or "")
                if wanted is None or itemid in wanted:
                    yield enrich_case_row(row)
                    if wanted is not None:
                        wanted.discard(itemid)
                        if not wanted:
                            return

            buffer = buffer[idx:]

            if eof:
                tail = buffer.strip()
                if tail in ("", "]"):
                    return
                if not progressed:
                    raise ValueError(f"Failed to finish parsing {path}; trailing buffer starts with: {tail[:120]!r}")


def load_cases_by_itemid(
    itemids: Iterable[str],
    case_store_dir: Path = CASE_STORE_DIR,
    fallback_cases_json: Path | None = UNSTRUCTURED_CASES,
    backfill_store: bool = True,
) -> dict[str, dict[str, Any]]:
    wanted = [str(x) for x in itemids]
    rows: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for itemid in wanted:
        row = load_case_from_store(itemid, case_store_dir=case_store_dir)
        if row is None:
            missing.append(itemid)
        else:
            rows[itemid] = row

    if missing and fallback_cases_json is not None and fallback_cases_json.exists():
        for row in iter_cases_from_cases_json(fallback_cases_json, wanted_itemids=missing):
            itemid = str(row.get("itemid") or "")
            rows[itemid] = row
            if backfill_store:
                write_case_to_store(row, case_store_dir=case_store_dir, force=False)

    return rows


def backfill_case_store_appendix(
    case_store_dir: Path = CASE_STORE_DIR,
    *,
    force_refresh_derived: bool = False,
) -> dict[str, Any]:
    updated = 0
    unchanged = 0
    became_nonempty = 0
    became_empty = 0
    changed_nonempty = 0
    report_rows: list[dict[str, Any]] = []
    for path in sorted(case_store_dir.glob("*.json")):
        row = load_json(path)
        if not isinstance(row, dict):
            continue
        enriched = enrich_case_row(row, force_refresh_derived=force_refresh_derived)
        old_text = row.get("appendix_table_text")
        new_text = enriched.get("appendix_table_text")
        old_clean = old_text.strip() if isinstance(old_text, str) else ""
        new_clean = new_text.strip() if isinstance(new_text, str) else ""
        changed = row != enriched
        if not changed:
            unchanged += 1
        case_became_nonempty = False
        case_became_empty = False
        case_changed_nonempty = False
        if not old_clean and new_clean:
            became_nonempty += 1
            case_became_nonempty = True
        elif old_clean and not new_clean:
            became_empty += 1
            case_became_empty = True
        elif old_clean and new_clean and old_clean != new_clean:
            changed_nonempty += 1
            case_changed_nonempty = True
        if changed:
            path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
            updated += 1
        report_rows.append(
            {
                "itemid": str(row.get("itemid") or path.stem),
                "old_len": len(old_clean),
                "new_len": len(new_clean),
                "changed": changed,
                "became_nonempty": case_became_nonempty,
                "became_empty": case_became_empty,
                "changed_nonempty": case_changed_nonempty,
            }
        )
    return {
        "updated": updated,
        "unchanged": unchanged,
        "became_nonempty": became_nonempty,
        "became_empty": became_empty,
        "changed_nonempty": changed_nonempty,
        "rows": report_rows,
    }

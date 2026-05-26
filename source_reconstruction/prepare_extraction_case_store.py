#!/usr/bin/env python3
"""Create a local extraction case store from downloaded HUDOC DOCX files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", required=True, type=Path)
    parser.add_argument("--hudoc-docx-dir", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path, help="Local workspace root to receive unstructured/ cases.")
    parser.add_argument("--echrod-metadata", type=Path, default=None, help="Optional echrod_metadata_subset.csv.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def paragraph_text(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        tag = local_name(node.tag)
        if tag == "t":
            parts.append(node.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def cell_text(cell: ET.Element) -> str:
    paragraphs = [paragraph_text(par) for par in cell.findall("w:p", W_NS)]
    return "\n".join(par for par in paragraphs if par).strip()


def table_rows(table: ET.Element) -> list[list[str]]:
    return [[cell_text(tc) for tc in tr.findall("w:tc", W_NS)] for tr in table.findall("w:tr", W_NS)]


def format_table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows).strip()


def build_docx_lossless_record(docx_path: Path) -> dict:
    with ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    body = root.find("w:body", W_NS)
    if body is None:
        raise ValueError(f"word/document.xml missing body in {docx_path}")

    blocks: list[dict] = []
    table_count = 0
    paragraph_count = 0
    other_block_count = 0
    after_appendix_marker = False
    appendix_table_indices: list[int] = []

    for block_index, child in enumerate(body):
        tag = local_name(child.tag)
        if tag == "p":
            text = paragraph_text(child)
            if text:
                upper = text.strip().upper()
                if upper == "APPENDIX" or upper.startswith("APPENDIX "):
                    after_appendix_marker = True
            blocks.append({"block_index": block_index, "type": "paragraph", "text": text})
            paragraph_count += 1
        elif tag == "tbl":
            rows = table_rows(child)
            payload = {
                "block_index": block_index,
                "type": "table",
                "table_index": table_count,
                "after_appendix_marker": after_appendix_marker,
                "rows": rows,
                "text": format_table(rows),
            }
            blocks.append(payload)
            if after_appendix_marker:
                appendix_table_indices.append(table_count)
            table_count += 1
        else:
            blocks.append({"block_index": block_index, "type": "other", "tag": tag})
            other_block_count += 1

    return {
        "source_docx": str(docx_path),
        "block_count": len(blocks),
        "paragraph_count": paragraph_count,
        "table_count": table_count,
        "other_block_count": other_block_count,
        "appendix_table_indices": appendix_table_indices,
        "blocks": blocks,
    }


def heading_to_section_name(text: str) -> str:
    upper = text.strip().upper()
    if "FOR THESE REASONS" in upper:
        return "conclusion"
    if "ARTICLE 41" in upper or "ARTICLE 50" in upper or "JUST SATISFACTION" in upper:
        return "law"
    if "RELEVANT" in upper and "LAW" in upper:
        return "relevant_law"
    if "THE FACTS" in upper or "CIRCUMSTANCES" in upper:
        return "facts"
    if upper in {"PROCEDURE", "PROCEEDINGS", "PROCEEDINGS BEFORE THE COURT"}:
        return "procedure"
    if upper == "THE LAW" or "ALLEGED VIOLATION" in upper:
        return "law"
    if "APPENDIX" in upper:
        return "appendix"
    return "section"


def looks_like_heading(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned or len(cleaned) > 140:
        return False
    letters = [char for char in cleaned if char.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.65


def sections_from_docx_lossless(docx_lossless: dict) -> list[dict]:
    sections: list[dict] = []
    current = {"section_name": "introduction", "content": "INTRODUCTION", "elements": []}
    for block in docx_lossless.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        if block.get("type") == "paragraph" and looks_like_heading(text):
            if current["elements"]:
                sections.append(current)
            current = {"section_name": heading_to_section_name(text), "content": text, "elements": []}
        else:
            current["elements"].append({"section_name": "paragraph", "content": text, "elements": []})
    if current["elements"] or not sections:
        sections.append(current)
    return sections


def read_csv_by_itemid(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("itemid") or "").strip(): row for row in reader if str(row.get("itemid") or "").strip()}


def read_case_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "itemid" not in reader.fieldnames:
            raise ValueError(f"{path} must contain itemid")
        return [row for row in reader if str(row.get("itemid") or "").strip()]


def main() -> int:
    args = parse_args()
    rows = read_case_index(args.case_index)
    if args.limit is not None:
        rows = rows[: args.limit]
    echrod = read_csv_by_itemid(args.echrod_metadata)

    unstructured_root = args.out_root / "unstructured"
    case_store = unstructured_root / "cases_by_itemid"
    case_store.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    missing: list[str] = []
    for row in rows:
        itemid = str(row.get("itemid") or "").strip()
        docx_path = args.hudoc_docx_dir / f"{itemid}.docx"
        if not docx_path.exists():
            missing.append(itemid)
            continue

        out_path = case_store / f"{itemid}.json"
        if out_path.exists() and not args.overwrite:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            docx_lossless = build_docx_lossless_record(docx_path)
            meta = echrod.get(itemid, {})
            payload = {
                **meta,
                "itemid": itemid,
                "hudoc_url": row.get("hudoc_url"),
                "judgementdate": row.get("judgementdate") or meta.get("judgementdate"),
                "respondent": row.get("respondent_state") or meta.get("country"),
                "content": {"document": sections_from_docx_lossless(docx_lossless)},
                "docx_lossless": docx_lossless,
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        cases.append(payload)

    cases_json = unstructured_root / "cases.json"
    cases_json.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "case_index_rows": len(rows),
        "written_cases": len(cases),
        "missing_docx_count": len(missing),
        "missing_docx_itemids_preview": missing[:100],
        "contains_raw_text": True,
        "outputs": [str(cases_json), str(case_store)],
    }
    (unstructured_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

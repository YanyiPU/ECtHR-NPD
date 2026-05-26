#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


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
    non_empty = [par for par in paragraphs if par]
    if non_empty:
        return "\n".join(non_empty).strip()
    return ""


def table_rows(table: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.findall("w:tr", W_NS):
        rows.append([cell_text(tc) for tc in tr.findall("w:tc", W_NS)])
    return rows


def format_table_rows(rows: list[list[str]]) -> str:
    formatted_rows = [" | ".join(cell.strip() for cell in row) for row in rows]
    return "\n".join(formatted_rows).strip()


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
            blocks.append(
                {
                    "block_index": block_index,
                    "type": "paragraph",
                    "text": text,
                }
            )
            paragraph_count += 1
            continue

        if tag == "tbl":
            rows = table_rows(child)
            text = format_table_rows(rows)
            table_payload = {
                "block_index": block_index,
                "type": "table",
                "table_index": table_count,
                "after_appendix_marker": after_appendix_marker,
                "rows": rows,
                "text": text,
            }
            blocks.append(table_payload)
            if after_appendix_marker:
                appendix_table_indices.append(table_count)
            table_count += 1
            continue

        blocks.append(
            {
                "block_index": block_index,
                "type": "other",
                "tag": tag,
            }
        )
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


def format_pipeline_c_appendix_text(docx_lossless: dict | None) -> str:
    if not isinstance(docx_lossless, dict):
        return ""
    blocks = docx_lossless.get("blocks")
    if not isinstance(blocks, list):
        return ""

    appendix_tables = [
        block for block in blocks if isinstance(block, dict) and block.get("type") == "table" and block.get("after_appendix_marker")
    ]
    candidate_tables = appendix_tables or [
        block for block in blocks if isinstance(block, dict) and block.get("type") == "table"
    ]
    if not candidate_tables:
        return ""

    parts: list[str] = []
    for block in candidate_tables:
        parts.append(f"APPENDED TABLE {block.get('table_index', 0) + 1}")
        text = str(block.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()

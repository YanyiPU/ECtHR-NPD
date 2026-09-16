#!/usr/bin/env python3
"""Create a local extraction case store from downloaded HUDOC DOCX files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from source_utils import (decode_structure, file_hash, ingestion_lock, index_records,
                          merge_case_records, normalize_metadata, read_index, value_hash, write_json)


W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", required=True, type=Path)
    parser.add_argument("--hudoc-docx-dir", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path, help="Local workspace root to receive unstructured/ cases.")
    parser.add_argument("--echrod-metadata", type=Path, default=None, help="Optional echrod_metadata_subset.csv.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-precedence", choices=["error", "index", "echrod"], default="error",
                        help="Resolve conflicting nonempty metadata explicitly; default rejects conflicts.")
    parser.add_argument("--resume-interrupted", action="store_true",
                        help="Roll forward a pending journal only with the exact same source batch and verified file hashes.")
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
        document = zf.getinfo("word/document.xml")
        if document.file_size > 128 * 1024 * 1024:
            raise ValueError("DOCX XML exceeds 128 MiB safety limit")
        root = ET.fromstring(zf.read(document))
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


def read_csv_by_itemid(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    return index_records(read_index(path), str(path))


def read_case_index(path: Path) -> list[dict[str, str]]:
    return read_index(path)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    with ingestion_lock(args.out_root):
        return ingest(args)


def combine_metadata(row: dict, meta: dict, precedence: str = "error") -> tuple[dict, list[str]]:
    left, right = normalize_metadata(meta), normalize_metadata(row)
    conflicts = [key for key in set(meta) & set(row)
                 if key != "metadata_normalization_issues"
                 and decode_structure(meta[key]) is not None and decode_structure(row[key]) is not None
                 and left.get(key) != right.get(key)]
    if conflicts and precedence == "error":
        raise ValueError(f"{row['itemid']}: conflicting metadata fields {sorted(conflicts)}; choose --metadata-precedence explicitly")
    first, second = (row, meta) if precedence == "echrod" else (meta, row)
    combined = dict(first)
    combined.update({key: value for key, value in second.items() if decode_structure(value) is not None})
    return normalize_metadata(combined), sorted(conflicts)


def ingest(args: argparse.Namespace) -> int:
    rows = read_case_index(args.case_index)
    if args.limit is not None:
        rows = rows[: args.limit]
    echrod = read_csv_by_itemid(args.echrod_metadata)

    unstructured_root = args.out_root / "unstructured"
    case_store = unstructured_root / "cases_by_itemid"
    cases: list[dict] = []
    errors: list[dict] = []
    for row in rows:
        itemid = str(row.get("itemid") or "").strip()
        docx_path = args.hudoc_docx_dir / f"{itemid}.docx"
        if not docx_path.exists():
            errors.append({"itemid": itemid, "error": "missing_docx"})
            continue
        try:
            source_hash = file_hash(docx_path)
            docx_lossless = build_docx_lossless_record(docx_path)
            if file_hash(docx_path) != source_hash:
                raise ValueError("DOCX source changed while being parsed; retry from a stable source file")
            meta, metadata_conflicts = combine_metadata(row, echrod.get(itemid, {}), args.metadata_precedence)
            payload = {
                **meta,
                "itemid": itemid,
                "respondent": meta.get("respondent") or meta.get("respondent_state") or meta.get("country"),
                "content": {"document": sections_from_docx_lossless(docx_lossless)},
                "docx_lossless": docx_lossless,
                "source_metadata": {"case_index": row, "echrod": echrod.get(itemid)},
                "source_provenance": {
                    "docx_sha256": source_hash,
                    "case_index_record_sha256": value_hash(row),
                    "echrod_record_sha256": value_hash(echrod[itemid]) if itemid in echrod else None,
                    "metadata_precedence": args.metadata_precedence,
                    "metadata_conflicts": metadata_conflicts,
                },
            }
            cases.append(payload)
        except Exception as exc:
            errors.append({"itemid": itemid, "error": f"{type(exc).__name__}: {exc}"})
    if not rows:
        errors.append({"error": "empty_input_batch"})
    if errors:
        print(json.dumps({"status": "rejected", "errors": errors, "dataset_changed": False}, indent=2))
        return 1
    cases_json = unstructured_root / "cases.json"
    pending = unstructured_root / "ingestion_pending.json"
    journal = None
    if pending.exists():
        if not args.resume_interrupted:
            raise RuntimeError(f"Unresolved interrupted ingestion journal: {pending}; inspect and use --resume-interrupted with identical inputs")
        journal = json.loads(pending.read_text(encoding="utf-8"))
        requested = {row["itemid"]: value_hash(row) for row in cases}
        expected = {change["itemid"]: change["record_sha256"] for change in journal["changes"]}
        if requested != expected:
            raise ValueError("Interrupted-batch input hashes do not match; no data were changed")
        aggregate_hash = file_hash(cases_json) if cases_json.exists() else None
        if aggregate_hash not in {journal["previous_cases_sha256"], journal["expected_cases_sha256"]}:
            raise ValueError("Aggregate changed outside the interrupted batch; manual reconciliation required")
    elif args.resume_interrupted:
        raise ValueError("No pending journal exists; omit --resume-interrupted")
    existing = json.loads(cases_json.read_text(encoding="utf-8")) if cases_json.exists() else []
    if not isinstance(existing, list):
        raise ValueError("Existing cases.json must be a JSON array; no files were changed")
    # Recover per-item records not yet indexed in the aggregate, never silently drop them.
    # appendix_table_text is an explicitly derived cache added by case_store.py;
    # it is not a second source. All other differences remain conflicts.
    def source_record(record: dict) -> dict:
        return {key: value for key, value in record.items() if key != "appendix_table_text"}

    existing_by_id = {key: source_record(row) for key, row in index_records(existing, str(cases_json)).items()}
    permitted_recovery = {change["itemid"]: {change["previous_record_sha256"], change["record_sha256"]}
                          for change in journal["changes"]} if journal else {}
    for path in sorted(case_store.glob("*.json")):
        record = source_record(json.loads(path.read_text(encoding="utf-8")))
        indexed = index_records([record], str(path))
        itemid = next(iter(indexed))
        if path.stem != itemid:
            raise ValueError(f"Case-store filename/record mismatch: {path}")
        if itemid in existing_by_id and value_hash(existing_by_id[itemid]) != value_hash(record):
            permitted = permitted_recovery.get(itemid, set())
            if not (value_hash(record) in permitted and value_hash(existing_by_id[itemid]) in permitted):
                raise ValueError(f"Aggregate/per-item conflict for {itemid}; reconcile explicitly before ingestion")
        elif itemid in permitted_recovery and value_hash(record) not in permitted_recovery[itemid]:
            raise ValueError(f"Per-item source changed outside interrupted batch: {itemid}")
        existing_by_id[itemid] = record
    if journal and (file_hash(cases_json) if cases_json.exists() else None) == journal["previous_cases_sha256"]:
        # Newly written per-item files must be reinserted in original input order,
        # not the filename sort order encountered during recovery.
        for change in journal["changes"]:
            if change["previous_record_sha256"] is None:
                existing_by_id.pop(change["itemid"], None)
    merged, changes = merge_case_records(list(existing_by_id.values()), cases, overwrite=args.overwrite or bool(journal))
    if journal and value_hash(merged) != journal["expected_cases_sha256"]:
        raise ValueError("Recovered aggregate differs from the approved interrupted transaction")
    before_hash = file_hash(cases_json) if cases_json.exists() else None
    # Preflight completes before any data writes. Individual replacements are atomic.
    # The pending journal is retained if interrupted between files. Recovery is
    # explicit, hash-checked roll-forward; this is not a multi-file atomic swap.
    manifest = {"schema_version": 1, "visibility": "internal_only",
                "case_index_sha256": file_hash(args.case_index),
                "echrod_metadata_sha256": file_hash(args.echrod_metadata) if args.echrod_metadata else None,
                "previous_cases_sha256": before_hash, "changes": changes,
                "expected_cases_sha256": value_hash(merged),
                "total_cases": len(merged), "contains_raw_text": True,
                "contains_personal_information": True}
    manifest["batch_id"] = value_hash({"case_index_sha256": manifest["case_index_sha256"],
                                      "records": [change["record_sha256"] for change in changes]})
    if journal:
        manifest = journal
    else:
        write_json(pending, manifest)
    for row in cases:
        out_path = case_store / f"{row['itemid']}.json"
        if not out_path.exists() or value_hash(json.loads(out_path.read_text(encoding="utf-8"))) != value_hash(row):
            write_json(out_path, row)
    if not cases_json.exists() or value_hash(existing) != value_hash(merged):
        write_json(cases_json, merged)
    manifest["cases_sha256"] = file_hash(cases_json)
    write_json(unstructured_root / "ingestion_manifest.json", manifest)
    summary = {
        "case_index_rows": len(rows),
        "total_cases": len(merged),
        "added_cases": sum(change["action"] == "added" for change in changes),
        "updated_cases": sum(change["action"] == "updated" for change in changes),
        "unchanged_cases": sum(change["action"] == "unchanged" for change in changes),
        "contains_raw_text": True,
        "contains_personal_information": True,
        "outputs": [str(cases_json), str(case_store)],
    }
    write_json(unstructured_root / "summary.json", summary)
    pending.unlink()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

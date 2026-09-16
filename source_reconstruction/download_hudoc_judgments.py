#!/usr/bin/env python3
"""Download public HUDOC judgments by released itemid.

The script writes source documents only to a user-supplied local directory.
It does not require credentials and does not modify the release bundle.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from source_utils import atomic_write, file_hash, read_index, write_csv, write_json


DOCX_TEMPLATE = "https://hudoc.echr.coe.int/app/conversion/docx/?library=ECHR&id={itemid}&filename={filename}&logEvent=False"
HTML_TEMPLATE = "https://hudoc.echr.coe.int/app/conversion/html/?library=ECHR&id={itemid}&filename={filename}&logEvent=False"
USER_AGENT = "ECtHR-NPD-reconstruction/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", required=True, type=Path, help="CSV with at least an itemid column.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Local output directory for downloaded documents.")
    parser.add_argument("--format", choices=["docx", "html"], default="docx")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of cases to attempt.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between requests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first attempt for transient HTTP/network failures.")
    return parser.parse_args()


def read_case_rows(path: Path) -> list[dict[str, str]]:
    return read_index(path)


def hudoc_conversion_url(itemid: str, fmt: str) -> str:
    filename = urllib.parse.quote(f"{itemid}.{fmt}")
    template = DOCX_TEMPLATE if fmt == "docx" else HTML_TEMPLATE
    return template.format(itemid=urllib.parse.quote(itemid), filename=filename)


def validate_document(payload: bytes, fmt: str) -> None:
    if not payload:
        raise ValueError("Empty response document")
    if fmt == "docx":
        with ZipFile(io.BytesIO(payload)) as archive:
            document = archive.getinfo("word/document.xml")
            if document.file_size > 128 * 1024 * 1024:
                raise ValueError("DOCX XML exceeds 128 MiB safety limit")
            root = ET.fromstring(archive.read(document))
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            if root.tag != namespace + "document" or root.find(namespace + "body") is None:
                raise ValueError("DOCX lacks a Word document/body")
            if not any((node.text or "").strip() for node in root.iter(namespace + "t")):
                raise ValueError("DOCX has no readable text")
    else:
        text = payload.decode("utf-8", errors="replace").lower()
        if "<html" not in text or "<body" not in text or not any(marker in text for marker in ("judgment", "judgement", "arrêt", "decision", "décision")):
            raise ValueError("Response is not recognizable judgment HTML; inspect manually")


def download(url: str, dst: Path, retries: int = 2) -> tuple[str, int]:
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read(128 * 1024 * 1024 + 1)
            if len(payload) > 128 * 1024 * 1024:
                raise ValueError("Response exceeds 128 MiB safety limit")
            validate_document(payload, dst.suffix.lstrip("."))
            atomic_write(dst, payload)
            return "ok", len(payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    if args.retries < 0 or args.retries > 10 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--retries must be 0..10 and --limit must be positive")
    rows = read_case_rows(args.case_index)
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise ValueError("No input cases; no downloads attempted")
    status_path = args.out_dir / "download_status.csv"
    status_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        itemid = str(row["itemid"]).strip()
        dst = args.out_dir / f"{itemid}.{args.format}"
        url = hudoc_conversion_url(itemid, args.format)
        record = {"itemid": itemid, "url": url, "path": str(dst), "status": "", "bytes": "0", "sha256": "", "error": ""}

        if args.dry_run:
            record.update({"status": "dry_run"})
        else:
            try:
                if dst.exists() and not args.overwrite:
                    validate_document(dst.read_bytes(), args.format)
                    record.update({"status": "exists_validated", "bytes": str(dst.stat().st_size), "sha256": file_hash(dst)})
                else:
                    status, byte_count = download(url, dst, retries=args.retries)
                    record.update({"status": status, "bytes": str(byte_count), "sha256": file_hash(dst)})
            except urllib.error.HTTPError as exc:
                record.update({"status": "http_error", "error": f"{exc.code}: {exc.reason}"})
            except Exception as exc:
                record.update({"status": "error", "error": repr(exc)})
            time.sleep(max(args.sleep, 0.0))

        status_rows.append(record)
        if idx % 100 == 0:
            print(f"attempted {idx}/{len(rows)}")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "records": status_rows}, indent=2))
        return 0
    write_csv(status_path, status_rows, ["itemid", "url", "path", "status", "bytes", "sha256", "error"])
    write_json(args.out_dir / "download_manifest.json", {"schema_version": 1,
               "case_index_sha256": file_hash(args.case_index), "records": status_rows})
    ok = sum(1 for row in status_rows if row["status"] in {"ok", "exists_validated"})
    print(f"completed={ok} attempted={len(status_rows)} status={status_path}")
    return 0 if ok == len(status_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

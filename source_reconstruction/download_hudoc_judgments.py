#!/usr/bin/env python3
"""Download public HUDOC judgments by released itemid.

The script writes source documents only to a user-supplied local directory.
It does not require credentials and does not modify the release bundle.
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


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
    return parser.parse_args()


def read_case_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "itemid" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an itemid column")
        return [row for row in reader if (row.get("itemid") or "").strip()]


def hudoc_conversion_url(itemid: str, fmt: str) -> str:
    filename = urllib.parse.quote(f"{itemid}.{fmt}")
    template = DOCX_TEMPLATE if fmt == "docx" else HTML_TEMPLATE
    return template.format(itemid=urllib.parse.quote(itemid), filename=filename)


def download(url: str, dst: Path) -> tuple[str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    dst.write_bytes(payload)
    return "ok", len(payload)


def main() -> int:
    args = parse_args()
    rows = read_case_rows(args.case_index)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / "download_status.csv"
    status_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        itemid = str(row["itemid"]).strip()
        dst = args.out_dir / f"{itemid}.{args.format}"
        url = hudoc_conversion_url(itemid, args.format)
        record = {"itemid": itemid, "url": url, "path": str(dst), "status": "", "bytes": "0", "error": ""}

        if dst.exists() and not args.overwrite:
            record.update({"status": "exists", "bytes": str(dst.stat().st_size)})
        elif args.dry_run:
            record.update({"status": "dry_run"})
        else:
            try:
                status, byte_count = download(url, dst)
                record.update({"status": status, "bytes": str(byte_count)})
            except urllib.error.HTTPError as exc:
                record.update({"status": "http_error", "error": f"{exc.code}: {exc.reason}"})
            except Exception as exc:
                record.update({"status": "error", "error": repr(exc)})
            time.sleep(max(args.sleep, 0.0))

        status_rows.append(record)
        if idx % 100 == 0:
            print(f"attempted {idx}/{len(rows)}")

    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["itemid", "url", "path", "status", "bytes", "error"])
        writer.writeheader()
        writer.writerows(status_rows)

    ok = sum(1 for row in status_rows if row["status"] in {"ok", "exists", "dry_run"})
    print(f"completed={ok} attempted={len(status_rows)} status={status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Discover a explicitly scoped INTERNAL HUDOC judgment metadata index.

Only the official metadata-search endpoint is contacted. No judgment text or
model requests are made. Completion means two consistent traversals of the
specified search, not all HUDOC, a transactional snapshot, or dataset acceptance.
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from source_utils import ingestion_lock, validate_itemid, value_hash, write_csv, write_json

ENDPOINT = "https://hudoc.echr.coe.int/app/query/results"
COLLECTIONS = ("GRANDCHAMBER", "CHAMBER", "COMMITTEE")
# ENG/HEJUD is deliberately the only live-verified language/type combination.
SELECT = ("itemid", "appno", "extractedappno", "docname", "judgementdate", "kpdate", "doctype", "doctypebranch",
          "documentcollectionid", "documentcollectionid2", "languageisocode", "article", "violation", "nonviolation",
          "conclusion", "representedby", "respondent", "importance", "originatingbody", "separateopinion", "scl",
          "introductiondate", "decisiondate", "ecli", "typedescription", "applicability", "rulesofcourt",
          "externalsources", "kpthesaurus", "publishedby", "isplaceholder")
OUTPUT_FIELDS = [*SELECT, "hudoc_judgementdate_original", "discovery_query_sha256", "discovery_metadata_sha256"]
USER_AGENT = "ECtHR-NPD-scoped-metadata-discovery/1.0"


def make_config(start: date, until: date, collections: list[str], language: str,
                document_type: str, page_size: int = 100) -> dict:
    if until <= start:
        raise ValueError("date-until must be after date-from")
    if not collections or set(collections) - set(COLLECTIONS) or len(collections) != len(set(collections)):
        raise ValueError("Choose explicit, distinct supported document collections")
    if language != "ENG" or document_type != "judgments":
        raise ValueError("Only the independently verified ENG judgment scope is supported")
    if not 1 <= page_size <= 500:
        raise ValueError("page-size must be 1..500")
    return {"schema_version": 1, "endpoint": ENDPOINT, "date_from_inclusive": start.isoformat(),
            "date_until_exclusive": until.isoformat(), "date_field": "kpdate", "collections": sorted(collections),
            "language": language, "document_type": document_type, "doctype": "HEJUD",
            "sort": "itemid Ascending", "page_size": page_size, "select": list(SELECT)}


def query_text(config: dict) -> str:
    collection = " OR ".join(f'documentcollectionid2="{name}"' for name in config["collections"])
    # Date-only bounds were verified against the live endpoint; alternate
    # timestamp serializations are deliberately not assumed interchangeable.
    return (f'(contentsitename=ECHR) AND (doctype={config["doctype"]}) '
            f'AND (languageisocode="{config["language"]}") AND ({collection}) '
            f'AND (kpdate>="{config["date_from_inclusive"]}" AND kpdate<"{config["date_until_exclusive"]}")')


def request_url(config: dict, start: int, length: int | None = None) -> str:
    return ENDPOINT + "?" + urllib.parse.urlencode({"query": query_text(config), "select": ",".join(config["select"]),
          "sort": config["sort"], "start": start, "length": length or config["page_size"]})


def fetch_json(url: str, *, retries: int = 3, timeout: float = 30, ca_bundle: Path | None = None) -> dict:
    if not 0 <= retries <= 10 or not 0 < timeout <= 60:
        raise ValueError("Retries must be 0..10 and timeout must be >0 and <=60 seconds")
    if not url.startswith(ENDPOINT + "?"):
        raise ValueError("Only the fixed official metadata endpoint is allowed")
    context = ssl.create_default_context(cafile=str(ca_bundle) if ca_bundle else None)
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                payload = response.read(32 * 1024 * 1024 + 1)
                if urllib.parse.urlparse(response.geturl()).hostname != "hudoc.echr.coe.int":
                    raise ValueError("Unexpected metadata redirect host")
            if len(payload) > 32 * 1024 * 1024:
                raise ValueError("Metadata page exceeds 32 MiB response limit")
            return json.loads(payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise RuntimeError("TLS verification failed; configure a trusted --ca-bundle, never disable verification") from exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After", "") if isinstance(exc, urllib.error.HTTPError) else ""
            delay = min(float(retry_after), 30) if retry_after.isdigit() else min(2 ** attempt, 8)
            time.sleep(delay)
    raise AssertionError("unreachable")


def decode_page(payload: dict) -> tuple[int, list[dict]]:
    if not isinstance(payload, dict) or payload.get("message"):
        raise ValueError("Invalid/error HUDOC response envelope")
    total = payload.get("resultcount")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0 or not isinstance(payload.get("results"), list):
        raise ValueError("HUDOC resultcount/results contract changed")
    rows = []
    for result in payload["results"]:
        if not isinstance(result, dict) or not isinstance(result.get("columns"), dict):
            raise ValueError("HUDOC result lacks columns object")
        # Rank is server-injected and was observed to vary between equal queries;
        # it is not source metadata and must not destabilize integrity checks.
        row = {key: value for key, value in result["columns"].items() if key != "rank"}
        row["itemid"] = validate_itemid(row.get("itemid"))
        rows.append(row)
    return total, rows


def source_date(value: object) -> date:
    text = str(value or "").strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(text, "%d/%m/%Y %H:%M:%S").date()
        except ValueError:
            raise ValueError(f"Unknown HUDOC source date format: {text!r}") from None


def validate_scope(row: dict, config: dict) -> None:
    collections = set(str(row.get("documentcollectionid2") or "").split(";"))
    if (row.get("languageisocode") != config["language"] or row.get("doctype") != config["doctype"]
            or "JUDGMENTS" not in collections or not collections.intersection(config["collections"])):
        raise ValueError(f"Out-of-scope language/document collection: {row['itemid']}")
    actual = source_date(row.get("kpdate"))
    if not date.fromisoformat(config["date_from_inclusive"]) <= actual < date.fromisoformat(config["date_until_exclusive"]):
        raise ValueError(f"Out-of-window metadata response: {row['itemid']}")
    if source_date(row.get("judgementdate")) != actual:
        raise ValueError(f"Judgment date and kpdate disagree: {row['itemid']}; review required")


def page_fingerprint(total: int, rows: list[dict]) -> str:
    return value_hash({"resultcount": total, "rows": rows})


def verify_checkpoint(state: dict, config: dict) -> None:
    if state.get("config") != config or state.get("query_sha256") != value_hash(config):
        raise ValueError("Checkpoint scope/config differs; do not resume a different search")
    rows, offset, seen = [], 0, set()
    for page in state["pages"]:
        if page["start"] != offset or page["sha256"] != page_fingerprint(state["expected_total"], page["rows"]):
            raise ValueError("Checkpoint page hash or offset is invalid")
        for row in page["rows"]:
            validate_scope(row, config)
            if row["itemid"] not in seen:
                seen.add(row["itemid"])
                rows.append(row)
        offset += len(page["rows"])
    if offset != state["next_start"] or value_hash(rows) != value_hash(state["records"]):
        raise ValueError("Checkpoint record/offset integrity failed")


def run_discovery(config: dict, out_dir: Path, fetch, *, resume: bool = False, restart: bool = False,
                  max_pages: int = 1000, request_delay: float = 0.5) -> dict:
    if max_pages <= 0 or request_delay < 0 or request_delay > 30 or (resume and restart):
        raise ValueError("Invalid page limit/delay or mutually exclusive resume/restart")
    checkpoint = out_dir / "discovery_checkpoint.json"
    manifest_path = out_dir / "discovery_manifest.json"
    index_path = out_dir / "hudoc_case_index.csv"
    fields = OUTPUT_FIELDS
    if not checkpoint.exists() and out_dir.exists():
        unexpected = [entry.name for entry in out_dir.iterdir() if entry.name != ".ingestion.lock"]
        if unexpected:
            raise ValueError("Output directory is nonempty and has no owned discovery checkpoint; choose a dedicated empty directory")
    if checkpoint.exists() and not (resume or restart):
        raise ValueError("Discovery output already exists; use --resume or explicit --restart")
    if resume:
        if not checkpoint.exists():
            raise ValueError("No discovery checkpoint to resume")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        verify_checkpoint(state, config)
        if state.get("restart_required"):
            raise ValueError("Source drift/integrity issue requires explicit --restart")
    else:
        state = {"config": config, "query_sha256": value_hash(config), "phase": "collect", "next_start": 0,
                 "expected_total": None, "pages": [], "records": [], "duplicate_ids": [], "order_issues": [],
                 "verify_index": 0, "started_at_utc": datetime.now(timezone.utc).isoformat(), "restart_required": False}
    manifest = {"scope": config, "query": query_text(config), "query_sha256": value_hash(config),
                "status": "in_progress", "output_ready": False, "visibility": "internal_only",
                "all_hudoc_coverage": False, "transactional_snapshot": False,
                "acceptance_for_dataset": False, "expected_total": state["expected_total"],
                "unique_records": len(state["records"]), "source_drift_checked_by": "full_second_traversal"}
    write_json(manifest_path, manifest)
    # A restart invalidates only this command's generated index; a header-only
    # file cannot be mistaken for a stale successful harvest by ingestion helpers.
    if state["phase"] != "complete":
        write_csv(index_path, [], fields)
    write_json(checkpoint, state)
    requests = 0
    try:
        if state["phase"] == "complete":
            # Explicit resume revalidates every prior page before returning the
            # existing harvest; it never silently certifies a stale checkpoint.
            state["phase"], state["verify_index"] = "verify", 0
            write_csv(index_path, [], fields)
        elif resume:
            # Recheck all cached pages on every resume, including partial collection.
            for page in state["pages"]:
                if requests >= max_pages:
                    raise RuntimeError("Page budget insufficient to recheck cached prefix; increase --max-pages")
                total, rows = decode_page(fetch(request_url(config, page["start"], page["requested_length"])))
                requests += 1
                if page_fingerprint(total, rows) != page["sha256"]:
                    state["restart_required"] = True
                    raise ValueError("HUDOC changed within the cached pagination prefix; restart required")
                time.sleep(request_delay)
        while state["phase"] == "collect":
            if requests >= max_pages:
                raise RuntimeError("Per-run page budget reached; resume to continue")
            total, rows = decode_page(fetch(request_url(config, state["next_start"])))
            requests += 1
            if state["expected_total"] is None:
                state["expected_total"] = total
            if total != state["expected_total"]:
                state["restart_required"] = True
                raise ValueError("HUDOC resultcount changed during traversal; restart required")
            if len(rows) > config["page_size"] or state["next_start"] + len(rows) > total:
                raise ValueError("HUDOC page exceeds requested/remaining result count")
            if not rows and state["next_start"] < total:
                state["restart_required"] = True
                raise ValueError("HUDOC ended before reported total; scope traversal incomplete")
            known = {row["itemid"]: row for row in state["records"]}
            last = state["records"][-1]["itemid"] if state["records"] else None
            for row in rows:
                validate_scope(row, config)
                itemid = row["itemid"]
                if itemid in known:
                    state["duplicate_ids"].append(itemid)
                    if value_hash(known[itemid]) != value_hash(row):
                        state["restart_required"] = True
                        raise ValueError(f"Conflicting metadata for duplicate ID {itemid}")
                    continue
                if last is not None and itemid <= last:
                    state["order_issues"].append(itemid)
                known[itemid] = row
                state["records"].append(row)
                last = itemid
            state["pages"].append({"start": state["next_start"], "requested_length": config["page_size"],
                                   "rows": rows, "sha256": page_fingerprint(total, rows)})
            state["next_start"] += len(rows)
            if state["next_start"] == total:
                state["phase"] = "verify"
            write_json(checkpoint, state)
            time.sleep(request_delay)
        while state["verify_index"] < len(state["pages"]):
            if requests >= max_pages:
                raise RuntimeError("Per-run page budget reached during verification; resume to continue")
            page = state["pages"][state["verify_index"]]
            total, rows = decode_page(fetch(request_url(config, page["start"], page["requested_length"])))
            requests += 1
            if page_fingerprint(total, rows) != page["sha256"]:
                state["restart_required"] = True
                raise ValueError("HUDOC page changed during verification; restart required")
            state["verify_index"] += 1
            write_json(checkpoint, state)
            time.sleep(request_delay)
        if state["duplicate_ids"] or state["order_issues"] or len(state["records"]) != state["expected_total"]:
            state["restart_required"] = True
            raise ValueError("Deduplication/order/total mismatch prevents a complete scoped index")
        outputs = []
        for row in sorted(state["records"], key=lambda row: row["itemid"]):
            normalized = {key: row.get(key) for key in SELECT}
            normalized["hudoc_judgementdate_original"] = row.get("judgementdate")
            normalized["judgementdate"] = source_date(row["judgementdate"]).isoformat()
            normalized["discovery_query_sha256"] = state["query_sha256"]
            normalized["discovery_metadata_sha256"] = value_hash(row)
            outputs.append(normalized)
        write_csv(index_path, outputs, fields)
        state["phase"] = "complete"
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        state.pop("last_error", None)
        manifest.update({"status": "complete_for_explicit_scope_at_observation", "output_ready": True,
                         "expected_total": state["expected_total"], "unique_records": len(outputs),
                         "duplicate_count": 0, "pages_verified": len(state["pages"]),
                         "records_sha256": value_hash(outputs), "completed_at_utc": state["completed_at_utc"]})
    except Exception as exc:
        if isinstance(exc, (ValueError, KeyError, TypeError)):
            state["restart_required"] = True
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        manifest.update({"status": "incomplete", "error": state["last_error"], "expected_total": state["expected_total"],
                         "unique_records": len(state["records"]), "duplicate_count": len(state["duplicate_ids"]),
                         "restart_required": state["restart_required"], "next_start": state["next_start"]})
    write_json(checkpoint, state)
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-from", type=date.fromisoformat, required=True)
    parser.add_argument("--date-until", type=date.fromisoformat, required=True, help="Exclusive end date")
    parser.add_argument("--collections", nargs="+", choices=COLLECTIONS, required=True)
    parser.add_argument("--language", choices=["ENG"], required=True)
    parser.add_argument("--document-type", choices=["judgments"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1000, help="Per-run collection plus verification request budget")
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--ca-bundle", type=Path, help="Trusted TLS CA bundle if the Python environment lacks system certificates")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--restart", action="store_true", help="Explicitly replace this command's checkpoint/index/manifest and start a fresh search")
    args = parser.parse_args()
    config = make_config(args.date_from, args.date_until, args.collections, args.language, args.document_type, args.page_size)
    fetch = lambda url: fetch_json(url, retries=args.retries, timeout=args.timeout, ca_bundle=args.ca_bundle)
    with ingestion_lock(args.out_dir):
        result = run_discovery(config, args.out_dir, fetch, resume=args.resume, restart=args.restart,
                               max_pages=args.max_pages, request_delay=args.request_delay)
    print(json.dumps(result, indent=2))
    return 0 if result["output_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

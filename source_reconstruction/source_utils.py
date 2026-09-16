"""Offline, standard-library helpers for INTERNAL source ingestion.

Source metadata is not anonymised. These helpers deliberately retain evidence
and distinguish missing/invalid structured metadata from an observed empty list.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def validate_itemid(value: Any) -> str:
    itemid = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", itemid) or itemid in {".", ".."}:
        raise ValueError(f"Missing or unsafe itemid: {itemid!r}")
    return itemid


def index_records(rows: list[dict], source: str = "records") -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{source}: each record must be an object")
        itemid = validate_itemid(row.get("itemid"))
        if itemid in indexed:
            raise ValueError(f"{source}: duplicate itemid {itemid}")
        indexed[itemid] = {**row, "itemid": itemid}
    return indexed


def read_index(path: Path) -> list[dict]:
    csv.field_size_limit(64 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not ({"itemid", "case_id"} & set(fields)):
            raise ValueError(f"{path} must contain unique columns and itemid or HUDOC case_id")
        rows = list(reader)
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: malformed CSV row")
        itemid = (row.get("itemid") or "").strip()
        case_id = (row.get("case_id") or "").strip()
        if not itemid:
            # The sole public case identifier is the HUDOC item identifier.
            # Do not silently accept old surrogate-ID releases as source IDs.
            if not re.fullmatch(r"\d{3}-\d+", case_id):
                raise ValueError(f"{path}: case_id must be a HUDOC identifier (e.g. 001-12345)")
            row["itemid"] = case_id
        elif re.fullmatch(r"\d{3}-\d+", case_id) and case_id != itemid:
            raise ValueError(f"{path}: case_id/itemid mismatch")
    return list(index_records(rows, str(path)).values())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def value_hash(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    """Replace one file atomically; do not truncate a good file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json_bytes(value))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    atomic_write(path, buffer.getvalue().encode("utf-8"))


@contextmanager
def ingestion_lock(root: Path):
    """Single-writer local lock (macOS/Linux). Never break another writer's lock."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".ingestion.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another ingestion writer holds {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def decode_structure(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() in {"null", "none", "nan", "n/a"}:
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except ValueError:
            # Some historical CSV exports use Python list/dict repr, not JSON.
            try:
                parsed = ast.literal_eval(text)
                return parsed if isinstance(parsed, (list, dict)) else value
            except (ValueError, SyntaxError):
                return value
    return value


def normalize_metadata(row: dict) -> dict:
    result = {key: decode_structure(value) for key, value in row.items()}
    issues: list[str] = []
    for key in ("article", "scl", "representedby", "parties"):
        value = result.get(key)
        if value is not None and not isinstance(value, list):
            if isinstance(value, str) and not value.lstrip().startswith(("[", "{")):
                result[key] = [part.strip() for part in value.split(";") if part.strip()]
            else:
                result[key] = None
                issues.append(f"{key}: invalid list metadata")
    # Mentioned articles may be recovered from explicit positive one-hot fields.
    # These are NOT violated articles and must not drive the challenging selector.
    if result.get("article") is None:
        articles = [key.split("=", 1)[1] for key, value in result.items()
                    if key.startswith("article=") and str(value).lower() in {"1", "1.0", "true"}]
        if articles:
            result["article"] = articles
    conclusions = result.get("conclusion")
    if conclusions is not None and (not isinstance(conclusions, list)
                                    or any(not isinstance(entry, dict) or not entry.get("type") for entry in conclusions)):
        result["__conclusion"] = row.get("conclusion")
        result["conclusion"] = None
        issues.append("conclusion: not a structured list of typed findings; manual/source normalization required")
    for key in ("appno", "conclusion", "representedby", "article", "scl"):
        result.setdefault(key, None)
    result["metadata_normalization_issues"] = issues
    return result


def merge_case_records(existing: list[dict], incoming: list[dict], *, overwrite: bool = False) -> tuple[list[dict], list[dict]]:
    """Stable-order, idempotent merge. Changes require explicit overwrite."""
    merged = index_records(existing, "existing case store")
    new = index_records(incoming, "incoming batch")
    changes: list[dict] = []
    for itemid, row in new.items():
        old = merged.get(itemid)
        old_hash = value_hash(old) if old is not None else None
        new_hash = value_hash(row)
        if old is not None and old_hash != new_hash and not overwrite:
            raise ValueError(f"Source conflict for {itemid}; inspect changes and use --overwrite only if intended")
        action = "added" if old is None else ("unchanged" if old_hash == new_hash else "updated")
        merged[itemid] = row
        changes.append({"itemid": itemid, "action": action, "previous_record_sha256": old_hash,
                        "record_sha256": new_hash, "source_provenance": row.get("source_provenance")})
    return list(merged.values()), changes

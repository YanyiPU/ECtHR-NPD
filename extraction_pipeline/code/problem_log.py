"""Threadsafe append-only writer for `problems.jsonl`.

Every (itemid, stage) pair that fails — for any reason — is recorded here so
that a later batch run can resume only the broken parts. Problems include both
system errors (timeouts, HTTP 429, malformed payloads) and quality issues
(schema validation failures, B/C beneficiary count mismatch, sparse D output).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


# Problem categories. Keep this list small and stable so the 2nd-run loader can
# group records by `category` without having to learn new strings.
CATEGORY_API = "api_error"                  # ApiCallError raised by the client
CATEGORY_SCHEMA = "schema_validation"        # parsed JSON did not pass schema
CATEGORY_RECONCILIATION = "quality_reconciliation"  # B/C applicant vs beneficiary mismatch
CATEGORY_D_SPARSE = "quality_d_sparse"       # D legal_analysis returned with no signal
CATEGORY_EXCEPTION = "exception"             # any other unexpected exception


class ProblemLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        itemid: str,
        stage: str,
        category: str,
        detail: Any,
        *,
        http_status: int | None = None,
        kind: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": time.time(),
            "itemid": itemid,
            "stage": stage,
            "category": category,
            "detail": detail,
        }
        if kind is not None:
            record["kind"] = kind
        if http_status is not None:
            record["http_status"] = http_status
        if extras:
            record.update(extras)
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def load_problem_itemids(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group problem records by itemid for 2nd-run loaders.

    Returns `{itemid: [record, ...]}` with each record sorted by ts ascending.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return grouped
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            itemid = str(record.get("itemid") or "")
            if not itemid:
                continue
            grouped.setdefault(itemid, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda r: r.get("ts") or 0)
    return grouped

#!/usr/bin/env python3
"""Verify the audited ECtHR-NPD source-release candidate by SHA-256.

The repository may not yet have a public tag or remote revision.  This
content manifest gives reviewers a deterministic local identity for the code,
configuration, prompts, and the linked dataset manifest while publication is
blocked.  It deliberately does not grant a license or publish anything.

    python3 scripts/verify_source_manifest.py
    python3 scripts/verify_source_manifest.py --mode publish
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "RELEASE_SOURCE_MANIFEST.v1.json"


class SourceManifestError(Exception):
    """The local source candidate cannot be identified safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestError("manifest file paths must be non-empty strings")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise SourceManifestError(f"unsafe manifest path: {value!r}")
    return relative


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or (ROOT / "CURRENT_CODE_MANIFEST.json" if (ROOT / "CURRENT_CODE_MANIFEST.json").exists() else MANIFEST_PATH)
    if path is None and not manifest_path.exists() and (ROOT / "manifest.json").exists():
        manifest_path = ROOT / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceManifestError(f"missing source manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise SourceManifestError(f"invalid source manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SourceManifestError("source manifest must be an object")
    if manifest.get("format_version") == "two-tables-1":
        # A reduced release has its own exact inventory, not the excluded
        # private historical code/data manifest. Access approval is separate.
        return {
            "manifest_version": "public-two-table-source-1",
            "release_candidate_id": "two-table-v" + str(manifest["version"]),
            "status": "LOCAL_VALIDATED_NOT_PUBLICATION_APPROVAL",
            "integrity": {"algorithm": "sha256", "files": manifest["file_sha256"]},
            "identity": {"dataset_version": manifest["version"]},
            "publication_requirements": ["Separate access, rights and compliance approval"],
            "_two_table_release": True,
        }
    required = {"manifest_version", "release_candidate_id", "status", "integrity", "identity", "publication_requirements"}
    missing = sorted(required - set(manifest))
    if missing:
        raise SourceManifestError("source manifest missing keys: " + ", ".join(missing))
    integrity = manifest["integrity"]
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256" or not isinstance(
        integrity.get("files"), dict
    ):
        raise SourceManifestError("source manifest must declare SHA-256 integrity.files")
    return manifest


def verify(manifest: dict[str, Any]) -> int:
    if manifest.get("_two_table_release"):
        spec = importlib.util.spec_from_file_location("release_public_tables", ROOT / "code/public_tables.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate(ROOT)
    files = manifest["integrity"]["files"]
    if not files:
        raise SourceManifestError("source manifest integrity.files cannot be empty")
    for raw_relative, expected_hash in files.items():
        relative = safe_relative_path(raw_relative)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise SourceManifestError(f"invalid SHA-256 for {relative}")
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SourceManifestError(f"source file resolves outside release root: {relative}") from exc
        if not path.is_file():
            raise SourceManifestError(f"missing source-manifest file: {relative}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise SourceManifestError(f"SHA-256 mismatch for {relative}: expected {expected_hash}, got {actual_hash}")
    print("PASS: source-release content manifest")
    print(f"  candidate: {manifest['release_candidate_id']}")
    print(f"  pinned files: {len(files)}")
    print(f"  status: {manifest['status']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "publish"), default="audit")
    parser.add_argument("--manifest", type=Path, help="Explicit source manifest; otherwise current-code manifest when present, legacy frozen manifest otherwise.")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        verify(manifest)
        if args.mode == "publish" and manifest["status"] != "READY_FOR_PUBLICATION":
            raise SourceManifestError("publish gate failed: source manifest is not READY_FOR_PUBLICATION")
        return 0
    except SourceManifestError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

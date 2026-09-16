#!/usr/bin/env python3
"""Create a fail-closed, manifest-allowlisted Hugging Face upload bundle.

The release directory deliberately retains a few superseded audit artifacts
for local review.  This tool never mirrors the directory wholesale: it copies
only files approved by ``RELEASE_MANIFEST.v2.json`` and the Hub schema's
required metadata.  It is intentionally separate from uploading; no network
operation is performed here.

Examples:

    # After the release owner has cleared the manifest blockers.
    python3 scripts/build_hf_upload_bundle.py --dataset-release /approved/public-view \
        --out-dir /safe/path/ecthr-npd

    # Local inspection of the current blocked candidate only.
    python3 scripts/build_hf_upload_bundle.py --dataset-release /legacy/candidate \
        --out-dir /safe/path/ecthr-npd-rc2 \\
        --allow-blocked-candidate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_RELATIVE = Path("metadata") / "RELEASE_MANIFEST.v2.json"


class UploadBundleError(Exception):
    """A candidate cannot be safely assembled for upload."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UploadBundleError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UploadBundleError(f"Invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UploadBundleError(f"{label} must be a JSON object: {path}")
    return value


def safe_relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise UploadBundleError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or candidate == Path("."):
        raise UploadBundleError(f"Unsafe {label}: {value!r}")
    return candidate


def ensure_inside_release(release: Path, relative: Path, *, label: str) -> Path:
    relative = safe_relative_path(str(relative), label=label)
    source = (release / relative).resolve()
    try:
        source.relative_to(release.resolve())
    except ValueError as exc:
        raise UploadBundleError(f"{label} resolves outside dataset_release: {relative}") from exc
    if not source.is_file():
        raise UploadBundleError(f"Missing required source file: dataset_release/{relative}")
    return source


def blocked_ids(manifest: dict[str, Any]) -> list[str]:
    blockers = manifest.get("publication_blockers")
    if not isinstance(blockers, list):
        raise UploadBundleError("Manifest publication_blockers must be a list")
    ids: list[str] = []
    for blocker in blockers:
        if not isinstance(blocker, dict) or not isinstance(blocker.get("id"), str):
            raise UploadBundleError("Each publication blocker must be an object with an id")
        if str(blocker.get("status", "")).upper() != "RESOLVED":
            ids.append(blocker["id"])
    return ids


def planned_files(release: Path, manifest: dict[str, Any], manifest_relative: Path) -> list[Path]:
    manifest_relative = safe_relative_path(str(manifest_relative), label="manifest")
    integrity = manifest.get("integrity")
    schema = manifest.get("schema")
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256":
        raise UploadBundleError("Manifest must declare integrity.algorithm = sha256")
    expected_hashes = integrity.get("files")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise UploadBundleError("Manifest integrity.files must be a non-empty object")
    if not isinstance(schema, dict):
        raise UploadBundleError("Manifest schema must be an object")

    exclusions = manifest.get("upload_exclusions")
    if not isinstance(exclusions, list):
        raise UploadBundleError("Manifest upload_exclusions must be a list")
    excluded = {
        safe_relative_path(entry.get("path") if isinstance(entry, dict) else None, label="upload exclusion")
        for entry in exclusions
    }

    selected: set[Path] = set()
    for raw_path, expected_hash in expected_hashes.items():
        relative = safe_relative_path(raw_path, label="integrity file")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise UploadBundleError(f"Invalid SHA-256 entry for {relative}")
        if relative in excluded:
            raise UploadBundleError(f"Manifest integrity file is also excluded from upload: {relative}")
        source = ensure_inside_release(release, relative, label="integrity file")
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise UploadBundleError(
                f"SHA-256 mismatch for dataset_release/{relative}: expected {expected_hash}, got {actual_hash}"
            )
        selected.add(relative)

    schema_relative = safe_relative_path(schema.get("path"), label="schema path")
    schema_hash = schema.get("sha256")
    if not isinstance(schema_hash, str) or len(schema_hash) != 64:
        raise UploadBundleError("Manifest schema.sha256 must be a SHA-256 digest")
    if schema_relative in excluded:
        raise UploadBundleError(f"Schema is excluded from upload: {schema_relative}")
    schema_source = ensure_inside_release(release, schema_relative, label="schema")
    if sha256(schema_source) != schema_hash:
        raise UploadBundleError(f"SHA-256 mismatch for dataset_release/{schema_relative}")
    selected.add(schema_relative)
    selected.add(manifest_relative)

    schema_object = load_object(schema_source, "Hub schema")
    viewer = schema_object.get("hub_viewer_contract")
    if not isinstance(viewer, dict):
        raise UploadBundleError("Hub schema missing hub_viewer_contract")
    required_metadata = viewer.get("required_metadata")
    if not isinstance(required_metadata, list) or not required_metadata:
        raise UploadBundleError("Hub schema required_metadata must be a non-empty list")
    for raw_path in required_metadata:
        relative = safe_relative_path(raw_path, label="required metadata")
        if relative in excluded:
            raise UploadBundleError(f"Required metadata is excluded from upload: {relative}")
        ensure_inside_release(release, relative, label="required metadata")
        # Every required metadata file other than the manifest and separately
        # hashed schema must also be independently integrity-pinned.
        if relative not in {manifest_relative, schema_relative} and relative not in selected:
            raise UploadBundleError(
                f"Required metadata is not integrity-pinned in manifest: {relative}"
            )
        selected.add(relative)

    for relative in selected:
        if relative in excluded:
            raise UploadBundleError(f"Upload plan includes excluded path: {relative}")
        ensure_inside_release(release, relative, label="upload plan")
    return sorted(selected, key=lambda item: item.as_posix())


def build_bundle(release: Path, manifest_relative: Path, output: Path, *, allow_blocked: bool) -> list[Path]:
    release = Path(release).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    manifest_relative = safe_relative_path(str(manifest_relative), label="manifest")
    # These packages retain HUDOC IDs, demographic data and/or restricted
    # companion relationships. An inspection override is never privacy consent.
    if any((release / path).exists() for path in (
        "VERSION_MANIFEST.json", "release_contract.json", "clean/VERSION_MANIFEST.json",
        "paper_reference/clean/VERSION_MANIFEST.json",
    )):
        raise UploadBundleError("Internal dataset versions are not public-upload bundles; build a separately reviewed privacy-safe projection")
    try:
        output.relative_to(release)
    except ValueError:
        pass
    else:
        raise UploadBundleError("Output directory must not be inside dataset_release")
    manifest_path = ensure_inside_release(release, manifest_relative, label="manifest")
    manifest = load_object(manifest_path, "release manifest")
    unresolved = blocked_ids(manifest)
    ready = manifest.get("status") == "READY_FOR_PUBLICATION" and not unresolved
    if not ready and not allow_blocked:
        detail = ", ".join(unresolved) if unresolved else f"status={manifest.get('status')!r}"
        raise UploadBundleError(
            "Refusing to assemble a public-upload bundle before the release gate is clear: " + detail
        )

    files = planned_files(release, manifest, manifest_relative)
    if Path("DO_NOT_UPLOAD_INTERNAL_INSPECTION_ONLY.txt") in files:
        raise UploadBundleError("Manifest uses the reserved inspection-warning path")
    if output.exists():
        if not output.is_dir():
            raise UploadBundleError(f"Output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise UploadBundleError(f"Refusing to write into non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ensure_inside_release(release, relative, label="upload plan"), destination)
    if not ready:
        warning = Path("DO_NOT_UPLOAD_INTERNAL_INSPECTION_ONLY.txt")
        (output / warning).write_text(
            "DO NOT UPLOAD. This is a local inspection copy of a blocked legacy candidate.\n"
            "It is not privacy-safe certification, rights clearance, or publication approval.\n"
            "Unresolved blockers: " + (", ".join(unresolved) or str(manifest.get("status"))) + "\n",
            encoding="utf-8",
        )
        files.append(warning)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest-allowlisted Hugging Face upload bundle")
    parser.add_argument("--dataset-release", required=True, help="Explicit separately approved public projection or legacy inspection candidate; internal versions are refused")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_RELATIVE),
        help="Manifest path relative to dataset_release (default: metadata/RELEASE_MANIFEST.v2.json)",
    )
    parser.add_argument("--out-dir", required=True, help="New or empty output directory for the upload contents")
    parser.add_argument(
        "--allow-blocked-candidate",
        action="store_true",
        help="Allow a local inspection bundle when publication blockers remain; never bypasses manifest/hash checks.",
    )
    args = parser.parse_args()

    release = Path(args.dataset_release).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    try:
        manifest_relative = safe_relative_path(args.manifest, label="manifest")
        files = build_bundle(
            release,
            manifest_relative,
            output,
            allow_blocked=args.allow_blocked_candidate,
        )
    except (UploadBundleError, OSError, ValueError, TypeError) as error:
        print(f"UPLOAD BUNDLE REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Built {len(files)} allowlisted file(s) in {output}")
    if args.allow_blocked_candidate:
        print("Candidate-only mode: this output is not publication approval.")


if __name__ == "__main__":
    main()

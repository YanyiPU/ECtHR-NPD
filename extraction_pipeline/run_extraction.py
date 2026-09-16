#!/usr/bin/env python3
"""Run source reconstruction and candidate extraction using local HUDOC DOCX files.

The default is offline. This does not change the published cohort or accept labels.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "source_reconstruction"))
from source_utils import file_hash, read_index, write_csv, write_json


def run(args: argparse.Namespace) -> dict:
    cohort_rows = read_index(args.case_index)
    available = {row["itemid"] for row in cohort_rows}
    selected = args.itemids or [row["itemid"] for row in cohort_rows]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select one or more unique HUDOC identifiers")
    if set(selected) - available:
        raise ValueError("Selected HUDOC identifiers are outside the supplied case index")
    if any(not re.fullmatch(r"\d{3}-\d+", itemid) for itemid in selected):
        raise ValueError("The workflow requires HUDOC identifiers, not project surrogate IDs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        raise ValueError("run-name must be a simple filename component")
    workspace = args.workspace_root.resolve()
    if workspace == ROOT or ROOT in workspace.parents:
        raise ValueError("Use an extraction workspace outside the distributed dataset directory")
    documents = [args.hudoc_docx_dir.resolve() / f"{itemid}.docx" for itemid in selected]
    missing = [path.name for path in documents if not path.is_file()]
    if missing:
        raise ValueError(f"Missing local DOCX files: {', '.join(missing[:10])}")
    sys.path.insert(0, str(HERE / "code"))
    from openai_compatible_client import schema_errors
    for path in sorted((HERE / "schemas").glob("pipeline_*.schema.json")):
        schema_errors({}, json.loads(path.read_text(encoding="utf-8")))
    model_configuration = None
    if args.mode == "llm":
        from openai_compatible_client import OpenAICompatibleClient
        # Read configuration before performing local writes; this makes no request.
        client = OpenAICompatibleClient.from_env()
        model_configuration = {key: getattr(client, key) for key in (
            "model", "timeout_seconds", "use_json_schema", "default_temperature", "enable_thinking", "thinking_budget", "seed")}
        model_configuration["provider_host"] = urlsplit(client.base_url).hostname
    manifest_path = workspace / "workflow_runs" / f"{args.run_name}.json"
    run_path = (workspace / "extraction" / "runs" / "pipeline_c_backbone" / args.run_name if args.mode == "regex"
                else workspace / "extraction" / "outputs" / "runs" / "holistic" / args.run_name)
    if manifest_path.exists() or run_path.exists():
        raise ValueError("Run name already exists; choose a new run name")
    selected_index = workspace / "workflow_runs" / f"{args.run_name}.source_index.csv"
    base = [sys.executable, "-B"]
    prepare = base + [str(ROOT / "source_reconstruction" / "prepare_extraction_case_store.py"),
        "--case-index", str(selected_index), "--hudoc-docx-dir", str(args.hudoc_docx_dir.resolve()),
        "--out-root", str(workspace)]
    if args.source_metadata is not None:
        prepare += ["--echrod-metadata", str(args.source_metadata.resolve())]
    scaffold = base + [str(HERE / "code" / "build_extraction_layers.py"),
        "--workspace-root", str(workspace), "--itemids", *selected]
    program = "run_pipeline_c_backbone.py" if args.mode == "regex" else "holistic_extractor.py"
    extract = base + [str(HERE / "code" / program), "--workspace-root", str(workspace),
        "--run-name", args.run_name, "--concurrency", str(args.concurrency), "--itemids", *selected]
    if args.mode == "regex":
        extract += ["--regex-only"]
    else:
        extract += ["--max-retries", str(args.max_retries)]
    tracked = sorted([*HERE.glob("code/*.py"), *HERE.glob("prompts/*"), *HERE.glob("schemas/*.json"),
                      *ROOT.glob("source_reconstruction/*.py"), Path(__file__).resolve()])
    report = {"status": "planned", "mode": args.mode, "case_ids": selected,
        "python_version": sys.version.split()[0], "jsonschema_version": version("jsonschema"),
        "model_configuration": model_configuration,
        "case_index_sha256": file_hash(args.case_index),
        "source_metadata_sha256": file_hash(args.source_metadata) if args.source_metadata else None,
        "documents_sha256": {p.name: file_hash(p) for p in documents},
        "implementation_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in tracked},
        "source_index_contains_targets": False, "stages": [],
        "acceptance_status": "candidate_requires_dataset_review", "paper_scores_reproduced": False}
    if args.dry_run:
        return {**report, "commands": [prepare, scaffold, extract]}
    # Pass only source identifiers. Published award labels/demographics must not
    # become evidence for a fresh source extraction.
    write_csv(selected_index, [{"itemid": itemid} for itemid in selected], ["itemid"])
    write_json(manifest_path, report)
    for name, command in (("prepare", prepare), ("scaffold", scaffold), (args.mode, extract)):
        result = subprocess.run(command, check=False)
        report["stages"].append({"name": name, "exit_code": result.returncode})
        report["status"] = "failed" if result.returncode else "running"
        write_json(manifest_path, report)
        if result.returncode:
            return report
    report["status"] = "candidate_extraction_completed"
    write_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", required=True, type=Path,
                        help="CSV with HUDOC case_id or itemid; use data/case_level.csv to select the paper cohort")
    parser.add_argument("--itemids", nargs="+", help="Optional explicit subset of the supplied index")
    parser.add_argument("--hudoc-docx-dir", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--source-metadata", type=Path,
                        help="Optional original HUDOC/ECHROD metadata CSV, not the benchmark label table")
    parser.add_argument("--mode", choices=["regex", "llm"], default="regex")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, choices=range(11), default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

> The distributed cohort is represented by `data/case_level.csv` and `data/applicant_level.csv`. Raw source documents and extraction runs are separate local working artifacts. The public `case_id` is the original HUDOC item identifier.

# Source Reconstruction

These are source-ingestion helpers, not an automatic dataset publication pipeline.
They accept a CSV with real HUDOC `itemid` values or the public `case_id`
column, including `data/case_level.csv`; no private case crosswalk is needed. Source
documents can be retrieved from HUDOC subject to applicable source terms.

This directory contains code only. The scripts write raw or derived judgment
text and potentially identifying metadata to a user-supplied local directory such
as `your_path/hudoc_judgments_docx` or `your_path/extraction_workspace`; those
local files are **internal-only** and are not part of the public bundle. Full
metadata, raw text and working audit artifacts are not distributed. The public
HUDOC identifiers remain directly linkable to judgments; masking applicant names
does not establish anonymity. See [the executable workflow](../docs/EXTRACTION.md).

## Inputs

- A case-index CSV with HUDOC `itemid` or `case_id`, source metadata if available,
  and optionally a HUDOC URL. The local runner projects the public case table to
  source IDs only so benchmark labels are not fed back into extraction.
- `your_path/ECHROD/echr_database`: optional local ECHR-OD/ECHROD export used
  to recover HUDOC metadata for the same `itemid` values. The local export may
  not cover recent cases; missing requested IDs cause a nonzero exit.
- HUDOC public conversion endpoints, accessed by `itemid`.

## Steps

Run the following commands from the repository root. These are explicit
source-download/provider operations, not checks to run during a documentation
audit. For a new batch, use the reviewed planner's `candidate_index.csv`;
for a full source reconstruction, supply an approved complete internal index.

Download the reviewed batch of public HUDOC judgments:

```bash
python3 source_reconstruction/download_hudoc_judgments.py \
  --case-index your_path/update_plan/candidate_index.csv \
  --out-dir your_path/hudoc_judgments_docx \
  --format docx
```

Build the complete **internal** ECHR-OD/ECHROD metadata subset:

```bash
python3 source_reconstruction/build_echrod_subset.py \
  --case-index your_path/update_plan/candidate_index.csv \
  --echrod-root your_path/ECHROD/echr_database \
  --out-dir your_path/echrod_source_index
```

Prepare a local extraction case store from the downloaded DOCX files:

```bash
python3 source_reconstruction/prepare_extraction_case_store.py \
  --case-index your_path/update_plan/candidate_index.csv \
  --echrod-metadata your_path/echrod_source_index/echrod_metadata_subset.csv \
  --hudoc-docx-dir your_path/hudoc_judgments_docx \
  --out-root your_path/extraction_workspace
```

Then run the extraction code directly with `--workspace-root`. Historical
`structured/cases_core.json` is optional; it is only a comparison reference.
No working-code copy or symlink is needed. The one-command offline route and
reviewed two-CSV projection are documented in [EXTRACTION.md](../docs/EXTRACTION.md).

```bash
python3 extraction_pipeline/code/build_extraction_layers.py \
  --workspace-root your_path/extraction_workspace --itemids YOUR_AUTHORISED_HUDOC_ID
EXTRACTION_API_BASE=your_api_base \
EXTRACTION_API_KEY=your_api_key \
EXTRACTION_MODEL=your_model \
python3 extraction_pipeline/code/holistic_extractor.py \
  --workspace-root your_path/extraction_workspace \
  --itemids YOUR_AUTHORISED_HUDOC_ID --run-name local_check
```

Strict model inputs must satisfy the experiment-input contract in the root
documentation; a directory name alone does not establish source review or
absence of target leakage. The reconstruction workflow above is for source audit,
label extraction, and rebuilding extraction sidecars; raw judgment text and
Article 41 material must not be fed to benchmark prediction baselines.

## Incremental update guarantees

- Duplicate/empty/unsafe IDs are rejected. A partial batch with missing or invalid
  DOCX files does not update the case store and returns a nonzero exit code.
- `prepare_extraction_case_store.py` merges by `itemid`, keeps existing order,
  and appends new records in input order. It does not replace the aggregate with
  just the new batch. An identical batch leaves `cases.json` unchanged.
- Changed source records require explicit `--overwrite`. Conflicting nonempty
  case-index/ECHROD fields require explicit `--metadata-precedence index` or
  `--metadata-precedence echrod`; both originals and the decision remain recorded.
- Unexpected disagreements between the aggregate and per-item records are
  rejected even with `--overwrite`. The known `appendix_table_text` derived cache
  is excluded from source equality; no other source differences are ignored.
- One local writer holds a lock. Each data file is replaced atomically; a pending
  journal protects against a crash between files. After inspection, rerun the
  identical batch with `--resume-interrupted` to roll forward. Recovery verifies
  source-record and aggregate hashes and rejects unrelated changes. Readers must
  not consume a workspace while `unstructured/ingestion_pending.json` exists.
- `ingestion_manifest.json` records old/new record hashes, DOCX and metadata
  hashes, affected IDs/actions and the aggregate checksum. It represents the
  latest batch, not a historical release archive. Retain approved prior release
  manifests outside the current working dataset when historical reproducibility
  is required.
- The downloader validates DOCX ZIP/XML/text before replacing a file; existing
  downloads are checked too. It retries transient network errors, records errors,
  and returns nonzero on failed items. `--dry-run` prints a plan without writing.
  HTML validation is heuristic and does not certify judgment identity/content.

## Metadata and person counts

The default metadata exporter retains all supplied fields, including `appno`,
structured `conclusion`, representation and original values. JSON/list-like
exports are normalized. Unparseable conclusion text remains evidence and is
marked unknown; it is not guessed into structured findings. Positive
`article=...` flags supply **mentioned** articles only, not violated articles.
An ECHROD export that never contained structured findings cannot recover them;
a complete source export or manual/source-supported normalization is required.

Missing metadata counts/booleans remain `null`, not zero/one/false. The scaffold
provides `metadata_missing_fields` and normalization issues. Its diagnostic
`challenging_metadata_eligible` uses Grand Chamber **OR** (more than one distinct
application number **AND** more than one distinct violated article **code**).
Unknown inputs remain null unless the other OR arm already establishes true;
the complexity arm is also exposed separately. This does not relabel the
historical released split or automatically assign a case to test.

`num_application_numbers` is not the number of people. The compatibility field
`num_applicants_proxy` now explicitly means application-number count and can be
null. Initial `facts_procedure.num_applicants` is null until person extraction;
it is never filled from application counts. The original DOCX remains the source
of truth: the text/table projection and section-heading parser are heuristic,
not complete preservation of formatting, merged-cell semantics or legal meaning.

## Scoped live discovery and offline quarterly planning

The [live discovery helper](discover_hudoc_cases.py) can first obtain an internal
index for explicitly selected **English judgments** from Grand Chamber, Chamber
and/or Committee collections. Dates, language, type and collections are mandatory.
See [LIVE_DISCOVERY.md](LIVE_DISCOVERY.md) for the verified endpoint, exact scope,
pagination, two-pass verification, safe output handling and resume/restart commands.
Only use its index when `discovery_manifest.json` reports `output_ready: true`.

Then supply that index, or another current source-index CSV with ISO judgment
dates, to the separate **offline** planner:

```bash
python3 source_reconstruction/plan_incremental_ingestion.py \
  --case-index your_path/latest_internal_hudoc_index.csv \
  --existing-cases your_path/extraction_workspace/unstructured/cases.json \
  --date-from 2026-01-01 --date-until 2026-04-01 \
  --out-dir your_path/update_plan
```

The date interval is start-inclusive/end-exclusive. The script writes only
`candidate_index.csv` and a deterministic hash-addressed `ingestion_plan.json`.
It never changes the current dataset, downloads cases, calls an LLM, or reassigns
splits. Invalid dates are errors rather than silent exclusions. With
`--include-known-revisions`, known cases outside the date window are also checked
for metadata revisions; `--docx-dir` additionally checks retained document hashes.
Without a current document, unchanged metadata is **source-unverified**, not proof
that the judgment has not changed. If rows are merely missing from the supplied
index, the planner cannot establish whether HUDOC omitted them.

**Scope limitation:** live discovery certifies observed consistency of its explicit
query, not all-HUDOC completeness or an immutable snapshot. A judgment-date window
can miss older judgments indexed late, later translations or historic corrections;
overlap/backfill and historical re-harvest require an explicit project policy.
Scientific label review, tabular acceptance, privacy review and publication approval
remain separate gates. Native HUDOC narrative conclusions are not automatically
equivalent to ECHROD structured findings.

## Offline regression tests

From the release root:

```bash
python3 -B -m unittest discover -s tests -p test_source_reconstruction.py -v
python3 -B -m unittest discover -s tests -p test_hudoc_discovery.py -v
```

Tests use only synthetic DOCX/CSV fixtures and mocked responses. The separately
documented tiny live metadata probes verify the observed search interface, not a
complete quarterly harvest, future API availability or real-case extraction accuracy.

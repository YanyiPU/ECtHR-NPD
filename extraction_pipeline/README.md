> The distributed data are the single paper-cohort `data/case_level.csv` and `data/applicant_level.csv`, with original HUDOC case IDs. Extraction working artifacts stay in a separate local workspace.

# Extraction Pipeline

This directory contains the current extraction implementation for deriving
case-level structured sidecars and Article 41 label candidates from local
HUDOC source documents. It is separate from benchmark prediction baselines:
raw judgment text, Article 41 text, operative clauses, award snippets, and
claim amounts may be read here for source extraction, but they must not be
used as benchmark prediction inputs. The code and schema checks do not certify
historical extraction equivalence or permission to release source outputs.

## Layout

- `code/`: deterministic scaffold builder, OpenAI-compatible client,
  B/C/D/E extraction stages, and the holistic stage orchestrator.
- `prompts/`: system prompts for facts/procedure, compensation, legal
  analysis, and reasoning extraction.
- `schemas/`: JSON schemas for the extraction stages.

## Local Source Setup

Start with [EXTRACTION.md](../docs/EXTRACTION.md): it provides the
`run_extraction.py` offline local-DOCX workflow and `export_reviewed.py`
projection into exactly the two canonical CSVs. The latter requires complete,
source-reviewed canonical fields; it does not infer missing historical data.

From the repository root, install the extraction-only requirements in the
selected environment before using the LLM stages:

```bash
python3 -m pip install -r requirements-extraction.txt
```

`jsonschema` is mandatory: a missing dependency or invalid
schema fails closed, before a provider request. A separate download/extraction
environment may be used; the prediction requirements are unchanged.

Use `source_reconstruction/` to download HUDOC DOCX files and prepare a
local workspace with `unstructured/cases.json` and
`unstructured/cases_by_itemid/*.json`. Point the extraction stages at that
workspace using `--workspace-root`; no code copy is required. Keep this
operational workspace outside the maintained dataset folder.

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

The API variables above are placeholders, not credentials or an authorization
to send data. Existing source evidence and run artifacts are internal; their
absence from this code directory is not a blanket publication certificate.

## Reliability and acceptance contract

- Explicit `--itemids` work without the historical split files in B, B-backbone,
  C and E. This selects cases, not a new train/dev/test allocation.
- Application-number count is never substituted for person count. B-backbone
  returns `insufficient_evidence` if explicit person-count evidence is absent,
  instead of defaulting to one or counting title fragments. C can use a
  successful B person-count candidate, but never the application-count proxy.
- `--max-retries N` means at most **N additional attempts per stage** (0–10),
  after a first attempt. B/C/E standalone CLIs and the holistic CLI default to
  3. Direct Python stage calls default to 1. The combined D/E stage has one
  shared budget and does not double-count its usage. Chat POSTs have no hidden
  transport retries. Transient timeouts, connection errors, 429 and server
  errors retry with bounded backoff; schema failures retry with validation
  feedback. Authentication/client errors and malformed API payloads fail.
- Holistic `success` means all B/C/D/E stage schemas passed, **not** scientific
  acceptance of labels. Results remain candidates requiring dataset review.
  Schema-failing C values are never promoted into the combined award result.
- Holistic `partial_success` results are retained only under the run's
  `quarantine/` directory, not promoted to `outputs/cases/`. They are excluded
  from `completed_cases` and are attempted again on `--resume`. Historical
  partial files already written by older code are not automatically deleted;
  consumers must check the run metadata rather than glob all old outputs.
- Standalone failed cases and holistic partial/failed cases produce exit status
  1 after writing their summary. An automation must not treat them as a
  successful completed batch. Exit status 0 is still not dataset publication
  approval; label and privacy review remain separate.
- B-backbone resume preserves earlier successes in rebuilt `by_split/` files.
  B/E mixed per-case and JSONL inputs merge by ID, with per-case inputs taking
  precedence; finding a fallback does not discard cases present only locally.
- C `--regex-only` needs no API environment variables and makes zero API calls.
  Its cross-validation is explicitly `not_cross_validated_regex_only`, with
  `flag_for_review=true`; deterministic agreement is not asserted without a
  second independent extraction.

All source text, unreviewed applicant-level output, failed candidates and provider
errors belong in an internal extraction workspace. HUDOC case IDs are public.
Do not publish these output folders as a privacy-safe dataset. Provider use
also needs an approved data-handling arrangement and explicit cost approval.

These safeguards are covered by offline regression tests in
`../tests/test_extraction_reliability.py`. They do not establish provider
compatibility, extraction accuracy, full historical reproduction, or quarterly
end-to-end readiness. Run a separately approved adjudicated sample before
ingesting newly extracted labels.

Candidate allocation indices are not verified identities. Holistic output
does not infer missing indices from row order, fill a person's name from an
array position, or assign joint/group amounts to one person. The internal
`per_applicant_mapping_audit` retains source indices separately from normalized
candidate indices and leaves `verified_applicant_id` null.

## Internal tabular export

The canonical route is `export_reviewed.py`, documented in
[EXTRACTION.md](../docs/EXTRACTION.md). The older [export_tabular.py utility](EXPORT_TABULAR.md)
is retained for internal candidate/applicant/allocation audits. Its historical
43-column internal-schema adapter is not the root public schema; its staging
tables must not be copied into the distributed data directory. Neither exporter
automatically changes the dataset or reproduces the paper's scores.

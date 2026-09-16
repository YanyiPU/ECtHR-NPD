# Source extraction and reviewed table projection

The distributed dataset is the single retained paper cohort of 14,575 cases in
`data/case_level.csv` and their retained source records in `data/applicant_level.csv`.
`case_id` is the HUDOC item identifier. The following workflow reconstructs
source evidence and extraction candidates for those IDs; it never reselects the
paper cohort, changes splits, or replaces published targets automatically.

## Local documents to candidates

Use Python 3.10 or later with `requirements-extraction.txt` installed. The
`jsonschema` dependency is required even for deterministic candidate runs.
The runner checks all stage schemas before writing outputs. It does not install
packages or download judgments, tokenizers, or models. Token counts use the
documented character-count approximation by default; the separate optional
`EXTRACTION_TOKEN_COUNTER=tiktoken` setting can trigger a vocabulary download.

Place the local documents at `/path/to/documents/<HUDOC-ID>.docx`. From the
repository root, select an ID present in the published case table:

```sh
python3 -B extraction_pipeline/run_extraction.py \
  --case-index data/case_level.csv \
  --itemids 001-210465 \
  --hudoc-docx-dir /path/to/documents \
  --workspace-root /path/to/extraction_workspace \
  --run-name source_check --mode regex
```

Replace the illustrative ID with the case you are checking. Omit `--itemids` only
when all selected cohort documents are locally available. `--dry-run` checks
inputs and prints the command plan without creating files. The workspace must
be outside the dataset directory. The installed code is used directly; copying
or symlinking the extraction code into the workspace is unnecessary.

The runner creates an ID-only source index, so the existing dataset's target
amounts and demographics are not used as source evidence. Add
`--source-metadata /path/to/original_hudoc_metadata.csv` to supply actual source
metadata. Missing source metadata remains unknown. The distributed descriptive
fields are not automatically treated as newly recovered HUDOC evidence.

Stages and outputs:

| Stage | Local output | Meaning |
| --- | --- | --- |
| Source preparation | `unstructured/cases.json`, `unstructured/cases_by_itemid/<ID>.json` | DOCX text/table projection and source checksums |
| Segmentation | `extraction/outputs/case_features_labels.jsonl` and stage input sidecars | Heuristic facts, merits, Article 41/50, operative, and appendix separation |
| Deterministic compensation | `extraction/runs/pipeline_c_backbone/<run>/results_unique.jsonl` | Candidate award evidence; `not_cross_validated_regex_only` and review flag remain set |
| Workflow record | `workflow_runs/<run>.json` | Input/document/code/prompt/schema hashes, dependency versions, stage exit codes, model settings when used |

The DOCX projection retains paragraph/table order but does not preserve every
formatting or merged-cell interpretation. Failed or incomplete evidence does
not become a zero target. Application-number count is never substituted for a
person count. Group and joint awards are never evenly split among people.

## Optional model extraction

The same command with `--mode llm` calls the configured provider and can incur
cost. Set `EXTRACTION_API_BASE`, `EXTRACTION_API_KEY`, and `EXTRACTION_MODEL` in
the execution environment; do not put credentials in scripts or output files.
The runner records the model, provider hostname, temperature, seed and relevant
decoding settings, but never the API key. Select your own model explicitly:
the available material does not establish a uniquely reproducible historical
extraction provider/model configuration.

The holistic model stage runs facts/applicants (B), compensation (C), merits (D),
and case reasoning (E), with JSON schema checks and bounded retries. Successful
outputs are under `extraction/outputs/runs/holistic/<run>/<ID>.meta.json`.
Schema success still means candidate extraction. Partial results are quarantined,
not promoted as complete. Missing operative validation is marked unknown and
requires review, including when a model supplies a converted non-EUR amount.

Individual entry points also accept an external workspace:

```sh
python3 -B extraction_pipeline/code/build_extraction_layers.py \
  --workspace-root /path/to/extraction_workspace --itemids 001-210465
python3 -B extraction_pipeline/code/holistic_extractor.py \
  --workspace-root /path/to/extraction_workspace \
  --itemids 001-210465 --run-name model_check --concurrency 1
```

Source discovery and downloading remain separate optional operations in
`source_reconstruction/`. Its CSV readers accept either source `itemid` or
public HUDOC `case_id`. Retrieval today does not recreate the historical source
snapshot or exclusion cascade.

## Eight-check review and two-table projection

Appendix B.3 describes these checks: head separation, applicant/beneficiary
consistency, per-applicant sum consistency, no-claim/positive-award incompatibility,
claim/award consistency, currency normalization, operative recoverability, and
the manually audited bundled-award exception. Automated diagnostics cover parts
of this process; they do not replace the source review or reproduce the historical
230-candidate / 72-retained exception ledger.

`export_reviewed.py` accepts an explicitly reviewed JSONL file. Each line contains:

- `case`: exactly the case fields in the root `schema.json`, with string values.
- `applicants`: a list of records containing exactly the applicant fields in that
  schema. An explicit `unknown` is preserved where the schema permits it. Missing
  fields are rejected, not constructed from incomplete candidates.
- `review`: `status=accepted`, `adjudicator`, `source_anchor`,
  `canonical_fields_reviewed=true`, relative `source_document`,
  `source_document_sha256`, and `record_sha256` from
  `export_reviewed.record_hash(case, applicants)` before name masking.
- `review.checks`: exactly the eight keys in `export_reviewed.CHECKS`. Each has
  `status` (`pass` or `not_applicable`) and a nonempty `evidence` explanation with
  source anchors; an exception or non-applicable status needs a source-based
  reason. `unknown`, `fail`, or omitted checks block projection.
- For zero targets, `review.zero_rationale` is one of `finding_sufficient`,
  `no_claim`, `unsubstantiated`, `rule_60_non_compliance`,
  `domestic_award_covers`, or `applicant_deceased_no_heir`.
- Amount changes require `target_status=corrected_amount` and a nonempty
  `review.correction_rationale` supported by the source.

The review payload is an assertion by its preparer; hashes bind records to
documents but do not authenticate a reviewer or verify legal interpretation.
Inspect all identifying text, including names occurring inside nationality text.
The exporter masks supplied applicant names and repeated occurrences, retains
HUDOC IDs and descriptive values, and adds no predictor encodings.

```sh
python3 -B extraction_pipeline/export_reviewed.py \
  --input /path/to/reviewed_canonical_records.jsonl \
  --cohort-case-table data/case_level.csv \
  --source-root /path/to/documents \
  --out /path/to/reviewed_staging
```

The staging directory contains exactly `data/case_level.csv` and
`data/applicant_level.csv`, and is checked using the root public-table validator.
It can contain a reviewed subset for inspection. It is not a second dataset and
is never appended automatically: out-of-cohort cases and altered split/view
assignments are rejected. Existing output directories are not overwritten.
Keep the review input and printed provenance report with the internal audit.

The older `export_tabular.py` remains an internal candidate/allocation audit
utility, with a historical internal-schema adapter. Its multi-table outputs are
not the public two-table contract; use `export_reviewed.py` for canonical staging.

## What this establishes, and what remains unavailable

The commands and synthetic tests establish a working local source-to-candidate
workflow and strict reviewed projection. They do not establish that rerunning
today's extraction gives the paper's historical labels or reported scores.
The published tables preserve the original cohort and documented corrections;
mixed or unresolved historical labels remain visibly marked by `target_status`.

Historical source snapshot completeness, full extraction run manifests, original
provider responses, source-specific zero/exception review ledgers, complete
identity/allocation adjudication, and original benchmark input snapshots are
not reconstructed by these utilities. Appendix B.1/Table 10's exclusion counts
cannot be recovered merely by running against the retained 14,575 IDs.
The paper's Appendix B.3 validated-target claim therefore requires separate
evidence; missing artifacts are not proof every historical label is wrong.

Reconstructed facts sidecars are also not certified prediction inputs. The
experiment bridge requires a separately reviewed `facts_text` JSONL with
`review_status=accepted_prediction_input`, source document checksum and source
anchor. Article 41/50 claims, award tables, and operative payment text belong to
target construction, not the strict benchmark predictors (Appendix B.2/Table 11).

Offline checks (synthetic fixtures and mocked providers only):

```sh
python3 -B -m unittest discover -s tests -p test_unified_extraction.py -v
python3 -B -m unittest discover -s tests -p test_source_reconstruction.py -v
python3 -B -m unittest discover -s tests -p test_extraction_reliability.py -v
python3 -B -m unittest discover -s tests -p test_extraction_export.py -v
```

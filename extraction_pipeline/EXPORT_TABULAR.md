> This audit utility writes local working tables. The distributed dataset has only `data/case_level.csv` and `data/applicant_level.csv`; use the canonical projection documented below for those fields.

# Internal extraction-to-tabular exporter

For the current public HUDOC-ID two-table contract, use
[`export_reviewed.py`](../docs/EXTRACTION.md). This document describes only an
internal audit utility and its historical internal-schema adapter. The public
dataset does not require a private case-ID crosswalk. If this audit utility is
used with current IDs, its explicit crosswalk can map each HUDOC ID to itself.

`export_tabular.py` converts explicitly selected holistic extraction records
into three candidate tables and three independently adjudicated clean-batch
tables. **It is not an anonymization tool, a complete 43-column feature builder,
an automatic quarterly append, or a public-release tool.**

`clean` here means a source-adjudicated **new-batch subset**, not the full
historical paper feature package. `y_amount_eur` and `y_binary` use the current
model-loader target names. The generated `schema_adapter.json` maps them to
the historical 43-column internal case schema and explicitly marks the other
39 columns as unconstructed/unvalidated. `itemid` requires a private crosswalk
join; `split` is either explicitly supplied or pending. Missing features are
never fabricated or filled with zeros. The reference is the header-only
snapshot `schemas/current_case_columns.v20260913.json`, which must be revised
deliberately if the maintained case schema changes.

## Candidate pass: no scientific promotion

Run from the repository root. Install the dependency with
`python3 -m pip install -r requirements-extraction.txt` in the selected
environment, then:

```sh
python3 extraction_pipeline/export_tabular.py \
  --input your_path/holistic_run/YOUR_FIRST_AUTHORISED_HUDOC_ID.meta.json \
  --input your_path/holistic_run/YOUR_SECOND_AUTHORISED_HUDOC_ID.meta.json \
  --case-crosswalk your_path/private_case_crosswalk.csv \
  --out your_path/candidate_batch
```

Repeat `--input` for each specifically selected record. The exporter does not
glob all historical runs and will reject duplicate itemids. Use holistic
`*.meta.json` or full result JSON with `result`/`layers.merged`. A standalone
`merged.json` can be exported as a blocked candidate, but cannot pass promotion
without the selected run status and per-stage evidence. Files without any
merged result are rejected rather than silently omitted.

The private case crosswalk requires `itemid,case_id`, both one-to-one and
nonempty. The exporter **never creates or changes an ID**. Reuse the maintained
crosswalk and add new IDs through your separately approved ID-management
process. Candidate tables retain internal itemids. Neither these IDs nor
content fingerprints provide anonymity.

## Adjudicated pass: explicit source evidence

After human source review, rerun to a new staging directory with:

```sh
python3 extraction_pipeline/export_tabular.py \
  --input your_path/holistic_run/YOUR_AUTHORISED_HUDOC_ID.meta.json \
  --case-crosswalk your_path/private_case_crosswalk.csv \
  --case-adjudication your_path/case_adjudication.csv \
  --applicant-crosswalk your_path/applicant_adjudication.csv \
  --allocation-adjudication your_path/allocation_adjudication.csv \
  --source-root your_path/approved_source_documents \
  --split-manifest your_path/approved_splits.csv \
  --out your_path/adjudicated_batch
```

The optional applicant/allocation/split files can be omitted. Omitting them
does not authorize identity linking, allocation promotion, or split creation.
An existing output directory is never overwritten. Use the project's
controlled-apply process to update the single maintained latest dataset after
reviewing these outputs; do not create extra public dataset copies.

### Case adjudication CSV

Required for clean case promotion:

- `itemid`, `record_sha256`: copy the hash from the candidate case table; it
  binds the decision to the exact current merged extraction JSON.
- `review_status=accepted`, `decision=include_pure_npd`,
  `heads_status=pure_npd`, `award_scope=case_total`.
- `case_total_npd_eur`: an explicit, finite, nonnegative, cent-exact value.
  Empty does not mean zero.
- `currency=EUR`, `source_currency=EUR`: this exporter does not perform FX
  conversion. A contrary extracted currency needs
  `currency_resolution=verified_eur_in_source` and `correction_rationale`.
- `source_document_path`, `source_document_sha256`, `source_anchor`,
  `adjudicator`: an existing source document, its SHA-256, the exact paragraph,
  operative point or appendix locator, and the responsible reviewer.
  Paths resolve within `--source-root`, or the case-ledger directory if no
  root is supplied. Absolute/traversal/symlink paths escaping that root are
  rejected **before content hashing**. The exporter checks the file hash; it
  does not claim to understand or verify the reviewer's legal interpretation.
- `allocation_coverage`: `complete`, `partial`, `none_found`, or `unknown`.
  This is the reviewer's source assessment; `allocation_export_status`
  separately reports whether the exported allocation set supports it.
- `zero_rationale`: required for zero targets and restricted to the paper's
  six categories: `no_claim`, `unsubstantiated`, `rule_60_non_compliance`,
  `domestic_award_covers`, `applicant_deceased_no_heir`, `finding_sufficient`.
  Free text or expanded categories are not silently mixed into this taxonomy.
  An extension would require a separately versioned schema and documentation.
- `correction_rationale`: required if the adjudicated total differs from the
  extraction or if a mixed-head candidate is corrected. Such a candidate also
  needs `mixed_candidate_resolution=verified_pure_npd_amount`. A genuinely
  mixed or unresolved source must remain blocked; this flag is only for a
  reviewed extraction error with a demonstrably separate NPD amount.

Clean promotion additionally requires selected-run `status=success`, all
B/C/D/E stage statuses successful, the current candidate-acceptance marker,
and fresh **real schema and stage-semantic validation** of B/C/D/E outputs.
A ledger cannot bypass partial, missing or invalid extraction evidence.

### Applicant identity and attribute crosswalk

Each row requires `itemid,applicant_key,applicant_id,review_status,
source_document_sha256,source_anchor,adjudicator`.

`applicant_key` is the candidate table's SHA-256 fingerprint of the complete
source applicant object, not its row position. `applicant_id` is an explicitly
maintained stable ID; no automatic remapping or longitudinal identity inference
occurs. Set `review_status=accepted_identity_and_attributes` only after checking
**both the identity mapping and exported demographic attributes** against the
case's hashed source document. Partial applicant coverage must not be used to
derive whole-case demographic ratios. Unreviewed/padded records remain
candidates, not confirmed people. Duplicate indistinguishable candidate rows
are blocked and retain their multiplicity count.

### Allocation adjudication CSV

Each row requires `itemid,allocation_key,allocation_id,review_status,
head,currency,amount_eur,allocation_scope,applicant_id,
source_document_sha256,source_anchor,adjudicator,correction_rationale`.

- `allocation_key` binds the review to an exact candidate allocation object.
  Use a stable maintained `allocation_id`; no random ID is generated.
- Only `review_status=accepted`, `head=non_pecuniary`, `currency=EUR` are
  eligible. A changed amount/head needs a source-based correction rationale.
- Allowed scopes: `personal`, `joint`, `group`, `not_person_linked`.
- `personal` requires an independently accepted applicant crosswalk entry
  **for this same case**. An extraction `applicant_index` is not such a link.
- All other scopes require empty `applicant_id`. A joint/group amount remains
  **one allocation row**, never copied to each person or divided equally.
- Indistinguishable repeated primary/fallback allocations preserve
  `occurrences` and are not promoted until their ambiguity is resolved.
- Verified allocations exceeding the adjudicated case NPD are blocked.
  `complete` coverage additionally requires an exact sum and every candidate
  allocation reviewed. Under-allocation with explicitly `partial` coverage
  may be exported as a verified partial set. `none_found` or `unknown` never
  creates zero-valued personal records or speculative allocations.

An allocation conflict blocks the allocation set, not an independently
source-adjudicated case target. The clean case's `allocation_export_status`
then reads `blocked`; the audit records why. The applicant table intentionally
has no synthesized per-person target—explicit personal and group amounts
remain in the allocation table.

### Split manifest

Requires `case_id,split,split_policy_version`. Supported split labels are
`train`, `dev`, `test`, `ood`, `unassigned_pending_policy`. Omission results in
`unassigned_pending_policy`; no existing benchmark is re-split. A future
mapping to a historical split spelling must be an explicit adapter, not an
assumed equivalent.

## Output tree and fields

```text
batch/
├── candidate/
│   ├── case_level.csv          # 16 fields
│   ├── applicant_level.csv     # 12 fields
│   └── allocation_level.csv    # 12 fields
├── clean/
│   ├── case_level.csv          # 11 fields, not the full 43-column package
│   ├── applicant_level.csv     # 8 fields
│   └── allocation_level.csv    # 9 fields
├── audit/decisions.csv         # 5 fields
├── manifest.json              # exact columns, counts and CSV checksums
├── schema_adapter.json        # explicit mappings and missing-feature status
└── RESTRICTED.txt
```

Exact field order:

| CSV | Fields |
|---|---|
| candidate/case_level.csv | case_id, itemid, record_sha256, extraction_status, stage_validation_status, respondent_country, judgementdate, num_application_numbers, num_applicants_candidate, violated_article_codes_json, npd_eur_candidate, bundled_eur_candidate, split, split_policy_version, promotion_status, issues_json |
| candidate/applicant_level.csv | case_id, itemid, applicant_key, applicant_id, source_applicant_index, birth_year, sex, age_group, nationality, occurrences, mapping_status, issues_json |
| candidate/allocation_level.csv | case_id, itemid, allocation_key, head_candidate, eur_amount_candidate, source_applicant_index_candidate, source_kinds_json, occurrences, allocation_scope, applicant_id, promotion_status, issues_json |
| clean/case_level.csv | case_id, y_amount_eur, y_binary, currency, target_scope, zero_rationale, allocation_coverage, allocation_export_status, split, split_policy_version, review_status |
| clean/applicant_level.csv | case_id, applicant_id, birth_year, sex, age_group, nationality, split, review_status |
| clean/allocation_level.csv | case_id, allocation_id, applicant_id, head, amount_eur, currency, allocation_scope, split, review_status |
| audit/decisions.csv | case_id, record_type, record_key, decision, reasons_json |

The allocation adapter explicitly maps `amount_eur` to the maintained
`eur_non_pec_amount`, and links `case_id` to internal `itemid` only through the
private case crosswalk. Applicant/allocator review statuses are also explicit
adapter mappings, not silent renames.

No beneficiary-name columns, full judgment text, claim text, award-reasoning
text, or predictor feature matrix is exported. The bundle still contains
source IDs, dates, amounts, demographic quasi-identifiers and content
fingerprints: **all outputs, including clean tables, remain internal and
restricted**. Do not upload this bundle to GitHub/Hugging Face as a safe public
release. Record linkage and demographic review are not anonymity certificates.

## Verification boundary

Tests exercise real JSON schemas with synthetic, non-personal fixtures and
temporary source files. They verify blocked/promoted flows, exact monetary
arithmetic, identity boundaries, duplicate handling, source path containment,
stable IDs, deterministic output, and no implicit splitting. They do not
establish real-case extraction accuracy or validate a human ledger's factual
truth. Every remaining paper-feature, privacy and publication gate still
applies before a new batch joins the maintained dataset.

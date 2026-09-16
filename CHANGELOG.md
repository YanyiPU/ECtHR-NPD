# Unified release changes — 2026-09-16

Published at the author's request after local preview: one current dataset,
the original paper cohort, and amount-only target corrections. The public GitHub
and primary Hugging Face current trees are synchronized. Historical commits and
the private HF review repository are retained. Internal canonical evidence was
not overwritten by this publication.

## Data corrections

| HUDOC case ID | Case target | Applicant allocation now carried into the CSV |
|---|---:|---|
| 001-210465 | €800 → **€2,400** | 3 source-stated awards of €800 |
| 001-142400 | €12,000 → **€96,000** | 8 source-stated awards of €12,000 |
| 001-196605 | €6,000 unchanged | 2 × €3,000 restored |
| 001-152888 | €22,500 unchanged | 3 × €7,500 restored |
| 001-89058 | €6,000 unchanged | 2 × €3,000 restored |
| 001-72236 | €20,000 unchanged | 5 × €4,000 restored; two are applicant-estate units |

The two case-total fixes were already present in the previous review CSV; they
remain applied. All 23 applicant amounts above were `unknown` in that previous
CSV because the export rejected unlinked source indices/estate units. Source
recipient comparisons and explicit each-applicant award evidence now support
their restoration. This is not equal division of a joint award.

`verified_source_link` records an audited recipient match. `verified_source_allocation`
records a source-stated allocation but not independent person identification;
this covers the eight generic indexed records and two estates. Estate amounts
are not amounts for each heir. Internal evidence and names are not distributed.

## Five retained target issues — no fabricated replacement

| HUDOC case ID | Retained historical label | Remaining issue |
|---|---:|---|
| 001-95089 | €2,000 | Each-beneficiary sum combines NPD and costs. The supported aggregate is 109 × €2,000 = €218,000 **mixed-head**, not separable pure NPD; €138,000 wrongly used 69 applications as people. Historical €2,000 is not a verified case-total NPD. |
| 001-195534 | €2,000 | Per-person NPD/costs mixture with domestic-payment deductions; pure-NPD case total unverified. |
| 001-71339 | €2,300 | Global NPD/costs mixture; pure-NPD portion unverified. |
| 001-71742 | €2,440 | Reasoning/operative head and award-unit conflict; complete total unverified. |
| 001-228159 | €0 | Confirmed positive €11,700 NPD component disproves the historical zero, but does not establish the complete case total. The retained zero is a flagged historical label, not a verified zero award. |

All five cases and their 24 retained applicant rows remain. Their `target_status`
identifies mixed or unresolved labels; applicant amounts remain unknown.
`y_binary` is derived from the retained benchmark label and is not a validated
legal zero/nonzero finding for 001-228159. These cases remain a substantive
paper-alignment limitation, not a resolved amount correction.

## Publication schema

- One current release; no v1.0/v1.1 directories or separate cohorts.
- Original HUDOC `case_id` restored in both tables. Private custom-ID/name
  mappings are not shipped. Applicant IDs remain stable source-record keys.
- `applicant_name` is `[MASKED]` throughout. Original nationality text restored;
  `nationality_scope` separately retains single/multiple/stateless/unknown.
- Full `judgment_date` and `hudoc_application_count` included so chronology and
  the metadata-based Challenging rule can be checked without a private mapping.
- `npd_award_scope` distinguishes individual/estate units. No raw names, claim
  narratives, judgments, audit ledgers or ratio/one-hot predictor columns added.
- Previously corrected readable importance labels and missing-value handling
  retained. No case re-splitting, de-duplication, exclusions or invented rows.
- The historical `num_applicants` proxy is retained, not newly certified. It
  differs from retained applicant-row count in 50 cases, and may differ greatly
  from people in the judgment (e.g. 001-95089). The 44,581 rows are source units,
  not a complete verified population of individuals.

## Code and documentation

- New public-table experiment bridge removes the need for private encoded
  matrices for documented new runs; preprocessing occurs within code only.
- Added source/extraction orchestration and explicit reviewed two-table export.
- Hardened missing cross-source validation evidence and removed accidentally
  injected publication boilerplate from runtime prompts.
- Corrected zero-shot instructions that invited claim-side information.
- Added offline regression tests, current commands, a complete field dictionary,
  a file tree, content hashes and an explicit paper-reproduction boundary.

See `docs/EXTRACTION.md`, `docs/EXPERIMENTS.md` and `docs/VALIDATION.md` for exact
coverage. None of these repairs recreates unavailable historical run artifacts.

## Repository synchronization

The current trees replace obsolete encoded tables, split-file duplicates,
unbound generated priors and the old two-version release notes with this one
coherent package. Prior files remain recoverable from repository history.
Name masking, HUDOC IDs, the two CSVs and all research source code match the
reviewed preview. Only publication wording, inventory and manifest metadata
were adjusted for release. Repository visibility and existing tags are unchanged.

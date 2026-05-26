# ECtHR-NPD Public Dataset Release 20260526

This package contains the public dataset artifacts for ECtHR-NPD, a
benchmark for predicting case-level Article 41
non-pecuniary damage awards at the European Court of Human Rights.

## Main Dataset

- `data/ecthr_npd_cases.csv`: canonical public case-level table
  with public HUDOC identifiers, validated targets, chronological
  split/view tags, leakage-audited non-compensation structured
  annotations, and external macroeconomic covariates.
- `splits/case_index.csv`: split and diagnostic-view membership.
- `model_inputs/structured_tree/`: optional model-ready structured
  feature matrices for reproducing tree baselines. These are not the
  main dataset release.
- `model_inputs/external_factors/economic_covariates.csv`: respondent-
  state/year external economic covariates available to all model
  families under the strict input contract.
- `INPUT_CONTRACT.md`: common model-input boundary for structured,
  encoder, prompted, and agentic conditions.

## Counts

| split | n |
| --- | ---: |
| train | 10,217 |
| validation | 1,461 |
| test | 2,897 |
| total | 14,575 |

Within the test split, `test_view` partitions cases into ID (1,000) and
OOD (1,897). `test_challenging_view` is the overlapping Challenging
diagnostic view with 699 test cases.

## Exclusions

This public package does not include raw HUDOC judgment text, redacted
facts JSONL, Article 41 text, operative clauses, appendix award tables,
claim fields, award-audit tables, per-applicant award allocations,
applicant names, case names, local paths, model predictions, or provider
traces.

Text inputs reported in the paper are reproducible from HUDOC through
the accompanying redaction and evaluation code, subject to ECtHR terms,
but raw or redacted judgment text is not redistributed here.

## Use

Use the strict model inputs under `model_inputs/` for reproducible
baselines. Targets are provided for supervised training and evaluation
only; they are not model inputs.

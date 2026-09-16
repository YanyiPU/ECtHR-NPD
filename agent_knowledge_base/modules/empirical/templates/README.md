# Empirical Prior Templates

These legacy templates illustrate proposed wire formats for train-derived
priors. They do not define the current runtime v2 prior-manifest contract;
see the [empirical-prior README](../README.md).

This folder contains schemas and examples only. Actual prior files should be
generated offline from the training split.

## Recommended Prior Families

- `global_priors`
- `article_priors`
- `article_state_priors`
- `article_state_limb_priors`
- `zero_rate_priors`
- `zero_reason_priors`

## Build Rule

Do not populate these priors by hand. Build them only from the train split to
avoid contamination.

The public train-target schema does not provide `y_source` or a zero-reason
taxonomy. A zero-reason prior therefore requires an authorised supplemental
train-label source with documented provenance; it cannot be reconstructed from
the released `y_amount_eur` / `y_binary` targets alone.

## Recommended Statistics

- `sample_count`
- `zero_rate`
- `median`
- `iqr`
- `p10`
- `p90`

The legacy suggested zero-reason categories below are not the paper's current
six-category target taxonomy and must not be silently used to rebuild those
labels. They remain here only as historical template documentation:

- `no_claim`
- `finding_sufficient`
- `claim_rejected_procedural`
- `unclear_review_needed`

Current adjudicated new-batch targets require `no_claim`, `unsubstantiated`,
`rule_60_non_compliance`, `domestic_award_covers`,
`applicant_deceased_no_heir`, or `finding_sufficient`, with source evidence.
See the [export contract](../../../../extraction_pipeline/EXPORT_TABULAR.md).

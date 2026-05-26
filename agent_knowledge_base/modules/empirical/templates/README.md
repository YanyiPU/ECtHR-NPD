# Empirical Prior Templates

These templates define the expected wire format for train-split-derived priors.

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

## Recommended Statistics

- `sample_count`
- `zero_rate`
- `median`
- `iqr`
- `p10`
- `p90`

For zero-reason priors, store frequency over the controlled taxonomy:

- `no_claim`
- `finding_sufficient`
- `claim_rejected_procedural`
- `unclear_review_needed`

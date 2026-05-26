# Mode Contract

## Target Input Mode: `award_redacted_full_info`

Base target input:

- the same standard input row used by zeroshot prompting:
  - `combined_input_text`
  - `oracle_violated_articles`
  - optional component fields such as `introduction_text`, `procedure_text`,
    `facts_text`, and `safe_appendix_text`

Allowed attached context:

- extraction sidecars located by target `itemid`, after target non-pecuniary
  award redaction
- Article 41 claim-side information, including non-pecuniary claim state and
  claimed amount
- government response and claim contestation
- metadata such as respondent state, judgment date, formation, and importance
- extracted factors such as violation type, subtype, duration, applicant count,
  vulnerability, domestic remedies, and remedial measures
- shared article modules and policy modules
- train-only empirical priors

Blocked target-case inputs:

- final non-pecuniary award amount
- direct final non-pecuniary award fields such as:
  - `safe_non_pec_eur`
  - `award_eur`
  - `award_non_pec_*`
  - `raw_extractor_non_pec_*`
  - `fx_fill_non_pec_*`
  - target-derived binary or zero-award labels
  - target per-applicant non-pecuniary allocation totals
- final target judgment spans replaced by
  `[TARGET_NON_PECUNIARY_AWARD_REDACTED]`

## Experiment Settings

### `zero_shot`

- no retrieved reference cases
- `retrieval_trace = []`
- shared KB modules and train-only empirical priors may still be used

### `few_shot`

- controller supplies retrieved train reference cases
- every reference case must satisfy
  `retrieved_judgment_date < target_judgment_date`
- reference case non-pecuniary awards may be used because they are not the target
  label
- `retrieval_trace` must be non-empty when references are available

## Leakage Tiers

- `award_redacted_full_info`
  allowed in the v3 target setting after target non-pecuniary award redaction
- `article41_structured`
  allowed in v3 because claims and Article 41 reasoning are not blocked unless
  they directly disclose the target final non-pecuniary award
- `forbidden`
  target final non-pecuniary award labels and direct target-award derivatives

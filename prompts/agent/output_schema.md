# Output Schema

## Final Prediction Object

The final model-visible prediction is pure regression. Return exactly one
non-negative case-level EUR amount:

```json
{
  "award_eur": 7500.0
}
```

## Requirements

- `award_eur` is always required.
- `award_eur` must be numeric and non-negative.
- `0` is a valid numeric prediction.
- Do not output any additional field for zero/non-zero status, classification,
  zero-award reason, or a derived label.
- The amount is the total case-level non-pecuniary damages award across all
  applicants, not a per-applicant amount unless the case has only one applicant.

## Input Contract

- The base target input is the same standard `combined_input_text` used by the
  earlier zeroshot prompting experiment.
- Target extraction sidecars are attached by `itemid` after target
  non-pecuniary award redaction.
- Redaction markers are not evidence for either a positive or zero award.

## Explanation Rules

- Reason internally about zero-award pathways, article severity, applicant
  aggregation, cross-article overlap, claim context, priors, and calibration.
- Do not expose the internal reasoning in the final JSON.
- Do not use test-set label distribution or quotas to choose 0.

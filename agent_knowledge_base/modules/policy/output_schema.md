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

- The public default is `strict_react`: strictly redacted standard input,
  oracle violated articles, safe metadata, non-claim/non-award extracted hints,
  train-only priors, and temporally prior train-reference anchors only.
- The strict packet must not contain target Article 41 / just-satisfaction
  text, target claims or claimed amounts, operative clauses, target outcome or
  zero-reason fields, target labels, or label provenance.
- Broader target-input contracts are restricted diagnostics only and must not
  be represented as strict-input or submitted-paper reproductions.
- Redaction markers are not evidence for either a positive or zero award.

## Explanation Rules

- Reason internally about permitted facts, article severity, applicant
  aggregation, cross-article overlap, train-only priors, and calibration. Do
  not infer blocked claim or Article 41 information from its absence.
- Do not expose the internal reasoning in the final JSON.
- Do not use test-set label distribution or quotas to choose 0.

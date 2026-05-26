# Zero-Award Rules

An ECtHR violation does not automatically imply a positive NPD award.

## Main Zero-Award Pathways

### 1. No claim by the applicant

Typical effect:

- NPD remains `0`

Notes:

- do not infer a claim from suffering alone
- in v3, claim-layer information may be available; use observed claim state
  rather than absence of unredacted award text

### 2. Finding of violation is sufficient

Typical effect:

- NPD remains `0`

Notes:

- frequent in some Article 6 settings
- sometimes linked to short delay, minor procedural harm, retrial, or other
  non-monetary redress

### 3. Procedural rejection of the claim

Typical effect:

- NPD remains `0`

Notes:

- late submission
- non-compliance with formal claim requirements
- other procedural reasons for rejecting the monetary head

### 4. Unclear but plausibly zero

Typical effect:

- use `unclear_review_needed`

Notes:

- do not default to a positive amount when the zero path is plausible but the
  evidence is thin
- do not use the redaction marker itself as evidence for either zero or positive
  award

## Regression Reminder

Zero-award reasoning is legal context for continuous amount calibration. Do not
output a separate binary decision. If the visible facts and legal context support
no compensable non-pecuniary amount, set the single regression output
`award_eur` to `0`.

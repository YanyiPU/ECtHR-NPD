# Claim Rules

Claim-side information is allowed in the v3 award-redacted setting because the
target final non-pecuniary award label is separately redacted.

## Role

Claim information affects:

- zero-award plausibility
- ceiling logic
- whether a zero award is legally plausible
- how strongly the Court's Article 41 reasoning supports an amount

It should not replace legal reasoning, severity assessment, empirical priors, or
few-shot reference comparison.

## Main Cases

### No claim

- strong support for calibrating the continuous amount to `0`
- must be based on observed claim-layer information, not on absence of text after
  redaction

### Specific claim amount

- does not force a positive award
- if the case is positive, the claim can act as a ceiling or scale signal
- claimed amount is not the target award label

### Leave it to the Court

- means the claim ceiling is effectively open
- does not itself imply a positive award

### Procedurally defective claim

- supports calibrating the continuous amount to `0`

## Caution

Do not treat `[TARGET_NON_PECUNIARY_AWARD_REDACTED]` as evidence of the hidden
amount. The marker only indicates that a target award disposition span was
removed.

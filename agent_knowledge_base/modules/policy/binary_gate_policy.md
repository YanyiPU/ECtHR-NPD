# Deprecated Binary Gate Policy

This module is retained only for archive compatibility. It must not be selected
for current pure-regression experiments.

Current policy:

- final visible output is `{"award_eur": <non-negative number>}`
- `0` is a valid continuous regression value
- zero-award reasons are legal context for amount calibration, not a separate
  classifier output
- redaction markers are not evidence for either a positive or zero award

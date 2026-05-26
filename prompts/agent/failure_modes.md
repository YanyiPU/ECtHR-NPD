# Failure Modes

## FM1. Positive-Amount Bias On Zero Cases

Symptom:

- The model predicts a small positive amount whenever a violation exists.

Mitigation:

- treat `0` as a valid continuous regression value
- remind the model that violation alone does not imply positive compensation
- reason internally over zero-award pathways without outputting a classifier

## FM2. Violation -> Award Shortcut

Symptom:

- The model treats a Convention violation as sufficient for positive NPD.

Mitigation:

- use `zero_award_rules`
- use `finding_sufficient_guidance`
- keep zero reasoning inside continuous amount calibration

## FM3. Numeric Drift

Symptom:

- The model produces plausible legal reasoning but poorly calibrated EUR values.

Mitigation:

- use empirical priors
- use fallback by sample size
- make amount calibration explicit in internal reasoning

## FM4. Article Overlap Double Counting

Symptom:

- Multi-article cases are handled by naive summation.

Mitigation:

- use `cross_article_synthesis`
- require overlap-aware internal reasoning
- calibrate one case-level total rather than summing every article mechanically

## FM5. Overconfident Micro-Bucket Use

Symptom:

- The system over-trusts tiny buckets with weak support.

Mitigation:

- expose sample count
- expose fallback use
- prefer hierarchical fallback

## FM6. Applicant Count Linear Multiplication

Symptom:

- The model estimates a plausible per-applicant amount and mechanically
  multiplies it by the number of applicants, inflating mass, joined, derivative,
  repetitive, or global-award cases.

Mitigation:

- calibrate one final case-level total
- treat applicant count as context, not a default multiplier
- distinguish individualized harm from shared, derivative, overlapping, or
  weakly visible harm
- use safe appendix rows as applicant-structure context only; redacted award
  spans and award-derived fields remain unusable

## FM6. Redaction-Marker Leakage

Symptom:

- The model treats `[TARGET_NON_PECUNIARY_AWARD_REDACTED]` as evidence of a
  hidden amount, zero award, or positive award.

Mitigation:

- require internal amount reasoning to cite allowed facts, claims, priors, or
  few-shot references
- do not cite the redaction marker as substantive evidence
- validate that direct target award fields are absent from target inputs

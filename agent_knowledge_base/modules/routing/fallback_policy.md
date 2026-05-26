# Fallback Policy

## Unsupported Articles

If no dedicated article module exists:

- use `generic_article_fallback`
- do not invent article-specific numeric rules
- rely on broader empirical priors

## Weak Empirical Support

If the selected bucket has insufficient support:

- back off to the next broader bucket
- log the fallback in `calibration_sources_used`

## Calibration Uncertainty

If zero-award plausibility or positive-amount calibration is highly uncertain:

- avoid a confident positive amount unless strong support exists
- remember that `0` is a valid continuous regression output

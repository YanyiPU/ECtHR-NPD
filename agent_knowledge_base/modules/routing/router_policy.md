# Router Policy

The router constructs a bounded candidate set. The model may choose which
provided evidence is useful, but file access and reference-case retrieval remain
controller-owned for reproducibility and leakage control.

## Mandatory Modules

Always load:

- `system_policy`
- `mode_contracts`
- `output_schema`
- `failure_modes`

## Regression Guidance Modules

Always load:

- `zero_award_rules`
- `finding_sufficient_guidance`
- `claim_rules`

## Article Modules

Load one article module per violated article.

Special routing:

- Article 6 must route through `art6_limb_router` first
- unsupported articles should fall back to `generic_article_fallback`

## Cross-Article Modules

Load `cross_article_synthesis` if:

- more than one violated article exists
- or Article 13 / 14 is present

## Empirical Priors

Select the narrowest valid train-only bucket with sufficient support:

- `article × state × limb`
- `article × limb`
- `article × state`
- `article`
- `global`

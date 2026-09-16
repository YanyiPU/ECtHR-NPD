# External Factors

The readable public case table contains `gdp_per_capita_current_usd` and
`gdp_constant_2015_usd`, matched to respondent state and judgment year.
No separate case-linked economic CSV is distributed in this directory.
Experiment preparation computes their log1p transforms at runtime and includes
only those two economic predictors in the strict serialization.

Where the controller uses external factors, its controller-owned projection
allows only the GDP fields. Prepared `strict_react` inputs carry the transforms
in the strict serialization. This module does not expose or permit
joins on HUDOC URLs, split membership, test-view membership, or
challenging-view membership.

This module contains external covariates only. It does not contain
targets, claim amounts, Article 41 text, operative clauses, direct award
snippets, predictions, provider traces, split/view tags, or HUDOC URLs.

# External Economic Covariates

`economic_covariates.csv` contains respondent-state/year macroeconomic
covariates used by the paper's input contract.

These fields are safe external factors rather than compensation outcomes:

- `gdp_per_capita_current_usd`
- `gdp_constant_2015_usd`
- `gdp_per_capita_log1p`
- `gdp_constant_2015_log1p`

The structured-tree feature matrices include the log-transformed GDP
columns. Prompted and agentic conditions can serialize or retrieve the
same table by `itemid` or by respondent-state/year. This table does not
contain Article 41 text, operative clauses, claim amounts, award
snippets, target values, predictions, or target-derived fields.

# Shared Prediction-Input Policy

All model families follow the shared prediction-input policy.

Allowed inputs:

- Case metadata: respondent state/country code, judgment date-derived
  year/month, HUDOC decision body, court formation, case importance,
  separate-opinion flag, representation flag, applicant counts, and
  violation metadata that excludes award-related material.
- Violated articles: the violated-article list, article count, and
  article-indicator columns.
- Case facts: redacted case facts that exclude Article 41 award-related
  material when reconstructed from
  HUDOC, or serialized structured fact inputs in this public release
  such as applicant aggregates, violation-type aggregates, and violation
  duration features.
- External factors: respondent-state/year economic covariates in
  `model_inputs/external_factors/economic_covariates.csv`; tree-ready
  matrices include the log GDP fields.

Excluded inputs:

- Article 41 / Article 50 compensation text
- operative clauses and appendix award tables
- claimed amounts and claim-state fields
- direct award snippets, award-side fields, and zero-award rationales
- raw extractor outputs, repair/debug fields, and target-derived fields
- target labels and split/view labels as model features
- applicant names, case names, app numbers, ECLI, or local provenance

Model mapping:

- Tree baselines read `model_inputs/structured_tree/features/*.csv`,
  which serialize the allowed input groups.
- Encoder settings describe text/serialized-input runs that must follow
  the same policy; raw judgment text is not redistributed here.
- Prompt templates and the ReAct agent consume the same case metadata,
  violated articles, case facts or serialized inputs, and external
  factors. Empirical priors are train-split resources for the agent
  condition only.

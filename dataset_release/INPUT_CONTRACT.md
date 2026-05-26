# Model Input Contract

All model families use the same strict input boundary.

Allowed inputs:

- Safe metadata: respondent state/country code, judgment date-derived
  year/month, HUDOC decision body, court formation, case importance,
  separate-opinion flag, representation flag, applicant counts, and
  non-compensation violation metadata.
- Violated articles: the violated-article list, article count, and
  article-indicator columns.
- Case facts: redacted non-compensation facts when reconstructed from
  HUDOC, or serialized structured fact inputs in this public release
  such as applicant aggregates, violation-type aggregates, and violation
  duration features.
- External factors: respondent-state/year economic covariates in
  `model_inputs/external_factors/economic_covariates.csv`; tree-ready
  matrices include the log GDP fields.

Excluded strict inputs:

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
  the same boundary; raw judgment text is not redistributed here.
- Prompt templates and the ReAct agent consume the same safe metadata,
  violated articles, case facts or serialized inputs, and external
  factors. Empirical priors are train-split resources for the agent
  condition only.

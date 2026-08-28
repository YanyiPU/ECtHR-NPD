# Shared Prediction-Input Policy

All model families in this bundle follow the shared prediction-input policy:

- Case metadata: respondent state/country code, judgment date-derived
  year/month, HUDOC decision body, court formation where available,
  case importance, separate-opinion flag, representation flag,
  applicant counts, and violation metadata that excludes award-related material.
- Violated articles: violated-article list/count or article-indicator
  features.
- Case facts: redacted case facts that exclude Article 41 award-related
  material for text/agent settings
  when reconstructed from HUDOC, or serialized structured inputs in this
  public release such as applicant aggregates, violation-type aggregates,
  and violation-duration features.
- External factors: respondent-state/year economic covariates in
  `dataset_release/model_inputs/external_factors/economic_covariates.csv`;
  tree-ready matrices include `gdp_per_capita_log1p` and
  `gdp_constant_2015_log1p`.

Excluded inputs:

- Article 41 / Article 50 compensation text
- operative clauses and appendix award tables
- claimed amounts and claim-state fields
- direct award snippets, award-side fields, and zero-award rationales
- target labels, prediction files, split/view labels as model features,
  and target-derived fields
- applicant names, case names, app numbers, ECLI, or local paths

The CatBoost/XGBoost/LightGBM tree baselines consume the serialized
structured version of this contract via `baselines.data.data_loader` and
use direct log-scale pure regression with no separate zero/positive
stage. The BM25, BGE-M3 dense, and BGE-M3 sparse retrieval baselines
consume user-supplied procedure/facts plus case metadata and apply
train-only temporal retrieval. Encoder baselines use the same
policy either from user-supplied text inputs or from serialized public
inputs. Prompted and agentic conditions must use the same policy even when
their inputs are prompt serializations rather than CSV
feature matrices.

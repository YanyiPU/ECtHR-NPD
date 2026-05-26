# Train-Only Empirical Priors

The actual empirical knowledge base for the agent condition is the set of
train-derived CSV tables in this directory:

- `article_award_distribution_train.csv`: article-level award priors.
- `country_award_distribution_train.csv`: respondent-state award priors.
- `article_country_award_distribution_train.csv`: article-by-state priors.
- `article_single_violation_stats_train.csv`: legacy article-level
  compatibility table used by the older controller path.

The `templates/` subdirectory intentionally contains only schemas and
examples for prior wire formats. It is not expected to contain the full
train priors. No test labels, prediction outputs, traces, or provider
metadata are included in these empirical priors.

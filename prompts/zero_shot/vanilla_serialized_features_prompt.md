# Vanilla Serialized-Features Regression

## System Prompt

```text
You are a legal professional predicting European Court of Human Rights non-pecuniary damages under Article 41.
```

## User Prompt Template

```text
Predict the total non-pecuniary damages award in EUR for the case described below.

This is a continuous regression task: output one case-level EUR amount.

Use only:
- the serialized extracted features below
- the provided violated articles below

The features include case metadata, merits findings, non-identifying applicant summaries, and macro-economic context. Claims, Article 41 reasoning, operative provisions, allocation fields, and target-derived fields are excluded.

The serialized features are the case facts available for this run. The absence of raw judgment narrative or raw Article 41 text is not a reason to predict 0 EUR. Make a best calibrated estimate from the structured fields.

Adjust the amount using the visible case-specific features: violation type, duration, applicant structure, respondent context, and merits descriptors.

The award can be 0 EUR or a positive integer EUR amount. Base the prediction only on the visible features. Do not use dataset prevalence, quotas, external retrieval, named-case recall, or hidden award cues. Do not infer claim defects from absent Article 41 material. Predict one case-level total without summing amounts across Articles or multiplying by applicant count.

`award_eur` must be an integer amount in original EUR scale. It is not log space, log1p, thousands, a normalized score, or a probability.

Provided violated articles:
{provided_violated_articles}

Case features:
{combined_input_text}

Respond with a JSON object: {{"award_eur": <integer EUR amount>}}
```

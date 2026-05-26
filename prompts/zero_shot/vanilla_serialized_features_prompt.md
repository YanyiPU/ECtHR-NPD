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

The features include case metadata, violation descriptors, applicant structure, reasoning descriptors, macro-economic context, and claim-side fields. Read the field names and values carefully.

The serialized features are the case facts available for this run. The absence of raw judgment narrative or raw Article 41 text is not a reason to predict 0 EUR. Make a best calibrated estimate from the structured fields.

Adjust the amount using the extracted case-specific features: violation type, severity, duration, number and structure of applicants, applicant vulnerability, respondent context, reasoning descriptors, and claim-side fields.

The award can be 0 EUR or a positive integer EUR amount. As a rough benchmark/training-distribution prior, about one third of cases have a 0 EUR non-pecuniary damages award and the remaining cases have positive integer awards. Use this only as calibration; base the prediction on the serialized case-specific features.

`award_eur` must be an integer amount in original EUR scale. It is not log space, log1p, thousands, a normalized score, or a probability.

Provided violated articles:
{provided_violated_articles}

Case features:
{combined_input_text}

Respond with a JSON object: {{"award_eur": <integer EUR amount>}}
```

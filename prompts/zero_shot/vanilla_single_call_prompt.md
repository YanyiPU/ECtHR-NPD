# Vanilla Single-Call Regression

## System Prompt

```text
You are a legal professional predicting European Court of Human Rights non-pecuniary damages under Article 41.
```

## User Prompt Template

```text
Below is a standard European Court of Human Rights case input and the violated articles provided for this benchmark case.

Predict the total non-pecuniary damages award in EUR as one case-level continuous regression amount.

Use only:
- the standard case input below
- the provided violated articles below

The correct non-pecuniary damages award may be 0 EUR. Do not assume that a finding of violation necessarily leads to a positive non-pecuniary award.

`award_eur` must be a non-negative integer amount in original EUR scale. It is not log space, log1p, thousands, a normalized score, or a probability.

Provided violated articles:
{provided_violated_articles}

Standard case input:
{combined_input_text}

Respond with a JSON object containing only the field "award_eur".
```

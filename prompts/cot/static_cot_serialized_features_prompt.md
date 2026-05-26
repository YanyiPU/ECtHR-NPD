# Static CoT Strict Serialized-Features Single-Stage Regression

## System Prompt

```text
You are a legal expert assessing European Court of Human Rights non-pecuniary damages under Article 41 from strict serialized case features. Make a best estimate. Reason concisely. Do not refuse.
```

## User Prompt Template

```text
Below are strict serialized extracted features and the violated articles for this benchmark case. Predict the total non-pecuniary damages award in EUR as one case-level continuous regression amount.

ALLOWED INPUTS
- the strict serialized extracted features below
- the provided violated articles below
- external factors only when they appear inside the serialized features, such as respondent country, judgment date fields, and GDP fields

The serialized extracted features are the complete case input for this run and replace raw legal text. Do not complain that raw judgment narrative, raw Article 41 text, raw operative clauses, claim text, or compensation reasoning text are absent.

STRICT INPUT POLICY
- This is a strict A41-free, target-hidden serialized feature run intended to be comparable with strict text runs.
- Article 41 text, operative clauses, direct award snippets, claimed amounts, claim state, no-claim/court-to-determine status, costs-and-expenses reasoning, pecuniary-damage reasoning, default-interest language, domestic-redress discussion, Article 46 remedial-measure discussion, Article 41 precedent lists, and reasoning-layer compensation hints are not provided.
- The absence of claim fields or reasoning fields is a design choice, not evidence of no claim, weak claim, Rule 60 non-compliance, finding sufficient, or zero award.
- Do not infer Article 41 content or claim-side defects from silence.
- Use only the visible structured features and provided violated articles.

PROHIBITED
- external retrieval, precedent lookup, examples, web search, tools
- inferring award amounts from named cases, citations, application numbers, or case IDs that may appear in the input
- using hidden target labels, dataset prevalence, quotas, or base rates to choose the amount
- treating absent Article 41, absent claim amount, absent claim state, or absent reasoning fields as a zero-award signal

TASK FRAMING
This is a single-stage regression baseline. Output one numeric amount, `award_eur`. Do not split the task into a zero/non-zero classifier and a regressor. Do not produce a binary award decision in any field.

OUTPUT SCALE
`award_eur` is an integer in original EUR scale. It is not log space, log1p, thousands, a normalized score, or a probability. Awards can range from 0 to over 1,000,000 EUR. Output exactly 0 when the calibrated amount is zero; otherwise output an integer of at least 500 EUR. Do not default to 500, 1,000, 10,000, or a generic median without case-specific calibration.

ZERO AS A CONTINUOUS VALUE
0 EUR is a valid regression output, not a separate decision; treat it as the lowest-magnitude end of the continuous range. Lean toward a substantially lower or 0 EUR amount only when the visible serialized features support weak individualized non-pecuniary harm, such as a narrow procedural or technical violation, low-severity article pattern, minimal or unclear individualized applicant impact, shared or derivative applicant structure, or mainly systemic/procedural harm without visible personal burden. Do not infer any zero reason from absent Article 41 or absent claim fields. A violation finding, serious article, or multiple violations can support a positive amount but does not require one.

POSITIVE AWARD BURDEN
A violation finding alone does not justify a positive non-pecuniary award. Before selecting a positive amount, compare 0 EUR with a positive award using the serialized case features. A positive award requires visible support in the structured fields, such as serious article pattern, substantive severity, detention or liberty impact, death or injury, ill-treatment, child or vulnerable applicant cues, serious family or private-life burden, prolonged proceedings, enforcement failure, property interference with personal burden, discrimination with individualized burden, or comparable concrete distress.

EXTERNAL FACTORS
External factors in the serialized features are allowed decision-time covariates. Use respondent country, judgment year/month, and GDP fields only as contextual calibration signals. Do not use them as a direct multiplier, exchange-rate formula, or substitute for visible harm. Do not use external factors to infer hidden Article 41 text, claims, or prior awards.

ARTICLE CODE HANDLING
The provided violated article list is already the fixed list of Convention or Protocol provisions for this case. Use the codes as labels; do not spend reasoning tokens second-guessing, decoding, or validating unfamiliar codes. Common labels include: Art 2 life, Art 3 ill-treatment, Art 5 liberty, Art 6 fair trial, Art 8 private/family life, Art 9 religion, Art 10 expression, Art 11 assembly/association, Art 13 remedy, Art 14 discrimination, P1-1 property, P1-2 education, P1-3 elections, P4-2 movement, P4-4 collective expulsion, P7-1 expulsion safeguards, P7-2 criminal appeal, P7-4 ne bis in idem, and P12-1 general discrimination. If a code is unfamiliar, keep the code unchanged and rely on the visible serialized features and harm pattern.

MULTI-VIOLATION
If multiple violated articles are provided, identify the dominant article from the serialized harm pattern. A rough severity ordering can be used only as a weak tie-breaker when the visible harm pattern is ambiguous: Art 2 > Art 3 > Art 5 > Art 6 > Art 8 > Art 10 > Art 11 > P1-1 > Art 13 > Art 14. Visible harm and serialized case features override this ordering. The total reflects all violations jointly; do not sum per-article amounts.

MULTI-APPLICANT
Predict one case-level total. Do not multiply linearly by applicant count. If the serialized features show shared, derivative, joined, or appendix-listed applicants without separate individualized harm evidence, anchor on a single case-level amount with at most a modest upward adjustment. If the features show distinct individualized harm across applicants, choose a higher case-level band qualitatively, but only to the extent the input supports.

Never calculate or state a per-applicant rate times applicant count. Even when multiple applicants have distinct harm, first choose one total case-level EUR band for the joined case, then select one point in that band.

For very large joined applications or long applicant lists, do not enumerate applicants, do not reason row-by-row, and do not calculate a per-applicant rate times a headcount. Treat the case as a collective case-level award unless the visible serialized features clearly identify separate individualized harm. Joined applications, many applicants, or large per-applicant structure do not by themselves justify a high aggregate award; a large joined case can still calibrate to 0 EUR. Use the shared structured pattern, dominant violation, and degree of individualization to choose one case-level band, then output promptly.

REASONING FIELDS
Fill `continuous_reasoning` with the five audit fields concisely. Each field should be no more than 60 tokens unless the case materially requires more. Keep content non-overlapping across fields.

- `case_factors`: provided violated articles, dominant violation, serialized case metadata, external factors, applicant structure, and visible harm cues relevant to NPD.
- `zero_award_assessment`: visible serialized reasons for or against 0 EUR. If selecting a positive amount, state why 0 EUR is rejected; if selecting 0, state the visible reason for 0. Do not write a yes/no classifier. Do not mention absent claim fields as a zero reason.
- `damages_assessment`: strength of compensable non-pecuniary harm, gravity of the violation, individualized impact, duration/structure cues if visible, and external-factor context if relevant.
- `applicant_aggregation`: how the case-level total treats multiple applicants, joined applications, and any visible individualization.
- `amount_calibration`: a concrete EUR band in the form "X-Y EUR" with numeric endpoints chosen for this case, and the selected point within that band, with one line of justification anchored in the gravity assessment. If selecting 0, write "0-0 EUR; select 0 EUR". If selecting a positive amount while 0 is plausible, compare 0 EUR against the positive band and state why the positive band is better calibrated.

Then write `final_award_reasoning` as one short paragraph of no more than 80 tokens explaining why the chosen `award_eur` is the calibrated amount, including why 0 is appropriate or why 0 is rejected and the positive value sits at that specific level within the calibration band.

Then output `award_eur`.

Do not repeat uncertainty or calibration sentences. If several amounts are plausible, choose the best calibrated integer within the stated band and return the JSON object.

PROVIDED VIOLATED ARTICLES
{provided_violated_articles}

STRICT SERIALIZED EXTRACTED FEATURES
{combined_input_text}

OUTPUT SCHEMA
{output_schema}

Return only one valid JSON object matching `output_schema` exactly. No markdown, no code fences, no commentary, no extra fields.
```

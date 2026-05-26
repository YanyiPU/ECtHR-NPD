# Static CoT Single-Stage Regression

## System Prompt

```text
You are a legal expert assessing European Court of Human Rights non-pecuniary damages under Article 41. Make a best estimate. Reason concisely. Do not refuse.
```

## User Prompt Template

```text
Below is a standard ECtHR case input and the violated articles for this benchmark case. Predict the total non-pecuniary damages award in EUR as one case-level continuous regression amount.

ALLOWED INPUTS
- the standard case input below
- the provided violated articles below

PROHIBITED
- external retrieval, precedent lookup, examples, web search, tools
- inferring award amounts from named cases, citations, or application numbers that may appear in the input
- using dataset prevalence, quotas, or base rates to choose the amount

TASK FRAMING
This is a single-stage regression baseline. Output one numeric amount, `award_eur`. Do not split the task into a zero/non-zero classifier and a regressor. Do not produce a binary award decision in any field.

OUTPUT SCALE
`award_eur` is an integer in original EUR scale. It is not log space, log1p, thousands, a normalized score, or a probability. Awards can range from 0 to over 1,000,000 EUR. Output exactly 0 when the calibrated amount is zero; otherwise output an integer of at least 500 EUR. Do not default to 500, 1,000, 10,000, or a generic median without case-specific calibration.

ZERO AS A CONTINUOUS VALUE
0 EUR is a valid regression output, not a separate decision; treat it as the lowest-magnitude end of the continuous range. Lean toward a substantially lower or 0 EUR amount only when the visible facts in the standard input support reasons such as: a finding of violation operating as sufficient just satisfaction; weak or absent compensable non-pecuniary harm; no visible causal link between the violation and individualized non-pecuniary harm; harm already addressed by visible domestic redress, compensation, sentence reduction, reopening, or other remedial measures; a narrow procedural or technical violation without visible individualized distress, prolonged delay, detention, or other substantial practical impact; mainly pecuniary loss or costs; weakly individualized distress; the standard input visibly states that the applicant is deceased and no heir or eligible relative continues the relevant claim; the standard input visibly states lack of victim status for the relevant harm; or harm already captured by another finding. Do not infer any of these from silence. A finding of violation, visible distress, or serious facts can support a positive amount but does not require one.

POSITIVE AWARD BURDEN
A violation finding alone does not justify a positive non-pecuniary award. Before selecting a positive amount, compare 0 EUR with a positive award using the visible case facts. A positive award requires visible individualized non-pecuniary harm such as detention or liberty impact, death or injury, ill-treatment, serious family or private-life burden, prolonged proceedings with distress, enforcement failure, property interference with personal burden, or comparable concrete distress. If the visible facts do not justify individualized non-pecuniary harm, calibrate to 0 EUR.

CLAIM CEILING
In this strict baseline, claim text and claim ceilings are not provided. Do not infer a claim ceiling and do not infer no-claim, late claim, Rule 60 non-compliance, unsubstantiated claim, or other claim defects from absent Article 41 material. Treat claim-side defects as zero-calibration reasons only if they are expressly visible in the allowed input.

ARTICLE CODE HANDLING
The provided violated article list is already the fixed list of Convention or Protocol provisions for this case. Use the codes as labels; do not spend reasoning tokens second-guessing, decoding, or validating unfamiliar codes. Common labels include: Art 2 life, Art 3 ill-treatment, Art 5 liberty, Art 6 fair trial, Art 8 private/family life, Art 9 religion, Art 10 expression, Art 11 assembly/association, Art 13 remedy, Art 14 discrimination, P1-1 property, P1-2 education, P1-3 elections, P4-2 movement, P4-4 collective expulsion, P7-1 expulsion safeguards, P7-2 criminal appeal, P7-4 ne bis in idem, and P12-1 general discrimination. If a code is unfamiliar, keep the code unchanged and rely on the visible facts and harm pattern.

MULTI-VIOLATION
If multiple violated articles are provided, identify the dominant article from the visible harm pattern. A rough severity ordering can be used only as a weak tie-breaker when the visible harm pattern is ambiguous: Art 2 > Art 3 > Art 5 > Art 6 > Art 8 > Art 10 > Art 11 > P1-1 > Art 13 > Art 14. Visible harm and case facts override this ordering. The total reflects all violations jointly; do not sum per-article amounts.

MULTI-APPLICANT
Predict one case-level total. Do not multiply linearly by applicant count. If the input shows shared, derivative, joined, or appendix-listed applicants without separate individualized harm evidence, anchor on a single case-level amount with at most a modest upward adjustment. If the input shows distinct individualized harm across applicants, choose a higher case-level band qualitatively, but only to the extent the input supports.

Never calculate or state a per-applicant rate times applicant count. Even when multiple applicants have distinct harm, first choose one total case-level EUR band for the joined case, then select one point in that band.

For very large joined applications or long appendix tables, do not enumerate applicants, do not reason row-by-row, and do not calculate a per-applicant rate times a headcount. Treat the case as a collective case-level award unless the visible facts clearly identify separate individualized harm. Joined applications, many application numbers, or appendix length do not by themselves justify a high aggregate award; a large joined case can still calibrate to 0 EUR. Use the shared fact pattern, dominant violation, and degree of individualization to choose one case-level band, then output promptly.

REASONING FIELDS
Fill `continuous_reasoning` with the five audit fields concisely. Each field should be no more than 60 tokens unless the case materially requires more. Keep content non-overlapping across fields.

- `case_factors`: provided violated articles, dominant violation, visible harm, duration, and the applicant structure relevant to NPD.
- `zero_award_assessment`: visible reasons for or against 0 EUR. If selecting a positive amount, state why 0 EUR is rejected; if selecting 0, state the visible reason for 0. Do not write a yes/no classifier.
- `damages_assessment`: strength of compensable non-pecuniary harm, gravity of the violation, and any visible just-satisfaction context that pushes the amount upward, downward, or to 0 EUR.
- `applicant_aggregation`: how the case-level total treats multiple applicants, joined applications, and any visible individualization.
- `amount_calibration`: a concrete EUR band in the form "X-Y EUR" with numeric endpoints chosen for this case, and the selected point within that band, with one line of justification anchored in the gravity assessment. If selecting 0, write "0-0 EUR; select 0 EUR". If selecting a positive amount while 0 is plausible, compare 0 EUR against the positive band and state why the positive band is better calibrated.

Then write `final_award_reasoning` as one short paragraph of no more than 80 tokens explaining why the chosen `award_eur` is the calibrated amount, including why 0 is appropriate or why 0 is rejected and the positive value sits at that specific level within the calibration band.

Then output `award_eur`.

Do not repeat uncertainty or calibration sentences. If several amounts are plausible, choose the best calibrated integer within the stated band and return the JSON object.

PROVIDED VIOLATED ARTICLES
{provided_violated_articles}

STANDARD CASE INPUT
{combined_input_text}

OUTPUT SCHEMA
{output_schema}

Return only one valid JSON object matching `output_schema` exactly. No markdown, no code fences, no commentary, no extra fields.
```

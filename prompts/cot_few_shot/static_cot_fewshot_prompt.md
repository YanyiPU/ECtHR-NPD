# Static CoT Few-Shot Target-Strict Regression

## System Prompt

```text
You are a legal expert assessing European Court of Human Rights non-pecuniary damages under Article 41. Use the provided temporally prior train references only as calibration examples. Make one best case-level EUR estimate. Reason concisely. Do not refuse.
```

## User Prompt Template

```text
Below is a target ECtHR case input, provided violated articles, target external factors when available, and temporally prior train few-shot references. Predict the target case's total non-pecuniary damages award in EUR as one case-level continuous regression amount.

LEAKAGE LABEL
target-strict, full-info-train static CoT few-shot ablation

ALLOWED TARGET INPUTS
- the target standard case input below
- the provided violated articles below
- target external factors below when provided, limited to macro/time-value calibration

PROHIBITED TARGET INPUTS
- target raw Article 41 / Article 50 / just-satisfaction text
- target operative clauses or direct award snippets
- target claimed amount, claim state, no-claim, or court-to-determine fields
- target government submissions, finding-sufficient cues, equitable-basis cues, zero-award reason, pecuniary/costs/default-interest/tax reasoning, domestic-redress discussion, Article 46 measures, appended-table references, or Article 41 precedent anchors
- any target-derived award field

TRAIN REFERENCE POLICY
The few-shot references are supervised train cases selected before inference. They may expose train-only full information, including train claim amounts, train award amounts, train zero-award reasons, train Article 41-derived reasoning metadata, government submissions, finding-sufficient or equitable-basis metadata, raw train awards, and train awards adjusted to the target year. This is intentional for this ablation.

Use train references as calibration anchors, not as target facts. Prefer time-value-adjusted reference awards when those values are provided; use raw train awards for provenance. Do not mechanically average the examples. Compare the target's visible facts, violation pattern, applicant structure, respondent country, decision body, and time context against the references.

PROHIBITED MODEL BEHAVIOR
- external retrieval, precedent lookup, examples not provided here, web search, tools
- inferring the target award from target case names, citations, item IDs, application numbers, or your internal memory
- using test-set prevalence, quotas, or base rates to choose the amount
- importing claim-side or Article 41-derived facts from train references into the target case unless those facts are visibly present in the allowed target input

PARAMETERISED TARGET-KNOWLEDGE CHECK
Do not try to identify or recall the target case. However, if you already recognize the target case, target case name, application number, item ID, facts, or recalled award amount from parameterised model knowledge, record that in `parametric_target_knowledge_check`. If you recall a target award or related amount, write the recalled amount there for audit, but do not use it to select `award_eur`. The prediction must be based on the visible target input, provided violated articles, target external factors, and train references only.

TASK FRAMING
This is a single-stage regression baseline. Output one numeric amount, `award_eur`. Do not split the task into a zero/non-zero classifier and a regressor. Do not produce a binary award decision in any field.

OUTPUT SCALE
`award_eur` is an integer in original EUR scale. It is not log space, log1p, thousands, a normalized score, or a probability. Awards can range from 0 to over 1,000,000 EUR. Output exactly 0 when the calibrated amount is zero; otherwise output an integer of at least 500 EUR. Do not default to 500, 1,000, 10,000, or a generic median without case-specific calibration.

ZERO AS A CONTINUOUS VALUE
0 EUR is a valid regression output, not a separate decision; treat it as the lowest-magnitude end of the continuous range. Consider the provided zero-award train analogue. Lean toward a substantially lower or 0 EUR amount only when the target's visible facts support reasons such as: a finding of violation operating as sufficient just satisfaction; weak or absent compensable non-pecuniary harm; no visible causal link between the violation and individualized non-pecuniary harm; harm already addressed by visible domestic redress, compensation, sentence reduction, reopening, or other remedial measures; a narrow procedural or technical violation without visible individualized distress, prolonged delay, detention, or other substantial practical impact; mainly pecuniary loss or costs; weakly individualized distress; the standard input visibly states that the applicant is deceased and no heir or eligible relative continues the relevant claim; the standard input visibly states lack of victim status for the relevant harm; or harm already captured by another finding. Do not infer any of these from silence or from train references alone.

POSITIVE AWARD BURDEN
A violation finding alone does not justify a positive non-pecuniary award. Before selecting a positive amount, compare 0 EUR with a positive award using the visible target case facts and the zero-award reference. A positive award requires visible individualized non-pecuniary harm such as detention or liberty impact, death or injury, ill-treatment, serious family or private-life burden, prolonged proceedings with distress, enforcement failure, property interference with personal burden, or comparable concrete distress.

CLAIM CEILING
The target claim text and target claim ceiling are not provided. Do not infer a target claim ceiling and do not infer no-claim, late claim, Rule 60 non-compliance, unsubstantiated claim, or other target claim defects from absent Article 41 material. Claim ceilings shown in train references are calibration examples only and do not bind the target.

ARTICLE CODE HANDLING
The provided violated article list is already the fixed list of Convention or Protocol provisions for the target case. Use the codes as labels; do not spend reasoning tokens second-guessing, decoding, or validating unfamiliar codes. If a code is unfamiliar, keep the code unchanged and rely on the visible facts, target external context, and train-reference calibration.

MULTI-VIOLATION
If multiple violated articles are provided, identify the dominant article from the visible target harm pattern. The total reflects all violations jointly; do not sum per-article amounts. Train references with exact or overlapping article sets are stronger anchors than references with only broad severity similarity.

MULTI-APPLICANT
Predict one target case-level total. Do not multiply linearly by applicant count. Use train references with similar applicant-count bands as calibration aids, but final prediction remains one case-level amount.

FEW-SHOT CALIBRATION AUDIT
Use the few-shot reference set as follows:
- identify the most comparable positive-award references
- explicitly consider at least one zero-award analogue if provided
- use any high-similarity different-country or different-formation calibration anchor as a robustness check, not as the main anchor when closer same-country examples exist
- prefer adjusted-to-target-year awards for amount calibration when available
- explain which reference item IDs drove the final band in `few_shot_reference_audit`

REASONING FIELDS
Fill `continuous_reasoning` with the five audit fields concisely. Each field should be no more than 70 tokens unless the case materially requires more.

- `case_factors`: provided violated articles, dominant violation, visible target harm, duration, applicant structure, and relevant target external/time context.
- `zero_award_assessment`: visible target reasons for or against 0 EUR, with explicit comparison to the zero-award reference if provided. Do not write a yes/no classifier.
- `damages_assessment`: strength of compensable non-pecuniary harm, gravity of the target violation, and how train references push the amount upward, downward, or to 0 EUR.
- `applicant_aggregation`: how the case-level total treats multiple applicants, joined applications, and visible individualization.
- `amount_calibration`: a concrete EUR band in the form "X-Y EUR" with numeric endpoints chosen for this target case, and the selected point within that band. If selecting 0, write "0-0 EUR; select 0 EUR".

Fill `few_shot_reference_audit` with counts and the reference item IDs actually used for calibration. If no few-shot references are provided, state that explicitly and set counts to 0.

Fill `parametric_target_knowledge_check` honestly. If no recognition or recalled target amount occurs, set recognized fields to false or null as the schema requires.

Then write `final_award_reasoning` as one short paragraph of no more than 90 tokens explaining why the chosen `award_eur` is calibrated, including why 0 is appropriate or why 0 is rejected and the positive value sits at that specific level within the calibration band.

PROVIDED VIOLATED ARTICLES
{provided_violated_articles}

TARGET EXTERNAL FACTORS
{external_factors_context}

TARGET STANDARD CASE INPUT
{combined_input_text}

TRAIN FEW-SHOT REFERENCES
{few_shot_references}

OUTPUT SCHEMA
{output_schema}

Return only one valid JSON object matching `output_schema` exactly. No markdown, no code fences, no commentary, no extra fields.
```

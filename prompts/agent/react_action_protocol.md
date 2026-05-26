# ReAct Action Protocol

This protocol turns the KB from single prompt assembly into a bounded
reasoning/action loop. The controller owns all file access, retrieval,
redaction, and leakage checks. The model may only request one allowed action at
a time.

## Action Object

Every non-final model turn must return exactly one JSON action object:

```json
{
  "thought_summary": "Short reason for the next action, not a private chain-of-thought transcript.",
  "action": "inspect_case",
  "action_input": {}
}
```

The controller executes the action and appends an `observation` to the trace.
The model then receives the updated trace and may request another action.

## Allowed Actions

- `inspect_case`
  - Returns the target case overview and available-information catalog.
  - Under lazy context policy, it does not dump full target payload values.
- `query_target_information`
  - Returns selected target fields by source, path prefix, or field-name
    substring.
  - This is the primary way to let the model decide which target information it
    needs without loading every field into the first observation.
- `search_modules`
  - Returns selected policy, routing, empirical, and article modules.
- `load_module`
  - Returns controller-approved module text for requested module ids.
- `load_relevant_modules`
  - Loads the controller-selected modules in one action.
  - Prefer this default path over separate `search_modules` and `load_module`
    unless a custom subset is needed.
- `resolve_empirical_priors`
  - Returns train-only empirical calibration anchors and fallback trace.
  - Uses the current chronological train split distribution tables in this
    order: article×country, article, country, then global article-weighted
    fallback.
- `retrieve_train_references`
  - Returns temporally prior train references only when `inference_mode` is
    `few_shot`.
  - Uses domain multiple filtering first, then similarity ranking.
  - Returns both positive references and zero / finding-sufficient references
    when available.
  - Retrieved cases expose reference metadata, known train
    `reference_non_pec_eur`, and feature-source catalogs by default.
- `query_reference_features`
  - Returns selected feature-row values for retrieved reference cases.
  - Raw judgment text and award/label-derived feature columns are removed.
  - If `sources` is omitted, the controller defaults to
    `split_case_features`, `reasoning_layer_features`, and
    `applicant_features`.
  - Generated reasoning summaries such as
    `reasoning_layer.award_reasoning_summary` are blocked. They are LLM
    extraction outputs, not source-text quotes.
- `assess_zero_positive_evidence`
  - Compares visible target claim/applicant signals, train-only empirical
    zero rates, positive references, and zero / finding-sufficient references.
  - Returns a calibration recommendation:
    `zero_plausible`, `positive_plausible`, `ambiguous`, or
    `insufficient_evidence`.
  - This is not a classifier output. It is an internal calibration step before
    the final continuous `award_eur` prediction.
- `assess_aggregation_pattern`
  - Must be called after `assess_zero_positive_evidence` and before
    `final_predict` in few-shot ReAct runs.
  - Returns a controller-owned aggregation observation with:
    `num_applicants`, `application_count`, `appendix_row_count`,
    `individualized_applicant_rows`, applicant and application count bands,
    joined / repetitive application signals, train-only applicant-band priors,
    same-band retrieved references, and a high-award p75 / p90 anchor.
  - Target structure is computed from non-award structural metadata and
    applicant/facts rows. Target final awards, per-applicant award allocations,
    and target label-derived fields are not used.
  - Train priors use only `train.csv` joined to `prediction_values/train.csv`,
    exclude the target itemid, and apply the target-date temporal filter when
    available.
  - The model should use this observation to decide whether the target is a
    `single_case_band`, `small_group_band`, `large_joined_case_band`, or
    `mass_joined_case_band`. It must not mechanically multiply a per-applicant
    amount.
- `leakage_check`
  - Runs a controller-side scan over visible target-case state.
  - Optional debugging action. The controller also runs this gate
    automatically when `final_predict` is requested.
- `final_predict`
  - Ends the loop with the final prediction object.
  - The controller rejects final prediction if the automatic leakage gate fails.

No other action is allowed. File paths, raw repo search, external browsing, and
ad hoc retrieval are blocked unless implemented as controller-owned actions.

## ReAct Modes

### `strict_react`

Use this as the benchmark-safe default. The target case may expose:

- facts/procedure style standard input after strict redaction
- oracle violated articles
- safe metadata
- non-award, non-claim extracted hints
- shared strict-safe KB modules
- train-only empirical priors
- temporally prior train references in `few_shot`

The target case must not expose raw Article 41 text, operative clauses, direct
award snippets, claimed amounts, or target-derived fields.

### `award_redacted_react`

Use this only as an explicitly labelled extracted-information / relaxed
ablation. It follows the v3 `award_redacted_full_info` contract: target final
non-pecuniary awards and direct target-award derivatives remain redacted, while
Article 41 structured claim-side information may be supplied after redaction.

### `full_info_award_blind_react`

Use this as the broad-information agent mode. The agent may access rich target
features, structured Article 41 claim information, external factors, training
reference features, and train-only award distributions, but it must remain blind
to the target final awards.

Blocked target fields include:

- final non-pecuniary award labels and direct derivatives
- other final award heads, including pecuniary, costs, bundled, and total award
  fields
- total/bundled final award labels and direct derivatives
- per-applicant award/allocation rows
- raw extractor final-award fields
- target-safe label fields derived from final awards
- raw text snippets that reveal target final award amounts
- target citation-list fields such as `all_scl_citations` when they can trigger
  target award-value leakage audits

Allowed target fields include claim-side structured information, government
arguments, procedural/factual features, legal-analysis features, metadata, and
external as-of features, as long as they do not directly reveal target final
awards.

In this relaxed mode, target zero-award reason fields may be inspected when they
do not expose a numeric final award, including main-table
`exclusion_reason_codes`, `include_reason`,
`award_non_pec_satisfaction_sufficient`, and
`award_non_pec_dismissed_reason`. These fields are leakage-sensitive and are
not part of strict baselines. Generated reasoning summaries such as
`reasoning_layer.award_reasoning_summary` are not allowed as prediction input or
zero-award evidence because they are model-written summaries, not source-text
extractions.

If a numeric non-pecuniary claim amount is visible, the controller treats it as
a final-award cap and applies that cap automatically at `final_predict`.

## Final Action

The final turn must be:

```json
{
  "thought_summary": "Short calibration summary.",
  "action": "final_predict",
  "action_input": {
    "award_eur": 7500.0
  }
}
```

The controller records the trace but the benchmark prediction remains pure
regression:

```json
{"award_eur": 7500.0}
```

Do not output a zero/non-zero classifier, zero reason field, or any target label
derivative in the final prediction.

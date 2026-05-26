# ECtHR NPD Prediction Knowledge Base v3

This is a separate knowledge base from `npd_prediction_knowledge_base_v2_agentic`.
It implements an award-redacted agentic design for non-pecuniary damages
prediction.

The benchmark no longer compares strict versus relaxed input modes. There is one
target-case input policy:

- keep the base target input identical to the earlier zeroshot prompting
  experiment
- let the controller locate the corresponding extraction sidecars by `itemid`
  and attach them after redacting the target non-pecuniary award label and direct
  target-award derivatives

The only experiment settings are:

- `zero_shot`: no retrieved reference cases
- `few_shot`: deterministic multiple-layer filtering retrieves temporally prior
  train cases; their reference non-pecuniary awards may be shown as examples

## Core Contract

Target case:

- blocked:
  - final non-pecuniary award amount
  - direct target non-pecuniary award fields such as `safe_non_pec_eur`,
    `award_eur`, `award_non_pec_*`, and `raw_extractor_non_pec_*`
  - target judgment text spans that directly disclose the final non-pecuniary
    award disposition or amount
- allowed:
  - the standard zeroshot prompting input:
    `combined_input_text` plus oracle violated articles
  - target extraction sidecars located by `itemid`, including Article 41 claim
    information, government arguments, metadata, extracted factors, and shared
    KB modules after target non-pecuniary award redaction
  - train-only empirical priors

Few-shot reference cases:

- retrieved only from train cases
- must satisfy `retrieved_judgment_date < target_judgment_date`
- may expose their known reference non-pecuniary awards
- expose only tabular / extraction feature rows, not raw judgment text
- remove award and label-derived feature columns from the feature rows; the
  reference award remains only as the explicit few-shot anchor
- must be recorded in `retrieval_trace`

## Flow

1. Read the same standard input row used by zeroshot prompting.
2. Use `itemid` to locate target extraction sidecars.
3. Redact target non-pecuniary award labels and direct derivatives from the
   attached extraction context.
4. Route to article modules and shared policy modules.
5. Resolve train-only empirical priors when available.
6. For `few_shot`, retrieve temporally valid train reference cases.
7. Run internal zero-award reasoning, article assessment, calibration, and
   synthesis, then output one continuous EUR amount.

The model may decide which shared information is useful, but it may not recover,
guess from, or quote the redacted target non-pecuniary award label. The final
visible output is pure regression: `{"award_eur": <non-negative number>}`.

### Empirical Priors

Train-only award distributions are generated from the current chronological
split:

- `dataset_release/data/ecthr_npd_cases.csv`
- `dataset_release/model_inputs/structured_tree/targets/train.csv`

The generated tables are:

- `modules/empirical/article_award_distribution_train.csv`
- `modules/empirical/country_award_distribution_train.csv`
- `modules/empirical/article_country_award_distribution_train.csv`

`resolve_empirical_priors` selects the narrowest supported anchor first:
article×country, then article, then country, then global article-weighted
fallback. The legacy `article_single_violation_stats_train.csv` is still written
for compatibility with `orchestrator_v2.py`.

## ReAct Controller

`react_orchestrator.py` adds a controller-owned ReAct loop beside the original
single-prompt `orchestrator_v2.py` baseline. The model does not read files or
perform retrieval directly. Instead, each turn returns one JSON action object:

```json
{
  "thought_summary": "Short reason for the next action.",
  "action": "inspect_case",
  "action_input": {}
}
```

The controller executes only whitelisted actions, records the observation, and
continues until `final_predict`.

By default the controller now uses lazy context access:

- `inspect_case` returns a target overview and information catalog, not the full
  target payload. It also returns recommended target and reference query
  templates to avoid source-selector trial and error.
- `query_target_information` retrieves selected target fields by source, path
  prefix, or field-name substring.
- `load_relevant_modules` loads the controller-selected legal / policy modules
  in one action; the older `search_modules` + `load_module` path remains
  available for custom subsets.
- `retrieve_train_references` returns reference ids, ranking traces, known train
  reference awards, and feature-source counts.
- `query_reference_features` retrieves selected feature-row values for chosen
  retrieved references. If no source list is supplied, it defaults to
  `split_case_features`, `reasoning_layer_features`, and `applicant_features`.
- `assess_zero_positive_evidence` compares visible target claim/applicant
  signals, train-only zero rates, positive references, and zero /
  finding-sufficient references before final prediction. In relaxed/full-info
  mode this includes visible main-table zero-reason fields such as
  `exclusion_reason_codes`, `include_reason`,
  `award_non_pec_satisfaction_sufficient`, and
  `award_non_pec_dismissed_reason`. Generated reasoning summaries such as
  `reasoning_layer.award_reasoning_summary` are blocked because they are
  model-written summaries, not source-text quotes.
- `assess_aggregation_pattern` runs after the zero/positive assessment and
  before `final_predict` in few-shot runs. It gives the model a compact
  controller-owned observation for case-level aggregation scale:
  `num_applicants`, `application_count`, `appendix_row_count`,
  `individualized_applicant_rows`, applicant/application count bands,
  joined/repetitive signals, train-only applicant-band priors, same-band
  retrieved references, and the selected high-award p75/p90 anchor. The action
  uses target structural metadata plus train-only labels; it does not expose
  target final awards or per-applicant award allocations.
- `leakage_check` is now optional for debugging. The controller automatically
  runs the same leakage gate when `final_predict` is requested, so the model
  does not need to spend a normal ReAct step on it.
- If a numeric non-pecuniary claim amount is visible in relaxed/full-info mode,
  the controller applies it as a cap at `final_predict`.
- The controller now blocks all target final-award heads, not only non-pecuniary
  awards. Pecuniary, costs, bundled, total, raw extractor, and per-applicant
  final-award fields are redacted. Raw text lines that reveal the target final
  award amount are also replaced with
  `[TARGET_FINAL_AWARD_TEXT_REDACTED]`.
- Target citation-list fields such as `all_scl_citations` are also blocked
  because they can contain numeric strings that trip the target award-value
  leakage gate without adding material prediction value.

This keeps all allowed information available while preventing the first ReAct
observation from dumping every target and reference field into the model
context.

ReAct modes:

- `strict_react`: benchmark-safe default. Blocks raw Article 41 text, operative
  clauses, direct award snippets, claimed amounts, and target-derived fields.
- `award_redacted_react`: explicit extracted-information / relaxed ablation
  following the v3 `award_redacted_full_info` contract.
- `full_info_award_blind_react`: broad-information agent mode. Allows rich
  target features and structured claim-side information while blocking target
  final awards, final-award derivatives, and per-applicant awards.

Dry run:

```bash
python .knowledge/npd_prediction_knowledge_base_v3_agentic_award_redacted/react_orchestrator.py \
  --case_file your_path/strict_case_inputs.jsonl \
  --react_mode strict_react \
  --inference_mode zero_shot \
  --dry_run
```

Live mode uses an OpenAI-compatible chat endpoint only for action selection and
final prediction; all retrieval, module loading, and leakage checks remain local
controller actions:

```bash
python .knowledge/npd_prediction_knowledge_base_v3_agentic_award_redacted/react_orchestrator.py \
  --case_file your_path/strict_case_inputs.jsonl \
  --react_mode strict_react \
  --inference_mode few_shot \
  --live \
  --api_base http://127.0.0.1:1234/v1 \
  --model qwen3.5-27b
```

Add `--provider_json_schema` only for endpoints that support OpenAI-style
`response_format: json_schema`.

### Reference Retrieval

The ReAct retriever is controller-owned. It first applies hard filters:

- train split only
- retrieved judgment date before target judgment date
- target itemid excluded
- reference label exists in `prediction_values/train.csv`
- at least one violated-article overlap

It then applies domain multiple filtering where each filter is kept only when it
does not over-narrow the pool:

- exact article set
- same respondent country
- same violation type
- same applicant-count band
- same formation
- same case importance
- five-year recency window

The remaining candidates are ranked by a weighted similarity score and returned
as positive references, zero / finding-sufficient references, and a balanced
reference list.

## Structure

```text
npd_prediction_knowledge_base_v3_agentic_award_redacted/
├── README.md
├── kb_index.json
├── orchestrator_v2.py
└── modules/
    ├── policy/
    ├── normative/
    │   └── articles/
    ├── routing/
    └── empirical/
        └── templates/
```

The orchestrator filename remains `orchestrator_v2.py` for import
compatibility, but the directory and KB index identify this as v3.

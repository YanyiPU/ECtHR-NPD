# Re-running experiments from the public tables

The package has one current dataset: 14,575 cases, with 10,217 train, 1,461 validation and 2,897 test cases. Use the package root as the dataset path. No version designation, private release contract, or pre-encoded 48/50-column matrix is required. Preserve the released splits and diagnostic view flags.

These commands support **new experiments** on the current targets. Two case totals were corrected, and the exact historical X0–X3 feature map, tree search candidates, reviewed FACTS snapshots, model checkpoints, provider snapshots, predictions and traces are not completely available. Running current code is not proof of reproducing the submitted paper's numerical results. See [paper alignment](PAPER_ALIGNMENT.md) and the root change log for retained source issues.

## Preparation and constant baselines

Run commands from the package root. Python 3.10+ with NumPy and pandas is sufficient for preparation, constants and structured kNN. Install optional tree/text/encoder dependencies only for those runners. Derived inputs and outputs must be outside this public package, which contains exactly two public CSV tables.

```bash
python -m pip install numpy pandas
python code/public_tables.py .
python code/prepare_experiments.py prepare --dataset-root . --output-dir ../experiment-inputs
python code/prepare_experiments.py constants --dataset-root . --output-dir ../constant-results
```

The second experiment command estimates zero, mean, median, positive-only mean and positive-only median from training labels only, then reports the shared metric suite on validation/test and ID/OOD/Challenging. It recalculates current-data values rather than copying Table 19. The legacy reported train-median test MAE has a known €1 conventional-rounding difference; do not alter labels to force agreement.

Preparation writes a manifest with public-table hashes, implementation/prompt hashes, transformations, and output hashes. `itemid` in derived files is an unchanged alias of the public HUDOC `case_id`; it supports joins and is dropped before model fitting or prompt rendering. `train_targets.csv` alone is the reference-label source. Validation/test labels are separate files.

The active `public_structured` profile has 20 predictors: respondent country, court decision body, importance, separate-opinion and representation flags, applicant and HUDOC application counts, judgment year/month, violation count/type/Articles/duration, four readable demographic summaries, and two log-transformed GDP measures. The stored public values stay unencoded. Runtime categorical vocabularies and numerical imputation are fit on train only. Demographic summary sets are treated as readable categories, not reconstructed historical ratios. Allocation/beneficiary fields, names, identifiers, split/view flags, target status, and every target are excluded. The applicant table is not used to derive features from award allocations.

## Tree models and structured retrieval

```bash
python -m pip install catboost xgboost lightgbm scikit-learn
python code/baselines/tree_models/train.py --dataset-release . --model catboost --example-fixed-parameters --output-dir ../tree-results
python code/baselines/retrieval/tabular_knn.py --dataset-release . --metadata ../experiment-inputs/case_index.csv --features ../experiment-inputs/strict_features.csv --profile ../experiment-inputs/knn_profile.json --train-targets ../experiment-inputs/train_targets.csv --metric euclidean --output-dir ../knn-results
```

Replace `catboost` with `xgboost` or `lightgbm` to run the other tree models. The fixed-parameter example does not reconstruct historical tuning. For a new validation search, replace `--example-fixed-parameters` with `--selection-candidates /path/to/candidates.json`. The JSON accepts a list of `{"name":"candidate_name","params":{...}}` objects, or an object keyed by model name containing such lists. Specify and publish all candidate values before evaluating test. Trees train on `log1p(y)`, inverse-transform and clip at zero, select by validation MAE with MedAE tie-break, and refit on train only (Table 20). Never select the feature profile or hyperparameters using test results.

Structured kNN uses top 20, median aggregation, training-only references, strictly earlier judgment dates, at least one shared violated Article and target exclusion, with the train median as fallback (Table 21). The supplied dates, Articles and splits are checked against the public dataset. Euclidean/cosine metric, train-only scaling and categorical choices are documented new-run choices; unavailable historical preprocessing is not inferred. Runtime and coverage records are saved with predictions.

## Reviewed raw FACTS and text models

The extraction workflow in `docs/EXTRACTION.md` reconstructs source material. Its raw segmented sections are not automatically accepted prediction inputs. Review each case to remove claims, Article 41/50 reasoning, operative payment clauses, appendices containing awards, names and other identifying text, then supply one CSV/JSONL record per public case:

```json
{"case_id":"001-100018","facts_text":"Reviewed FACTS text...","review_status":"accepted_prediction_input","source_document_sha256":"<64 hexadecimal characters>","source_anchor":"FACTS paragraphs 1–20"}
```

The review marker records the researcher's acceptance; it is not an automated certificate. A valid source fingerprint and anchor, nonempty text, unique IDs and exact full-cohort coverage are mandatory. Preserve the review artifact and underlying source documents locally for audit. No missing cases are silently dropped.

```bash
python code/prepare_experiments.py prepare --dataset-root . --facts-inputs /path/to/reviewed_facts.jsonl --output-dir ../text-experiment-inputs
python -m pip install bm25s==0.3.8
python code/baselines/retrieval/bm25_pfme_knn.py --dataset-release . --documents ../text-experiment-inputs/retrieval_documents.jsonl --train-targets ../text-experiment-inputs/train_targets.csv --output-dir ../bm25-results
```

`code/baselines/retrieval/bge_m3_knn.py` accepts the same documents/targets with `--mode dense` or `--mode sparse`; use `--model-name-or-path /path/to/local/bge-m3` for a local model. BGE-M3 requires its optional model runtime and weights. No text model is downloaded or executed by preparation.

For Table 22 encoder architecture, use a pre-existing local model directory, a bf16-capable CUDA device, and explicitly chosen learning rate, epochs and seed:

```bash
python code/baselines/encoder/train_paper.py --dataset-release . --case-index ../text-experiment-inputs/case_index.csv --targets ../text-experiment-inputs/targets.csv --text-inputs ../text-experiment-inputs/encoder_facts.jsonl --architecture modernbert --model-name-or-path /path/to/local/ModernBERT-base --learning-rate 0.00002 --epochs 3 --seed 42 --acknowledge-new-run --output-dir ../encoder-results
```

The displayed learning rate and epoch count are new-run examples, not recovered historical settings. Use `legal_longformer` and an appropriate local backbone for two 4,096-token chunks. Its LegalBERT initialization provenance must be documented separately. Add `--features ../text-experiment-inputs/strict_features.csv --profile ../text-experiment-inputs/feature_profile.json --mlp-activation gelu` for a new public-profile late-fusion experiment. The runner implements log-target standardization on train, SmoothL1, specified pooling, MLP dimensions, optimizer/warmup, batch accumulation and validation selection. The generic `encoder/train.py` is an example scaffold, not this Table 22 implementation. A serialization is never substituted silently for raw FACTS.

## Prompting and ReAct

Preparation renders `zero_shot_requests.jsonl` and `cot_requests.jsonl` for validation and test. Send **only each record's `messages`** to the provider; its wrapper identifier and split are audit metadata. Calls are intentionally not made by preparation. Validate the JSON schemas in `prompts/`, disable tools, request temperature 0 and seed 42 where supported and maximum 4,096 output tokens, and record the provider, model snapshot, access date, effective decoding settings, failures and retries. Convert outputs to `itemid,predicted_award_eur` for evaluation. Missing/invalid responses must be reported, not imputed to zero or silently dropped.

The strict serialized zero-shot prompt has been corrected to remove claim-side inputs and an unsupported dataset prevalence prior. Templates are current source implementations. They are not recovered exact historical prompt packets.

Retrieved few-shot CoT additionally needs reviewed training Article 41 rationales, up to five temporally prior article-overlapping references, the documented positive/zero reference mix and a retained selection trace (Appendix C.6). Those rationales and historical selections are unavailable from the two public tables. They must be supplied and reviewed separately; a rationale-free run is a separate condition.

To rebuild priors and run the controller offline:

```bash
python agent_knowledge_base/build_train_article_priors.py --dataset-release . --case-rows ../experiment-inputs/train_cases.csv --train-labels ../experiment-inputs/train_targets.csv --output-dir ../agent-priors
python agent_knowledge_base/react_orchestrator.py --dataset-release . --train_csv ../experiment-inputs/train_cases.csv --train_label_csv ../experiment-inputs/train_targets.csv --prior_dir ../agent-priors --case_file ../experiment-inputs/agent_cases.jsonl --case_id 001-214040 --paper_preset --dry_run --trace_out ../agent-dry-trace.json
```

The displayed ID is a test-split smoke case, not a scored result. Priors are hash-bound to exact public training metadata and labels. The dry run produces an action trace without provider calls or a scored prediction. Live ReAct uses the shipped compatible client by default; optionally override its module through `ECTHR_NPD_OPENAI_COMPATIBLE_CLIENT`. Supply an authorized endpoint, model, API key and documented provider settings when explicitly requesting `--live`. The preset has 12 steps and top-5 references (Tables 24–25), strict leakage gates and separately labeled expanded diagnostics. Public structured target packets differ from the unrecovered historical raw-text packets.

## Evaluation and evidence limits

```bash
python code/baselines/evaluate.py --predictions /path/to/test_predictions.csv --ground-truth ../experiment-inputs/test_targets.csv --dataset-release . --output-dir ../evaluation-results
python -m unittest discover -s tests -p 'test_unified_experiments.py' -v
```

The evaluator aligns exact IDs, rejects duplicate/missing/nonfinite records, validates labels against the selected public dataset and reports EUR metrics plus zero/positive diagnostics. `case_id` and legacy `itemid` are accepted aliases. The prediction file must cover exactly the ground-truth file supplied; do not use the complete cohort's labels when evaluating a test-only prediction file. Use the fixed public view flags when reporting ID, OOD and overlapping Challenging results.

Paper anchors: Table 11 / Appendix C.1 specify the input boundary; Tables 17–18 the split/constant protocol; Tables 20–25 the model families; Table 28 names X0–X3 but does not provide the exact column mapping. The historical 50-feature matrix contained two allocation-derived predictors with unverified facts-side lineage, which this public adapter excludes. Therefore no current result should be labeled historical X1 or an exact reproduction of Table 3 without additional original-run evidence.

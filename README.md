# ECtHR-NPD Code/Data Release

This bundle packages the public artifacts for ECtHR-NPD: the dataset
release, shared data loader, baseline code, prompt templates, latest
encoder settings, the redacted ReAct agent knowledge base, source
reconstruction helpers, and the extraction pipeline.

## Contents

- `dataset_release/`: canonical public dataset release with 14,575 cases,
  validated NPD targets, chronological split and diagnostic-view tags,
  public HUDOC IDs/URLs, leakage-audited structured features, and
  structured tree-ready inputs plus respondent-state/year external
  economic covariates.
- `code/`: shared data loader, pure-regression tree baseline trainer,
  encoder baseline trainer/loader, BM25/BGE-M3 retrieval baselines, and
  evaluation/audit helpers. The included loader reads only
  permitted model input files from `dataset_release/model_inputs/` and
  keeps targets separate.
- `prompts/`: zero-shot, CoT, CoT + few-shot, and agent prompt
  templates.
- `model_settings/`: sanitized latest tree, retrieval, encoder, and
  prompting settings from the synchronized experiment package.
- `agent_knowledge_base/`: redacted ReAct knowledge base and train-prior
  tables, ReAct controller code, and external covariates, without
  figures/heatmaps.
- `requirements.txt`: full Python package dependencies grouped by baseline
  family.
- `scripts/smoke_test_release.py`: quick local validation for reviewers
  after unzipping the shared folder.
- `source_reconstruction/`: HUDOC and ECHR-OD/ECHROD reconstruction
  helpers. These scripts retrieve or index public source documents into a
  user-supplied local workspace; raw judgment text is not in the bundle.
- `extraction_pipeline/`: release-safe extraction code, prompts, and
  schemas for rebuilding structured sidecars and Article 41 label
  candidates from local HUDOC source documents.
- `INPUT_CONTRACT.md`: shared prediction-input policy for structured, encoder,
  prompted, and agentic conditions.

## Excluded

The bundle excludes raw HUDOC judgment text, redacted per-case text input
files, Article 41/Article 50 text, operative award clauses, claim amount
fields as model inputs, result tables, figures/heatmaps, appendix drafts,
per-case prediction archives, provider traces/logs, checkpoints,
embeddings, neighbor traces, API keys/provider credentials, local paths,
and applicant-level identifying information. Generic code paths that read
credentials from user-supplied environment variables may remain; no
credential values are packaged.

Targets are labels for supervised training and evaluation; they must not
be used as model inputs. The extraction pipeline is included for source
reconstruction and label audit only; it is not a baseline input generator.

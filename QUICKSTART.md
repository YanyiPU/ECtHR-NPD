# Reviewer Quickstart

Run these commands from the unzipped release folder.

## 1. Minimal Inspection

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test_release.py
```

This verifies the public dataset, split counts, structured loader,
encoder loader fallback, external-factor table, and sanitized release
layout. It does not download HUDOC documents, call any API, or train
heavy models.

## 2. Structured Tree Baselines

Install the full dependency set if you want to run the packaged tree,
retrieval, or encoder code:

```bash
python -m pip install -r requirements.txt
python code/baselines/tree_models/train.py --model catboost --dataset-release dataset_release
python code/baselines/tree_models/train.py --model xgboost --dataset-release dataset_release
python code/baselines/tree_models/train.py --model lightgbm --dataset-release dataset_release
```

The tree commands run directly from the shared folder because the
model-ready structured matrices and targets are included.

## 3. Text, Retrieval, Agent, And Extraction Conditions

These components require external resources by design:

- HUDOC source documents must be reconstructed locally with
  `source_reconstruction/`; raw judgment text is not redistributed.
- BM25 and BGE-M3 retrieval require user-supplied case text with
  Article 41 award material removed or serialized model inputs that
  conform to the shared policy.
- Encoder text reproduction requires either serialized public model
  inputs or user-supplied case text with Article 41 award material
  removed, plus model downloads/checkpoints.
- The extraction pipeline and ReAct controller require user-provided
  API credentials and model/provider choices at runtime.

Use `REPRODUCIBILITY.md` for the full command map and `INPUT_CONTRACT.md`
for the shared prediction-input policy.

# ECtHR-NPD

Code and documentation for *How Much is a Human Right Worth? ECtHR-NPD: A Benchmark for Predicting Non-Pecuniary Damage Awards*.

**Data:** [Hugging Face dataset](https://huggingface.co/datasets/YanyiPU716/ECtHR-NPD)  
**Paper:** final proceedings citation and link forthcoming

## Overview

ECtHR-NPD is a benchmark for predicting the case-level Article 41
non-pecuniary damage (NPD) award in European Court of Human Rights
(ECtHR) cases. The public release contains 14,575 validated case-level
Euro targets, including valid zero awards, with chronological training,
validation, and test splits.

| Split | Cases |
| --- | ---: |
| Train | 10,217 |
| Validation | 1,461 |
| Test | 2,897 |

The test split also includes ID, OOD, and Challenging diagnostic-view
annotations. ECtHR-NPD is intended for research on legal NLP and
empirical legal analysis; it is not designed for legal advice, settlement
valuation, or automated judicial decision-making.

## Data release

The canonical data are hosted on
[Hugging Face](https://huggingface.co/datasets/YanyiPU716/ECtHR-NPD).
The dataset release includes:

- a complete extracted case-level dataset (`ecthr_npd_cases.csv`);
- chronological `train.csv`, `validation.csv`, and `test.csv` files;
- respondent-state/year economic covariates (`economic_covariates.csv`);
- split and diagnostic-view annotations in the case-level files; and
- a dataset card and source-terms note.

It does **not** redistribute raw HUDOC judgment text, Article 41 or
Article 50 award material, claim amounts, operative clauses, applicant
names, model outputs, provider traces, or agent priors. Public HUDOC
identifiers and URLs allow users to retrieve source judgments subject to
the Court's terms.

## Repository contents

- `dataset_release/` — release bundle, including data copies used by the
  packaged code and model-family-specific reproducibility inputs.
- `code/` — data loading, evaluation helpers, and baseline implementations.
- `prompts/` — prompt templates used for prompted conditions.
- `model_settings/` — configuration files for the reported model families.
- `source_reconstruction/` — helpers for retrieving and indexing public
  source material in a user-supplied local workspace.
- `extraction_pipeline/` — extraction code, prompts, and schemas.
- `agent_knowledge_base/` — redacted resources and controller code for the
  agent condition.
- `scripts/smoke_test_release.py` — a lightweight consistency check for the
  public release.

## Prediction-input policy

All model conditions should follow the
[shared prediction-input policy](INPUT_CONTRACT.md). Permitted inputs are
case metadata, violated-article information, case facts that exclude
award-related material, and respondent-state/year external economic
covariates. Award-related text, operative clauses, claim amounts, target
labels, target-derived fields, and split/view labels must not be used as
model inputs.

## Installation and quick check

```bash
git clone https://github.com/YanyiPU/ECtHR-NPD.git
cd ECtHR-NPD
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test_release.py
```

The packaged structured-data baseline can be run with:

```bash
python code/baselines/tree_models/train.py --model catboost --dataset-release dataset_release
```

## Reproducibility scope

The release supports inspection of the dataset and code, and rerunning the
packaged structured-data baselines. It does not claim one-command
reproduction of every reported experiment: retrieval and text-encoder
settings require user-supplied award-free text inputs; source
reconstruction requires public source documents; and live agent or
extraction runs require user-provided models or credentials. Raw judgment
text, model checkpoints, predictions, and provider traces are intentionally
not redistributed.

## Source terms and responsible use

The release redistributes derived tabular data and supporting code, not
ECtHR judgment text. Use of official judgments retrieved through HUDOC
remains subject to the Court's applicable terms. See
[LICENSE_AND_SOURCE_TERMS.md](dataset_release/LICENSE_AND_SOURCE_TERMS.md)
and the Hugging Face dataset card for details.

## Citation

If you use ECtHR-NPD, please cite the accompanying paper:

> Yanyi Pu, Damian Gonzalez-Salzberg, Zheng Yuan, and Nikos Aletras.
> *How Much is a Human Right Worth? ECtHR-NPD: A Benchmark for
> Predicting Non-Pecuniary Damage Awards.* Proceedings citation forthcoming.

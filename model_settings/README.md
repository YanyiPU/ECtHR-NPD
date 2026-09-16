# Experiment settings

Run new experiments from the single public dataset root with `--dataset-release .`; no dataset-version selection or private feature matrix is required. See [EXPERIMENTS.md](../docs/EXPERIMENTS.md) for complete commands.

The active public feature representation is defined in `code/baselines/data/public_adapter.py`: 20 predictors derived from readable case columns. It excludes identifiers, split/view fields, targets, beneficiary/allocation information, and target status. GDP transforms and judgment month are computed at runtime. Training alone determines imputation and categorical vocabularies.

Settings under `tree/strict_trainonly_50_feature_tree_regression/` are retained historical configuration records. The directory name and internal 48-column schema are not the public feature contract and are not required inputs. The tree runner detects the public tables and selects the public feature schema. Historical X0–X3 column mappings and validation candidate manifests are unavailable; neither schema is certified as historical X1.

Retrieval scripts implement training-only references, strictly earlier judgment dates, shared violated Articles, top-20 median aggregation, and a training-median fallback. Structured kNN records its explicit new-run metric and preprocessing choices. Text retrieval requires reviewed FACTS inputs.

`encoder/train_paper.py` implements the specified Table 22 architecture for new experiments using local backbone files and reviewed FACTS. Historical checkpoints, exact training arguments, and provider prediction artifacts are not supplied. Prompt templates and controller settings are source implementations, not archived requests or evidence of reproduced paper scores.

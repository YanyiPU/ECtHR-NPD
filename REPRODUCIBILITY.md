# Reproducibility Notes

These notes are written for the shared release folder after unzipping.
Start with the smoke test:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test_release.py
```

The public dataset release is already included under `dataset_release/`.
Repository-internal build scripts used to assemble the shared folder are
not required for reviewer inspection.

Data loader:

```python
from baselines.data.data_loader import load_structured_tree_splits

splits = load_structured_tree_splits("dataset_release")
X_train = splits["train"].X
y_train = splits["train"].y_amount_eur
```

Tree-ready model inputs:

```text
dataset_release/model_inputs/structured_tree/features/{train,val,test}.csv
dataset_release/model_inputs/structured_tree/targets/{train,val,test}.csv
dataset_release/model_inputs/external_factors/economic_covariates.csv
```

Tree and retrieval code:

```bash
python code/baselines/tree_models/train.py --model catboost --dataset-release dataset_release
python code/baselines/tree_models/train.py --model xgboost --dataset-release dataset_release
python code/baselines/tree_models/train.py --model lightgbm --dataset-release dataset_release
python code/baselines/encoder/train.py --dataset-release dataset_release --model-name-or-path your_path/or_hf_model_id --text-inputs your_path/strict_case_inputs.jsonl
python code/baselines/retrieval/bm25_pfme_knn.py --documents your_path/strict_case_inputs.jsonl --train-targets dataset_release/model_inputs/structured_tree/targets/train.csv
python code/baselines/retrieval/bge_m3_knn.py --mode dense --documents your_path/strict_case_inputs.jsonl --train-targets dataset_release/model_inputs/structured_tree/targets/train.csv --external-factors dataset_release/model_inputs/external_factors/economic_covariates.csv
python code/baselines/retrieval/bge_m3_knn.py --mode sparse --documents your_path/strict_case_inputs.jsonl --train-targets dataset_release/model_inputs/structured_tree/targets/train.csv --external-factors dataset_release/model_inputs/external_factors/economic_covariates.csv
```

The CatBoost/XGBoost/LightGBM script is a direct pure-regression trainer
using the latest `strict_trainonly_50_feature_tree_regression` settings,
with no separate zero/positive stage. The BM25 and BGE-M3 retrieval
scripts require user-supplied procedure/facts inputs that exclude
Article 41 award-related material because raw judgment text is not
redistributed; retrieval inputs should serialize case metadata, violated
articles, case facts, and external factors under the shared
prediction-input policy.

The encoder script can run from serialized public model inputs when
`--text-inputs` is omitted. To reproduce text-encoder runs, provide a
text file with Article 41 award material removed at
`your_path/strict_case_inputs.jsonl`.

Source reconstruction and extraction:

```bash
python source_reconstruction/download_hudoc_judgments.py --case-index dataset_release/data/ecthr_npd_cases.csv --out-dir your_path/hudoc_judgments_docx --format docx
python source_reconstruction/build_echrod_subset.py --case-index dataset_release/data/ecthr_npd_cases.csv --echrod-root your_path/ECHROD/echr_database --out-dir your_path/echrod_source_index
python source_reconstruction/prepare_extraction_case_store.py --case-index dataset_release/data/ecthr_npd_cases.csv --echrod-metadata your_path/echrod_source_index/echrod_metadata_subset.csv --hudoc-docx-dir your_path/hudoc_judgments_docx --out-root your_path/extraction_workspace
cp -R extraction_pipeline your_path/extraction_workspace/extraction
cd your_path/extraction_workspace
python extraction/code/build_extraction_layers.py --itemids 001-000000
EXTRACTION_API_BASE=your_api_base EXTRACTION_API_KEY=your_api_key EXTRACTION_MODEL=your_model python extraction/code/holistic_extractor.py --itemids 001-000000 --run-name local_check
```

These commands rebuild local raw/derived source workspaces only. The
release itself does not redistribute HUDOC judgment text, extraction
outputs, provider traces, or credential values.

The ReAct controller code is included without credential values. Provider
credentials must be supplied by the user environment when running live
inference. The release includes prompt templates and evaluation helpers
for already produced model outputs.

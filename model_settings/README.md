# Model Settings

This folder records the current tree, retrieval, and encoder settings
from the latest synchronized tree/retrieval/encoder package. It includes
settings and training/config files only; prediction CSVs,
metrics tables, checkpoints, embeddings, and neighbor traces are not
included.

- `tree/strict_trainonly_50_feature_tree_regression/`: CatBoost,
  XGBoost, and LightGBM pure-regression tree setting. The public code
  includes CatBoost, XGBoost, and LightGBM reproduction scripts.
- `retrieval/strict_bm25_pfme_knn/`: strict BM25 PFME-KNN retrieval
  setting using train-only corpus and temporal filtering.
- `retrieval/strict_bge_m3_dense_text_knn/`: strict BGE-M3 dense KNN
  retrieval setting using train-only corpus and temporal filtering.
- `retrieval/strict_bge_m3_sparse_text_knn/`: strict BGE-M3 sparse KNN
  retrieval setting using train-only corpus and temporal filtering.
- `encoder/`: current strict encoder and late-fusion ablation settings.
- `prompting/`: zero-shot, CoT, and CoT + few-shot prompting settings.

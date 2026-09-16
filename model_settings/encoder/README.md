# Encoder implementations

`code/baselines/encoder/train_paper.py` implements Table 22's architecture for new experiments: ModernBERT with one 8,192-token input, Legal-Longformer with two 4,096-token chunks and mean CLS pooling, train-standardized log targets, SmoothL1, and the specified late-fusion dimensions. See [EXPERIMENTS.md](../../docs/EXPERIMENTS.md) for complete public-table and reviewed-FACTS commands.

The public profile is a new 20-predictor representation. Historical X0–X3 exact maps, complete hyperparameters, reviewed text snapshots, original checkpoints, predictions and LegalBERT initialization evidence are not recovered. The settings JSON files are records, not a certificate of exact historical training. The generic `encoder/train.py` remains a separate example scaffold.

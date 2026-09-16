# Training-only empirical priors

No empirical CSV tables are distributed in this directory. Its `templates/` child holds wire-format schemas/examples only.

Use `code/prepare_experiments.py prepare` and then `agent_knowledge_base/build_train_article_priors.py` as shown in [EXPERIMENTS.md](../../../docs/EXPERIMENTS.md). The builder derives article, country, article-country and legacy article compatibility tables from the public train subset, in a fresh directory outside the dataset package. Supply that directory as `--prior_dir` to the controller.

A manifest pins the exact training IDs, metadata, labels and table hashes. Priors resolve from article-country to article, country and global article-weighted fallback. Validation/test labels never contribute. Historical `y_source` values are not in the public targets, so label-source counts remain blank with an explicit unavailable status; unknown is not fabricated as zero.

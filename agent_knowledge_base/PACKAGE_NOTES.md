# Package notes

This package provides controller source, policy modules and schemas. Build current training-only empirical priors from the single public dataset using [EXPERIMENTS.md](../docs/EXPERIMENTS.md); no precomputed prior CSV or historical run artifacts are distributed here.

The public dataset keeps HUDOC case IDs and masks applicant names. Strict feature exclusion addresses target leakage; it does not make the source-linked dataset anonymous. The controller's public-table inputs use the documented new feature representation, not a recovered historical input snapshot.

Amount distributions and zero rates can be recomputed from train labels. Historical label-source counts cannot, and are explicitly marked unavailable. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for remaining historical reproduction limits.

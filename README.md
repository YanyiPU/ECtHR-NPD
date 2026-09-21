---
language:
- en
pretty_name: ECtHR-NPD
configs:
- config_name: case_level
  data_files:
  - split: all
    path: data/case_level.csv
- config_name: applicant_level
  data_files:
  - split: all
    path: data/applicant_level.csv
---

# ECtHR-NPD — unified current release

One current dataset, not separate v1.0/v1.1 releases. This author-approved
distribution is available for manual review. It preserves the paper's
14,575-case cohort and original partitions, with documented amount corrections.
It is not a byte-identical copy of the targets used for the original experiments.

## Data

- `data/case_level.csv`: 14,575 cases; 33 columns.
- `data/applicant_level.csv`: 44,581 applicant source units; 14 columns.
- Join on `case_id`, the original HUDOC item identifier. `applicant_id` remains a
  stable source-record key, not an independently identified natural person.
- Train / validation / test: 10,217 / 1,461 / 2,897. ID / OOD: 1,000 / 1,897.
  Challenging: 699, overlapping the test pool, not a fourth physical split.
- Names are `[MASKED]`. Dates, exact amounts, birth years, nationality text and
  other structured attributes remain. This is **name-masked, not anonymous**.
- No ratio, one-hot, log-transformed or numerically encoded predictor columns.
  Amounts, years, counts and durations are natural numeric values. `y_binary` is
  the existing outcome label, not a predictor encoding.
- `unknown` is missing or unresolved, never an invented zero. Estate awards are
  identified by `npd_award_scope`; no amount is divided among heirs or groups.

`split` supplies logical partitions in each CSV. Hugging Face config `all` means
the full physical table, not the training split. See `CSV_TREE.md` for every
column and `DATA_DICTIONARY.md` for meanings.

## Important target limitation

Five previously discussed cases remain in the cohort. Three have inseparable
mixed-head awards and two have unresolved targets. Their historical benchmark
labels are explicitly marked by `target_status`; they are **not newly verified
pure-NPD totals**. The paper's universal target-validation/exclusion statements
therefore cannot be claimed for every retained row. See `CHANGELOG.md`.

## Reproduction entry points

Run from this folder using Python 3.10+; set `PYTHONDONTWRITEBYTECODE=1` if desired.

```bash
python -B code/public_tables.py .
python -B -m unittest discover -s tests
```

Tests require the dependencies appropriate to their tested components. Install
`requirements-extraction.txt` for extraction, and consult the experiment guide
for optional modelling environments rather than assuming all GPU packages are
needed to read the CSVs. The table validator itself uses the standard library.

- [Extraction and source reconstruction](docs/EXTRACTION.md): use the released
  HUDOC IDs to obtain sources; create extraction candidates; run explicit review
  checks before exporting two readable tables.
- [Experiments](docs/EXPERIMENTS.md): prepare model-only encodings locally,
  fit/select on train/validation, evaluate on fixed test views, and keep target
  evidence out of predictors. The public CSVs themselves remain unencoded.
- [Paper alignment and limits](docs/PAPER_ALIGNMENT.md): separates implemented
  methods, tested behaviours, missing historical evidence and unrun experiments.
- [Validation record](docs/VALIDATION.md): tests and actual checks for this ZIP.

Original checkpoints, predictions, provider responses, exact historical feature
maps and complete manual-review ledgers have not all been recovered. This code
supports transparent new runs; it does **not** authenticate the paper's reported
scores. New readable-feature projections must not be called the original X1.
Five known target ambiguities also remain, even though the cohort size matches.

Only two CSVs are distributed. Raw judgments, name mappings, review evidence and
API logs are not bundled; local source reconstruction may create identifying
files and send material to a configured provider only on explicit invocation.
The extraction/experiment workflow does not upload its generated artifacts. Legacy internal-contract
helpers retained for regression testing do not define extra public releases.

## Maintenance and reuse

Quarterly updates are intended to add newly eligible cases after review, with a
dated change log and content hashes. This is not a claim of an unattended service
or that the March 2026 source snapshot has already been extended. Cite the paper
and the exact content revision, since corrections can change results even when
the number of cases is unchanged. Please report suspected extraction errors.

See `LICENSE_AND_SOURCE_TERMS.md`: this distribution does not select or grant a new
open licence, nor assert compliance approval. No personal-name mapping is needed
or included to join the two tables.

## Citation

Please cite the paper using the following BibTeX entry ([download](CITATION.bib))
and identify the dataset commit used in your research.

```bibtex
@inproceedings{pu-etal-2026-ecthr-npd,
  author    = {Pu, Yanyi and Gonzalez-Salzberg, Damian A. and Yuan, Zheng and Aletras, Nikolaos},
  title     = {How Much is a Human Right Worth? {ECtHR-NPD}: A Benchmark for Predicting Non-Pecuniary Damage Awards},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year      = {2026},
  month     = oct,
  address   = {Budapest, Hungary},
  note      = {24--29 October 2026}
}
```

## Where to review this release

- GitHub: https://github.com/YanyiPU/ECtHR-NPD
- Hugging Face: https://huggingface.co/datasets/YanyiPU716/ECtHR-NPD

Both current branches carry the same two tables and matching research source
package. The original 172-file inventories and content hashes were verified
across GitHub and Hugging Face after synchronization. The subsequent GitHub-only
citation update leaves the data and executable code unchanged. Use the data and
code together from either repository for manual review.
Older encoded tables and separate version-policy files are superseded in the
current tree, not erased from repository history. The older private HF review
repository is not this current public release. Publication does not resolve
the five known target ambiguities or missing historical reproduction evidence.

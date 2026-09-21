# ECtHR-NPD

**How Much is a Human Right Worth? ECtHR-NPD: A Benchmark for Predicting Non-Pecuniary Damage Awards**

Yanyi Pu · Damian A. Gonzalez-Salzberg · Zheng Yuan · Nikolaos Aletras

[Dataset](https://huggingface.co/datasets/YanyiPU716/ECtHR-NPD) · [Data documentation](DATA_DICTIONARY.md) · [Experiments](docs/EXPERIMENTS.md) · [Citation](#citation)

## Dataset summary

ECtHR-NPD is a benchmark for predicting non-pecuniary damage awards at the European Court of Human Rights. It addresses a continuous monetary prediction task: estimating the case-level award in nominal euros from case information, including cases with zero awards.

The dataset contains **14,575 cases**, chronological train/validation/test splits, and ID, OOD and Challenging test views. We provide linked case-level and applicant-level tables, together with extraction, modelling and evaluation code. The dataset also supports research on legal information extraction, remedy allocation and empirical patterns in judicial awards, subject to its coverage and measurement limitations.

## Data

| Table | Rows | Columns | Contents |
|---|---:|---:|---|
| [case_level.csv](https://github.com/YanyiPU/ECtHR-NPD/blob/main/data/case_level.csv) | 14,575 | 33 | Case characteristics, award labels and split annotations |
| [applicant_level.csv](https://github.com/YanyiPU/ECtHR-NPD/blob/main/data/applicant_level.csv) | 44,581 | 14 | Applicant/source-unit characteristics and available award allocations |

Join the tables on `case_id`, the original HUDOC identifier. Applicant rows are source records and may represent estates, organisations or groups; they are not a verified count of distinct people. Features are supplied as readable categories and natural numeric values, without ratio or one-hot predictor encodings.

| Partition | Cases |
|---|---:|
| Train | 10,217 |
| Validation | 1,461 |
| Test | 2,897 |

The test set comprises **1,000 ID** and **1,897 OOD** cases. The **699-case Challenging view** overlaps this test set; it is not an additional split. Each CSV contains its split assignments, so separate split files are unnecessary.

Names are masked, but HUDOC identifiers and other potentially identifying attributes remain: **the dataset is not anonymous**. Five retained award labels have known unresolved or mixed-head issues, identified by `target_status`. See the [data documentation](DATA_DICTIONARY.md) and [known issues](CHANGELOG.md) before use.

## Quick start

Clone the repository and install pandas to read the tables:

```bash
git clone https://github.com/YanyiPU/ECtHR-NPD.git
cd ECtHR-NPD
python -m pip install pandas
```

```python
import pandas as pd

cases = pd.read_csv("data/case_level.csv", dtype={"case_id": str}, keep_default_na=False)
applicants = pd.read_csv("data/applicant_level.csv", dtype={"case_id": str}, keep_default_na=False)

train = cases.loc[cases["split"] == "train"]
test = cases.loc[cases["split"] == "test"]
linked = applicants.merge(cases[["case_id", "judgment_year"]], on="case_id", validate="many_to_one")
```

`unknown` denotes missing or unresolved information, not zero. See the [data documentation](DATA_DICTIONARY.md) for full field definitions and the [paper-alignment notes](docs/PAPER_ALIGNMENT.md) for the Challenging selector. Both tables are also available on [Hugging Face](https://huggingface.co/datasets/YanyiPU716/ECtHR-NPD).

## Experiments

The paper compares constant predictors, gradient-boosted trees, retrieval baselines, fine-tuned encoder language models, prompted language models and knowledge-augmented agents.

- [Experiment guide](docs/EXPERIMENTS.md): environments, feature preparation, model families and evaluation.
- [Extraction guide](docs/EXTRACTION.md): source retrieval, candidate extraction and reviewed table export.

Award-related evidence must be kept out of prediction inputs. Fit preprocessing and retrieval priors on training data, select settings on validation data, and evaluate on the fixed test views.

The current release includes documented label corrections. Historical feature maps, model checkpoints and provider-run artifacts are incomplete, so current runs must not be presented as exact reproductions of the paper's reported scores. Details are in [reproducibility notes](docs/PAPER_ALIGNMENT.md).

## Updates and contributions

We plan to update the dataset every three months with newly eligible cases and reviewed corrections. Consult the [changelog](CHANGELOG.md) and record the dataset revision when reporting results.

LLM-assisted extraction can introduce errors. We welcome corrections, new research applications and improvements to the code through [GitHub Issues](https://github.com/YanyiPU/ECtHR-NPD/issues) or pull requests. For a suspected data error, include the HUDOC case ID, affected field and a source reference; do not post applicant names or private extraction records.

## Citation

Please cite the paper using the following BibTeX entry ([download](CITATION.bib)) and identify the dataset commit used in your research.

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

## Source terms and responsible use

Underlying Court source material: **© ECHR-CEDH**. The Court permits reproduction of its website information and texts for private use or information/education connected with its activities, provided reproduction is free of charge and the source is credited. Other uses, particularly commercial use, require prior written permission. See the Court's [copyright and disclaimer](https://www.echr.coe.int/copyright-and-disclaimer) and our [source attribution and reuse terms](LICENSE_AND_SOURCE_TERMS.md).

These conditions are not an unrestricted dataset or code licence. ECtHR-NPD is a research resource, not a tool for legal advice or automated adjudication.

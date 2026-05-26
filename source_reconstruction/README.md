# Source Reconstruction

Raw ECtHR judgment texts are not redistributed in this release. The public
source documents can be retrieved from HUDOC using the released `itemid`
values and HUDOC URLs, subject to the source terms that apply to the user.

This directory contains release-safe reconstruction helpers only. The scripts
write raw or derived judgment text only to a user-supplied local directory such
as `your_path/hudoc_judgments_docx` or `your_path/extraction_workspace`; those
local files are not part of the public bundle.

## Inputs

- `dataset_release/data/ecthr_npd_cases.csv`: released case index with
  `itemid`, `hudoc_url`, split tags, labels, and safe metadata.
- `your_path/ECHROD/echr_database`: optional local ECHR-OD/ECHROD export used
  to recover public HUDOC metadata for the same `itemid` values.
- HUDOC public conversion endpoints, accessed by `itemid`.

## Steps

Download public HUDOC judgments:

```bash
python source_reconstruction/download_hudoc_judgments.py \
  --case-index dataset_release/data/ecthr_npd_cases.csv \
  --out-dir your_path/hudoc_judgments_docx \
  --format docx
```

Build a non-text ECHR-OD/ECHROD source index:

```bash
python source_reconstruction/build_echrod_subset.py \
  --case-index dataset_release/data/ecthr_npd_cases.csv \
  --echrod-root your_path/ECHROD/echr_database \
  --out-dir your_path/echrod_source_index
```

Prepare a local extraction case store from the downloaded DOCX files:

```bash
python source_reconstruction/prepare_extraction_case_store.py \
  --case-index dataset_release/data/ecthr_npd_cases.csv \
  --echrod-metadata your_path/echrod_source_index/echrod_metadata_subset.csv \
  --hudoc-docx-dir your_path/hudoc_judgments_docx \
  --out-root your_path/extraction_workspace
```

Then run the extraction pipeline from a workspace where `extraction/` and
`unstructured/` are siblings:

```bash
cp -R extraction_pipeline your_path/extraction_workspace/extraction
cd your_path/extraction_workspace
python extraction/code/build_extraction_layers.py --itemids 001-000000
EXTRACTION_API_BASE=your_api_base \
EXTRACTION_API_KEY=your_api_key \
EXTRACTION_MODEL=your_model \
python extraction/code/holistic_extractor.py --itemids 001-000000 --run-name local_check
```

The strict modeling inputs in `dataset_release/model_inputs/` are already
Article-41-free. The reconstruction workflow above is for source audit,
label extraction, and rebuilding extraction sidecars; raw judgment text and
Article 41 material must not be fed to strict prediction baselines.

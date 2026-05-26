# Extraction Pipeline

This directory contains the release-safe extraction code used to derive
case-level structured sidecars and Article 41 label candidates from local
HUDOC source documents. It is separate from strict prediction baselines:
raw judgment text, Article 41 text, operative clauses, award snippets, and
claim amounts may be read here for source extraction, but they must not be
used as strict model inputs.

## Layout

- `code/`: deterministic scaffold builder, OpenAI-compatible client,
  B/C/D/E extraction stages, and the holistic stage orchestrator.
- `prompts/`: system prompts for facts/procedure, compensation, legal
  analysis, and reasoning extraction.
- `schemas/`: JSON schemas for the extraction stages.

## Local Source Setup

Use `source_reconstruction/` to download HUDOC DOCX files and prepare a
local workspace with `unstructured/cases.json` and
`unstructured/cases_by_itemid/*.json`. Then copy or symlink this directory
as `your_path/extraction_workspace/extraction` so that `extraction/` and
`unstructured/` are siblings.

```bash
cd your_path/extraction_workspace
python extraction/code/build_extraction_layers.py --itemids 001-000000

EXTRACTION_API_BASE=your_api_base \
EXTRACTION_API_KEY=your_api_key \
EXTRACTION_MODEL=your_model \
python extraction/code/holistic_extractor.py --itemids 001-000000 --run-name local_check
```

The API variables above are placeholders. No credential values, provider
traces, extraction outputs, raw judgments, or local paths are included in
this release.

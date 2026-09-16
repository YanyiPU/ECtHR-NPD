# Validation performed for this release

Date: 2026-09-16. These are actual offline checks, not a claim of historical
experiment reproduction or legal/privacy clearance.

## Dataset

- Two CSVs only: 14,575 × 33 case table; 44,581 × 14 applicant table.
- Original train/validation/test assignments: 10,217 / 1,461 / 2,897.
  ID/OOD: 1,000 / 1,897. The released metadata recomputes all 699 Challenging flags.
- Unique HUDOC case keys and applicant source-record keys; complete foreign keys;
  applicant split agrees with case split. Dates and source application counts agree.
- Two corrected case totals and 23 restored applicant/estate allocations checked
  against retained source-review evidence. Two estate units are explicitly labeled.
- 19,495 source records now have known amounts. No retained known-amount sum
  exceeds its case label. A lower sum alone is not classified as an error.
- All applicant-name fields are `[MASKED]`. All 353 distinct original nationality
  values were subjected to a targeted name check; none was confirmed to contain
  personal names. Nationality values match source text except trimming and the
  explicit missing marker. This is not universal name-recognition certification.
- Five flagged target issues and their 24 applicant records remain. No unsupported
  pure-NPD totals, zero awards, individuals, or equal joint shares were invented.
- Previous review/source files used to build the reviewed candidate were hash-checked as
  unchanged. Public-package scanning found no credential-pattern or personal
  home-directory literals; raw judgments, mappings, logs and bytecode are excluded.

## Offline tests and runtime smoke

`python -B -m unittest discover -s tests`: **200 tests discovered; 193 passed,
7 skipped; zero failures or errors**. Skips are explicitly optional historical
private-fixture tests; those fixtures are not shipped. Synthetic fixtures and
mocked transports cover live-client paths without making provider requests.

The executed local environment was Python 3.14.6 with NumPy 2.5.3, pandas 3.0.5,
SciPy 1.18.1, scikit-learn 1.9.1, jsonschema 4.26.0 and torch 2.14.0. Direct tested
dependencies are recorded in `requirements-offline-tested.txt`; this is not the
historical paper environment or a full dependency lock for GPU experiments.
All 60 Python source files also passed syntax parsing.

| Check | Actual result |
|---|---|
| Public-table schema/linkage/metadata selector validator | Passed on complete tables |
| Synthetic DOCX → scaffold → deterministic extraction candidates | Passed end-to-end |
| Reviewed-record → two-CSV projection | Passed positive and rejection-path tests |
| Complete public dataset → new-run experiment inputs | 14,575 records prepared |
| Constant baselines and all fixed views | Completed from current training labels; train median €3,200 |
| Empirical prior build | Exactly 10,217 train cases; metadata/labels/artifacts hash-bound |
| ReAct preset dry run, test case 001-214040 | Eight events; leakage gate passed; no prediction or API call |
| Shipped compatible-client fallback | Mocked-transport regression passed |

The generated experiment inputs, priors and dry-run traces are local test artifacts,
not extra public CSVs. Their commands are documented in `EXPERIMENTS.md`.

## Not performed or not established

No live provider call, paid extraction, model-weight download, full-corpus
re-extraction or full GPU training was performed. Tree/text/encoder scores from
the paper were not reproduced. Raw FACTS review is researcher-attested, not an
automatic source-content or leakage certification. The original manual-review,
feature-selection, provider and model-run evidence remains incomplete.

See `PAPER_ALIGNMENT.md` for the unresolved scientific claims. In particular,
preserving the paper's row counts while retaining five mixed/unresolved targets
does not make the dataset strictly compliant with every paper statement.

The same 200-test suite was rerun before synchronization: 193 passed, 7 optional historical-fixture tests skipped, no failures or errors.

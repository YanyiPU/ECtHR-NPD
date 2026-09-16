> The distributed dataset contains the single paper cohort in `data/case_level.csv` and `data/applicant_level.csv`, with public HUDOC case IDs. Discovery output and raw judgment text are local working artifacts; discovery does not change the retained cohort.

# Scoped live HUDOC metadata discovery

`discover_hudoc_cases.py` now supplies a live source-index step before offline
planning/downloading. It contacts only the official metadata-search endpoint,
not judgment-conversion endpoints or an LLM. Output contains source IDs, case
names and potentially identifying metadata: **internal-only, not a public table**.

## Verified, deliberately narrow scope

The supported combination is **English (`ENG`) judgments (`HEJUD`)**, selected
from explicitly named Grand Chamber, Chamber and/or Committee collections.
Language, document type, collections and both dates must be supplied. French,
other translations, decisions, communicated cases, Commission material and
other document collections are not silently included.

The query interval is `date-from <= kpdate < date-until`, using date-only ISO
values. Each returned record is also checked against language/type/collection,
the date bounds, and its actual judgment-date field. A mismatch blocks completion
instead of silently discarding a record while claiming complete coverage.

```bash
python3 source_reconstruction/discover_hudoc_cases.py \
  --date-from 2026-01-01 --date-until 2026-04-01 \
  --language ENG --document-type judgments \
  --collections GRANDCHAMBER CHAMBER COMMITTEE \
  --out-dir your_path/current_discovery
```

Output:

```text
current_discovery/
├── discovery_checkpoint.json    # private page/record evidence and resume state
├── discovery_manifest.json      # exact query, totals, integrity/completion status
└── hudoc_case_index.csv         # populated only after successful verification
```

Use a dedicated empty output directory. Existing nonempty directories without
this command's checkpoint are rejected, including when `--restart` is supplied.
No unrelated files are removed. An owned directory requires `--resume` or explicit
`--restart`; restart replaces only this command's checkpoint, index and manifest.
During a pending/failed run the index is header-only, so a stale successful index
cannot silently enter the next ingestion step.

Only consume the index when `discovery_manifest.json` has `output_ready: true`.
An exit code of 1 means incomplete; a partial checkpoint is not a releaseable index.
The CSV includes original judgment-date text, normalized ISO judgment date,
query identity and a metadata-record hash. The checkpoint retains source metadata
without inventing structured legal findings from narrative conclusions.

## Pagination, verification and recovery

- Requests use zero-based `start`, bounded `length` and `itemid Ascending`.
- Rows are deduplicated by real HUDOC `itemid`. Duplicate, contradictory or
  out-of-order results prevent a complete-index claim; source IDs are not people.
- Totals must remain unchanged. A short response advances by its actual length;
  an empty response before the reported total is a failure.
- After collection, **every page is fetched again** and metadata hashes compared.
  The server-injected ranking score is excluded because it is not stable metadata.
- A resumed partial run revalidates its cached prefix before continuing. A resumed
  complete run revalidates all pages. Drift requires explicit restart, not merging
  records collected under inconsistent search results.
- HTTP/network retries are bounded. TLS is always verified. If the Python runtime
  lacks the appropriate CA chain, provide a trusted `--ca-bundle`; there is no
  insecure verification bypass.
- `--max-pages` limits requests per run, including verification. Resuming needs
  a budget large enough to recheck the cached prefix before making progress.
- Output/checkpoint writes are atomic and the CLI holds a local single-writer lock.

Two consistent traversals establish observed consistency of **this query**, not
an immutable HUDOC snapshot. Changes between checks, index lag, omissions in
HUDOC itself, or later record amendments cannot be ruled out. Success is reported
as `complete_for_explicit_scope_at_observation`; neither all-HUDOC coverage nor
scientific acceptance for the dataset is certified.

## Quarterly policy still required

Judgment date is **not the indexing/publication date**. Only searching judgments
dated within the latest quarter can miss an older judgment indexed late, a later
translation, or a correction to a historic record. The CLI does not claim access
to a verified change feed or modification timestamp.

An operational policy should therefore explicitly select an overlap/backfill
window and periodic historical re-harvest, compare source hashes, and deduplicate
against the retained internal store. No finite recent-date overlap proves that
all older late additions were captured. The appropriate window remains a project
decision; this command does not rewrite historical splits or add cases automatically.
French-only or otherwise excluded material requires a separately verified scope.

## Primary-source endpoint evidence

Checked 2026-09-13 with read-only public-client inspection and tiny metadata probes:

1. [Official HUDOC client](https://hudoc.echr.coe.int/shared/compiled.js?v=1775044138827)
   defines `/app/query/results`, query/select/sort/start/length request construction,
   zero-based query indexing, metadata/result fields, date comparators, and
   `kpdate Ascending` sort syntax. The inspected UTF-8 text, with normalized
   line endings, had SHA-256
   `940619982172351229fc5d5e677959e0a439267f14eb6f188da4e5acc6eafc38`.
2. [Official metadata endpoint](https://hudoc.echr.coe.int/app/query/results)
   returned `resultcount` and `results[].columns`. The explicit English judgment
   scope across the three collections for `[2020-01-01, 2020-02-01)` reported **97**
   results; pages `start=0,length=2` and `start=2,length=2` had distinct, ordered IDs
   and matching totals. Only those tiny pages and one known-case metadata record
   were inspected; no complete cohort was harvested into the dataset.
3. [Official HUDOC manual](https://www.echr.coe.int/documents/d/echr/HUDOC_Manual_ENG)
   describes inclusive UI date filters. This CLI uses an independently tested
   exclusive upper-bound comparator and verifies dates locally.
4. [Official HUDOC scope description](https://www.echr.coe.int/hudoc-database)
   distinguishes collections and notes coverage limits. This command deliberately
   does not treat the whole service as one complete judgment population.

The endpoint is an observed public web-client interface, **not a promised stable
public API**. Its behavior may change. The code rejects incompatible response
schemas instead of silently proceeding. Full end-to-end quarterly collection and
label extraction have not been run on real cases here.

## Tests

```bash
python3 -B -m unittest discover -s tests -p test_hudoc_discovery.py -v
```

15 offline tests cover scope/query generation, pagination, stable output, rank
normalization, two-pass drift checks, duplicates, total changes, partial/resume
behavior, checkpoint tampering, empty scopes, premature termination, rate-budget
limits, unrelated preexisting output protection, transient retries and TLS failure.

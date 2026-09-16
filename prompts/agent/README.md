# ReAct Policy-Template Mirror

The files in this directory are readable copies of the agent policy templates.
The controller loads its runtime modules through
`agent_knowledge_base/kb_index.json`, not from this mirror. This directory is
therefore useful for inspection and documentation, but is not a release of
historical model prompts, provider requests, predictions, or traces.

## Controller path and dry-run use

From the repository root, inspect the supported interface with:

```bash
python3 agent_knowledge_base/react_orchestrator.py --help
```

No `PYTHONPATH` setting is needed for that invocation. A dry run requires a
caller-supplied, authorised award-free case file and produces an assembly trace
with no model call and no prediction. It cannot reproduce a submitted-paper
agent result.

The controller's current source default is `strict_react`. Its target packet
excludes target Article 41 / just-satisfaction text, target claims and claimed
amounts, target outcome/zero-reason fields, target labels, and label
provenance. Its default target-query template is limited to that strict packet;
the blocked information cannot be recovered through a query action.

Every non-strict ReAct mode is a restricted diagnostic and requires
`--allow_restricted_diagnostic`. These modes must not be described as strict
benchmark-input runs, as reproductions of submitted-paper metrics, or as a
recovered submitted-paper configuration. The legacy `orchestrator_v2.py` is
also restricted and requires `--restricted_diagnostic` when run directly.

## Live calls and historical reproducibility

The controller uses the shipped `extraction_pipeline/code/openai_compatible_client.py`
for `--live`; `ECTHR_NPD_OPENAI_COMPATIBLE_CLIENT` can optionally select a custom
client module. A caller must supply compatible credentials, endpoint details,
an explicit model and an authorized case input. The package does not provide the historical model
snapshots, provider/run manifests, prompt packets, retrieval selections,
predictions, or traces needed to recreate Table 3 or Table 5 exactly.

See [`../../agent_knowledge_base/RELEASE_NOTES.md`](../../agent_knowledge_base/RELEASE_NOTES.md)
for the retained-versus-restricted artifact record.

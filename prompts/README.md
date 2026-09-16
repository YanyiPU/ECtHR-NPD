# Prompt templates

The public preparation command in [EXPERIMENTS.md](../docs/EXPERIMENTS.md) renders zero-shot and static-CoT requests from the same 20 strict public predictors used by the tree runner. It writes requests locally and makes no provider calls. Only the `messages` array is model input; the enclosing HUDOC identifier and split are alignment metadata.

- `zero_shot/`: one case-level nonnegative EUR amount, JSON field `award_eur`.
- `cot/`: static reasoning scaffold and output schemas. The claim-aware template is a separate diagnostic, not the common benchmark condition.
- `cot_few_shot/`: separate retrieved-reference condition; reference awards/rationales must come from training cases, strictly earlier dates, and overlapping violated Articles.
- `agent/`: a readable mirror of controller policy modules in `agent_knowledge_base/modules/policy/`.

Target inputs exclude claims, Article 41 reasoning, operative clauses, award allocations, zero rationales, label provenance, and targets. The serialized zero-shot template has been repaired to remove an incompatible invitation to include claim-side fields and a dataset-prevalence prior. This corrected template is a new-run implementation, not a recovered historical request.

The public tables contain no raw FACTS or Article 41 rationale text. New few-shot runs that follow Appendix C.6 need separately reviewed training rationales and a documented reference-selection manifest; the two CSVs alone do not reconstruct that condition. Provider snapshots, exact historical requests, predictions, and traces remain unavailable.

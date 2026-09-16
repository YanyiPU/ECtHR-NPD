# Controller and prompt implementation status

The current package supports new runs from one public dataset through the commands in [EXPERIMENTS.md](../docs/EXPERIMENTS.md). Preparation, train-only priors and offline ReAct traces are executable without private encoded matrices or provider calls. An offline trace is not a model prediction.

The following historical Table 3/Table 5 evidence is unavailable: exact model-visible inputs and requests, complete reference selections/ranking traces, provider/model snapshots and access-date manifests, per-case predictions, action/observation traces and checkpoint artifacts. Raw judgment documents and Article 41 materials must be reconstructed and reviewed separately. Source implementations and passing tests do not establish historical score reproduction.

The current strict path excludes target claims, Article 41 reasoning, operative clauses, award allocations, targets and zero rationales. Expanded diagnostics require explicitly supplied broader inputs and separate reporting. The repaired serialized zero-shot prompt removes claim-side input instructions and an unsupported dataset prevalence prior; this repair changes the current runnable template and does not claim to recover the exact historical prompt.

The public `--paper_preset` implements a maximum of 12 controller steps and five training references. The general legacy default remains 10 steps unless the preset or an explicit step count is selected. Live calls use the shipped compatible client (with optional module override) and require explicit provider configuration and credentials.

# Prompt Templates

Prompt templates are grouped by paper condition.

- `zero_shot/`: direct zero-shot prompt templates and JSON schema.
- `cot/`: static CoT templates. `static_cot_claimaware_prompt.md` is an
  explicitly labelled relaxed claim-aware sensitivity ablation; it is not
  a strict-input baseline.
- `cot_few_shot/`: CoT + few-shot templates. Templates that expose
  train-only references are ablations and must be reported separately
  from strict zero-shot/static-CoT baselines.
- `agent/`: ReAct system, action, output, mode, binary-gate, and failure
  policy prompt modules copied from the agent knowledge base.

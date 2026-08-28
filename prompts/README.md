# Prompt Templates

Prompt templates are grouped by paper condition.

- `zero_shot/`: direct zero-shot prompt templates and JSON schema.
- `cot/`: static CoT templates. `static_cot_claimaware_prompt.md` is an
  explicitly labelled relaxed claim-aware sensitivity ablation; it is not
  a baseline governed by the shared prediction-input policy.
- `cot_few_shot/`: CoT + few-shot templates. Templates that expose
  train-only references are ablations and must be reported separately
  from zero-shot/static-CoT baselines governed by the shared policy.
- `agent/`: ReAct system, action, output, mode, binary-gate, and failure
  policy prompt modules copied from the agent knowledge base.

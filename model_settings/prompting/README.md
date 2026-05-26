# Prompting Settings

The prompted decoder-LM conditions are represented as three settings:

- `zero_shot.json`: zero-shot prompting.
- `cot.json`: chain-of-thought prompting.
- `cot_few_shot.json`: chain-of-thought plus train-only few-shot examples.

Agent prompt modules are packaged under `prompts/agent/` and the full
redacted ReAct resources are under `agent_knowledge_base/`.

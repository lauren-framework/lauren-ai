---
name: chain-of-thought-prompting
description: Elicits step-by-step reasoning from models to improve accuracy on complex tasks. Use when the task involves multi-step logic, math, code analysis, or planning; when zero-shot answers are unreliable; or when you need to extract and inspect the model's reasoning separately from its final answer.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Chain-of-Thought Prompting Strategy

## Zero-shot CoT

Append a simple instruction to trigger step-by-step reasoning:

```python
# NOTE: future annotations are allowed, but tool signature types must resolve at import time.

COT_SUFFIX = "\n\nThink through this step by step before answering."

def add_cot(prompt: str) -> str:
    return prompt + COT_SUFFIX
```

## Structured CoT with XML tags

Use XML tags to separate reasoning from the final answer — the model is trained
to honor them and extraction is reliable:

```python
import re

STRUCTURED_COT_TEMPLATE = """\
Analyze the following problem:
{problem}

Think through it carefully:
1. Understand the problem
2. Identify key facts
3. Apply reasoning
4. Arrive at conclusion

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>"""


def extract_cot_answer(response: str) -> tuple[str, str]:
    """Extract <reasoning> and <answer> blocks from a structured CoT response.

    Returns (reasoning, answer).  When the tags are absent the full response
    is returned as the answer with an empty reasoning string.
    """
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else response.strip()
    return reasoning, answer
```

## System-prompt CoT instruction

Embed the CoT instruction in the agent system prompt so it applies to every turn:

```python
from lauren_ai import agent

COT_SYSTEM = """\
You are a helpful problem-solving assistant.

When answering questions, always reason step by step inside <reasoning> tags,
then give your final answer inside <answer> tags.

Format:
<reasoning>
[your step-by-step reasoning here]
</reasoning>

<answer>
[your final answer here]
</answer>"""

@agent(model="claude-sonnet-4-6", system=COT_SYSTEM)
class ReasoningAgent: ...
```

## Few-shot CoT

Provide worked examples where the model shows its reasoning:

```python
FEW_SHOT_COT_SYSTEM = """\
Solve math word problems step by step.

Example:
Q: A store has 24 apples. It sells 8. How many remain?
A: <reasoning>
Start: 24 apples.
Sold: 8 apples.
Remaining: 24 - 8 = 16.
</reasoning>
<answer>16</answer>

Now solve the user's problem in the same format."""
```

## Post-processing the response

```python
from lauren_ai import AgentRunnerBase, LLMConfig
from lauren_ai._transport._mock import MockTransport  # replace with real transport

async def run_with_cot(problem: str) -> tuple[str, str]:
    runner = AgentRunnerBase(transport=transport)
    response = await runner.run(ReasoningAgent(), problem)
    reasoning, answer = extract_cot_answer(response.content)
    return reasoning, answer
```

## Pitfalls

- **Don't truncate the output** — CoT responses are longer than direct answers.
  Set `max_tokens_per_turn` high enough (2048+ for complex tasks).
- **Structured tags require exact matching** — instruct the model to use
  `<reasoning>` and `<answer>` literally; do not vary the tag names per turn.
- **Zero-shot CoT suffix works best at the end** of the user message, not the
  system prompt.
- For extended thinking (Claude), use the `thinking=True` transport parameter
  instead — it is token-free and separated from the visible response.

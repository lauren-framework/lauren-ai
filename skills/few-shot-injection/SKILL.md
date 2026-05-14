---
name: few-shot-injection
description: Injects labeled input/output examples into system prompts or message history to steer model behavior with in-context learning. Use when the task has a consistent input/output format, when zero-shot instructions are insufficient, or when you need dynamic example selection per request.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Few-Shot Example Injection

## Built-in support: `FewShotPromptTemplate`

Lauren-AI ships `FewShotPromptTemplate` and `FewShotExample` in `lauren_ai._prompts`.
Use them for static example sets baked into the prompt:

```python
# NOTE: future annotations are allowed, but tool signature types must resolve at import time.
from lauren_ai import FewShotPromptTemplate, FewShotExample

tpl = FewShotPromptTemplate(
    prefix="Classify sentiment:\n",
    examples=[
        FewShotExample(input="Great product!", output="positive"),
        FewShotExample(input="Terrible experience.", output="negative"),
    ],
    example_template="{input} -> {output}",
    suffix="Input: {review}\nSentiment:",
    input_variables=["review"],
)
msg = tpl.render(review="Pretty good overall.")
```

## Custom builder for more control

When you need to choose examples dynamically or format them as message history:

```python
from dataclasses import dataclass

@dataclass
class FewShotExample:
    input: str
    output: str


class FewShotPromptBuilder:
    def __init__(self, task_description: str, examples: list[FewShotExample]):
        self._task = task_description
        self._examples = examples

    def build_system_prompt(self) -> str:
        lines = [self._task, "", "Examples:"]
        for i, ex in enumerate(self._examples, 1):
            lines.append(f"\nExample {i}:")
            lines.append(f"Input: {ex.input}")
            lines.append(f"Output: {ex.output}")
        lines.append("\nNow answer the user's input in the same format.")
        return "\n".join(lines)

    def build_messages(self) -> list[dict]:
        """Inject examples as alternating user/assistant message history."""
        messages = []
        for ex in self._examples:
            messages.append({"role": "user", "content": ex.input})
            messages.append({"role": "assistant", "content": ex.output})
        return messages
```

## System-prompt injection

```python
builder = FewShotPromptBuilder(
    task_description="You are a sentiment classifier. Reply with 'positive' or 'negative'.",
    examples=[
        FewShotExample("I love this!", "positive"),
        FewShotExample("This is awful.", "negative"),
    ],
)
system = builder.build_system_prompt()

from lauren_ai import agent

@agent(model="claude-haiku-4-5", system=system)
class SentimentAgent: ...
```

## Message-history injection

Injecting as message pairs is preferred when:
- The task benefits from seeing the model's "prior" responses.
- The examples are long enough that a system prompt becomes unwieldy.

```python
from lauren_ai import Message

example_messages = builder.build_messages()
# Prepend to actual conversation:
# [*example_messages, {"role": "user", "content": actual_query}]
```

## Dynamic example selection

Select examples nearest to the current query (BM25, cosine similarity, etc.):

```python
def select_examples(query: str, pool: list[FewShotExample], k: int = 3) -> list[FewShotExample]:
    """Simple keyword-overlap selector — replace with embedding similarity for production."""
    scored = [(sum(w in ex.input.lower() for w in query.lower().split()), ex) for ex in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]
```

## Pitfalls

- Keep examples concise — long examples consume tokens that could be used for
  the actual response.
- Maintain consistent formatting between examples and the suffix/instruction;
  mismatches confuse the model.
- For classification tasks, balance positive and negative examples to avoid bias.
- The built-in `FewShotPromptTemplate.render(extra_examples=[...])` supports
  dynamic augmentation on top of the static `examples` list.

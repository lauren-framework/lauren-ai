---
name: output-guardrails
description: Attaches post-generation content filters to agents to redact PII, enforce topic scope, cap response length, or block harmful content before it reaches users. Use when agents generate sensitive content, when compliance requires PII removal, or when responses must stay within a defined topic area.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Output Guardrails & Content Filtering

## Quick start

```python
# NOTE: future annotations are allowed, but tool signature types must resolve at import time.
from lauren_ai import agent, use_guardrails
from lauren_ai import PIIRedactor, LengthFilter, PromptInjectionFilter

@agent(model="claude-opus-4-6", system="You are a helpful assistant.")
@use_guardrails(
    input=[PromptInjectionFilter()],
    output=[PIIRedactor(), LengthFilter(max_chars=2000)],
)
class SafeAgent: ...
```

## Built-in output guardrails

| Class | What it does |
|-------|--------------|
| `PIIRedactor()` | Replaces emails, phone numbers, SSNs, credit cards with `[REDACTED]` |
| `LengthFilter(max_chars=N)` | Blocks responses longer than N characters |
| `TopicFilter(allowed_topics=[...])` | Blocks responses not matching allowed topics |
| `LLMGuardrail(policy="...")` | Evaluates response with a secondary LLM call |

## Custom output guardrail

Implement the `OutputGuardrail` protocol — an `async check(response, ctx)` method
returning a `GuardrailDecision`:

```python
from lauren_ai import guardrail
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision


@guardrail(kind="output")
class TopicScopeGuard:
    """Block responses that don't mention any of the allowed topics."""

    def __init__(self, allowed_topics: list[str]):
        self._topics = allowed_topics

    async def check(self, response: str, ctx: GuardrailContext) -> GuardrailDecision:
        lower = response.lower()
        if any(topic.lower() in lower for topic in self._topics):
            return GuardrailDecision(action="pass", guardrail_name="TopicScopeGuard")
        return GuardrailDecision(
            action="block",
            violation=f"Response is out of scope. Allowed topics: {', '.join(self._topics)}",
            guardrail_name="TopicScopeGuard",
        )
```

## `GuardrailDecision` fields

| Field | Type | Description |
|-------|------|-------------|
| `action` | `"pass"` \| `"block"` \| `"modify"` | What the runner should do |
| `modified_content` | `str \| None` | Replacement text when `action="modify"` |
| `violation` | `str \| None` | Human-readable reason; used as response when `action="block"` |
| `guardrail_name` | `str` | For logging/debugging |

## Decorator stacking order

`@use_guardrails()` sits between `@agent()` and `@use_tools()`:

```python
@agent(model="claude-opus-4-6")   # outermost
@use_guardrails(
    input=[PromptInjectionFilter()],
    output=[PIIRedactor()],
)
@use_tools(my_tool)                # innermost
class MyAgent: ...
```

## How output guardrails execute

1. LLM generates a response.
2. Runner calls each output guardrail in order.
3. First non-`"pass"` decision wins:
   - `"modify"` → `modified_content` replaces the response in memory and in
     the returned `AgentResponse.content`.
   - `"block"` → `violation` text becomes the response; loop ends.
4. Clean responses pass through unchanged.

## Streaming note

In streaming mode (`run_stream`), all chunks are yielded to the caller first
(real-time), then guardrails run on the fully assembled text.  If a guardrail
fires, a sentinel `CompletionChunk(guardrail_override="...")` is appended to
the stream — the frontend should replace the displayed content with this value.

## Pitfalls

- `PIIRedactor` uses regex patterns — it will miss novel PII formats.  For
  GDPR compliance, add a secondary NER-based check.
- `TopicFilter` without `embed_fn` uses keyword matching — pass an
  `embed_fn` for semantic similarity in production.
- Output guardrails run **after** the LLM call — they prevent harmful content
  from reaching users but do not save tokens.  For cost savings, use input
  guardrails to prevent unnecessary completions.

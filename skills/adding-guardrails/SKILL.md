---
name: adding-guardrails
description: Adds content safety filters to lauren-ai agents using @use_guardrails for input and output checks. Use when an agent needs prompt injection protection, PII redaction, topic filtering, length limiting, or LLM-powered content evaluation. Also covers creating custom guardrail classes injectable via @guardrail.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.


# Adding Guardrails to Agents

## Quick start

```python
# NOTE: Do NOT add `from __future__ import annotations` to tool files.
from lauren_ai import agent, tool, use_guardrails, use_tools, PromptInjectionFilter, PIIRedactor, LengthFilter

@tool()
async def lookup_policy(topic: str) -> dict:
    """Look up a company policy.

    Args:
        topic: The policy topic to retrieve.
    """
    return {"topic": topic, "policy": "..."}

@agent(model="claude-opus-4-6", system="You are a compliant assistant.")
@use_guardrails(
    input=[PromptInjectionFilter(), PIIRedactor()],
    output=[LengthFilter(max_chars=8000)],
)
@use_tools(lookup_policy)
class SafeAgent: ...
```

Run it:

```python
from lauren_ai import AgentRunnerBase, LLMConfig

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
runner = AgentRunnerBase(transport=..., tools={}, config=cfg)
result = await runner.run(SafeAgent(), "What is the refund policy?")
print(result.content)
```

---

## Two guardrail decorators — critical distinction

| Decorator | Purpose | Applied to |
|-----------|---------|------------|
| `@use_guardrails(input=[...], output=[...])` | Attaches guardrail **instances** to an `@agent()` class | Agent class |
| `@guardrail(kind="input"\|"output"\|"any")` | Marks a class as a **DI-injectable guardrail provider** (also applies `@injectable`) | Guardrail class |

These are two entirely different decorators for two different purposes.  Do not
confuse them:

- Use `@use_guardrails` on your `@agent()` class to plug in filter instances.
- Use `@guardrail` on a standalone class to make it injectable through the DI
  container so that it can be resolved and passed into `@use_guardrails`.

Import them as separate symbols:

```python
from lauren_ai import use_guardrails, guardrail
```

---

## Built-in guardrails

| Class | Direction | What it does |
|-------|-----------|--------------|
| `PromptInjectionFilter()` | input | Blocks jailbreak and prompt-override attempts |
| `PIIRedactor()` | input (or output) | Redacts email, phone, SSN, credit card with `[REDACTED]` |
| `LengthFilter(max_chars=N)` | output | Truncates output to at most N characters |
| `TopicFilter(blocked=[...])` | input or output | Blocks messages containing listed keywords |
| `LLMGuardrail(policy="...")` | input or output | Uses a separate LLM call to evaluate against a policy |

Full documentation and examples for each: see [guardrails.md](guardrails.md).

---

## Decorator stacking position

`@use_guardrails()` must sit **between** `@agent()` (above) and `@use_tools()`
(below).  The full correct order:

```python
@agent(model="claude-opus-4-6")   # outermost — applied last
@remember(...)                     # optional
@use_guardrails(input=[...])       # between remember and use_tools
@use_tools(my_tool)                # innermost — applied first
class MyAgent: ...
```

Python applies decorators bottom-up, so `@use_tools` runs first, then
`@use_guardrails`, then `@remember`, and finally `@agent` which reads all
collected metadata.  Swapping the order silently drops tools or guardrails.

---

## Conditional guardrails

`None` entries in the lists are silently dropped, which enables feature-flag
or config-driven guard selection:

```python
import os
from lauren_ai import agent, use_guardrails, PromptInjectionFilter, PIIRedactor

pii_guard = PIIRedactor() if os.getenv("ENABLE_PII_REDACTION") else None

@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[PromptInjectionFilter(), pii_guard],  # pii_guard dropped if None
    output=[],
)
class ConditionalAgent: ...
```

---

## DI-injectable guardrails with `@guardrail`

When your guardrail requires injected dependencies (database clients, config,
etc.), define it as a class decorated with `@guardrail`:

```python
from lauren_ai import guardrail, InputGuardrail, GuardrailContext, GuardrailDecision

@guardrail(kind="input")
class AllowlistFilter(InputGuardrail):
    """Blocks users not on the approved list."""

    def __init__(self, db_client: UserDatabase) -> None:
        self._db = db_client

    async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
        user_id = ctx.metadata.get("user_id")
        if user_id and not await self._db.is_allowed(user_id):
            return GuardrailDecision.block("User not on allowlist.")
        return GuardrailDecision.allow()
```

`@guardrail(kind="input")` automatically applies `@injectable`, so the DI
container can resolve `AllowlistFilter` with its dependencies and pass it to
`@use_guardrails`:

```python
@agent(model="claude-opus-4-6")
@use_guardrails(input=[AllowlistFilter])   # pass the class, DI resolves it
@use_tools(my_tool)
class ProtectedAgent: ...
```

---

## Reference

See [guardrails.md](guardrails.md) for:

- Full API for each built-in guardrail
- Implementing the `InputGuardrail` / `OutputGuardrail` protocol
- `GuardrailDecision` API (`allow()`, `allow(replacement=...)`, `block(reason)`)
- Stacking behaviour (first `block` short-circuits remaining guards)
- `GuardrailViolated` exception handling

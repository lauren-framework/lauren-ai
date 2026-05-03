---
name: building-lauren-ai-agents
description: Guides building production-ready AI agents with the lauren-ai framework — decorators, tools with DI, guardrails, memory tiers, multi-agent teams, secure ToolContext-based authentication, and zero-network tests. Use when implementing @agent, @tool, @guardrail, @remember, @team, MockTransport tests, or ToolContext.execution_context authentication in a lauren-ai codebase.
---

# Building Agents with lauren-ai

## Quick start

```python
# NOTE: Do NOT add `from __future__ import annotations` to tool files.
from lauren_ai import agent, guardrail, tool, use_tools, PromptInjectionFilter

@tool()
async def get_price(item: str) -> dict:
    """Look up an item's price.

    Args:
        item: Product name to look up.
    """
    return {"item": item, "price": 9.99}

@agent(model="openai/gpt-4o-mini", system="You are a helpful shop assistant.")
@guardrail(input=[PromptInjectionFilter()])
@use_tools(get_price)
class ShopAgent: ...
```

Run it:

```python
from lauren_ai import AgentRunner, LLMConfig, LLMModule

cfg = LLMConfig(provider="openai", model="openai/gpt-4o-mini")
runner = AgentRunner(transport=..., config=cfg)
result = await runner.run(ShopAgent(), "How much is a Widget?")
print(result.content)
```

---

## Safety invariants — read before writing any tool

| Rule | Why |
|------|-----|
| **Never add `from __future__ import annotations` to tool files** | `@tool()` reads `__annotations__` at decoration time; PEP 563 breaks schema generation silently |
| **Never trust LLM-supplied identity** | Use `ctx.execution_context.request.state.get("user_id")` — see [security.md](security.md) |
| **Always apply `@agent` outermost (top), `@use_tools` innermost (bottom)** | Decorators run bottom-up; wrong order silently drops tools or guardrails |
| **Always call decorators with parentheses** | `@agent()` not `@agent` — bare form raises `DecoratorUsageError` |
| **Class-form tools must define `run()`, not `__call__()`** | `@tool()` looks for `run`; a `__call__` is silently ignored |

---

## Decorator order (required)

```python
@agent(...)       # ← top / outermost — applied last, reads all metadata
@remember(...)    # ← optional
@guardrail(...)   # ← optional
@use_tools(...)   # ← bottom / innermost — applied first, sets USE_TOOLS_META
class MyAgent: ...
```

---

## Reference files

| File | Contents |
|------|----------|
| [agents.md](agents.md) | `@agent`, lifecycle hooks, streaming, delegation, `AgentConfig` params |
| [tools.md](tools.md) | Function-form, class-form (`run()`), `ToolContext`, HITL, caching |
| [security.md](security.md) | **Authentication via `ToolContext → execution_context → request.state`** |
| [guardrails.md](guardrails.md) | Built-in and custom guardrails, stacking, `GuardrailDecision` |
| [memory.md](memory.md) | Short-term, conversation, user (`@remember`), vector stores |
| [testing.md](testing.md) | `MockTransport`, `AgentTestClient`, zero-network tests |
| [teams.md](teams.md) | Multi-agent teams, coordinator vs collaborate, streaming events |

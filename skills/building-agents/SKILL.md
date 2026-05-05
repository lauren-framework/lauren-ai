---
name: building-agents
description: Builds AI agents with the lauren-ai framework using @agent, @use_tools, and AgentRunner. Use when defining an agent class, setting up lifecycle hooks, streaming responses, implementing agent delegation, or configuring the agentic loop with tools, guardrails, and memory.
---

# Building Agents with lauren-ai

## Quick start

```python
# agent.py — from __future__ import annotations IS allowed here (no @tool)
from __future__ import annotations

from lauren_ai import agent, use_tools
from .tools import get_price

@agent(model="claude-opus-4-6", system="You are a helpful shop assistant.")
@use_tools(get_price)
class ShopAgent: ...
```

Run it:

```python
from lauren_ai import AgentRunnerBase, LLMConfig

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
runner = AgentRunnerBase(transport=transport, tools={}, config=cfg)
result = await runner.run(ShopAgent(), "How much is a Widget?")
print(result.content)
```

---

## Decorator order (mandatory)

Decorators are applied **bottom-up** in Python. Always write them top-to-bottom
in this order so each decorator reads the metadata set by those below it:

```
@agent()           ← outermost (top)   — applied last, reads all metadata
@remember()        ← optional
@use_guardrails()  ← optional
@use_tools(...)    ← innermost (bottom) — applied first, sets USE_TOOLS_META
class MyAgent: ...
```

Full example with all optional layers:

```python
from lauren_ai import agent, remember, use_guardrails, use_tools
from lauren_ai import PromptInjectionFilter, PIIRedactor
from .tools import search_database, send_email
from .memory import user_memory_store

@agent(model="claude-opus-4-6", system="You are a personal assistant.", max_turns=10)
@remember(store=user_memory_store, extract=True, inject=True, top_k=5)
@use_guardrails(
    input=[PromptInjectionFilter(), PIIRedactor()],
)
@use_tools(search_database, send_email)
class PersonalAssistant: ...
```

---

## Safety invariants

| Rule | Why it matters |
|------|---------------|
| **`@agent` always outermost (top), `@use_tools` always innermost (bottom)** | Wrong order means `@agent` reads metadata before `@use_tools` writes it — tools silently disappear |
| **Always call decorators with parentheses** | `@agent()` not `@agent` — bare form raises `DecoratorUsageError` at decoration time |
| **Use `@use_guardrails()` to attach guardrails to agents** | `@guardrail()` is for making standalone DI-injectable guardrail classes; `@use_guardrails()` wires them into an agent |
| **Never trust LLM-supplied identity in tools** | Use `ctx.execution_context.request.state.get("user_id")` — see [security.md](../securing-agents/security.md) |
| **`from __future__ import annotations` is safe in agent files but forbidden in tool files** | `@tool()` reads `__annotations__` at decoration time; PEP 563 breaks schema generation in tool files |

---

## AgentConfig parameters

All parameters are passed as keyword arguments to `@agent()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model string, e.g. `"claude-opus-4-6"` |
| `system` | `str` | `""` | System prompt for the agent |
| `max_turns` | `int` | `10` | Maximum agentic loop iterations |
| `max_tokens_per_turn` | `int` | `4096` | Maximum output tokens per LLM call |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0–1.0) |
| `memory_window_tokens` | `int` | `40_000` | Rolling context window size in tokens |
| `max_cost_usd` | `float \| None` | `None` | Hard cost cap; raises `AgentBudgetExceededError` |
| `parallel_tool_calls` | `bool` | `True` | Execute multiple tool calls concurrently |
| `tool_error_policy` | `str` | `"return_error"` | `"raise"` \| `"return_error"` \| `"skip"` |

---

## Reference files

| File | Contents |
|------|----------|
| [agents.md](agents.md) | Full reference: minimal agent, lifecycle hooks, streaming, delegation, all AgentConfig params |
| [../building-tools/tools.md](../building-tools/tools.md) | Function-form and class-form tools, HITL, caching, built-in skills |
| [../securing-agents/security.md](../securing-agents/security.md) | Authentication via `ToolContext.execution_context.request.state` |

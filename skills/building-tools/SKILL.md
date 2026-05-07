---
name: building-tools
description: Creates tools for lauren-ai agents using @tool() on async functions or classes. Use when implementing a @tool() function or class, working with ToolContext for DI injection, adding caching, HITL approval, or using built-in skills like WebSearchTool, HttpFetchTool, or CodeExecutionTool.
---

# Building Tools with lauren-ai

## Critical rule — no PEP 563 in tool files

**Never add `from __future__ import annotations` to any file that defines `@tool()`.**

The `@tool()` decorator calls `inspect.signature()` at decoration time to build
the JSON schema sent to the LLM.  PEP 563 lazy evaluation converts all
annotations to plain strings, silently breaking schema generation.

Add this comment at the top of every tool file:

```python

# The @tool() decorator uses inspect.signature() at decoration time to build
# the JSON schema, and PEP 563 lazy evaluation breaks that introspection.
```

---

## Quick start — function-form tool

```python

from lauren_ai import tool

@tool()
async def get_price(item: str) -> dict:
    """Look up an item's current price.

    Args:
        item: Product name to look up.
    """
    return {"item": item, "price": 9.99}
```

Attach it to an agent:

```python
from lauren_ai import agent, use_tools

@agent(model="claude-opus-4-6", system="You are a helpful shop assistant.")
@use_tools(get_price)
class ShopAgent: ...
```

---

## Quick start — class-form tool

Use a class when the tool needs injected dependencies.  The method must be
named `run()` — not `__call__()`.  The `@tool()` decorator looks for `run`;
a `__call__` is silently ignored and the decorator will raise `ValueError`.

```python

from lauren_ai import tool, ToolContext

@tool()
class SendEmailTool:
    """Send an email to a recipient.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text (plain text).
    """

    def __init__(self, smtp_client: SmtpClient) -> None:
        self._smtp = smtp_client

    async def run(
        self,
        to: str,
        subject: str,
        body: str,
        ctx: ToolContext | None = None,
    ) -> dict:
        await self._smtp.send(to=to, subject=subject, body=body)
        return {"sent": True, "to": to}
```

---

## ToolContext — DI injection, not LLM input

`ToolContext` is **never** included in the JSON schema sent to the LLM.
`ToolExecutor` injects it automatically at call time.  Use it to:

- Access the agent's `AgentContext` via `ctx.agent_context`
- Read the authenticated user identity from `ctx.execution_context.request.state`
- Identify the current turn number via `ctx.turn`

```python
@tool()
async def whoami(ctx: ToolContext | None = None) -> dict:
    """Return the authenticated user's ID.

    (No additional parameters — all identity comes from context.)
    """
    if ctx is None or ctx.execution_context is None:
        return {"user_id": None}
    state = getattr(ctx.execution_context.request, "state", None)
    user_id = state.get("user_id") if state else None
    return {"user_id": user_id}
```

The generated schema contains zero parameters:

```json
{
  "name": "whoami",
  "input_schema": {"type": "object", "properties": {}, "required": []}
}
```

---

## Schema generation rules

| Python annotation | JSON Schema type | Required? |
|-------------------|-----------------|-----------|
| `str` | `"string"` | yes (unless Optional) |
| `int` / `float` | `"number"` | yes (unless Optional) |
| `bool` | `"boolean"` | yes (unless Optional) |
| `dict` | `"object"` | yes (unless Optional) |
| `list` | `"array"` | yes (unless Optional) |
| `Optional[T]` / `T \| None` | type T | no |
| `ctx: ToolContext` / `ctx: ToolContext \| None = None` | — | excluded entirely |
| parameter with default value | type T | no |

Parameter descriptions come from the `Args:` section of the Google-style
docstring.  Always document every parameter.

---

## Safety invariants

| Rule | Why |
|------|-----|
| **`run()` not `__call__()`** | `@tool()` looks for `run`; a `__call__` method is silently ignored |
| **No `from __future__ import annotations`** | PEP 563 breaks `inspect.signature()` used by `@tool()` |
| **Decorate with `@tool()` (parentheses required)** | Bare `@tool` raises `DecoratorUsageError` |
| **Never use LLM-supplied identity** | Read from `ctx.execution_context.request.state` — see [security.md](../securing-agents/security.md) |
| **Always return a `dict`** | LLM receives the JSON-serialised return value as the tool result |

---

## Reference files

| File | Contents |
|------|----------|
| [tools.md](tools.md) | Full reference: schema rules, HITL approval, caching, `on_tool_result` hook, built-in skills |
| [../securing-agents/security.md](../securing-agents/security.md) | Secure tool patterns using `ToolContext.execution_context` |
| [../building-agents/agents.md](../building-agents/agents.md) | `@agent`, lifecycle hooks, attaching tools via `@use_tools` |

---
name: multi-turn-tool-state
description: Shows how to accumulate state across multiple tool calls within a single agent run using AgentContext.metadata. Use when building stateful tools like shopping carts, accumulators, or multi-step workflows where intermediate state must persist between tool invocations within a run.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Multi-Turn Tool Execution with State Carryover

## Critical rule — no PEP 563 in tool files

**Never add `from __future__ import annotations` to any file that defines `@tool()`.**

---

## Overview

Use `ctx.agent_context.metadata` to persist state across multiple tool calls
within a single `runner.run()` call. Read via `ctx.get_metadata(key, default)`,
write via `ctx.agent_context.metadata[key] = value`.

> **Important:** `AgentContext.metadata` persists for the duration of one
> `runner.run()` call (one user turn). It is NOT shared across separate
> `runner.run()` calls. For cross-run persistence, use a conversation store or
> external database.

> **Do not use `ctx.state`** for cross-tool-call state — `ctx.state` is reset
> per tool call, not per run.

---

## Shopping cart example (function-form tool)

```python
# tools/shopping_cart.py — NO from __future__ import annotations
from lauren_ai import tool, ToolContext

CART_KEY = "shopping_cart"

@tool()
async def shopping_cart_tool(
    action: str,
    item: str = "",
    quantity: int = 1,
    ctx: ToolContext | None = None,
) -> dict:
    """Manage a shopping cart with persistent state across tool calls.

    Args:
        action: 'add', 'remove', 'view', or 'clear'.
        item: Item name (for add/remove).
        quantity: Quantity (for add, default 1).
    """
    if ctx is None:
        return {"error": "No context provided"}

    cart: dict = dict(ctx.get_metadata(CART_KEY, {}))

    if action == "add":
        if not item:
            return {"error": "item is required for add"}
        cart[item] = cart.get(item, 0) + quantity
        ctx.agent_context.metadata[CART_KEY] = cart
        return {"added": item, "quantity": cart[item], "cart": dict(cart)}

    elif action == "remove":
        if item in cart:
            del cart[item]
            ctx.agent_context.metadata[CART_KEY] = cart
        return {"removed": item, "cart": dict(cart)}

    elif action == "view":
        return {"cart": dict(cart), "total_items": sum(cart.values())}

    elif action == "clear":
        ctx.agent_context.metadata[CART_KEY] = {}
        return {"cleared": True}

    return {"error": f"Unknown action: {action}"}
```

---

## How state flows

```
Turn 1:  add "apple" → ctx.agent_context.metadata["shopping_cart"] = {"apple": 1}
Turn 2:  add "banana" → ctx.agent_context.metadata["shopping_cart"] = {"apple": 1, "banana": 1}
Turn 3:  view → returns both items
```

`ctx.get_metadata(CART_KEY, {})` reads from `AgentContext.metadata` (after
checking tool-level static metadata via `@set_metadata` first).

---

## AgentRunner test pattern

Queue multiple tool calls in one `client.run()` — the runner processes them in
sequence within a single `AgentContext`, so `metadata` accumulates correctly.

```python
import json
from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


@agent(model=None, system="Shopping cart agent")
@use_tools(shopping_cart_tool)
class CartAgent(_Capture):
    def __init__(self):
        _Capture.__init__(self)


def _c(text):
    return Completion(id="c1", model="mock", content=text, tool_calls=[],
                      stop_reason="end_turn", usage=TokenUsage(10, 5))


def test_cart_accumulates_across_calls():
    agent_inst = CartAgent()
    client = TestClient(agent_inst)
    client.mock.queue_tool_use(
        "shopping_cart_tool", {"action": "add", "item": "Widget", "quantity": 2}
    )
    client.mock.queue_tool_use(
        "shopping_cart_tool", {"action": "add", "item": "Gadget", "quantity": 1}
    )
    client.mock.queue_response(_c("Cart updated."))
    client.run("Add Widget x2 and Gadget x1")

    add1 = json.loads(agent_inst.captured[0].content)
    add2 = json.loads(agent_inst.captured[1].content)
    assert add1["quantity"] == 2
    assert add2["cart"].get("Widget") == 2
    assert add2["cart"].get("Gadget") == 1
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `ToolContext` — `get_metadata`, `agent_context` |
| `src/lauren_ai/_agents/__init__.py` | `AgentContext.metadata` field |
| `src/lauren_ai/_agents/_runner.py` | `_execute_single_tool` — context creation |

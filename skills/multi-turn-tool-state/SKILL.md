---
name: multi-turn-tool-state
description: Shows how to accumulate state across multiple tool calls within a single agent run using ToolContext.state. Use when building stateful tools like shopping carts, accumulators, or multi-step workflows where intermediate state must persist between tool invocations.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Multi-Turn Tool Execution with State Carryover

## Critical rule — no PEP 563 in tool files

**Never add `from __future__ import annotations` to any file that defines `@tool()`.**

---

## Overview

`ToolContext.state` is a mutable `dict[str, Any]` that the agent runner
**shares across all tool calls within the same agent run**.  This allows
class-form tools to accumulate state without instance-level mutation — the
state lives in the context, not in the tool instance.

> **Important:** `ctx.state` is per-run, not per-tool.  Multiple tools sharing
> a run all see the same dict.  Use namespaced keys (e.g. `"shopping_cart"`)
> to avoid collisions.

---

## Shopping cart example

```python
# tools/shopping_cart.py — NO from __future__ import annotations
from lauren_ai import tool, ToolContext

@tool()
class ShoppingCartTool:
    """Manage a shopping cart with persistent state across tool calls.

    Args:
        action: 'add', 'remove', 'view', or 'clear'.
        item: Item name (for add/remove).
        quantity: Quantity to add (default 1).
    """
    CART_KEY = "shopping_cart"

    async def run(
        self, ctx: ToolContext, action: str, item: str = "", quantity: int = 1
    ) -> dict:
        cart: dict[str, int] = ctx.state.get(self.CART_KEY, {})

        if action == "add":
            if not item:
                return {"error": "item is required for add"}
            cart[item] = cart.get(item, 0) + quantity
            ctx.state[self.CART_KEY] = cart
            return {"added": item, "quantity": cart[item], "cart": cart}

        elif action == "remove":
            if item in cart:
                del cart[item]
                ctx.state[self.CART_KEY] = cart
            return {"removed": item, "cart": cart}

        elif action == "view":
            return {"cart": cart, "total_items": sum(cart.values())}

        elif action == "clear":
            ctx.state[self.CART_KEY] = {}
            return {"cleared": True}

        return {"error": f"Unknown action: {action}"}
```

---

## How state flows

```
Turn 1:  add "apple" → ctx.state["shopping_cart"] = {"apple": 1}
Turn 2:  add "banana" → ctx.state["shopping_cart"] = {"apple": 1, "banana": 1}
Turn 3:  view → returns both items
```

The runner creates `ToolContext(state={})` fresh per **run**, not per turn.
All tool calls within the same run share that `state` dict.

---

## Testing tool state directly (without full agent loop)

Create a mock context to unit-test tools without spinning up a runner:

```python
class MockContext:
    def __init__(self):
        self.state = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0

    def get_metadata(self, key, default=None):
        return default
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `ToolContext.state` field definition |
| `src/lauren_ai/_agents/_runner.py` | `_execute_single_tool` — context creation |

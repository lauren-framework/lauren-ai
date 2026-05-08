"""Integration tests for Skill 44: Multi-Turn Tool Execution with State Carryover.

Tests cover:
- Add item to cart → accumulates in ctx.state
- Add same item twice → quantity increases
- Add different items across calls → both present
- Remove item → removes from cart
- View cart → returns current contents and total
- Clear → resets to empty
- Error when item is missing for add action

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import pytest

from lauren_ai._tools import ToolContext
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import _add_to_tool_map


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------

from lauren_ai._tools import tool


@tool()
class ShoppingCartTool:
    """Manage a shopping cart with persistent state across tool calls.

    Args:
        action: 'add', 'remove', 'view', or 'clear'.
        item: Item name (for add/remove).
        quantity: Quantity (for add, default 1).
    """
    CART_KEY = "shopping_cart"

    async def run(
        self, ctx: ToolContext, action: str, item: str = "", quantity: int = 1
    ) -> dict:
        cart: dict = ctx.state.get(self.CART_KEY, {})

        if action == "add":
            if not item:
                return {"error": "item is required for add"}
            cart[item] = cart.get(item, 0) + quantity
            ctx.state[self.CART_KEY] = cart
            return {"added": item, "quantity": cart[item], "cart": dict(cart)}
        elif action == "remove":
            if item in cart:
                del cart[item]
                ctx.state[self.CART_KEY] = cart
            return {"removed": item, "cart": dict(cart)}
        elif action == "view":
            total_items = sum(cart.values())
            return {"cart": dict(cart), "total_items": total_items}
        elif action == "clear":
            ctx.state[self.CART_KEY] = {}
            return {"cleared": True}
        return {"error": f"Unknown action: {action}"}


# ---------------------------------------------------------------------------
# Mock context helper (shared state simulates same run)
# ---------------------------------------------------------------------------

class MockContext:
    def __init__(self):
        self.state = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShoppingCartAdd:
    async def test_add_single_item(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        result = await cart.run(ctx, "add", item="apple")
        assert result["added"] == "apple"
        assert result["quantity"] == 1

    async def test_add_accumulates_quantity(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        await cart.run(ctx, "add", item="apple")
        result = await cart.run(ctx, "add", item="apple")
        assert result["quantity"] == 2

    async def test_add_multiple_items_accumulate(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        await cart.run(ctx, "add", item="apple")
        await cart.run(ctx, "add", item="banana")
        result = await cart.run(ctx, "view")
        assert "apple" in result["cart"]
        assert "banana" in result["cart"]

    async def test_add_with_explicit_quantity(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        result = await cart.run(ctx, "add", item="milk", quantity=3)
        assert result["quantity"] == 3

    async def test_add_missing_item_returns_error(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        result = await cart.run(ctx, "add")
        assert "error" in result


class TestShoppingCartView:
    async def test_view_empty_cart(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        result = await cart.run(ctx, "view")
        assert result["cart"] == {}
        assert result["total_items"] == 0

    async def test_view_after_adds(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        await cart.run(ctx, "add", item="apple", quantity=2)
        await cart.run(ctx, "add", item="bread", quantity=1)
        result = await cart.run(ctx, "view")
        assert result["total_items"] == 3

    async def test_state_persists_across_invocations(self):
        """State dict is shared across calls on the same MockContext."""
        cart = ShoppingCartTool()
        ctx = MockContext()  # same context = same run state
        await cart.run(ctx, "add", item="item1")
        await cart.run(ctx, "add", item="item2")
        await cart.run(ctx, "add", item="item3")
        result = await cart.run(ctx, "view")
        assert len(result["cart"]) == 3


class TestShoppingCartRemove:
    async def test_remove_existing_item(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        await cart.run(ctx, "add", item="apple")
        result = await cart.run(ctx, "remove", item="apple")
        assert "apple" not in result["cart"]

    async def test_remove_nonexistent_item_no_error(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        result = await cart.run(ctx, "remove", item="nonexistent")
        assert "error" not in result


class TestShoppingCartClear:
    async def test_clear_resets_state(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        await cart.run(ctx, "add", item="apple")
        await cart.run(ctx, "add", item="banana")
        await cart.run(ctx, "clear")
        result = await cart.run(ctx, "view")
        assert result["cart"] == {}
        assert result["total_items"] == 0

    async def test_clear_returns_cleared_true(self):
        cart = ShoppingCartTool()
        ctx = MockContext()
        result = await cart.run(ctx, "clear")
        assert result["cleared"] is True


class TestShoppingCartAgentRunner:
    async def test_tool_invoked_via_runner(self):
        """Verify ShoppingCartTool works when dispatched through AgentRunner."""
        tools = {}
        _add_to_tool_map(tools, ShoppingCartTool)
        mock = MockTransport()
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools=tools, config=cfg)

        @use_tools(ShoppingCartTool)
        @agent(model="mock-model")
        class CartAgent: ...

        mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "apple"})
        mock.queue_response(_completion("Added apple to cart."))

        resp = await runner.run(CartAgent(), "Add apple to cart")
        assert resp.content == "Added apple to cart."

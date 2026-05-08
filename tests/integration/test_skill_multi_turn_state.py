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

import asyncio

from unittest.mock import MagicMock

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import tool, ToolContext, _add_to_tool_map
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------


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
# MockToolContext helper
# ---------------------------------------------------------------------------


def _tool_ctx(state=None):
    ctx = MagicMock()
    ctx.execution_context = None
    ctx.agent_context = MagicMock()
    ctx.agent_context.metadata = {}
    ctx.get_metadata = lambda k, d=None: ctx.agent_context.metadata.get(k, d)
    ctx.state = state if state is not None else {}
    ctx.tool_use_id = "t1"
    ctx.turn = 0
    return ctx


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShoppingCartAdd:
    def test_add_single_item(self):
        state = {}
        tool = ShoppingCartTool()
        ctx = _tool_ctx(state=state)
        result = asyncio.run(tool.run(ctx, "add", item="apple"))
        assert result["added"] == "apple"
        assert result["quantity"] == 1

    def test_add_accumulates_quantity(self):
        state = {}
        tool = ShoppingCartTool()
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="apple"))
        result = asyncio.run(tool.run(_tool_ctx(state=state), "add", item="apple"))
        assert result["quantity"] == 2

    def test_add_multiple_items_accumulate(self):
        state = {}
        tool = ShoppingCartTool()
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="apple"))
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="banana"))
        result = asyncio.run(tool.run(_tool_ctx(state=state), "view"))
        assert "apple" in result["cart"]
        assert "banana" in result["cart"]

    def test_add_with_explicit_quantity(self):
        state = {}
        tool = ShoppingCartTool()
        result = asyncio.run(tool.run(_tool_ctx(state=state), "add", item="milk", quantity=3))
        assert result["quantity"] == 3

    def test_add_missing_item_returns_error(self):
        state = {}
        tool = ShoppingCartTool()
        result = asyncio.run(tool.run(_tool_ctx(state=state), "add", item=""))
        assert "error" in result


class TestShoppingCartView:
    def test_view_empty_cart(self):
        state = {}
        tool = ShoppingCartTool()
        result = asyncio.run(tool.run(_tool_ctx(state=state), "view"))
        assert result["cart"] == {}
        assert result["total_items"] == 0

    def test_view_after_adds(self):
        state = {}
        tool = ShoppingCartTool()
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="apple", quantity=2))
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="bread", quantity=1))
        result = asyncio.run(tool.run(_tool_ctx(state=state), "view"))
        assert result["total_items"] == 3

    def test_state_persists_across_invocations(self):
        state = {}
        tool = ShoppingCartTool()
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="item1"))
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="item2"))
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="item3"))
        result = asyncio.run(tool.run(_tool_ctx(state=state), "view"))
        assert len(result["cart"]) == 3


class TestShoppingCartRemove:
    def test_remove_existing_item(self):
        state = {}
        tool = ShoppingCartTool()
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="apple"))
        result = asyncio.run(tool.run(_tool_ctx(state=state), "remove", item="apple"))
        assert "apple" not in result["cart"]

    def test_remove_nonexistent_item_no_error(self):
        state = {}
        tool = ShoppingCartTool()
        result = asyncio.run(tool.run(_tool_ctx(state=state), "remove", item="nonexistent"))
        assert "error" not in result


class TestShoppingCartClear:
    def test_clear_resets_state(self):
        state = {}
        tool = ShoppingCartTool()
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="apple"))
        asyncio.run(tool.run(_tool_ctx(state=state), "add", item="banana"))
        asyncio.run(tool.run(_tool_ctx(state=state), "clear"))
        result = asyncio.run(tool.run(_tool_ctx(state=state), "view"))
        assert result["cart"] == {}
        assert result["total_items"] == 0

    def test_clear_returns_cleared_true(self):
        state = {}
        tool = ShoppingCartTool()
        result = asyncio.run(tool.run(_tool_ctx(state=state), "clear"))
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

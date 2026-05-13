"""Integration tests for Skill 44: Multi-Turn Tool Execution with State Carryover.

Tests cover:
- Add item to cart → accumulates in AgentContext.metadata
- Add same item twice → quantity increases
- Add different items across calls → both present
- Remove item → removes from cart
- View cart → returns current contents and total
- Clear → resets to empty
- Error when item is missing for add action
- Multi-tool calls in one AgentRunner turn accumulate cart state

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import json

from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Tool definition — state stored in AgentContext.metadata (persists per run)
# ---------------------------------------------------------------------------

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
        total_items = sum(cart.values())
        return {"cart": dict(cart), "total_items": total_items}
    elif action == "clear":
        ctx.agent_context.metadata[CART_KEY] = {}
        return {"cleared": True}
    return {"error": f"Unknown action: {action}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text, *, n=1, stop="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock",
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(10, 5),
    )


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


@agent(model="mock-model", system="Shopping cart agent")
@use_tools(shopping_cart_tool)
class CartAgent(_Capture):
    def __init__(self):
        _Capture.__init__(self)


# ---------------------------------------------------------------------------
# Tests: add operations
# ---------------------------------------------------------------------------


class TestShoppingCartAdd:
    def test_add_single_item(self):
        """Adding one item yields quantity=1."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "apple"})
        client.mock.queue_response(_c("Added apple."))
        client.run("Add apple")
        result = json.loads(agent_inst.captured[0].content)
        assert result["added"] == "apple"
        assert result["quantity"] == 1

    def test_add_accumulates_quantity_across_tool_calls(self):
        """Two add calls for the same item in one run accumulate quantity."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "apple"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "apple"})
        client.mock.queue_response(_c("Added apples."))
        client.run("Add two apples")
        first = json.loads(agent_inst.captured[0].content)
        second = json.loads(agent_inst.captured[1].content)
        assert first["quantity"] == 1
        assert second["quantity"] == 2

    def test_add_multiple_items_accumulate(self):
        """Adding different items in one run — both appear in the cart."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "apple"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "banana"})
        client.mock.queue_response(_c("Added both."))
        client.run("Add apple and banana")
        second = json.loads(agent_inst.captured[1].content)
        assert "apple" in second["cart"]
        assert "banana" in second["cart"]

    def test_add_with_explicit_quantity(self):
        """Specifying quantity=3 stores 3 units."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "shopping_cart_tool", {"action": "add", "item": "milk", "quantity": 3}
        )
        client.mock.queue_response(_c("Added milk."))
        client.run("Add 3 milks")
        result = json.loads(agent_inst.captured[0].content)
        assert result["quantity"] == 3

    def test_add_missing_item_returns_error(self):
        """add with empty item returns an error."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": ""})
        client.mock.queue_response(_c("Error."))
        client.run("Add nothing")
        result = json.loads(agent_inst.captured[0].content)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: view
# ---------------------------------------------------------------------------


class TestShoppingCartView:
    def test_view_empty_cart(self):
        """Viewing an empty cart returns empty dict and total_items=0."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "view"})
        client.mock.queue_response(_c("Empty cart."))
        client.run("View cart")
        result = json.loads(agent_inst.captured[0].content)
        assert result["cart"] == {}
        assert result["total_items"] == 0

    def test_view_after_adds(self):
        """View after adding two items shows correct total."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "shopping_cart_tool", {"action": "add", "item": "apple", "quantity": 2}
        )
        client.mock.queue_tool_use(
            "shopping_cart_tool", {"action": "add", "item": "bread", "quantity": 1}
        )
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "view"})
        client.mock.queue_response(_c("Here is your cart."))
        client.run("Add items and view")
        view = json.loads(agent_inst.captured[2].content)
        assert view["total_items"] == 3

    def test_state_persists_across_invocations(self):
        """Multiple adds in one run — view shows all items."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "item1"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "item2"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "item3"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "view"})
        client.mock.queue_response(_c("Cart has 3 items."))
        client.run("Add 3 items then view")
        view = json.loads(agent_inst.captured[3].content)
        assert len(view["cart"]) == 3


# ---------------------------------------------------------------------------
# Tests: remove
# ---------------------------------------------------------------------------


class TestShoppingCartRemove:
    def test_remove_existing_item(self):
        """Adding then removing an item leaves it out of the cart."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "apple"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "remove", "item": "apple"})
        client.mock.queue_response(_c("Removed."))
        client.run("Add then remove apple")
        remove_result = json.loads(agent_inst.captured[1].content)
        assert "apple" not in remove_result["cart"]

    def test_remove_nonexistent_item_no_error(self):
        """Removing an item that doesn't exist returns no error."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "shopping_cart_tool", {"action": "remove", "item": "nonexistent"}
        )
        client.mock.queue_response(_c("Removed nothing."))
        client.run("Remove nonexistent")
        result = json.loads(agent_inst.captured[0].content)
        assert "error" not in result


# ---------------------------------------------------------------------------
# Tests: clear
# ---------------------------------------------------------------------------


class TestShoppingCartClear:
    def test_clear_resets_state(self):
        """After clear, viewing the cart shows empty."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "apple"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "add", "item": "banana"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "clear"})
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "view"})
        client.mock.queue_response(_c("Cart cleared."))
        client.run("Fill and clear cart")
        view = json.loads(agent_inst.captured[3].content)
        assert view["cart"] == {}
        assert view["total_items"] == 0

    def test_clear_returns_cleared_true(self):
        """clear action returns {cleared: True}."""
        agent_inst = CartAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("shopping_cart_tool", {"action": "clear"})
        client.mock.queue_response(_c("Cleared."))
        client.run("Clear cart")
        result = json.loads(agent_inst.captured[0].content)
        assert result["cleared"] is True

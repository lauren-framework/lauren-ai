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

from pydantic import BaseModel

from lauren import LaurenFactory, controller, delete, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai._tools import tool, ToolContext
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import _add_to_tool_map


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
# Controller — holds shared state across requests (singleton scope)
# ---------------------------------------------------------------------------


class _ActionRequest(BaseModel):
    action: str
    item: str = ""
    quantity: int = 1


@injectable(scope=Scope.SINGLETON)
class CartController:
    def __init__(self) -> None:
        self._tool = ShoppingCartTool()
        self._state: dict = {}  # simulates ToolContext.state carried across calls

    @property
    def _ctx(self) -> "_MockCtx":
        # Return a shared context that reuses the same state dict
        ctx = _MockCtx(state=self._state)
        return ctx

    async def action(self, req: _ActionRequest) -> dict:
        return await self._tool.run(self._ctx, req.action, req.item, req.quantity)

    async def clear(self) -> dict:
        result = await self._tool.run(self._ctx, "clear")
        return result

    def reset(self) -> None:
        self._state.clear()


@controller("/cart")
@injectable(scope=Scope.SINGLETON)
class CartHttpController:
    def __init__(self, cart: CartController) -> None:
        self._cart = cart

    @post("/action")
    async def action(self, body: Json[_ActionRequest]) -> dict:
        return await self._cart.action(body)

    @delete("/clear")
    async def clear(self) -> dict:
        return await self._cart.clear()


@module(controllers=[CartHttpController], providers=[CartController])
class CartModule: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCtx:
    def __init__(self, state: "dict | None" = None) -> None:
        self.state: dict = state if state is not None else {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(CartModule))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShoppingCartAdd:
    def test_add_single_item(self):
        client = build_app()
        r = client.post("/cart/action", json={"action": "add", "item": "apple"})
        assert r.status_code == 200
        data = r.json()
        assert data["added"] == "apple"
        assert data["quantity"] == 1

    def test_add_accumulates_quantity(self):
        client = build_app()
        client.post("/cart/action", json={"action": "add", "item": "apple"})
        r = client.post("/cart/action", json={"action": "add", "item": "apple"})
        assert r.status_code == 200
        assert r.json()["quantity"] == 2

    def test_add_multiple_items_accumulate(self):
        client = build_app()
        client.post("/cart/action", json={"action": "add", "item": "apple"})
        client.post("/cart/action", json={"action": "add", "item": "banana"})
        r = client.post("/cart/action", json={"action": "view"})
        assert r.status_code == 200
        cart = r.json()["cart"]
        assert "apple" in cart
        assert "banana" in cart

    def test_add_with_explicit_quantity(self):
        client = build_app()
        r = client.post("/cart/action", json={"action": "add", "item": "milk", "quantity": 3})
        assert r.status_code == 200
        assert r.json()["quantity"] == 3

    def test_add_missing_item_returns_error(self):
        client = build_app()
        r = client.post("/cart/action", json={"action": "add", "item": ""})
        assert r.status_code == 200
        assert "error" in r.json()


class TestShoppingCartView:
    def test_view_empty_cart(self):
        client = build_app()
        r = client.post("/cart/action", json={"action": "view"})
        assert r.status_code == 200
        data = r.json()
        assert data["cart"] == {}
        assert data["total_items"] == 0

    def test_view_after_adds(self):
        client = build_app()
        client.post("/cart/action", json={"action": "add", "item": "apple", "quantity": 2})
        client.post("/cart/action", json={"action": "add", "item": "bread", "quantity": 1})
        r = client.post("/cart/action", json={"action": "view"})
        assert r.status_code == 200
        assert r.json()["total_items"] == 3

    def test_state_persists_across_invocations(self):
        client = build_app()
        client.post("/cart/action", json={"action": "add", "item": "item1"})
        client.post("/cart/action", json={"action": "add", "item": "item2"})
        client.post("/cart/action", json={"action": "add", "item": "item3"})
        r = client.post("/cart/action", json={"action": "view"})
        assert r.status_code == 200
        assert len(r.json()["cart"]) == 3


class TestShoppingCartRemove:
    def test_remove_existing_item(self):
        client = build_app()
        client.post("/cart/action", json={"action": "add", "item": "apple"})
        r = client.post("/cart/action", json={"action": "remove", "item": "apple"})
        assert r.status_code == 200
        assert "apple" not in r.json()["cart"]

    def test_remove_nonexistent_item_no_error(self):
        client = build_app()
        r = client.post("/cart/action", json={"action": "remove", "item": "nonexistent"})
        assert r.status_code == 200
        assert "error" not in r.json()


class TestShoppingCartClear:
    def test_clear_resets_state(self):
        client = build_app()
        client.post("/cart/action", json={"action": "add", "item": "apple"})
        client.post("/cart/action", json={"action": "add", "item": "banana"})
        client.delete("/cart/clear")
        r = client.post("/cart/action", json={"action": "view"})
        assert r.status_code == 200
        data = r.json()
        assert data["cart"] == {}
        assert data["total_items"] == 0

    def test_clear_returns_cleared_true(self):
        client = build_app()
        r = client.delete("/cart/clear")
        assert r.status_code == 200
        assert r.json()["cleared"] is True


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

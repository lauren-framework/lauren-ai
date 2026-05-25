"""Integration tests for the ``@use_hooks()`` / ``global_tool_hooks`` feature.

Verifies that injectable hooks wire correctly through ``AgentModule.for_root()``
and fire during a real agent run driven by ``MockTransport``.
"""

from __future__ import annotations

from typing import Any

import pytest

from lauren_ai._tools import ToolContext, _add_to_tool_map, tool, use_hooks
from lauren_ai._tools._executor import ToolCall as ExecutorToolCall
from lauren_ai._tools._executor import ToolExecutor
from lauren_ai._tools._hooks import (
    AfterToolHookDecision,
    BeforeToolHookDecision,
    ErrorToolHookDecision,
    ToolCallContext,
    ToolHook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> ToolContext:
    return ToolContext(agent_context=None, tool_use_id="t1", turn=0)


def _tool_map(*tool_funcs, hook_instances: tuple = ()) -> dict:
    """Build a tool map; attach hook instances to all tools."""
    tm: dict = {}
    for fn in tool_funcs:
        _add_to_tool_map(tm, fn)
    if hook_instances:
        for name in tm:
            tm[name][1].resolved_hooks = hook_instances
    return tm


# ---------------------------------------------------------------------------
# Test 1: Per-tool hook fires via DI — wired through AgentModule
# ---------------------------------------------------------------------------


class TestPerToolHookViaDI:
    def test_per_tool_hook_wired_and_fires(self):
        calls: list[str] = []

        class AuditHook(ToolHook):
            async def before_tool_call(self, ctx: ToolCallContext) -> BeforeToolHookDecision:
                calls.append(f"before:{ctx.tool_name}")
                return BeforeToolHookDecision.proceed()

            async def after_tool_call(self, result: Any, ctx: ToolCallContext) -> AfterToolHookDecision:
                calls.append(f"after:{ctx.tool_name}")
                return AfterToolHookDecision.proceed()

        @use_hooks(AuditHook)
        @tool()
        async def get_price(item: str) -> str:
            """Return price of an item.

            Args:
                item: Item name.
            """
            return f"price:{item}"

        from lauren import LaurenFactory, controller, get, module
        from lauren.testing import TestClient

        from lauren_ai._agents import agent, use_tools
        from lauren_ai._config import LLMConfig
        from lauren_ai._module import AgentModule, LLMModule
        from lauren_ai._transport import Completion, TokenUsage

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("get_price", {"item": "apple"})
        mock.queue_response(
            Completion(
                id="c2",
                model="mock",
                content="Price is price:apple.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=10),
            )
        )

        @agent(model="mock-model")
        @use_tools(get_price)
        class PriceAgent:
            pass

        from lauren_ai._agents._runner import AgentRunner

        @controller("/test")
        class TestCtrl:
            def __init__(self, runner: AgentRunner, ag: PriceAgent) -> None:
                self._runner = runner
                self._ag = ag

            @get("/run")
            async def run_it(self) -> dict:
                await self._runner.run(self._ag, "What is the price of apple?")
                return {"ok": True}

        LLMModule_cls = LLMModule.for_root(cfg, transport_override=mock)
        AgentMod = AgentModule.for_root(
            agents=[PriceAgent],
            tools=[get_price],
            imports=LLMModule_cls,
        )

        @module(imports=[LLMModule_cls, AgentMod], controllers=[TestCtrl])
        class AppModule:
            pass

        r = TestClient(LaurenFactory.create(AppModule)).get("/test/run")
        assert r.status_code == 200

        assert "before:get_price" in calls
        assert "after:get_price" in calls


# ---------------------------------------------------------------------------
# Test 2: Global hook fires via AgentModule.for_root(global_tool_hooks=...)
# ---------------------------------------------------------------------------


class TestGlobalHookViaDI:
    def test_global_hook_fires_for_all_tools(self):
        global_calls: list[str] = []

        class GlobalAuditHook(ToolHook):
            async def before_tool_call(self, ctx: ToolCallContext) -> BeforeToolHookDecision:
                global_calls.append(ctx.tool_name)
                return BeforeToolHookDecision.proceed()

        @tool()
        async def tool_a_g(x: str) -> str:
            """Tool A.

            Args:
                x: Input.
            """
            return f"a:{x}"

        from lauren import LaurenFactory, controller, get, module
        from lauren.testing import TestClient

        from lauren_ai._agents import agent, use_tools
        from lauren_ai._config import LLMConfig
        from lauren_ai._module import AgentModule, LLMModule
        from lauren_ai._transport import Completion, TokenUsage

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("tool_a_g", {"x": "hello"})
        mock.queue_response(
            Completion(
                id="c2",
                model="mock",
                content="Done.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=10),
            )
        )

        @agent(model="mock-model")
        @use_tools(tool_a_g)
        class AgentA:
            pass

        from lauren_ai._agents._runner import AgentRunner

        @controller("/test2")
        class TestCtrl2:
            def __init__(self, runner: AgentRunner, ag: AgentA) -> None:
                self._runner = runner
                self._ag = ag

            @get("/run")
            async def run_it(self) -> dict:
                await self._runner.run(self._ag, "Run tool_a_g")
                return {"ok": True}

        LLMModule_cls = LLMModule.for_root(cfg, transport_override=mock)
        AgentMod = AgentModule.for_root(
            agents=[AgentA],
            tools=[tool_a_g],
            imports=LLMModule_cls,
            global_tool_hooks=[GlobalAuditHook],
        )

        @module(imports=[LLMModule_cls, AgentMod], controllers=[TestCtrl2])
        class AppModule2:
            pass

        r = TestClient(LaurenFactory.create(AppModule2)).get("/test2/run")
        assert r.status_code == 200

        assert "tool_a_g" in global_calls


# ---------------------------------------------------------------------------
# Tests 3-7: Drive via ToolExecutor directly (fast, no full DI needed)
# ---------------------------------------------------------------------------


class TestBeforeHookAbort:
    @pytest.mark.asyncio
    async def test_abort_skips_tool_execution(self):
        executed: list[bool] = []

        @tool()
        async def secret_tool(query: str) -> str:
            """Secret tool.

            Args:
                query: Query.
            """
            executed.append(True)
            return "secret"

        class BlockHook(ToolHook):
            async def before_tool_call(self, ctx: ToolCallContext) -> BeforeToolHookDecision:
                return BeforeToolHookDecision.abort(result="blocked by policy")

        tm = _tool_map(secret_tool, hook_instances=(BlockHook(),))
        executor = ToolExecutor(tools=tm)
        result = await executor.execute(
            ExecutorToolCall(tool_use_id="t1", name="secret_tool", input={"query": "x"}),
            _ctx(),
        )

        assert not executed
        assert "blocked by policy" in result.content


class TestErrorHookSuppression:
    @pytest.mark.asyncio
    async def test_suppressed_error_returns_fallback(self):
        @tool()
        async def broken_tool(q: str) -> str:
            """Always raises.

            Args:
                q: Input.
            """
            raise RuntimeError("db connection failed")

        class RecoverHook(ToolHook):
            async def on_tool_error(self, exc: Exception, ctx: ToolCallContext) -> ErrorToolHookDecision:
                return ErrorToolHookDecision.suppress_with("fallback response")

        tm = _tool_map(broken_tool, hook_instances=(RecoverHook(),))
        executor = ToolExecutor(tools=tm)
        result = await executor.execute(
            ExecutorToolCall(tool_use_id="t1", name="broken_tool", input={"q": "x"}),
            _ctx(),
        )

        assert "fallback response" in result.content


class TestHookOrdering:
    @pytest.mark.asyncio
    async def test_exact_hook_execution_order(self):
        order: list[str] = []

        class GlobalHook(ToolHook):
            async def before_tool_call(self, ctx):
                order.append("global_before")
                return BeforeToolHookDecision.proceed()

            async def after_tool_call(self, result, ctx):
                order.append("global_after")
                return AfterToolHookDecision.proceed()

        class PerToolHook(ToolHook):
            async def before_tool_call(self, ctx):
                order.append("pertool_before")
                return BeforeToolHookDecision.proceed()

            async def after_tool_call(self, result, ctx):
                order.append("pertool_after")
                return AfterToolHookDecision.proceed()

        @tool()
        async def ordered_tool(x: str) -> str:
            """Ordered tool.

            Args:
                x: Input.
            """
            order.append("tool")
            return x

        tm = _tool_map(ordered_tool, hook_instances=(PerToolHook(),))
        executor = ToolExecutor(tools=tm, global_hooks=[GlobalHook()])
        await executor.execute(
            ExecutorToolCall(tool_use_id="t1", name="ordered_tool", input={"x": "v"}),
            _ctx(),
        )

        assert order == [
            "global_before",
            "pertool_before",
            "tool",
            "pertool_after",
            "global_after",
        ]


class TestAfterHookReplace:
    @pytest.mark.asyncio
    async def test_after_hook_replace_overrides_result(self):
        @tool()
        async def raw_tool(x: str) -> str:
            """Raw tool.

            Args:
                x: Input.
            """
            return f"raw:{x}"

        class TransformHook(ToolHook):
            async def after_tool_call(self, result, ctx):
                return AfterToolHookDecision.replace("transformed!")

        tm = _tool_map(raw_tool, hook_instances=(TransformHook(),))
        executor = ToolExecutor(tools=tm)
        result = await executor.execute(
            ExecutorToolCall(tool_use_id="t1", name="raw_tool", input={"x": "hello"}),
            _ctx(),
        )

        assert "transformed!" in result.content


class TestModifyInputPropagates:
    @pytest.mark.asyncio
    async def test_modified_input_reaches_tool(self):
        received: list[str] = []

        @tool()
        async def echo_tool(msg: str) -> str:
            """Echo tool.

            Args:
                msg: Message.
            """
            received.append(msg)
            return msg

        class SanitizeHook(ToolHook):
            async def before_tool_call(self, ctx):
                return BeforeToolHookDecision.modify({"msg": "sanitized"})

        tm = _tool_map(echo_tool, hook_instances=(SanitizeHook(),))
        executor = ToolExecutor(tools=tm)
        await executor.execute(
            ExecutorToolCall(tool_use_id="t1", name="echo_tool", input={"msg": "original"}),
            _ctx(),
        )

        assert received == ["sanitized"]

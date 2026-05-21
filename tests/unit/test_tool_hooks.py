"""Unit tests for the ``@use_hooks()`` decorator and injectable tool hook system."""

from __future__ import annotations

import asyncio

import pytest

from lauren_ai._tools import TOOL_META, USE_HOOKS_META, ToolContext, ToolMeta, tool, use_hooks
from lauren_ai._tools._executor import ToolCall, ToolExecutor
from lauren_ai._tools._hooks import (
    _NO_REPLACE,
    AfterToolHookDecision,
    BeforeToolHookDecision,
    ErrorToolHookDecision,
    ToolCallContext,
    ToolHook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_context() -> ToolContext:
    return ToolContext(agent_context=None, tool_use_id="tid-1", turn=0)


def _make_hook_ctx(name: str = "my_tool", input: dict | None = None) -> ToolCallContext:
    return ToolCallContext(
        agent_context=None,
        tool_use_id="tid-1",
        turn=0,
        tool_name=name,
        tool_input=input or {},
    )


# ---------------------------------------------------------------------------
# ToolCallContext
# ---------------------------------------------------------------------------


class TestToolCallContext:
    def test_extends_tool_context(self):
        ctx = _make_hook_ctx()
        assert isinstance(ctx, ToolContext)

    def test_carries_tool_name_and_input(self):
        ctx = _make_hook_ctx("search", {"query": "hello"})
        assert ctx.tool_name == "search"
        assert ctx.tool_input == {"query": "hello"}


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


class TestBeforeToolHookDecision:
    def test_proceed(self):
        d = BeforeToolHookDecision.proceed()
        assert not d._aborted
        assert d._modified_input is None

    def test_abort(self):
        d = BeforeToolHookDecision.abort(result={"error": "blocked"})
        assert d._aborted
        assert d._abort_result == {"error": "blocked"}

    def test_modify(self):
        d = BeforeToolHookDecision.modify({"q": "new"})
        assert not d._aborted
        assert d._modified_input == {"q": "new"}


class TestAfterToolHookDecision:
    def test_proceed_no_replace(self):
        d = AfterToolHookDecision.proceed()
        assert d._replacement is _NO_REPLACE

    def test_replace(self):
        d = AfterToolHookDecision.replace("override")
        assert d._replacement == "override"


class TestErrorToolHookDecision:
    def test_reraise(self):
        d = ErrorToolHookDecision.reraise()
        assert not d._suppressed

    def test_suppress_with(self):
        d = ErrorToolHookDecision.suppress_with("fallback")
        assert d._suppressed
        assert d._fallback == "fallback"


# ---------------------------------------------------------------------------
# ToolHook base class default no-ops
# ---------------------------------------------------------------------------


class TestToolHookBase:
    def test_default_before_proceeds(self):
        async def _run():
            return await ToolHook().before_tool_call(_make_hook_ctx())

        result = asyncio.run(_run())
        assert isinstance(result, BeforeToolHookDecision)
        assert not result._aborted

    def test_default_after_proceeds(self):
        async def _run():
            return await ToolHook().after_tool_call("raw", _make_hook_ctx())

        result = asyncio.run(_run())
        assert isinstance(result, AfterToolHookDecision)
        assert result._replacement is _NO_REPLACE

    def test_default_error_reraised(self):
        async def _run():
            return await ToolHook().on_tool_error(ValueError("boom"), _make_hook_ctx())

        result = asyncio.run(_run())
        assert isinstance(result, ErrorToolHookDecision)
        assert not result._suppressed


# ---------------------------------------------------------------------------
# @use_hooks decorator
# ---------------------------------------------------------------------------


class TestUseHooksDecorator:
    def test_sets_attribute_on_fn_tool(self):
        class MyHook(ToolHook):
            pass

        @use_hooks(MyHook)
        @tool()
        async def fn_tool(x: str) -> str:
            """A tool.

            Args:
                x: Input.
            """
            return x

        assert getattr(fn_tool, USE_HOOKS_META) == (MyHook,)

    def test_hook_classes_in_tool_meta(self):
        """ToolMeta.hook_classes is synced when @use_hooks is applied above @tool()."""

        class MyHook(ToolHook):
            pass

        @use_hooks(MyHook)
        @tool()
        async def fn_tool2(x: str) -> str:
            """A tool.

            Args:
                x: Input.
            """
            return x

        meta: ToolMeta = getattr(fn_tool2, TOOL_META)
        assert meta.hook_classes == (MyHook,)

    def test_stacking_merges(self):
        class HookA(ToolHook):
            pass

        class HookB(ToolHook):
            pass

        @use_hooks(HookB)
        @use_hooks(HookA)
        @tool()
        async def fn_stacked(x: str) -> str:
            """A tool.

            Args:
                x: Input.
            """
            return x

        assert getattr(fn_stacked, USE_HOOKS_META) == (HookA, HookB)

    def test_stacking_syncs_tool_meta(self):
        class HookA(ToolHook):
            pass

        class HookB(ToolHook):
            pass

        @use_hooks(HookB)
        @use_hooks(HookA)
        @tool()
        async def fn_stacked2(x: str) -> str:
            """A tool.

            Args:
                x: Input.
            """
            return x

        meta: ToolMeta = getattr(fn_stacked2, TOOL_META)
        assert meta.hook_classes == (HookA, HookB)

    def test_non_hook_subclass_raises_when_applied(self):
        """@use_hooks raises when a non-ToolHook class is used."""
        from lauren_ai._exceptions import DecoratorUsageError

        class NotAHook:
            pass

        with pytest.raises(DecoratorUsageError):

            @use_hooks(NotAHook)  # type: ignore[arg-type]
            @tool()
            async def bad(x: str) -> str:
                """Bad.

                Args:
                    x: Input.
                """
                return x

    def test_bare_usage_raises(self):
        from lauren_ai._exceptions import DecoratorUsageError

        with pytest.raises(DecoratorUsageError):

            @use_hooks  # type: ignore[arg-type]
            @tool()
            async def bad2(x: str) -> str:
                """Bad.

                Args:
                    x: Input.
                """
                return x

    def test_auto_marks_injectable(self):
        class AutoHook(ToolHook):
            pass

        @use_hooks(AutoHook)
        @tool()
        async def fn_auto(x: str) -> str:
            """Auto hook.

            Args:
                x: Input.
            """
            return x

        assert hasattr(AutoHook, "__lauren_injectable__")


# ---------------------------------------------------------------------------
# ToolExecutor hook integration
# ---------------------------------------------------------------------------


class TestToolExecutorHooks:
    """Drive hooks through ToolExecutor without full DI."""

    def _simple_tool_map(self, fn_tool=None, hook_classes=()) -> dict:
        if fn_tool is None:

            @tool()
            async def target(query: str) -> str:
                """Simple tool.

                Args:
                    query: The query.
                """
                return f"result:{query}"

            fn_tool = target

        from lauren_ai._tools import _add_to_tool_map

        tm: dict = {}
        _add_to_tool_map(tm, fn_tool)
        name = list(tm.keys())[0]
        meta: ToolMeta = tm[name][1]
        meta.resolved_hooks = tuple(h() for h in hook_classes)
        return tm

    # -- before hooks --------------------------------------------------------

    def test_before_hook_proceed(self):
        async def _run():
            tm = self._simple_tool_map()
            executor = ToolExecutor(tools=tm)
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "hello"}),
                _make_tool_context(),
            )

        result = asyncio.run(_run())
        assert "result:hello" in result.content

    def test_before_hook_abort(self):
        class AbortHook(ToolHook):
            async def before_tool_call(self, ctx):
                return BeforeToolHookDecision.abort(result="aborted!")

        async def _run():
            tm = self._simple_tool_map(hook_classes=(AbortHook,))
            executor = ToolExecutor(tools=tm)
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "hello"}),
                _make_tool_context(),
            )

        result = asyncio.run(_run())
        assert "aborted!" in result.content

    def test_before_hook_modify_input(self):
        class ModifyHook(ToolHook):
            async def before_tool_call(self, ctx):
                return BeforeToolHookDecision.modify({"query": "modified"})

        async def _run():
            tm = self._simple_tool_map(hook_classes=(ModifyHook,))
            executor = ToolExecutor(tools=tm)
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "original"}),
                _make_tool_context(),
            )

        result = asyncio.run(_run())
        assert "result:modified" in result.content

    # -- after hooks ---------------------------------------------------------

    def test_after_hook_replace(self):
        class ReplaceHook(ToolHook):
            async def after_tool_call(self, result, ctx):
                return AfterToolHookDecision.replace("replaced!")

        async def _run():
            tm = self._simple_tool_map(hook_classes=(ReplaceHook,))
            executor = ToolExecutor(tools=tm)
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "hello"}),
                _make_tool_context(),
            )

        result = asyncio.run(_run())
        assert "replaced!" in result.content

    def test_after_hook_proceed_passthrough(self):
        class PassHook(ToolHook):
            async def after_tool_call(self, result, ctx):
                return AfterToolHookDecision.proceed()

        async def _run():
            tm = self._simple_tool_map(hook_classes=(PassHook,))
            executor = ToolExecutor(tools=tm)
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "hello"}),
                _make_tool_context(),
            )

        result = asyncio.run(_run())
        assert "result:hello" in result.content

    # -- error hooks ---------------------------------------------------------

    def test_error_hook_suppress(self):
        @tool()
        async def failing_tool(q: str) -> str:
            """Always fails.

            Args:
                q: Input.
            """
            raise ValueError("boom")

        class SuppressHook(ToolHook):
            async def on_tool_error(self, exc, ctx):
                return ErrorToolHookDecision.suppress_with("suppressed!")

        async def _run():
            from lauren_ai._tools import _add_to_tool_map

            tm: dict = {}
            _add_to_tool_map(tm, failing_tool)
            tm["failing_tool"][1].resolved_hooks = (SuppressHook(),)
            executor = ToolExecutor(tools=tm)
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="failing_tool", input={"q": "x"}),
                _make_tool_context(),
            )

        result = asyncio.run(_run())
        assert "suppressed!" in result.content

    def test_error_hook_reraise(self):
        @tool()
        async def failing_tool2(q: str) -> str:
            """Always fails.

            Args:
                q: Input.
            """
            raise ValueError("boom")

        class ReraiseHook(ToolHook):
            async def on_tool_error(self, exc, ctx):
                return ErrorToolHookDecision.reraise()

        async def _run():
            from lauren_ai._tools import _add_to_tool_map
            from lauren_ai._tools._executor import ToolExecutionError

            tm: dict = {}
            _add_to_tool_map(tm, failing_tool2)
            tm["failing_tool2"][1].resolved_hooks = (ReraiseHook(),)
            executor = ToolExecutor(tools=tm)
            with pytest.raises(ToolExecutionError):
                await executor.execute(
                    ToolCall(tool_use_id="t1", name="failing_tool2", input={"q": "x"}),
                    _make_tool_context(),
                )

        asyncio.run(_run())

    # -- global hooks --------------------------------------------------------

    def test_global_hook_receives_every_call(self):
        calls: list[str] = []

        class GlobalHook(ToolHook):
            async def before_tool_call(self, ctx):
                calls.append(ctx.tool_name)
                return BeforeToolHookDecision.proceed()

        async def _run():
            tm = self._simple_tool_map()
            executor = ToolExecutor(tools=tm, global_hooks=[GlobalHook()])
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "x"}),
                _make_tool_context(),
            )

        asyncio.run(_run())
        assert "target" in calls

    def test_global_before_runs_before_pertool(self):
        """Global before hook executes before per-tool before hook."""
        order: list[str] = []

        class GlobalBefore(ToolHook):
            async def before_tool_call(self, ctx):
                order.append("global")
                return BeforeToolHookDecision.proceed()

        class PerToolBefore(ToolHook):
            async def before_tool_call(self, ctx):
                order.append("pertool")
                return BeforeToolHookDecision.proceed()

        async def _run():
            tm = self._simple_tool_map(hook_classes=(PerToolBefore,))
            executor = ToolExecutor(tools=tm, global_hooks=[GlobalBefore()])
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "x"}),
                _make_tool_context(),
            )

        asyncio.run(_run())
        assert order == ["global", "pertool"]

    def test_pertool_after_runs_before_global(self):
        """Per-tool after hook executes before global after hook."""
        order: list[str] = []

        class GlobalAfter(ToolHook):
            async def after_tool_call(self, result, ctx):
                order.append("global")
                return AfterToolHookDecision.proceed()

        class PerToolAfter(ToolHook):
            async def after_tool_call(self, result, ctx):
                order.append("pertool")
                return AfterToolHookDecision.proceed()

        async def _run():
            tm = self._simple_tool_map(hook_classes=(PerToolAfter,))
            executor = ToolExecutor(tools=tm, global_hooks=[GlobalAfter()])
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "x"}),
                _make_tool_context(),
            )

        asyncio.run(_run())
        assert order == ["pertool", "global"]

    # -- tool_input updated on modify ----------------------------------------

    def test_modify_updates_hook_ctx_tool_input(self):
        """When a before hook returns modify(), subsequent hooks see new input."""
        seen_inputs: list[dict] = []

        class ModifyFirst(ToolHook):
            async def before_tool_call(self, ctx):
                return BeforeToolHookDecision.modify({"query": "modified"})

        class ObserveSecond(ToolHook):
            async def before_tool_call(self, ctx):
                seen_inputs.append(dict(ctx.tool_input))
                return BeforeToolHookDecision.proceed()

        async def _run():
            tm = self._simple_tool_map(hook_classes=(ModifyFirst, ObserveSecond))
            executor = ToolExecutor(tools=tm)
            return await executor.execute(
                ToolCall(tool_use_id="t1", name="target", input={"query": "original"}),
                _make_tool_context(),
            )

        asyncio.run(_run())
        assert seen_inputs == [{"query": "modified"}]

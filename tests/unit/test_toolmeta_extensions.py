"""Tests for ToolMeta and ToolContext extensions (prd-toolmeta-context-extensions)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai._tools import ToolContext, ToolMeta, ToolSchema, tool
from lauren_ai._tools._hooks import ToolCallContext
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._transport import Completion, TokenUsage, ToolCall as TransportToolCall


# ── helpers ───────────────────────────────────────────────────────────────────


def _minimal_meta(name: str = "my_tool", **kwargs) -> ToolMeta:
    return ToolMeta(
        name=name,
        description="A test tool.",
        parameters=ToolSchema(name=name, description="", input_schema={}),
        is_async=True,
        reads_context=False,
        **kwargs,
    )


def _mock_transport(completions):
    t = MagicMock()
    t.complete = AsyncMock(side_effect=list(completions))
    return t


def _end_completion():
    return Completion(
        id="c1",
        model="mock-model",
        content="done",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


# ── ToolMeta.label / display_label ───────────────────────────────────────────


class TestToolMetaLabel:
    def test_label_empty_by_default(self):
        m = _minimal_meta()
        assert m.label == ""

    def test_display_label_title_cases_name_when_label_empty(self):
        m = _minimal_meta(name="read_file")
        assert m.display_label == "Read File"

    def test_display_label_returns_label_when_set(self):
        m = _minimal_meta(name="read_file", label="Read a File")
        assert m.display_label == "Read a File"

    def test_tool_decorator_accepts_label(self):
        @tool(label="My Pretty Tool")
        async def my_pretty_tool(x: str) -> dict:
            """Tool.

            Args:
                x: Input.
            """
            return {}

        from lauren_ai._tools import TOOL_META

        meta = getattr(my_pretty_tool, TOOL_META)
        assert meta.label == "My Pretty Tool"
        assert meta.display_label == "My Pretty Tool"

    def test_display_label_on_underscore_name(self):
        m = _minimal_meta(name="git_log")
        assert m.display_label == "Git Log"


# ── ToolMeta.initial_state ────────────────────────────────────────────────────


class TestInitialState:
    def test_initial_state_none_by_default(self):
        m = _minimal_meta()
        assert m.initial_state is None

    def test_initial_state_is_callable(self):
        factory = lambda: {"k": 0}
        m = _minimal_meta(initial_state=factory)
        assert m.initial_state is factory

    def test_initial_state_produces_fresh_dict_each_call(self):
        m = _minimal_meta(initial_state=lambda: {"items": []})
        d1 = m.initial_state()
        d2 = m.initial_state()
        assert d1 is not d2
        d1["items"].append(1)
        assert d2["items"] == []

    def test_executor_seeds_state_before_dispatch(self):
        """Executor calls initial_state() and merges into ctx.state before dispatch."""
        seeded_values: list[dict] = []

        @tool(initial_state=lambda: {"start_ts": None, "retries": 0})
        async def spy(ctx: ToolContext) -> dict:
            """Spy.

            Args:
                None.
            """
            seeded_values.append(dict(ctx.state))
            return {}

        from lauren_ai._tools import TOOL_META, _add_to_tool_map  # noqa: PLC0415
        from lauren_ai._agents import agent, use_tools  # noqa: PLC0415

        @agent(model="mock-model")
        @use_tools(spy)
        class SpyAgent: ...

        transport = _mock_transport(
            [
                Completion(
                    id="c1",
                    model="mock-model",
                    content="",
                    tool_calls=[TransportToolCall(tool_use_id="tu1", name="spy", input={})],
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                ),
                _end_completion(),
            ]
        )
        tool_map = {}
        _add_to_tool_map(tool_map, spy)
        SpyAgent.__lauren_ai_agent__.tools = tool_map

        runner = AgentRunnerBase(transport=transport)
        asyncio.run(runner.run(SpyAgent(), "go"))

        assert len(seeded_values) == 1
        assert "start_ts" in seeded_values[0]
        assert seeded_values[0]["retries"] == 0

    def test_caller_set_state_values_win_over_initial_state(self):
        """Values pre-set in ctx.state before dispatch survive the seed merge."""
        captured: list[dict] = []

        @tool(initial_state=lambda: {"key": "from_factory", "other": "from_factory"})
        async def spy2(ctx: ToolContext) -> dict:
            """Spy2.

            Args:
                None.
            """
            captured.append(dict(ctx.state))
            return {}

        from lauren_ai._tools import _add_to_tool_map, ToolContext as TC  # noqa: PLC0415
        from lauren_ai._tools._executor import ToolExecutor, ToolCall as EC  # noqa: PLC0415

        tool_map = {}
        _add_to_tool_map(tool_map, spy2)

        ctx = TC(
            agent_context=MagicMock(),
            tool_use_id="tu1",
            turn=0,
            state={"key": "caller_wins"},  # caller pre-set
        )
        executor = ToolExecutor(tools=tool_map)
        asyncio.run(executor.execute(EC(tool_use_id="tu1", name="spy2", input={}), ctx))

        assert captured[0]["key"] == "caller_wins"  # caller wins
        assert captured[0]["other"] == "from_factory"  # factory fills gap


# ── ToolMeta.initial_tool_state ───────────────────────────────────────────────


class TestInitialToolState:
    def test_initial_tool_state_none_by_default(self):
        m = _minimal_meta()
        assert m.initial_tool_state is None

    def test_initial_tool_state_is_callable(self):
        factory = lambda: {"count": 0}
        m = _minimal_meta(initial_tool_state=factory)
        assert m.initial_tool_state is factory

    def test_tool_state_seeded_from_factory_at_run_start(self):
        """initial_tool_state() is called once; same dict reused across calls."""
        state_snapshots: list[dict] = []

        @tool(initial_tool_state=lambda: {"seen": set(), "count": 0})
        async def counter(ctx: ToolContext) -> dict:
            """Counter.

            Args:
                None.
            """
            ctx.tool_state["count"] += 1
            state_snapshots.append(dict(ctx.tool_state))
            return {"count": ctx.tool_state["count"]}

        from lauren_ai._tools import _add_to_tool_map  # noqa: PLC0415
        from lauren_ai._agents import agent, use_tools  # noqa: PLC0415
        from lauren_ai._transport import ToolCall as TransportTC  # noqa: PLC0415

        @agent(model="mock-model")
        @use_tools(counter)
        class CountAgent: ...

        transport = _mock_transport(
            [
                Completion(
                    id="c1",
                    model="mock-model",
                    content="",
                    tool_calls=[TransportTC(tool_use_id="tu1", name="counter", input={})],
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                ),
                Completion(
                    id="c2",
                    model="mock-model",
                    content="",
                    tool_calls=[TransportTC(tool_use_id="tu2", name="counter", input={})],
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                ),
                _end_completion(),
            ]
        )
        tool_map = {}
        _add_to_tool_map(tool_map, counter)
        CountAgent.__lauren_ai_agent__.tools = tool_map

        runner = AgentRunnerBase(transport=transport)
        asyncio.run(runner.run(CountAgent(), "count twice"))

        assert len(state_snapshots) == 2
        assert state_snapshots[0]["count"] == 1
        assert state_snapshots[1]["count"] == 2  # SAME dict, mutation from call 1 visible

    def test_tool_state_reset_between_separate_runs(self):
        """A new run() resets tool_state; mutations from previous run not visible."""
        counts: list[int] = []

        @tool(initial_tool_state=lambda: {"n": 0})
        async def bumper(ctx: ToolContext) -> dict:
            """Bumper.

            Args:
                None.
            """
            ctx.tool_state["n"] += 1
            counts.append(ctx.tool_state["n"])
            return {}

        from lauren_ai._tools import _add_to_tool_map  # noqa: PLC0415
        from lauren_ai._agents import agent, use_tools  # noqa: PLC0415
        from lauren_ai._transport import ToolCall as TransportTC  # noqa: PLC0415

        @agent(model="mock-model")
        @use_tools(bumper)
        class BumpAgent: ...

        def _one_call_transport():
            return _mock_transport(
                [
                    Completion(
                        id="cx",
                        model="mock-model",
                        content="",
                        tool_calls=[TransportTC(tool_use_id="tu1", name="bumper", input={})],
                        stop_reason="tool_use",
                        usage=TokenUsage(input_tokens=1, output_tokens=1),
                    ),
                    _end_completion(),
                ]
            )

        tool_map = {}
        _add_to_tool_map(tool_map, bumper)
        BumpAgent.__lauren_ai_agent__.tools = tool_map

        runner1 = AgentRunnerBase(transport=_one_call_transport())
        asyncio.run(runner1.run(BumpAgent(), "run1"))

        runner2 = AgentRunnerBase(transport=_one_call_transport())
        asyncio.run(runner2.run(BumpAgent(), "run2"))

        assert counts == [1, 1]  # each run starts fresh at 0


# ── ToolMeta.dependency_factory ──────────────────────────────────────────────


class TestDependencyFactory:
    def test_dependency_factory_none_by_default(self):
        m = _minimal_meta()
        assert m.dependency_factory is None

    def test_dependency_factory_result_in_ctx_dependencies(self):
        """dependency_factory() result available as ctx.dependencies."""
        seen_deps: list[dict] = []

        client_obj = object()

        @tool(dependency_factory=lambda: {"client": client_obj})
        async def dep_tool(ctx: ToolContext) -> dict:
            """Dep tool.

            Args:
                None.
            """
            seen_deps.append(dict(ctx.dependencies))
            return {}

        from lauren_ai._tools import _add_to_tool_map  # noqa: PLC0415
        from lauren_ai._agents import agent, use_tools  # noqa: PLC0415
        from lauren_ai._transport import ToolCall as TransportTC  # noqa: PLC0415

        @agent(model="mock-model")
        @use_tools(dep_tool)
        class DepAgent: ...

        transport = _mock_transport(
            [
                Completion(
                    id="c1",
                    model="mock-model",
                    content="",
                    tool_calls=[TransportTC(tool_use_id="tu1", name="dep_tool", input={})],
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                ),
                _end_completion(),
            ]
        )
        tool_map = {}
        _add_to_tool_map(tool_map, dep_tool)
        DepAgent.__lauren_ai_agent__.tools = tool_map

        runner = AgentRunnerBase(transport=transport)
        asyncio.run(runner.run(DepAgent(), "go"))

        assert seen_deps[0]["client"] is client_obj

    def test_dependency_factory_called_once_per_run(self):
        """Factory is called exactly once; same object returned on both calls."""
        call_count = [0]

        def factory():
            call_count[0] += 1
            return {"n": call_count[0]}

        captured_deps: list[dict] = []

        @tool(dependency_factory=factory)
        async def dep2(ctx: ToolContext) -> dict:
            """Dep2.

            Args:
                None.
            """
            captured_deps.append(ctx.dependencies)
            return {}

        from lauren_ai._tools import _add_to_tool_map  # noqa: PLC0415
        from lauren_ai._agents import agent, use_tools  # noqa: PLC0415
        from lauren_ai._transport import ToolCall as TransportTC  # noqa: PLC0415

        @agent(model="mock-model")
        @use_tools(dep2)
        class Dep2Agent: ...

        transport = _mock_transport(
            [
                Completion(
                    id="c1",
                    model="mock-model",
                    content="",
                    tool_calls=[TransportTC(tool_use_id="tu1", name="dep2", input={})],
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                ),
                Completion(
                    id="c2",
                    model="mock-model",
                    content="",
                    tool_calls=[TransportTC(tool_use_id="tu2", name="dep2", input={})],
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                ),
                _end_completion(),
            ]
        )
        tool_map = {}
        _add_to_tool_map(tool_map, dep2)
        Dep2Agent.__lauren_ai_agent__.tools = tool_map

        runner = AgentRunnerBase(transport=transport)
        asyncio.run(runner.run(Dep2Agent(), "twice"))

        assert call_count[0] == 1  # factory called once
        assert captured_deps[0] is captured_deps[1]  # same dict object


# ── ToolContext new fields ─────────────────────────────────────────────────────


class TestToolContextNewFields:
    def test_tool_state_defaults_to_empty_dict(self):
        ctx = ToolContext(agent_context=None, tool_use_id="t", turn=0)
        assert ctx.tool_state == {}

    def test_dependencies_defaults_to_empty_dict(self):
        ctx = ToolContext(agent_context=None, tool_use_id="t", turn=0)
        assert ctx.dependencies == {}

    def test_extras_defaults_to_empty_dict(self):
        ctx = ToolContext(agent_context=None, tool_use_id="t", turn=0)
        assert ctx.extras == {}

    def test_tool_state_accepts_dict(self):
        ctx = ToolContext(
            agent_context=None,
            tool_use_id="t",
            turn=0,
            tool_state={"x": 1},
        )
        assert ctx.tool_state["x"] == 1


# ── ToolContext removed fields ────────────────────────────────────────────────


class TestToolContextRemovedFields:
    def test_request_not_on_tool_context(self):
        ctx = ToolContext(agent_context=None, tool_use_id="t", turn=0)
        assert not hasattr(ctx, "request")

    def test_execution_context_not_on_tool_context(self):
        ctx = ToolContext(agent_context=None, tool_use_id="t", turn=0)
        assert not hasattr(ctx, "execution_context")

    def test_request_accessible_via_agent_context(self):
        mock_ac = MagicMock()
        mock_ac.request = object()
        ctx = ToolContext(agent_context=mock_ac, tool_use_id="t", turn=0)
        assert ctx.agent_context.request is mock_ac.request

    def test_execution_context_accessible_via_agent_context(self):
        sentinel = object()
        mock_ac = MagicMock()
        mock_ac.execution_context = sentinel
        ctx = ToolContext(agent_context=mock_ac, tool_use_id="t", turn=0)
        assert ctx.agent_context.execution_context is sentinel


# ── ToolCallContext inherits new fields ────────────────────────────────────────


class TestToolCallContextInheritance:
    def test_tool_call_context_has_tool_state(self):
        hctx = ToolCallContext(
            agent_context=None,
            tool_use_id="t",
            turn=0,
            tool_state={"k": "v"},
        )
        assert hctx.tool_state == {"k": "v"}

    def test_tool_call_context_no_request_field(self):
        hctx = ToolCallContext(agent_context=None, tool_use_id="t", turn=0)
        assert not hasattr(hctx, "request")

    def test_tool_call_context_no_execution_context_field(self):
        hctx = ToolCallContext(agent_context=None, tool_use_id="t", turn=0)
        assert not hasattr(hctx, "execution_context")


# ── @tool() decorator new kwargs ─────────────────────────────────────────────


class TestToolDecoratorNewKwargs:
    def test_decorator_accepts_all_new_kwargs(self):
        factory_s = lambda: {"a": 1}
        factory_ts = lambda: {"b": 2}
        factory_d = lambda: {"c": 3}

        @tool(
            label="My Tool",
            initial_state=factory_s,
            initial_tool_state=factory_ts,
            dependency_factory=factory_d,
        )
        async def my_tool(x: str) -> dict:
            """My tool.

            Args:
                x: Input.
            """
            return {}

        from lauren_ai._tools import TOOL_META

        meta = getattr(my_tool, TOOL_META)
        assert meta.label == "My Tool"
        assert meta.initial_state is factory_s
        assert meta.initial_tool_state is factory_ts
        assert meta.dependency_factory is factory_d

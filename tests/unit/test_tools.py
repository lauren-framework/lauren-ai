"""Unit tests for the @tool() decorator and schema generation."""

from __future__ import annotations

import pytest

from lauren_ai._agents import AgentContext
from lauren_ai._tools import TOOL_META, ToolContext, ToolResult, _add_to_tool_map, tool
from lauren_ai._tools._schema import type_to_json_schema


class TestToolDecorator:
    def test_basic_tool(self):
        @tool()
        async def my_tool(name: str) -> str:
            """Return a greeting.

            Args:
                name: Person's name.
            """
            return f"Hello, {name}!"

        meta = getattr(my_tool, TOOL_META)
        assert meta.name == "my_tool"
        assert "greeting" in meta.description.lower() or "Return" in meta.description

    def test_tool_name_override(self):
        @tool(name="custom_name")
        async def some_func(x: int) -> int:
            """A function.

            Args:
                x: Input value.
            """
            return x

        meta = getattr(some_func, TOOL_META)
        assert meta.name == "custom_name"

    def test_bare_usage_raises(self):
        with pytest.raises(Exception):  # noqa: B017 — DecoratorUsageError

            @tool
            async def bad_tool(x: str) -> str:
                return x

    def test_tool_schema_has_parameters(self):
        @tool()
        async def greet(name: str, times: int = 1) -> str:
            """Greet someone.

            Args:
                name: Name to greet.
                times: Number of times.
            """
            return name * times

        meta = getattr(greet, TOOL_META)
        # meta.parameters is a ToolSchema with input_schema key
        input_schema = meta.parameters["input_schema"]
        assert "properties" in input_schema
        assert "name" in input_schema["properties"]
        assert "times" in input_schema["properties"]

    def test_ctx_excluded_from_schema(self):
        @tool()
        async def func_with_ctx(query: str, ctx: ToolContext | None = None) -> str:
            """Search something.

            Args:
                query: The query string.
            """
            return query

        meta = getattr(func_with_ctx, TOOL_META)
        input_schema = meta.parameters["input_schema"]
        assert "ctx" not in input_schema.get("properties", {})

    def test_context_param_any_name_excluded_from_schema(self):
        """ToolContext param is excluded regardless of its name."""

        @tool()
        async def func(query: str, context: ToolContext | None = None) -> str:
            """Search something.

            Args:
                query: The query string.
            """
            return query

        meta = getattr(func, TOOL_META)
        input_schema = meta.parameters["input_schema"]
        assert "context" not in input_schema.get("properties", {})
        assert meta.reads_context is True

    def test_context_param_bare_annotation_any_name(self):
        """Bare ToolContext annotation (not Optional) with a non-default name."""

        @tool()
        async def func(task: str, agent_ctx: ToolContext = None) -> str:
            """Do a task.

            Args:
                task: The task description.
            """
            return task

        meta = getattr(func, TOOL_META)
        input_schema = meta.parameters["input_schema"]
        assert "agent_ctx" not in input_schema.get("properties", {})
        assert meta.reads_context is True

    def test_required_vs_optional(self):
        @tool()
        async def func(required: str, optional: int = 5) -> str:
            """A function.

            Args:
                required: Must be provided.
                optional: Has a default.
            """
            return required

        meta = getattr(func, TOOL_META)
        input_schema = meta.parameters["input_schema"]
        required_list = input_schema.get("required", [])
        assert "required" in required_list
        assert "optional" not in required_list


class TestToolResult:
    def test_ok_string(self):
        r = ToolResult.ok("hello", tool_use_id="tu1")
        assert r.content == "hello"
        assert not r.is_error

    def test_ok_dict(self):
        r = ToolResult.ok({"key": "value"}, tool_use_id="tu1")
        assert '"key"' in r.content
        assert not r.is_error

    def test_error(self):
        r = ToolResult.error("Something went wrong", tool_use_id="tu1")
        assert r.is_error
        assert "Something went wrong" in r.content


class TestToolClassInjectable:
    def test_tool_class_sets_injectable_meta(self):
        @tool()
        class MySimpleTool:
            """A simple tool.

            Args:
                x: Input value.
            """

            async def run(self, x: str) -> str:
                return x

        assert hasattr(MySimpleTool, "__lauren_injectable__"), "@tool() on a class must set __lauren_injectable__"

    def test_tool_class_default_scope_singleton(self):
        from lauren._di import INJECTABLE_META

        @tool()
        class SingletonTool:
            """A singleton tool.

            Args:
                x: Input.
            """

            async def run(self, x: str) -> str:
                return x

        from lauren import Scope

        inj_meta = getattr(SingletonTool, INJECTABLE_META)
        assert inj_meta.scope == Scope.SINGLETON

    def test_tool_class_explicit_injectable_scope_preserved(self):
        """@tool() on top of @injectable(scope=REQUEST) must not override scope."""
        from lauren import Scope, injectable
        from lauren._di import INJECTABLE_META

        @tool()
        @injectable(scope=Scope.REQUEST)
        class RequestTool:
            """A request-scoped tool.

            Args:
                x: Input.
            """

            async def run(self, x: str) -> str:
                return x

        inj_meta = getattr(RequestTool, INJECTABLE_META)
        assert inj_meta.scope == Scope.REQUEST

    def test_tool_class_injectable_not_double_applied(self):
        """Calling @tool() again on an already-injectable class must be idempotent."""
        from lauren import Scope, injectable

        @injectable(scope=Scope.SINGLETON)
        class AlreadyInjectable:
            """Already injectable.

            Args:
                x: Input.
            """

            async def run(self, x: str) -> str:
                return x

        # Apply @tool() after @injectable() — should not raise or double-wrap
        decorated = tool()(AlreadyInjectable)
        assert hasattr(decorated, "__lauren_injectable__")
        assert hasattr(decorated, TOOL_META)

    def test_tool_meta_still_set_after_injectable(self):
        @tool()
        class MetaTool:
            """A meta tool.

            Args:
                x: Input.
            """

            async def run(self, x: str) -> str:
                return x

        assert TOOL_META in MetaTool.__dict__, "TOOL_META must be on the class __dict__ after injectable"


class TestToolInheritance:
    def test_subclass_without_tool_decorator_raises_on_register(self):
        from lauren.exceptions import MetadataInheritanceError

        @tool()
        class Base:
            """Base tool.

            Args:
                x: Input.
            """

            async def run(self, x: str) -> str:
                return x

        class Sub(Base):
            pass

        with pytest.raises(MetadataInheritanceError, match="Sub"):
            _add_to_tool_map({}, Sub)

    def test_subclass_with_tool_decorator_ok(self):
        @tool()
        class BaseOk:
            """Base ok.

            Args:
                x: Input.
            """

            async def run(self, x: str) -> str:
                return x

        @tool(name="sub_ok_tool")
        class SubOk(BaseOk):
            """Sub ok.

            Args:
                x: Input.
            """

            async def run(self, x: str) -> str:
                return x

        tools = {}
        _add_to_tool_map(tools, SubOk)  # must not raise
        assert "sub_ok_tool" in tools

    def test_non_subclass_unaffected(self):
        """Registering an object with no TOOL_META raises ValueError,
        not MetadataInheritanceError."""

        class NoMeta:
            pass

        with pytest.raises(ValueError, match="does not have"):
            _add_to_tool_map({}, NoMeta)


class TestContextParamName:
    def test_context_param_named_ctx(self):
        @tool()
        async def standard_ctx(task: str, ctx: ToolContext | None = None) -> str:
            """A standard tool.

            Args:
                task: The task.
            """
            return task

        meta = getattr(standard_ctx, TOOL_META)
        assert meta.context_param_name == "ctx"

    def test_context_param_custom_name(self):
        @tool()
        async def custom_ctx(task: str, agent_ctx: ToolContext | None = None) -> str:
            """A tool with custom ctx name.

            Args:
                task: The task.
            """
            return task

        meta = getattr(custom_ctx, TOOL_META)
        assert meta.context_param_name == "agent_ctx"

    def test_context_param_name_none_when_no_context(self):
        @tool()
        async def no_ctx(task: str) -> str:
            """A tool with no context param.

            Args:
                task: The task.
            """
            return task

        meta = getattr(no_ctx, TOOL_META)
        assert meta.context_param_name is None

    def test_context_injected_by_param_name(self):
        """Executor must inject ToolContext under the annotation-found param name."""
        import asyncio

        from lauren_ai._tools._executor import ToolCall, ToolExecutor

        received: list[ToolContext] = []

        @tool()
        async def receives_agent_ctx(task: str, agent_ctx: ToolContext | None = None) -> str:
            """A tool.

            Args:
                task: The task.
            """
            received.append(agent_ctx)
            return task

        tools = {}
        _add_to_tool_map(tools, receives_agent_ctx)
        executor = ToolExecutor(tools=tools)

        dummy_ctx = ToolContext(
            agent_context=None,
            tool_use_id="tid1",
            turn=0,
        )
        call = ToolCall(tool_use_id="tid1", name="receives_agent_ctx", input={"task": "hello"})

        result = asyncio.run(executor.execute(call, dummy_ctx))
        assert not result.is_error
        assert len(received) == 1
        assert received[0] is dummy_ctx


class TestExecutionContext:
    def test_agent_context_accepts_execution_context(self):
        from lauren_ai._config import AgentConfig
        from lauren_ai._memory import ShortTermMemory

        sentinel = object()
        ctx = AgentContext(
            agent_id="a1",
            agent_run_id="r1",
            agent_class=object,
            config=AgentConfig(),
            memory=ShortTermMemory(),
            turn=0,
            metadata={},
            execution_context=sentinel,
        )
        assert ctx.execution_context is sentinel

    def test_tool_context_has_execution_context(self):
        sentinel = object()
        tc = ToolContext(
            agent_context=None,
            tool_use_id="tu1",
            turn=0,
            execution_context=sentinel,
        )
        assert tc.execution_context is sentinel

    def test_runner_passes_execution_context_to_agent_context(self):
        """AgentRunner.run(execution_context=x) must store x in the AgentContext."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lauren_ai._agents import agent
        from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
        from lauren_ai._transport import Completion, TokenUsage

        @agent(model="mock-model")
        class CapturCtxAgent:
            """Test agent."""

            captured_ctx: AgentContext | None = None

            async def on_start(self, ctx: AgentContext) -> None:
                CapturCtxAgent.captured_ctx = ctx

        mock_transport = MagicMock()
        mock_transport.complete = AsyncMock(
            return_value=Completion(
                id="c1",
                model="mock-model",
                content="done",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        )

        runner = AgentRunner(
            transport=mock_transport,
        )

        sentinel = object()
        instance = CapturCtxAgent()

        asyncio.run(runner.run(instance, "hello", execution_context=sentinel))

        assert CapturCtxAgent.captured_ctx is not None
        assert CapturCtxAgent.captured_ctx.execution_context is sentinel

    def test_execution_context_flows_to_tool_context(self):
        """execution_context set on AgentContext must be forwarded to ToolContext."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lauren_ai._agents import agent
        from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
        from lauren_ai._transport import Completion, TokenUsage
        from lauren_ai._transport import ToolCall as TransportToolCall

        tool_contexts: list[ToolContext] = []

        @tool()
        async def spy_tool(msg: str, ctx: ToolContext | None = None) -> str:
            """Spy tool.

            Args:
                msg: Message.
            """
            tool_contexts.append(ctx)
            return msg

        @agent(model="mock-model")
        class SpyAgent:
            """Spy agent."""

        # Two completions: first requests a tool call, second ends
        call1 = Completion(
            id="c1",
            model="mock-model",
            content="",
            tool_calls=[TransportToolCall(tool_use_id="tu1", name="spy_tool", input={"msg": "hi"})],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        call2 = Completion(
            id="c2",
            model="mock-model",
            content="done",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

        mock_transport = MagicMock()
        mock_transport.complete = AsyncMock(side_effect=[call1, call2])

        spy_tools = {}
        _add_to_tool_map(spy_tools, spy_tool)
        SpyAgent.__lauren_ai_agent__.tools = spy_tools

        runner = AgentRunner(
            transport=mock_transport,
        )

        sentinel = object()
        instance = SpyAgent()

        asyncio.run(runner.run(instance, "go", execution_context=sentinel))

        assert len(tool_contexts) == 1
        assert tool_contexts[0].execution_context is sentinel


class TestTypeToJsonSchema:
    def test_str(self):
        assert type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_of_str(self):
        result = type_to_json_schema(list[str])
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_dict_type(self):
        result = type_to_json_schema(dict)
        assert result == {"type": "object"}

    def test_optional_str(self):
        from typing import Optional

        result = type_to_json_schema(Optional[str])  # noqa: UP045
        assert result == {"type": "string"}

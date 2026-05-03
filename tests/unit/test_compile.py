"""Unit tests for _agents/_compile.py — validate_agent_class and validate_tool."""
from __future__ import annotations

import pytest

from lauren_ai._agents import USE_TOOLS_META, agent, use_tools
from lauren_ai._agents._compile import validate_agent_class, validate_tool
from lauren_ai._exceptions import AgentConfigError, ToolConfigError
from lauren_ai._tools import TOOL_META, ToolContext, tool

# ---------------------------------------------------------------------------
# validate_tool tests
# ---------------------------------------------------------------------------


class TestValidateTool:
    def test_valid_function_tool(self):
        @tool()
        async def my_tool(name: str, count: int) -> str:
            """A simple tool.

            Args:
                name: The name.
                count: The count.
            """
            return name * count

        meta = validate_tool(my_tool)
        assert meta.name == "my_tool"

    def test_raises_for_no_tool_meta(self):
        def plain_func(x: str) -> str:
            return x

        with pytest.raises(ToolConfigError) as exc_info:
            validate_tool(plain_func)
        assert "plain_func" in str(exc_info.value) or TOOL_META in str(exc_info.value)

    def test_raises_for_none(self):
        with pytest.raises(ToolConfigError):
            validate_tool(None)

    def test_raises_for_unannotated_parameter(self):
        # Create a tool with unannotated param — bypass @tool() decorator check
        # by manually injecting TOOL_META
        from lauren_ai._tools import ToolMeta

        # We'll define a function without annotation, then give it a fake TOOL_META
        def bad_func(x):
            """Tool with no annotation."""
            return str(x)

        # Manually attach fake TOOL_META
        fake_meta = ToolMeta(
            name="bad_func",
            description="",
            parameters={},
            is_async=False,
            reads_context=False,
        )
        setattr(bad_func, TOOL_META, fake_meta)

        with pytest.raises(ToolConfigError) as exc_info:
            validate_tool(bad_func)
        assert "annotation" in str(exc_info.value).lower() or "x" in str(exc_info.value)

    def test_class_form_tool_valid(self):
        @tool()
        class CalculatorTool:
            """Calculates things."""

            def run(self, expression: str) -> str:
                """Run the calculation.

                Args:
                    expression: Math expression.
                """
                return eval(expression)  # noqa

        meta = validate_tool(CalculatorTool)
        assert meta is not None

    def test_class_form_missing_run_raises(self):
        # Create class with TOOL_META but no run() method
        from lauren_ai._tools import ToolMeta

        fake_meta = ToolMeta(
            name="no_run",
            description="",
            parameters={},
            is_async=False,
            reads_context=False,
        )

        class NoRunTool:
            pass

        setattr(NoRunTool, TOOL_META, fake_meta)

        with pytest.raises(ToolConfigError) as exc_info:
            validate_tool(NoRunTool)
        assert "run" in str(exc_info.value).lower()

    def test_class_form_run_not_callable_raises(self):
        from lauren_ai._tools import ToolMeta

        fake_meta = ToolMeta(
            name="bad_run",
            description="",
            parameters={},
            is_async=False,
            reads_context=False,
        )

        class BadRunTool:
            run = "not_callable"

        setattr(BadRunTool, TOOL_META, fake_meta)

        with pytest.raises(ToolConfigError) as exc_info:
            validate_tool(BadRunTool)
        assert "callable" in str(exc_info.value).lower()

    def test_ctx_param_skipped(self):
        """ctx parameter should not require annotation."""
        @tool()
        async def tool_with_ctx(query: str, ctx: ToolContext=None) -> str:
            """A tool. Args: query: The query."""
            return query

        # Should pass without raising (ctx is allowed to be unannotated)
        meta = validate_tool(tool_with_ctx)
        assert meta is not None

    def test_varargs_skipped(self):
        """*args and **kwargs should not require annotation."""
        @tool()
        async def variadic_tool(name: str, *args, **kwargs) -> str:
            """A tool. Args: name: The name."""
            return name

        meta = validate_tool(variadic_tool)
        assert meta is not None


# ---------------------------------------------------------------------------
# validate_agent_class tests
# ---------------------------------------------------------------------------


class TestValidateAgentClass:
    def test_valid_agent_no_hooks(self):
        @agent(model="claude-opus-4-6")
        class SimpleAgent:
            """A simple agent."""

        meta = validate_agent_class(SimpleAgent)
        assert meta is not None
        assert meta.model == "claude-opus-4-6"

    def test_raises_for_non_agent_class(self):
        class PlainClass:
            pass

        with pytest.raises(AgentConfigError) as exc_info:
            validate_agent_class(PlainClass)
        assert "@agent()" in str(exc_info.value)

    def test_valid_with_on_start_hook(self):
        @agent()
        class AgentWithStart:
            async def on_start(self, ctx):
                pass

        meta = validate_agent_class(AgentWithStart)
        assert meta is not None

    def test_valid_with_on_finish_hook(self):
        @agent()
        class AgentWithFinish:
            async def on_finish(self, response, ctx):
                pass

        meta = validate_agent_class(AgentWithFinish)
        assert meta is not None

    def test_valid_with_on_tool_result_hook(self):
        @agent()
        class AgentWithToolResult:
            async def on_tool_result(self, result, ctx):
                pass

        meta = validate_agent_class(AgentWithToolResult)
        assert meta is not None

    def test_valid_with_on_turn_complete_hook(self):
        @agent()
        class AgentWithTurnComplete:
            async def on_turn_complete(self, completion, ctx):
                pass

        meta = validate_agent_class(AgentWithTurnComplete)
        assert meta is not None

    def test_invalid_hook_too_few_params_raises(self):
        @agent()
        class BadHookAgent:
            async def on_start(self):  # Missing ctx param
                pass

        with pytest.raises(AgentConfigError) as exc_info:
            validate_agent_class(BadHookAgent)
        assert "on_start" in str(exc_info.value)

    def test_invalid_hook_too_many_params_raises(self):
        @agent()
        class BadHookAgent2:
            async def on_start(self, ctx, extra_param):  # Too many params
                pass

        with pytest.raises(AgentConfigError) as exc_info:
            validate_agent_class(BadHookAgent2)
        assert "on_start" in str(exc_info.value)

    def test_invalid_on_finish_too_few_params_raises(self):
        @agent()
        class BadFinishAgent:
            async def on_finish(self, response):  # Missing ctx
                pass

        with pytest.raises(AgentConfigError) as exc_info:
            validate_agent_class(BadFinishAgent)
        assert "on_finish" in str(exc_info.value)

    def test_non_callable_hook_raises(self):
        @agent()
        class NonCallableHookAgent:
            on_start = "not_callable"

        with pytest.raises(AgentConfigError) as exc_info:
            validate_agent_class(NonCallableHookAgent)
        assert "callable" in str(exc_info.value).lower()

    def test_agent_with_valid_tools(self):
        @tool()
        async def valid_tool(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        @agent()
        @use_tools(valid_tool)
        class AgentWithTools:
            pass

        meta = validate_agent_class(AgentWithTools)
        assert meta is not None

    def test_agent_with_invalid_tool_raises_agent_config_error(self):
        # Attach a plain function (no TOOL_META) as a tool
        def not_a_tool(x: str) -> str:
            return x

        @agent()
        class AgentWithBadTool:
            pass

        # Manually inject a bad tool into USE_TOOLS_META
        setattr(AgentWithBadTool, USE_TOOLS_META, (not_a_tool,))

        with pytest.raises(AgentConfigError) as exc_info:
            validate_agent_class(AgentWithBadTool)
        assert "invalid tool" in str(exc_info.value).lower()

    def test_none_tool_skipped(self):
        """None entries in USE_TOOLS_META should be silently skipped."""
        @agent()
        class AgentWithNoneTool:
            pass

        setattr(AgentWithNoneTool, USE_TOOLS_META, (None, None))
        # Should not raise
        meta = validate_agent_class(AgentWithNoneTool)
        assert meta is not None

    def test_vararg_hooks_count_correctly(self):
        """*args and **kwargs in hooks shouldn't count toward positional params."""
        @agent()
        class AgentWithVarKwargs:
            async def on_start(self, ctx, **kwargs):  # ctx + kwargs → 1 positional
                pass

        meta = validate_agent_class(AgentWithVarKwargs)
        assert meta is not None

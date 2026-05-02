"""Unit tests for the @agent() and @use_tools() decorators."""
from __future__ import annotations

import pytest

from lauren_ai._agents import AGENT_META, AgentMeta, agent, use_tools
from lauren_ai._exceptions import DecoratorUsageError
from lauren_ai._tools import tool


class TestAgentDecorator:
    def test_basic_agent(self):
        @agent(model="claude-opus-4-6", system="Be helpful.")
        class MyAgent:
            pass

        meta = getattr(MyAgent, AGENT_META)
        assert isinstance(meta, AgentMeta)
        assert meta.model == "claude-opus-4-6"
        assert meta.system == "Be helpful."

    def test_model_none_inherits_at_runtime(self):
        @agent()
        class NoModelAgent:
            pass

        meta = getattr(NoModelAgent, AGENT_META)
        assert meta.model is None

    def test_bare_usage_raises(self):
        with pytest.raises(DecoratorUsageError):
            @agent
            class BadAgent:
                pass

    def test_docstring_as_system(self):
        @agent()
        class DocAgent:
            """I am a document summariser."""

        meta = getattr(DocAgent, AGENT_META)
        assert meta.system == "I am a document summariser."

    def test_explicit_system_overrides_docstring(self):
        @agent(system="Explicit system.")
        class DocAgent:
            """This should be ignored."""

        meta = getattr(DocAgent, AGENT_META)
        assert meta.system == "Explicit system."

    def test_config_max_turns(self):
        @agent(max_turns=3)
        class FastAgent:
            pass

        meta = getattr(FastAgent, AGENT_META)
        assert meta.config.max_turns == 3

    def test_config_temperature(self):
        @agent(temperature=0.2)
        class CoolAgent:
            pass

        meta = getattr(CoolAgent, AGENT_META)
        assert meta.config.temperature == pytest.approx(0.2)


class TestUseTools:
    def test_attach_single_tool(self):
        @tool()
        async def my_tool(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        # @agent() must be ABOVE @use_tools() so @use_tools() is applied first
        @agent()
        @use_tools(my_tool)
        class ToolAgent:
            pass

        meta = getattr(ToolAgent, AGENT_META)
        assert my_tool in meta.tool_classes

    def test_none_filtered(self):
        @agent()
        @use_tools(None, None)
        class Agent1:
            pass

        meta = getattr(Agent1, AGENT_META)
        assert len(meta.tool_classes) == 0

    def test_stacking_use_tools(self):
        @tool()
        async def tool_a(x: str) -> str:
            """Tool A. Args: x: Input."""
            return x

        @tool()
        async def tool_b(y: int) -> int:
            """Tool B. Args: y: Number."""
            return y

        @agent()
        @use_tools(tool_b)
        @use_tools(tool_a)
        class MultiToolAgent:
            pass

        meta = getattr(MultiToolAgent, AGENT_META)
        assert tool_a in meta.tool_classes
        assert tool_b in meta.tool_classes

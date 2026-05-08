"""Integration tests for Skill 6: Tool / Function Calling Definition & Registration.

Tests cover:
- @tool() function form: schema generation from annotations
- @tool() decorated function executes when called by agent
- Tool attached to agent via @use_tools()
- MockTransport queue_tool_use triggers tool execution
- Tool result fed back to LLM as final answer
- Tool with default parameter values
- Tool error propagation

NOTE: No `from __future__ import annotations` — @tool() needs live annotations
for schema generation.
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import TOOL_META, _add_to_tool_map, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Tool definitions (module level — no future annotations)
# ---------------------------------------------------------------------------


@tool()
async def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name.
        units: Temperature units — celsius or fahrenheit.
    """
    return {"city": city, "temp": 18, "units": units, "condition": "cloudy"}


@tool()
async def add_numbers(a: int, b: int) -> dict:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return {"result": a + b}


@tool()
async def always_fails(msg: str) -> dict:
    """A tool that always raises.

    Args:
        msg: Message to include in exception.
    """
    raise ValueError(f"Tool error: {msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str = "OK") -> Completion:
    return Completion(
        id="c1",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner_with_tools(*tool_funcs) -> tuple[AgentRunner, MockTransport]:
    mock = MockTransport()
    tools = {}
    for t in tool_funcs:
        _add_to_tool_map(tools, t)
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools=tools, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# TestToolDefinition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_tool_has_tool_meta_attribute(self):
        assert hasattr(get_weather, TOOL_META)

    def test_tool_meta_name_matches_function_name(self):
        meta = getattr(get_weather, TOOL_META)
        assert meta.name == "get_weather"

    def test_tool_meta_has_description(self):
        meta = getattr(get_weather, TOOL_META)
        assert "city" in meta.description.lower() or len(meta.description) > 0

    def test_tool_schema_has_input_schema(self):
        meta = getattr(get_weather, TOOL_META)
        # parameters is a plain dict with 'input_schema' key
        assert "input_schema" in meta.parameters

    def test_tool_schema_city_parameter(self):
        meta = getattr(get_weather, TOOL_META)
        props = meta.parameters["input_schema"].get("properties", {})
        assert "city" in props

    def test_tool_schema_units_parameter(self):
        meta = getattr(get_weather, TOOL_META)
        props = meta.parameters["input_schema"].get("properties", {})
        assert "units" in props

    def test_add_numbers_tool_has_meta(self):
        assert hasattr(add_numbers, TOOL_META)


# ---------------------------------------------------------------------------
# TestToolExecution
# ---------------------------------------------------------------------------


class TestToolExecution:
    async def test_tool_call_executes_function(self):
        runner, mock = _make_runner_with_tools(get_weather)

        @use_tools(get_weather)
        @agent(model="mock-model")
        class WeatherAgent: ...

        mock.queue_tool_use("get_weather", {"city": "London"})
        mock.queue_response(_completion("It is 18°C in London."))

        resp = await runner.run(WeatherAgent(), "Weather in London?")
        assert resp.content == "It is 18°C in London."

    async def test_tool_call_records_in_response(self):
        runner, mock = _make_runner_with_tools(get_weather)

        @use_tools(get_weather)
        @agent(model="mock-model")
        class WeatherAgent: ...

        mock.queue_tool_use("get_weather", {"city": "Paris"})
        mock.queue_response(_completion("Rainy in Paris."))

        resp = await runner.run(WeatherAgent(), "Weather in Paris?")
        assert len(resp.tool_calls_made) == 1

    async def test_tool_call_name_recorded(self):
        runner, mock = _make_runner_with_tools(get_weather)

        @use_tools(get_weather)
        @agent(model="mock-model")
        class WeatherAgent: ...

        mock.queue_tool_use("get_weather", {"city": "Berlin"})
        mock.queue_response(_completion("Cloudy in Berlin."))

        resp = await runner.run(WeatherAgent(), "Weather in Berlin?")
        assert resp.tool_calls_made[0].name == "get_weather"

    async def test_two_llm_calls_for_tool_turn(self):
        runner, mock = _make_runner_with_tools(get_weather)

        @use_tools(get_weather)
        @agent(model="mock-model")
        class WeatherAgent: ...

        mock.queue_tool_use("get_weather", {"city": "Tokyo"})
        mock.queue_response(_completion("Sunny in Tokyo."))

        await runner.run(WeatherAgent(), "Weather in Tokyo?")
        # Turn 1: tool_use, Turn 2: end_turn
        assert len(mock.calls) == 2

    async def test_add_numbers_tool_executes(self):
        runner, mock = _make_runner_with_tools(add_numbers)

        @use_tools(add_numbers)
        @agent(model="mock-model")
        class MathAgent: ...

        mock.queue_tool_use("add_numbers", {"a": 3, "b": 7})
        mock.queue_response(_completion("3 + 7 = 10."))

        resp = await runner.run(MathAgent(), "Add 3 and 7")
        assert resp.turns == 2


# ---------------------------------------------------------------------------
# TestToolErrorHandling
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    async def test_tool_error_return_error_policy(self):
        runner, mock = _make_runner_with_tools(always_fails)

        @use_tools(always_fails)
        @agent(model="mock-model", tool_error_policy="return_error")
        class ErrorAgent: ...

        mock.queue_tool_use("always_fails", {"msg": "test"})
        mock.queue_response(_completion("Tool failed but I handled it."))

        resp = await runner.run(ErrorAgent(), "do it")
        # With return_error policy, execution continues
        assert resp.content == "Tool failed but I handled it."

    async def test_tool_error_raise_policy(self):
        from lauren_ai._tools._executor import ToolExecutionError

        runner, mock = _make_runner_with_tools(always_fails)

        @use_tools(always_fails)
        @agent(model="mock-model", tool_error_policy="raise")
        class RaiseAgent: ...

        mock.queue_tool_use("always_fails", {"msg": "boom"})

        with pytest.raises(ToolExecutionError):
            await runner.run(RaiseAgent(), "do it")

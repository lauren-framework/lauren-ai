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
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


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
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model")
@use_tools(get_weather)
class WeatherAgent: ...


@agent(model="mock-model")
@use_tools(add_numbers)
class MathAgent: ...


@agent(model="mock-model", tool_error_policy="return_error")
@use_tools(always_fails)
class ErrorAgent: ...


@agent(model="mock-model", tool_error_policy="raise")
@use_tools(always_fails)
class RaiseAgent: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(content: str = "OK") -> Completion:
    return Completion(
        id="c1",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


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
    def test_tool_call_executes_function(self):
        client = TestClient(WeatherAgent())
        client.mock.queue_tool_use("get_weather", {"city": "London"})
        client.mock.queue_response(_c("It is 18°C in London."))
        result = client.run("Weather in London?")
        assert result.content == "It is 18°C in London."

    def test_tool_call_records_in_response(self):
        client = TestClient(WeatherAgent())
        client.mock.queue_tool_use("get_weather", {"city": "Paris"})
        client.mock.queue_response(_c("Rainy in Paris."))
        result = client.run("Weather in Paris?")
        assert len(result.tool_calls_made) == 1

    def test_tool_call_name_recorded(self):
        client = TestClient(WeatherAgent())
        client.mock.queue_tool_use("get_weather", {"city": "Berlin"})
        client.mock.queue_response(_c("Cloudy in Berlin."))
        result = client.run("Weather in Berlin?")
        assert result.tool_calls_made[0].name == "get_weather"

    def test_two_llm_calls_for_tool_turn(self):
        client = TestClient(WeatherAgent())
        client.mock.queue_tool_use("get_weather", {"city": "Tokyo"})
        client.mock.queue_response(_c("Sunny in Tokyo."))
        client.run("Weather in Tokyo?")
        assert len(client.calls) == 2

    def test_add_numbers_tool_executes(self):
        client = TestClient(MathAgent())
        client.mock.queue_tool_use("add_numbers", {"a": 3, "b": 7})
        client.mock.queue_response(_c("3 + 7 = 10."))
        result = client.run("Add 3 and 7")
        assert result.turns == 2


# ---------------------------------------------------------------------------
# TestToolErrorHandling
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    def test_tool_error_return_error_policy(self):
        client = TestClient(ErrorAgent())
        client.mock.queue_tool_use("always_fails", {"msg": "test"})
        client.mock.queue_response(_c("Tool failed but I handled it."))
        result = client.run("do it")
        assert result.content == "Tool failed but I handled it."

    def test_tool_error_raise_policy(self):
        from lauren_ai._tools._executor import ToolExecutionError

        client = TestClient(RaiseAgent())
        client.mock.queue_tool_use("always_fails", {"msg": "boom"})
        with pytest.raises(ToolExecutionError):
            client.run("do it")

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
from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
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
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content: str = "OK") -> Completion:
    return Completion(
        id="c1",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controller / Module
# ---------------------------------------------------------------------------


class _RunRequest(BaseModel):
    prompt: str = "hi"


@controller("/agent")
class ToolCallingController:
    def __init__(self, mock: MockTransport) -> None:
        tools = {}
        _add_to_tool_map(tools, get_weather)
        _add_to_tool_map(tools, add_numbers)
        _add_to_tool_map(tools, always_fails)
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools=tools, config=cfg)
        self._mock = mock

    @post("/run-weather")
    async def run_weather(self, body: Json[_RunRequest]) -> dict:
        @use_tools(get_weather)
        @agent(model="mock-model")
        class WeatherAgent: ...

        resp = await self._runner.run(WeatherAgent(), body.prompt)
        return {
            "content": resp.content,
            "tool_calls_made": [t.name for t in resp.tool_calls_made],
            "turns": resp.turns,
            "calls": len(self._mock.calls),
        }

    @post("/run-add")
    async def run_add(self, body: Json[_RunRequest]) -> dict:
        @use_tools(add_numbers)
        @agent(model="mock-model")
        class MathAgent: ...

        resp = await self._runner.run(MathAgent(), body.prompt)
        return {
            "content": resp.content,
            "turns": resp.turns,
        }

    @post("/run-fail-return")
    async def run_fail_return(self, body: Json[_RunRequest]) -> dict:
        @use_tools(always_fails)
        @agent(model="mock-model", tool_error_policy="return_error")
        class ErrorAgent: ...

        resp = await self._runner.run(ErrorAgent(), body.prompt)
        return {"content": resp.content}

    @post("/run-fail-raise")
    async def run_fail_raise(self, body: Json[_RunRequest]) -> dict:
        from lauren_ai._tools._executor import ToolExecutionError

        @use_tools(always_fails)
        @agent(model="mock-model", tool_error_policy="raise")
        class RaiseAgent: ...

        try:
            await self._runner.run(RaiseAgent(), body.prompt)
            return {"raised": False}
        except ToolExecutionError:
            return {"raised": True}


@module(
    controllers=[ToolCallingController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class ToolCallingModule: ...


def build_app() -> TestClient:
    _MOCK.reset()
    return TestClient(LaurenFactory.create(ToolCallingModule))


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
# TestToolExecution (via TestClient)
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_tool_call_executes_function(self):
        client = build_app()
        _MOCK.queue_tool_use("get_weather", {"city": "London"})
        _MOCK.queue_response(_completion("It is 18°C in London."))
        r = client.post("/agent/run-weather", json={"prompt": "Weather in London?"})
        assert r.status_code == 200
        assert r.json()["content"] == "It is 18°C in London."

    def test_tool_call_records_in_response(self):
        client = build_app()
        _MOCK.queue_tool_use("get_weather", {"city": "Paris"})
        _MOCK.queue_response(_completion("Rainy in Paris."))
        r = client.post("/agent/run-weather", json={"prompt": "Weather in Paris?"})
        assert r.status_code == 200
        assert len(r.json()["tool_calls_made"]) == 1

    def test_tool_call_name_recorded(self):
        client = build_app()
        _MOCK.queue_tool_use("get_weather", {"city": "Berlin"})
        _MOCK.queue_response(_completion("Cloudy in Berlin."))
        r = client.post("/agent/run-weather", json={"prompt": "Weather in Berlin?"})
        assert r.status_code == 200
        assert r.json()["tool_calls_made"][0] == "get_weather"

    def test_two_llm_calls_for_tool_turn(self):
        client = build_app()
        _MOCK.queue_tool_use("get_weather", {"city": "Tokyo"})
        _MOCK.queue_response(_completion("Sunny in Tokyo."))
        r = client.post("/agent/run-weather", json={"prompt": "Weather in Tokyo?"})
        assert r.status_code == 200
        assert r.json()["calls"] == 2

    def test_add_numbers_tool_executes(self):
        client = build_app()
        _MOCK.queue_tool_use("add_numbers", {"a": 3, "b": 7})
        _MOCK.queue_response(_completion("3 + 7 = 10."))
        r = client.post("/agent/run-add", json={"prompt": "Add 3 and 7"})
        assert r.status_code == 200
        assert r.json()["turns"] == 2


# ---------------------------------------------------------------------------
# TestToolErrorHandling (via TestClient)
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    def test_tool_error_return_error_policy(self):
        client = build_app()
        _MOCK.queue_tool_use("always_fails", {"msg": "test"})
        _MOCK.queue_response(_completion("Tool failed but I handled it."))
        r = client.post("/agent/run-fail-return", json={"prompt": "do it"})
        assert r.status_code == 200
        assert r.json()["content"] == "Tool failed but I handled it."

    def test_tool_error_raise_policy(self):
        client = build_app()
        _MOCK.queue_tool_use("always_fails", {"msg": "boom"})
        r = client.post("/agent/run-fail-raise", json={"prompt": "do it"})
        assert r.status_code == 200
        assert r.json()["raised"] is True

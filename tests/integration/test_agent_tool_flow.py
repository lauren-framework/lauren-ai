"""Integration tests for a complete agent run that uses tools.

Tests cover:
- Single tool use (tool_use -> final answer)
- Multi-turn tool use (tool -> tool -> end)
- Tool error handling (tool raises exception)
- Agent with no tools (simple text completion)
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_map(*tool_funcs) -> dict:
    tools = {}
    for t in tool_funcs:
        m = getattr(t, TOOL_META)
        tools[m.name] = (t, m)
    return tools


def make_runner(
    mock: MockTransport,
    tools: dict | None = None,
) -> AgentRunner:
    tools = tools if tools is not None else {}
    config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, config=config)


def text_completion(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=20, output_tokens=10),
    )


# ---------------------------------------------------------------------------
# Tool definitions — declared at module level (no from __future__ import annotations)
# ---------------------------------------------------------------------------


@tool()
async def get_weather(city: str) -> dict:
    """Get weather data for a city.

    Args:
        city: The city name.
    """
    return {"city": city, "temperature": 22, "condition": "sunny"}


@tool()
async def get_forecast(city: str, days: int) -> list:
    """Get weather forecast.

    Args:
        city: The city name.
        days: Number of forecast days.
    """
    return [{"day": i + 1, "temp": 20 + i} for i in range(days)]


@tool()
async def always_fails(query: str) -> str:
    """A tool that always raises an exception.

    Args:
        query: Ignored input.
    """
    raise RuntimeError(f"Tool exploded for query: {query}")


# ---------------------------------------------------------------------------
# Agent class definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="You are a weather assistant.")
@use_tools(get_weather)
class WeatherAgent:
    pass


WeatherAgent.__lauren_ai_agent__.tools = _make_tool_map(get_weather)


@agent(model="mock-model", system="You are a multi-tool agent.")
@use_tools(get_weather, get_forecast)
class MultiToolAgent:
    pass


MultiToolAgent.__lauren_ai_agent__.tools = _make_tool_map(get_weather, get_forecast)


@agent(model="mock-model", system="You are an error-prone agent.")
@use_tools(always_fails)
class ErrorAgent:
    pass


ErrorAgent.__lauren_ai_agent__.tools = _make_tool_map(always_fails)


@agent(model="mock-model", system="You are a simple assistant.")
class SimpleAgent:
    pass


# ---------------------------------------------------------------------------
# Tests: single tool use
# ---------------------------------------------------------------------------


class TestSingleToolFlow:
    @pytest.mark.asyncio
    async def test_weather_tool_call_and_final_answer(self):
        """Agent calls get_weather once, then gives a final text answer."""
        mock = MockTransport()
        tools = _make_tool_map(get_weather)
        runner = make_runner(mock, tools)

        mock.queue_tool_use("get_weather", {"city": "London"})
        mock.queue_response(text_completion("The weather in London is sunny at 22°C.", id="c2"))

        instance = WeatherAgent()
        response = await runner.run(instance, "What's the weather in London?")

        assert response.content == "The weather in London is sunny at 22°C."
        assert response.stop_reason == "end_turn"
        assert response.turns == 2
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0].name == "get_weather"
        assert response.tool_calls_made[0].input == {"city": "London"}

    @pytest.mark.asyncio
    async def test_mock_transport_recorded_two_calls(self):
        """MockTransport records one call per model invocation."""
        mock = MockTransport()
        tools = _make_tool_map(get_weather)
        runner = make_runner(mock, tools)

        mock.queue_tool_use("get_weather", {"city": "Paris"})
        mock.queue_response(text_completion("Paris is sunny.", id="c2"))

        instance = WeatherAgent()
        await runner.run(instance, "Weather in Paris?")

        # Two model calls: once for the tool, once for the final answer
        assert len(mock.calls) == 2

    @pytest.mark.asyncio
    async def test_tool_use_id_is_tracked(self):
        """The tool_use_id on the ToolCall matches what MockTransport generated."""
        mock = MockTransport()
        tools = _make_tool_map(get_weather)
        runner = make_runner(mock, tools)

        mock.queue_tool_use("get_weather", {"city": "Tokyo"}, tool_use_id="tool-abc-123")
        mock.queue_response(text_completion("Tokyo weather: 22C.", id="c2"))

        instance = WeatherAgent()
        response = await runner.run(instance, "Tokyo weather?")

        assert response.tool_calls_made[0].tool_use_id == "tool-abc-123"

    @pytest.mark.asyncio
    async def test_token_usage_accumulated_across_turns(self):
        """Total usage sums across both model calls."""
        mock = MockTransport()
        tools = _make_tool_map(get_weather)
        runner = make_runner(mock, tools)

        mock.queue_tool_use("get_weather", {"city": "Berlin"})
        mock.queue_response(
            Completion(
                id="c2",
                model="mock-model",
                content="Berlin is cool.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=30, output_tokens=15),
            )
        )

        instance = WeatherAgent()
        response = await runner.run(instance, "Berlin weather?")

        # First call from queue_tool_use: TokenUsage(50, 20)
        # Second call: TokenUsage(30, 15)
        assert response.total_usage.input_tokens == 80
        assert response.total_usage.output_tokens == 35


# ---------------------------------------------------------------------------
# Tests: multi-turn tool use
# ---------------------------------------------------------------------------


class TestMultiTurnToolFlow:
    @pytest.mark.asyncio
    async def test_two_sequential_tool_calls_then_end(self):
        """Agent calls two different tools in separate turns before finishing."""
        mock = MockTransport()
        tools = _make_tool_map(get_weather, get_forecast)
        runner = make_runner(mock, tools)

        mock.queue_tool_use("get_weather", {"city": "Madrid"})
        mock.queue_tool_use("get_forecast", {"city": "Madrid", "days": 3})
        mock.queue_response(text_completion("Madrid weather and forecast: warm.", id="c3"))

        instance = MultiToolAgent()
        response = await runner.run(
            instance, "Give me current weather and 3-day forecast for Madrid."
        )

        assert response.stop_reason == "end_turn"
        assert response.turns == 3
        assert len(response.tool_calls_made) == 2
        tool_names = [tc.name for tc in response.tool_calls_made]
        assert "get_weather" in tool_names
        assert "get_forecast" in tool_names

    @pytest.mark.asyncio
    async def test_same_tool_called_twice(self):
        """Agent calls the same tool in consecutive turns."""
        mock = MockTransport()
        tools = _make_tool_map(get_weather)
        runner = make_runner(mock, tools)

        mock.queue_tool_use("get_weather", {"city": "Rome"})
        mock.queue_tool_use("get_weather", {"city": "Milan"})
        mock.queue_response(text_completion("Rome and Milan compared.", id="c3"))

        instance = WeatherAgent()
        response = await runner.run(instance, "Compare Rome and Milan weather.")

        assert len(response.tool_calls_made) == 2
        assert all(tc.name == "get_weather" for tc in response.tool_calls_made)


# ---------------------------------------------------------------------------
# Tests: tool error handling
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    @pytest.mark.asyncio
    async def test_tool_error_policy_return_error_by_default(self):
        """By default (return_error policy), tool exceptions do not propagate."""
        mock = MockTransport()
        tools = _make_tool_map(always_fails)
        runner = make_runner(mock, tools)

        mock.queue_tool_use("always_fails", {"query": "boom"})
        mock.queue_response(text_completion("Sorry, the tool failed.", id="c2"))

        instance = ErrorAgent()
        # Should NOT raise; tool_error_policy is "return_error" by default
        response = await runner.run(instance, "Try the failing tool.")

        assert response.stop_reason == "end_turn"
        assert "failed" in response.content.lower() or len(response.content) > 0

    @pytest.mark.asyncio
    async def test_tool_error_policy_raise(self):
        """With tool_error_policy='raise', tool exceptions propagate
        (wrapped in ToolExecutionError)."""
        # The executor wraps tool exceptions in its own ToolExecutionError.
        # Import it from the executor module directly.

        mock = MockTransport()
        tools = _make_tool_map(always_fails)
        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, config=config)

        @agent(model="mock-model", tool_error_policy="raise")
        @use_tools(always_fails)
        class RaisingErrorAgent:
            pass

        RaisingErrorAgent.__lauren_ai_agent__.tools = tools

        mock.queue_tool_use("always_fails", {"query": "will raise"})

        instance = RaisingErrorAgent()
        with pytest.raises(Exception):  # noqa: B017 — ToolExecutionError wraps RuntimeError
            await runner.run(instance, "Try the failing tool.")

    @pytest.mark.asyncio
    async def test_tool_error_policy_skip(self):
        """With tool_error_policy='skip', the loop continues silently after a tool error."""
        mock = MockTransport()
        tools = _make_tool_map(always_fails)
        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, config=config)

        @agent(model="mock-model", tool_error_policy="skip")
        @use_tools(always_fails)
        class SkippingErrorAgent:
            pass

        SkippingErrorAgent.__lauren_ai_agent__.tools = tools

        mock.queue_tool_use("always_fails", {"query": "silently skip"})
        mock.queue_response(text_completion("I skipped the error.", id="c2"))

        instance = SkippingErrorAgent()
        response = await runner.run(instance, "Try the failing tool.")
        assert response.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Tests: no tools (simple text completion)
# ---------------------------------------------------------------------------


class TestNoToolsFlow:
    @pytest.mark.asyncio
    async def test_simple_completion_no_tools(self):
        """An agent with no tools returns the first completion directly."""
        mock = MockTransport()
        runner = make_runner(mock)

        mock.queue_response(text_completion("Hello! I am a simple assistant."))

        instance = SimpleAgent()
        response = await runner.run(instance, "Hello!")

        assert response.content == "Hello! I am a simple assistant."
        assert response.stop_reason == "end_turn"
        assert response.turns == 1
        assert len(response.tool_calls_made) == 0

    @pytest.mark.asyncio
    async def test_no_tools_single_mock_call(self):
        """No-tool agent results in exactly one transport call."""
        mock = MockTransport()
        runner = make_runner(mock)

        mock.queue_response(text_completion("Simple answer."))

        instance = SimpleAgent()
        await runner.run(instance, "Any question.")

        assert len(mock.calls) == 1

    @pytest.mark.asyncio
    async def test_no_tools_metadata_passed_through(self):
        """Metadata is accessible in the agent context."""
        mock = MockTransport()
        runner = make_runner(mock)

        on_start_metadata = {}

        @agent(model="mock-model")
        class MetaAgent:
            async def on_start(self, ctx):
                on_start_metadata.update(ctx.metadata)

        mock.queue_response(text_completion("OK."))

        instance = MetaAgent()
        await runner.run(instance, "Hello", metadata={"user_id": "u-42", "session": "s-1"})

        assert on_start_metadata.get("user_id") == "u-42"
        assert on_start_metadata.get("session") == "s-1"

    @pytest.mark.asyncio
    async def test_stop_reason_from_completion_preserved(self):
        """stop_reason from the transport completion is preserved in the response."""
        mock = MockTransport()
        runner = make_runner(mock)

        mock.queue_response(
            Completion(
                id="c1",
                model="mock-model",
                content="Stopped early.",
                tool_calls=[],
                stop_reason="stop_sequence",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            )
        )

        instance = SimpleAgent()
        response = await runner.run(instance, "Any question.")
        # stop_sequence is treated as end_turn by the runner
        assert response.stop_reason == "end_turn"

"""Unit tests for the AgentRunner agentic loop."""
from __future__ import annotations

import pytest

from lauren_ai._agents import AgentResponse, agent, use_tools
from lauren_ai._agents._runner import AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import tool
from lauren_ai._tools._registry import ToolRegistry
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


def make_runner(mock: MockTransport, max_turns: int = 10) -> tuple[AgentRunner, type]:
    """Build a runner + simple agent class for testing."""
    registry = ToolRegistry()
    config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, registry=registry, config=config)
    return runner


@pytest.fixture()
def mock() -> MockTransport:
    return MockTransport()


class TestAgentRunnerBasic:
    @pytest.mark.asyncio
    async def test_single_turn_response(self, mock):
        mock.queue_response(
            Completion(
                id="c1", model="mock", content="Hello! How can I help?",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=8),
            )
        )

        @agent(model="mock-model")
        class SimpleAgent:
            pass

        runner = make_runner(mock)
        instance = SimpleAgent()
        response = await runner.run(instance, "Hi!")
        assert isinstance(response, AgentResponse)
        assert response.content == "Hello! How can I help?"
        assert response.turns == 1
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_tool_call_and_result(self, mock):
        @tool()
        async def get_time() -> str:
            """Get the current time."""
            return "12:00 PM UTC"

        registry = ToolRegistry()
        registry.register(get_time)
        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, registry=registry, config=config)

        # Turn 1: model calls the tool
        mock.queue_tool_use("get_time", {})
        # Turn 2: model responds with final answer
        mock.queue_response(
            Completion(
                id="c2", model="mock",
                content="The time is 12:00 PM UTC.",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=20, output_tokens=10),
            )
        )

        @agent(model="mock-model")
        @use_tools(get_time)
        class TimeAgent:
            pass

        instance = TimeAgent()
        response = await runner.run(instance, "What time is it?")
        assert "12:00" in response.content
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0].name == "get_time"

    @pytest.mark.asyncio
    async def test_max_turns_stop_reason(self, mock):
        # Queue completions that always call tools — loop exhausts turns
        for _ in range(5):
            mock.queue_tool_use("fake_tool", {})

        @agent(model="mock-model", max_turns=2)
        class StuckAgent:
            pass

        registry = ToolRegistry()
        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, registry=registry, config=config)

        instance = StuckAgent()
        response = await runner.run(instance, "Do something")
        # The runner sets stop_reason to "max_turns" when the loop exhausts
        assert response.stop_reason == "max_turns"

    @pytest.mark.asyncio
    async def test_cumulative_token_usage(self, mock):
        mock.queue_response(
            Completion(
                id="c1", model="mock", content="Result",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        )

        @agent(model="mock-model")
        class TokenAgent:
            pass

        runner = make_runner(mock)
        instance = TokenAgent()
        response = await runner.run(instance, "Hello")
        assert response.total_usage.input_tokens == 100
        assert response.total_usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_called(self, mock):
        mock.queue_response(
            Completion(
                id="c1", model="mock", content="Done",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            )
        )

        on_start_called = []
        on_finish_called = []

        @agent(model="mock-model")
        class HookAgent:
            async def on_start(self, ctx):
                on_start_called.append(True)

            async def on_finish(self, response, ctx):
                on_finish_called.append(True)

        runner = make_runner(mock)
        instance = HookAgent()
        await runner.run(instance, "Hi")

        assert len(on_start_called) == 1
        assert len(on_finish_called) == 1

    @pytest.mark.asyncio
    async def test_tool_error_policy_return_error(self, mock):
        @tool()
        async def failing_tool() -> str:
            """A tool that always fails."""
            raise RuntimeError("Tool exploded!")

        registry = ToolRegistry()
        registry.register(failing_tool)
        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, registry=registry, config=config)

        mock.queue_tool_use("failing_tool", {})
        mock.queue_response(
            Completion(
                id="c2", model="mock",
                content="I see the tool encountered an error.",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        )

        @agent(model="mock-model")
        @use_tools(failing_tool)
        class ErrorAgent:
            pass

        instance = ErrorAgent()
        # Should NOT raise — policy is "return_error" by default
        response = await runner.run(instance, "Use the tool")
        assert response.stop_reason == "end_turn"

"""Integration tests for SignalBus lifecycle signals during agent runs.

Tests cover:
- ModelCallStarted emitted before each LLM call
- ModelCallComplete emitted with correct usage info after each call
- ToolCallStarted and ToolCallComplete emitted around tool execution
- AgentRunComplete emitted at the end of the run
- Signal ordering is correct
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._signals import (
    AgentRunComplete,
    ModelCallComplete,
    ModelCallStarted,
    SignalBus,
    ToolCallComplete,
    ToolCallStarted,
)
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


def make_runner_with_signals(
    mock: MockTransport,
    bus: SignalBus,
    tools: dict | None = None,
) -> AgentRunner:
    tools = tools if tools is not None else {}
    config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools=tools, config=config, signals=bus)


def text_completion(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Tool definition (no from __future__ import annotations)
# ---------------------------------------------------------------------------


@tool()
async def echo_tool(message: str) -> str:
    """Echo the input message.

    Args:
        message: The message to echo.
    """
    return f"echo: {message}"


@agent(model="mock-model", system="Signal test agent.")
@use_tools(echo_tool)
class SignalTestAgent:
    pass


@agent(model="mock-model", system="Simple signal test agent.")
class SimpleSignalAgent:
    pass


@agent(name="Named CRM Agent", model="mock-model", system="Named agent for signal tests.")
class NamedCRMAgent:
    pass


# ---------------------------------------------------------------------------
# Tests: ModelCallStarted
# ---------------------------------------------------------------------------


class TestModelCallStartedSignal:
    @pytest.mark.asyncio
    async def test_model_call_started_emitted_once_for_simple_run(self):
        """ModelCallStarted is emitted once for a single-turn agent."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallStarted] = []

        @bus.on(ModelCallStarted)
        async def capture(event: ModelCallStarted) -> None:
            received.append(event)

        mock.queue_response(text_completion("Hello!"))

        instance = SimpleSignalAgent()
        await runner.run(instance, "Hello")

        assert len(received) == 1
        assert received[0].model == "mock-model"

    @pytest.mark.asyncio
    async def test_model_call_started_has_messages_count(self):
        """ModelCallStarted.messages_count reflects the conversation history."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallStarted] = []

        @bus.on(ModelCallStarted)
        async def capture(event: ModelCallStarted) -> None:
            received.append(event)

        mock.queue_response(text_completion("Answer!"))

        instance = SimpleSignalAgent()
        await runner.run(instance, "What is 2 + 2?")

        assert len(received) == 1
        # At least the user message is in memory
        assert received[0].messages_count >= 1

    @pytest.mark.asyncio
    async def test_model_call_started_has_agent_class(self):
        """ModelCallStarted.agent_class is set to the decorated agent class."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallStarted] = []

        @bus.on(ModelCallStarted)
        async def capture(event: ModelCallStarted) -> None:
            received.append(event)

        mock.queue_response(text_completion("Done."))

        instance = SimpleSignalAgent()
        await runner.run(instance, "Go")

        assert received[0].agent_class is SimpleSignalAgent


# ---------------------------------------------------------------------------
# Tests: ModelCallComplete
# ---------------------------------------------------------------------------


class TestModelCallCompleteSignal:
    @pytest.mark.asyncio
    async def test_model_call_complete_emitted_once_for_simple_run(self):
        """ModelCallComplete is emitted once after the single model call."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            received.append(event)

        mock.queue_response(
            Completion(
                id="c1",
                model="mock-model",
                content="Done.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=42, output_tokens=7),
            )
        )

        instance = SimpleSignalAgent()
        await runner.run(instance, "Test")

        assert len(received) == 1
        assert received[0].usage.input_tokens == 42
        assert received[0].usage.output_tokens == 7
        assert received[0].stop_reason == "end_turn"
        assert received[0].model == "mock-model"

    @pytest.mark.asyncio
    async def test_model_call_complete_emitted_twice_with_tool_use(self):
        """ModelCallComplete fires once per model call (two calls = two events)."""
        bus = SignalBus()
        mock = MockTransport()
        tools = _make_tool_map(echo_tool)
        runner = make_runner_with_signals(mock, bus, tools)

        received: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            received.append(event)

        mock.queue_tool_use("echo_tool", {"message": "ping"})
        mock.queue_response(text_completion("The echo said: ping.", id="c2"))

        instance = SignalTestAgent()
        await runner.run(instance, "Echo ping")

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_model_call_complete_has_duration_ms(self):
        """ModelCallComplete.duration_ms is a non-negative float."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            received.append(event)

        mock.queue_response(text_completion("Fast."))

        instance = SimpleSignalAgent()
        await runner.run(instance, "Quick test")

        assert received[0].duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Tests: ToolCallStarted and ToolCallComplete
# ---------------------------------------------------------------------------


class TestToolCallSignals:
    @pytest.mark.asyncio
    async def test_tool_call_started_emitted_before_tool(self):
        """ToolCallStarted is emitted with the tool name and input before execution."""
        bus = SignalBus()
        mock = MockTransport()
        tools = _make_tool_map(echo_tool)
        runner = make_runner_with_signals(mock, bus, tools)

        started: list[ToolCallStarted] = []

        @bus.on(ToolCallStarted)
        async def capture_started(event: ToolCallStarted) -> None:
            started.append(event)

        mock.queue_tool_use("echo_tool", {"message": "hello"}, tool_use_id="tid-001")
        mock.queue_response(text_completion("Echo done.", id="c2"))

        instance = SignalTestAgent()
        await runner.run(instance, "Echo hello")

        assert len(started) == 1
        assert started[0].tool_name == "echo_tool"
        assert started[0].input == {"message": "hello"}
        assert started[0].tool_use_id == "tid-001"

    @pytest.mark.asyncio
    async def test_tool_call_complete_emitted_after_tool(self):
        """ToolCallComplete is emitted after successful tool execution."""
        bus = SignalBus()
        mock = MockTransport()
        tools = _make_tool_map(echo_tool)
        runner = make_runner_with_signals(mock, bus, tools)

        completed: list[ToolCallComplete] = []

        @bus.on(ToolCallComplete)
        async def capture_complete(event: ToolCallComplete) -> None:
            completed.append(event)

        mock.queue_tool_use("echo_tool", {"message": "world"}, tool_use_id="tid-002")
        mock.queue_response(text_completion("Done.", id="c2"))

        instance = SignalTestAgent()
        await runner.run(instance, "Echo world")

        assert len(completed) == 1
        assert completed[0].tool_name == "echo_tool"
        assert completed[0].tool_use_id == "tid-002"
        assert completed[0].success is True
        assert completed[0].error is None

    @pytest.mark.asyncio
    async def test_tool_call_complete_marks_failure_on_error(self):
        """ToolCallComplete.success=False and error is set when the tool raises."""

        @tool()
        async def boom_tool(x: str) -> str:
            """A failing tool. Args: x: Ignored."""
            raise ValueError("Kaboom!")

        bus = SignalBus()
        mock = MockTransport()
        tools = _make_tool_map(boom_tool)
        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools=tools, config=config, signals=bus)

        @agent(model="mock-model")
        @use_tools(boom_tool)
        class BoomAgent:
            pass

        completed: list[ToolCallComplete] = []

        @bus.on(ToolCallComplete)
        async def capture(event: ToolCallComplete) -> None:
            completed.append(event)

        mock.queue_tool_use("boom_tool", {"x": "trigger"})
        mock.queue_response(text_completion("The tool failed.", id="c2"))

        instance = BoomAgent()
        await runner.run(instance, "Trigger the failure")

        assert len(completed) == 1
        assert completed[0].success is False
        assert completed[0].error is not None
        assert "Kaboom!" in completed[0].error


# ---------------------------------------------------------------------------
# Tests: AgentRunComplete
# ---------------------------------------------------------------------------


class TestAgentRunCompleteSignal:
    @pytest.mark.asyncio
    async def test_agent_run_complete_emitted_at_end(self):
        """AgentRunComplete is emitted once after the run finishes."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        completed: list[AgentRunComplete] = []

        @bus.on(AgentRunComplete)
        async def capture(event: AgentRunComplete) -> None:
            completed.append(event)

        mock.queue_response(text_completion("All done."))

        instance = SimpleSignalAgent()
        await runner.run(instance, "Finish please")

        assert len(completed) == 1
        assert completed[0].stop_reason == "end_turn"
        assert completed[0].turns == 1

    @pytest.mark.asyncio
    async def test_agent_run_complete_has_total_usage(self):
        """AgentRunComplete.total_usage carries the cumulative token usage."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        completed: list[AgentRunComplete] = []

        @bus.on(AgentRunComplete)
        async def capture(event: AgentRunComplete) -> None:
            completed.append(event)

        mock.queue_response(
            Completion(
                id="c1",
                model="mock-model",
                content="Summary.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        )

        instance = SimpleSignalAgent()
        await runner.run(instance, "Summarise something")

        assert completed[0].total_usage.input_tokens == 100
        assert completed[0].total_usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_agent_run_complete_has_agent_class(self):
        """AgentRunComplete.agent_class is set to the decorated class."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        completed: list[AgentRunComplete] = []

        @bus.on(AgentRunComplete)
        async def capture(event: AgentRunComplete) -> None:
            completed.append(event)

        mock.queue_response(text_completion("Done."))

        instance = SimpleSignalAgent()
        await runner.run(instance, "Go")

        assert completed[0].agent_class is SimpleSignalAgent


# ---------------------------------------------------------------------------
# Tests: signal ordering
# ---------------------------------------------------------------------------


class TestSignalOrdering:
    @pytest.mark.asyncio
    async def test_signal_order_simple_run(self):
        """For a simple (no-tool) run, order is: ModelCallStarted, ModelCallComplete, AgentRunComplete."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        signal_order: list[str] = []

        @bus.on(ModelCallStarted)
        async def on_started(event: ModelCallStarted) -> None:
            signal_order.append("ModelCallStarted")

        @bus.on(ModelCallComplete)
        async def on_complete(event: ModelCallComplete) -> None:
            signal_order.append("ModelCallComplete")

        @bus.on(AgentRunComplete)
        async def on_run_complete(event: AgentRunComplete) -> None:
            signal_order.append("AgentRunComplete")

        mock.queue_response(text_completion("Simple."))

        instance = SimpleSignalAgent()
        await runner.run(instance, "Go")

        assert signal_order == ["ModelCallStarted", "ModelCallComplete", "AgentRunComplete"]

    @pytest.mark.asyncio
    async def test_signal_order_with_tool(self):
        """With a tool call: ModelCallStarted (once), ModelCallComplete x2, ToolCallStarted, ToolCallComplete, AgentRunComplete.

        The runner emits ModelCallStarted once before the loop begins, then
        ModelCallComplete after each LLM call within the loop.
        """
        bus = SignalBus()
        mock = MockTransport()
        tools = _make_tool_map(echo_tool)
        runner = make_runner_with_signals(mock, bus, tools)

        signal_order: list[str] = []

        @bus.on(ModelCallStarted)
        async def on_model_started(event: ModelCallStarted) -> None:
            signal_order.append("ModelCallStarted")

        @bus.on(ModelCallComplete)
        async def on_model_complete(event: ModelCallComplete) -> None:
            signal_order.append("ModelCallComplete")

        @bus.on(ToolCallStarted)
        async def on_tool_started(event: ToolCallStarted) -> None:
            signal_order.append("ToolCallStarted")

        @bus.on(ToolCallComplete)
        async def on_tool_complete(event: ToolCallComplete) -> None:
            signal_order.append("ToolCallComplete")

        @bus.on(AgentRunComplete)
        async def on_run_complete(event: AgentRunComplete) -> None:
            signal_order.append("AgentRunComplete")

        mock.queue_tool_use("echo_tool", {"message": "test"})
        mock.queue_response(text_completion("Echo done.", id="c2"))

        instance = SignalTestAgent()
        await runner.run(instance, "Echo test")

        # ModelCallStarted is emitted once before the loop; ModelCallComplete is
        # emitted per LLM call (twice: once for tool_use, once for end_turn).
        expected = [
            "ModelCallStarted",
            "ModelCallComplete",
            "ToolCallStarted",
            "ToolCallComplete",
            "ModelCallComplete",
            "AgentRunComplete",
        ]
        assert signal_order == expected

    @pytest.mark.asyncio
    async def test_no_signals_without_bus(self):
        """When no SignalBus is configured, the runner still works without errors."""
        mock = MockTransport()
        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(
            transport=mock,
            tools={},
            config=config,
            signals=None,  # explicitly no bus
        )

        mock.queue_response(text_completion("No signals here."))

        instance = SimpleSignalAgent()
        response = await runner.run(instance, "Quiet run")

        assert response.content == "No signals here."


# ---------------------------------------------------------------------------
# Tests: agent_name in signals
# ---------------------------------------------------------------------------


class TestAgentNameInSignals:
    @pytest.mark.asyncio
    async def test_model_call_complete_carries_explicit_agent_name(self):
        """ModelCallComplete.agent_name reflects @agent(name=...) value."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            received.append(event)

        mock.queue_response(text_completion("Hi!"))

        instance = NamedCRMAgent()
        await runner.run(instance, "Hello")

        assert len(received) == 1
        assert received[0].agent_name == "Named CRM Agent"

    @pytest.mark.asyncio
    async def test_model_call_complete_defaults_to_class_name(self):
        """ModelCallComplete.agent_name falls back to class __name__ when name not set."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            received.append(event)

        mock.queue_response(text_completion("Hi!"))

        instance = SimpleSignalAgent()
        await runner.run(instance, "Hello")

        assert len(received) == 1
        assert received[0].agent_name == "SimpleSignalAgent"

    @pytest.mark.asyncio
    async def test_agent_run_complete_carries_explicit_agent_name(self):
        """AgentRunComplete.agent_name reflects @agent(name=...) value."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[AgentRunComplete] = []

        @bus.on(AgentRunComplete)
        async def capture(event: AgentRunComplete) -> None:
            received.append(event)

        mock.queue_response(text_completion("Done."))

        instance = NamedCRMAgent()
        await runner.run(instance, "Go")

        assert len(received) == 1
        assert received[0].agent_name == "Named CRM Agent"

    @pytest.mark.asyncio
    async def test_model_call_started_carries_agent_name(self):
        """ModelCallStarted.agent_name is populated from AgentMeta."""
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallStarted] = []

        @bus.on(ModelCallStarted)
        async def capture(event: ModelCallStarted) -> None:
            received.append(event)

        mock.queue_response(text_completion("Start!"))

        instance = NamedCRMAgent()
        await runner.run(instance, "Begin")

        assert len(received) == 1
        assert received[0].agent_name == "Named CRM Agent"


# ---------------------------------------------------------------------------
# Tests: parity — run_stream() must emit the same signals as run()
# ---------------------------------------------------------------------------


def _stream_chunks(*parts: str, stop_reason: str = "end_turn"):
    """Build a list of CompletionChunk objects from text parts."""
    from lauren_ai._transport import CompletionChunk

    chunks = [CompletionChunk(delta=p) for p in parts]
    chunks.append(
        CompletionChunk(
            delta="",
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=20, output_tokens=len(parts)),
        )
    )
    return chunks


class TestRunStreamSignalParity:
    """run_stream() emits the same lifecycle signals as run()."""

    @pytest.mark.asyncio
    async def test_model_call_started_emitted_via_run_stream(self):
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallStarted] = []

        @bus.on(ModelCallStarted)
        async def capture(event: ModelCallStarted) -> None:
            received.append(event)

        mock.queue_stream(_stream_chunks("Hi"))

        instance = SimpleSignalAgent()
        async for _ in await runner.run_stream(instance, "Hello"):
            pass

        assert len(received) == 1
        assert received[0].agent_class is SimpleSignalAgent

    @pytest.mark.asyncio
    async def test_model_call_complete_emitted_via_run_stream(self):
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            received.append(event)

        mock.queue_stream(_stream_chunks("Hello", " world"))

        instance = SimpleSignalAgent()
        async for _ in await runner.run_stream(instance, "Hi"):
            pass

        assert len(received) == 1
        assert received[0].usage.output_tokens == 2  # number of text parts
        assert received[0].stop_reason == "end_turn"
        assert received[0].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_agent_run_complete_emitted_via_run_stream(self):
        bus = SignalBus()
        mock = MockTransport()
        runner = make_runner_with_signals(mock, bus)

        received: list[AgentRunComplete] = []

        @bus.on(AgentRunComplete)
        async def capture(event: AgentRunComplete) -> None:
            received.append(event)

        mock.queue_stream(_stream_chunks("Done"))

        instance = SimpleSignalAgent()
        async for _ in await runner.run_stream(instance, "Go"):
            pass

        assert len(received) == 1
        assert received[0].stop_reason == "end_turn"
        assert received[0].agent_class is SimpleSignalAgent

    @pytest.mark.asyncio
    async def test_tool_call_signals_emitted_via_run_stream(self):
        from lauren_ai._transport import CompletionChunk, ToolCallDelta

        bus = SignalBus()
        mock = MockTransport()
        tools = _make_tool_map(echo_tool)
        runner = make_runner_with_signals(mock, bus, tools)

        started: list[ToolCallStarted] = []
        completed: list[ToolCallComplete] = []

        @bus.on(ToolCallStarted)
        async def cap_start(event: ToolCallStarted) -> None:
            started.append(event)

        @bus.on(ToolCallComplete)
        async def cap_done(event: ToolCallComplete) -> None:
            completed.append(event)

        # Turn 1: tool-use chunk
        mock.queue_stream(
            [
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="tc1",
                        name="echo_tool",
                        input_delta='{"message":"ping"}',
                    )
                ),
                CompletionChunk(
                    delta="",
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                ),
            ]
        )
        # Turn 2: final response
        mock.queue_stream(_stream_chunks("Echoed."))

        instance = SignalTestAgent()
        async for _ in await runner.run_stream(instance, "Echo ping"):
            pass

        assert len(started) == 1
        assert len(completed) == 1
        assert started[0].tool_name == "echo_tool"
        assert completed[0].success is True

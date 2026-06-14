"""Unit tests for EventSink, serialize, and run-scoped sinks (PRD: pluggable-event-sink).

NOTE: No ``from __future__ import annotations`` — @tool() inspects real annotations.
"""

import dataclasses
import json

import pytest

from lauren_ai import EventSink, serialize
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._signals import AgentRunComplete, ModelCallComplete, ToolCallComplete
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------


class RecordingSink:
    """Minimal EventSink implementation — records signal instances in order."""

    def __init__(self) -> None:
        self.signals: list[object] = []

    async def on_signal(self, signal: object) -> None:
        self.signals.append(signal)


def names(sink: RecordingSink) -> list[str]:
    return [type(s).__name__ for s in sink.signals]


def _c(content: str = "ok", n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


@tool()
async def echo_tool(msg: str) -> str:
    """Echo a message.

    Args:
        msg: The message to echo.
    """
    return msg


@agent(model="mock")
@use_tools(echo_tool)
class EchoAgent: ...


# ---------------------------------------------------------------------------
# EventSink structural conformance
# ---------------------------------------------------------------------------


class TestEventSinkProtocol:
    async def test_recording_sink_satisfies_protocol(self):
        assert isinstance(RecordingSink(), EventSink)

    async def test_plain_class_with_on_signal_satisfies_protocol(self):
        class Anon:
            async def on_signal(self, signal: object) -> None:
                pass

        assert isinstance(Anon(), EventSink)

    async def test_missing_on_signal_does_not_satisfy(self):
        assert not isinstance(object(), EventSink)


# ---------------------------------------------------------------------------
# Constructor sinks
# ---------------------------------------------------------------------------


class TestConstructorSinks:
    async def test_sink_called_for_end_turn_run(self):
        mock = MockTransport()
        mock.queue_response(_c("hello"))
        sink = RecordingSink()
        runner = AgentRunnerBase(mock, event_sinks=[sink])

        await runner.run(EchoAgent(), "hi")

        assert "ModelCallStarted" in names(sink)
        assert "ModelCallComplete" in names(sink)
        assert "AgentRunComplete" in names(sink)
        assert names(sink)[-1] == "AgentRunComplete"

    async def test_sink_receives_tool_call_signals(self):
        mock = MockTransport()
        mock.queue_tool_use("echo_tool", {"msg": "x"})
        mock.queue_response(_c("done", n=2))
        sink = RecordingSink()
        runner = AgentRunnerBase(mock, event_sinks=[sink])

        await runner.run(EchoAgent(), "echo x")

        signal_names = names(sink)
        assert "ToolCallStarted" in signal_names
        assert "ToolCallComplete" in signal_names
        # AgentRunComplete must be last
        assert signal_names[-1] == "AgentRunComplete"

    async def test_sinks_fire_without_signal_bus(self):
        """signals=None must not short-circuit _emit when sinks exist."""
        mock = MockTransport()
        mock.queue_response(_c("hello"))
        sink = RecordingSink()
        runner = AgentRunnerBase(mock, signals=None, event_sinks=[sink])

        await runner.run(EchoAgent(), "hi")

        assert "AgentRunComplete" in names(sink)

    async def test_raising_sink_never_interrupts_run(self):
        class Bomb:
            async def on_signal(self, signal: object) -> None:
                raise RuntimeError("boom")

        mock = MockTransport()
        mock.queue_response(_c("ok"))
        good = RecordingSink()
        runner = AgentRunnerBase(mock, event_sinks=[Bomb(), good])

        response = await runner.run(EchoAgent(), "hi")
        assert response.content == "ok"
        assert "AgentRunComplete" in names(good)

    async def test_no_sinks_no_bus_is_noop(self):
        """Regression: legacy fast path preserved when neither sinks nor bus configured."""
        mock = MockTransport()
        mock.queue_response(_c("ok"))
        runner = AgentRunnerBase(mock)

        response = await runner.run(EchoAgent(), "hi")
        assert response.content == "ok"

    async def test_multiple_constructor_sinks_all_called(self):
        mock = MockTransport()
        mock.queue_response(_c("ok"))
        sink_a, sink_b = RecordingSink(), RecordingSink()
        runner = AgentRunnerBase(mock, event_sinks=[sink_a, sink_b])

        await runner.run(EchoAgent(), "hi")

        assert "AgentRunComplete" in names(sink_a)
        assert "AgentRunComplete" in names(sink_b)


# ---------------------------------------------------------------------------
# Run-scoped sinks
# ---------------------------------------------------------------------------


class TestRunScopedSinks:
    async def test_run_scoped_sink_sees_only_its_run(self):
        mock = MockTransport()
        mock.queue_response(_c("a"))
        mock.queue_response(_c("b"))
        per_run = RecordingSink()
        runner = AgentRunnerBase(mock)

        await runner.run(EchoAgent(), "first", event_sinks=[per_run])
        n_first = len(per_run.signals)

        await runner.run(EchoAgent(), "second")  # no sink

        assert n_first > 0
        assert len(per_run.signals) == n_first  # second run did not append

    async def test_constructor_sinks_precede_run_scoped_sinks(self):
        order: list[str] = []

        class Tagged:
            def __init__(self, tag: str) -> None:
                self._tag = tag

            async def on_signal(self, signal: object) -> None:
                order.append(self._tag)

        mock = MockTransport()
        mock.queue_response(_c("ok"))
        runner = AgentRunnerBase(mock, event_sinks=[Tagged("ctor")])

        await runner.run(EchoAgent(), "hi", event_sinks=[Tagged("run")])

        # Every signal: ctor fires before run
        assert len(order) > 0
        for i in range(0, len(order), 2):
            assert order[i] == "ctor"
            if i + 1 < len(order):
                assert order[i + 1] == "run"

    async def test_run_scoped_sinks_do_not_mutate_runner_state(self):
        mock = MockTransport()
        mock.queue_response(_c("ok"))
        mock.queue_response(_c("ok2"))
        per_run = RecordingSink()
        runner = AgentRunnerBase(mock)

        await runner.run(EchoAgent(), "first", event_sinks=[per_run])
        # Runner's own _event_sinks must remain empty
        assert runner._event_sinks == ()


# ---------------------------------------------------------------------------
# serialize()
# ---------------------------------------------------------------------------


class TestSerialize:
    async def test_includes_signal_type_discriminator(self):
        payload = serialize(AgentRunComplete(agent_id="r1", stop_reason="end_turn"))
        assert payload["signal_type"] == "AgentRunComplete"

    async def test_output_is_json_safe(self):
        payload = serialize(
            ToolCallComplete(
                tool_name="search",
                tool_use_id="tu_1",
                agent_id="run_abc",
                duration_ms=12.5,
                success=True,
                error=None,
            )
        )
        json.dumps(payload)  # must not raise

    async def test_all_signal_classes_json_safe(self):
        import inspect

        from lauren_ai import _signals as sig_module

        for _, cls in inspect.getmembers(sig_module, inspect.isclass):
            if not dataclasses.is_dataclass(cls) or cls.__name__ in ("LifecycleEvent",):
                continue
            try:
                instance = cls()
            except TypeError:
                continue
            payload = serialize(instance)
            assert payload["signal_type"] == cls.__name__
            json.dumps(payload)  # must not raise

    async def test_agent_class_becomes_qualified_name_string(self):
        payload = serialize(AgentRunComplete(agent_class=EchoAgent))
        assert isinstance(payload["agent_class"], str)
        assert "EchoAgent" in payload["agent_class"]

    async def test_none_fields_preserved(self):
        payload = serialize(
            ToolCallComplete(
                tool_name="t",
                tool_use_id="x",
                agent_id="a",
                duration_ms=1.0,
                success=True,
                error=None,
            )
        )
        assert payload["error"] is None

    async def test_non_dataclass_raises_type_error(self):
        with pytest.raises(TypeError):
            serialize(object())

    async def test_serialize_works_on_model_call_complete(self):
        from lauren_ai._transport import TokenUsage

        payload = serialize(
            ModelCallComplete(
                model="claude-sonnet",
                agent_id="a1",
                duration_ms=42.0,
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                stop_reason="end_turn",
                cost_usd=0.001,
            )
        )
        assert payload["signal_type"] == "ModelCallComplete"
        assert payload["model"] == "claude-sonnet"
        # Nested TokenUsage must also be serialised
        assert isinstance(payload["usage"], dict)
        json.dumps(payload)

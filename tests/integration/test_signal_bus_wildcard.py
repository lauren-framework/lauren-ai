"""Integration tests for SignalBus.on_any(), off_any(), and sink-before-bus ordering.

PRD: pluggable-event-sink-for-agent-runner — §8.2
"""

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._signals import (
    AgentRunComplete,
    ModelCallComplete,
    SignalBus,
    ToolCallStarted,
)
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


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
async def noop_tool(x: str) -> str:
    """No-op tool.

    Args:
        x: Any string.
    """
    return x


@agent(model="mock")
@use_tools(noop_tool)
class SimpleAgent: ...


# ---------------------------------------------------------------------------
# on_any / off_any / any_handler_count
# ---------------------------------------------------------------------------


class TestOnAny:
    async def test_wildcard_receives_every_event_type(self):
        bus = SignalBus()
        seen: list[str] = []

        @bus.on_any
        async def audit(event) -> None:
            seen.append(type(event).__name__)

        await bus.emit(ModelCallComplete(model="m"))
        await bus.emit(ToolCallStarted(tool_name="t"))
        await bus.emit(AgentRunComplete(agent_id="r1"))

        assert seen == ["ModelCallComplete", "ToolCallStarted", "AgentRunComplete"]

    async def test_wildcard_fires_when_no_typed_handler_exists(self):
        """The old `if not handlers: return` must not skip on_any."""
        bus = SignalBus()
        hits: list[object] = []

        @bus.on_any
        async def catch_all(event) -> None:
            hits.append(event)

        await bus.emit(ToolCallStarted(tool_name="lonely"))
        assert len(hits) == 1
        assert isinstance(hits[0], ToolCallStarted)

    async def test_typed_handlers_scheduled_before_wildcard(self):
        """Within one emit, typed handlers precede wildcard handlers."""
        bus = SignalBus()
        order: list[str] = []

        @bus.on(ModelCallComplete)
        async def typed(event) -> None:
            order.append("typed")

        @bus.on_any
        async def wild(event) -> None:
            order.append("wildcard")

        await bus.emit(ModelCallComplete(model="m"))
        assert order == ["typed", "wildcard"]

    async def test_off_any_unregisters_handler(self):
        bus = SignalBus()
        hits: list[object] = []

        async def handler(event) -> None:
            hits.append(event)

        bus.on_any(handler)
        bus.off_any(handler)
        await bus.emit(ToolCallStarted())
        assert hits == []
        assert bus.any_handler_count() == 0

    async def test_off_any_is_noop_for_unregistered_handler(self):
        bus = SignalBus()

        async def handler(event) -> None:
            pass

        bus.off_any(handler)  # must not raise
        assert bus.any_handler_count() == 0

    async def test_any_handler_count_tracks_registrations(self):
        bus = SignalBus()
        assert bus.any_handler_count() == 0

        async def h1(e) -> None:
            pass

        async def h2(e) -> None:
            pass

        bus.on_any(h1)
        assert bus.any_handler_count() == 1
        bus.on_any(h2)
        assert bus.any_handler_count() == 2
        bus.off_any(h1)
        assert bus.any_handler_count() == 1

    async def test_usable_as_bare_decorator(self):
        bus = SignalBus()
        hits: list[object] = []

        @bus.on_any
        async def catch(event) -> None:
            hits.append(event)

        await bus.emit(AgentRunComplete(agent_id="x"))
        assert len(hits) == 1

    async def test_raising_wildcard_handler_is_suppressed(self, capsys):
        bus = SignalBus()

        @bus.on_any
        async def bomb(event) -> None:
            raise ValueError("boom")

        await bus.emit(ToolCallStarted())  # must not raise
        # Main guarantee: no exception propagated. Stderr output is a nice-to-have.

    async def test_clear_no_arg_removes_wildcard_handlers(self):
        bus = SignalBus()
        hits: list[object] = []

        async def hit_handler(e) -> None:
            hits.append(e)

        bus.on_any(hit_handler)
        bus.clear()
        assert bus.any_handler_count() == 0
        await bus.emit(ToolCallStarted())
        assert hits == []

    async def test_clear_with_event_type_preserves_wildcard_handlers(self):
        bus = SignalBus()
        hits: list[object] = []

        @bus.on(ModelCallComplete)
        async def typed_h(event) -> None:
            pass

        @bus.on_any
        async def wild_h(event) -> None:
            hits.append(event)

        bus.clear(ModelCallComplete)
        assert bus.handler_count(ModelCallComplete) == 0
        assert bus.any_handler_count() == 1  # wildcard preserved

        await bus.emit(ModelCallComplete(model="m"))
        assert len(hits) == 1

    async def test_multiple_wildcard_handlers_all_invoked(self):
        bus = SignalBus()
        a_hits: list[object] = []
        b_hits: list[object] = []

        @bus.on_any
        async def ah(event) -> None:
            a_hits.append(event)

        @bus.on_any
        async def bh(event) -> None:
            b_hits.append(event)

        await bus.emit(AgentRunComplete(agent_id="x"))
        assert len(a_hits) == 1
        assert len(b_hits) == 1


# ---------------------------------------------------------------------------
# Sink-before-bus ordering
# ---------------------------------------------------------------------------


class TestSinkVsBusOrdering:
    async def test_sinks_complete_before_bus_handlers(self):
        """Sinks are awaited sequentially; SignalBus.emit fires after all sinks."""
        order: list[str] = []
        bus = SignalBus()

        @bus.on_any
        async def bus_side(event) -> None:
            order.append(f"bus:{type(event).__name__}")

        class OrderedSink:
            async def on_signal(self, signal) -> None:
                order.append(f"sink:{type(signal).__name__}")

        mock = MockTransport()
        mock.queue_response(_c("ok"))
        runner = AgentRunnerBase(mock, signals=bus, event_sinks=[OrderedSink()])
        await runner.run(SimpleAgent(), "hi")

        # Every odd index is a sink signal, every even is its matching bus signal
        assert len(order) > 0
        for i in range(0, len(order) - 1, 2):
            assert order[i].startswith("sink:"), f"expected sink at index {i}: {order}"
            assert order[i + 1].startswith("bus:"), f"expected bus at index {i + 1}: {order}"
            # Same signal type
            assert order[i][5:] == order[i + 1][4:]


# ---------------------------------------------------------------------------
# Downstream smoke test — KernelBridge pattern (informational, not a full E2E)
# ---------------------------------------------------------------------------


class TestKernelBridgePattern:
    async def test_bridge_collects_all_signal_types(self):
        """Simulate the Agenthicc KernelBridge pattern."""
        from lauren_ai import serialize

        collected: list[dict] = []

        class Bridge:
            async def on_signal(self, signal) -> None:
                collected.append(serialize(signal))

        mock = MockTransport()
        mock.queue_response(_c("ok"))
        runner = AgentRunnerBase(mock, event_sinks=[Bridge()])
        await runner.run(SimpleAgent(), "hi")

        assert len(collected) > 0
        types = [d["signal_type"] for d in collected]
        assert "ModelCallStarted" in types
        assert "AgentRunComplete" in types
        assert types[-1] == "AgentRunComplete"

        # All payloads are JSON-safe
        import json

        for payload in collected:
            json.dumps(payload)

"""Unit tests for the SignalBus and signal types."""
from __future__ import annotations

import pytest

from lauren_ai._signals import (
    AgentRunComplete,
    ModelCallComplete,
    ModelCallStarted,
    SignalBus,
)
from lauren_ai._transport import TokenUsage


class TestSignalBus:
    @pytest.mark.asyncio
    async def test_emit_and_receive(self):
        bus = SignalBus()
        received = []

        @bus.on(ModelCallStarted)
        async def handler(event):
            received.append(event)

        event = ModelCallStarted(model="claude-opus-4-6", messages_count=1)
        await bus.emit(event)
        assert len(received) == 1
        assert received[0] is event

    @pytest.mark.asyncio
    async def test_off_removes_handler(self):
        bus = SignalBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.on(ModelCallStarted)(handler)
        bus.off(ModelCallStarted, handler)

        await bus.emit(ModelCallStarted(model="mock", messages_count=0))
        assert received == []

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        bus = SignalBus()
        count = []

        @bus.on(ModelCallComplete)
        async def h1(event):
            count.append(1)

        @bus.on(ModelCallComplete)
        async def h2(event):
            count.append(2)

        usage = TokenUsage(input_tokens=100, output_tokens=50)
        await bus.emit(
            ModelCallComplete(
                model="mock",
                usage=usage,
                duration_ms=10.0,
                stop_reason="end_turn",
                cost_usd=0.01,
            )
        )
        assert len(count) == 2

    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_crash_bus(self):
        bus = SignalBus()

        @bus.on(ModelCallComplete)
        async def bad_handler(event):
            raise RuntimeError("Handler error!")

        usage = TokenUsage(input_tokens=10, output_tokens=5)
        # Should not raise
        await bus.emit(
            ModelCallComplete(
                model="mock",
                usage=usage,
                duration_ms=5.0,
                stop_reason="end_turn",
                cost_usd=0.0,
            )
        )

    @pytest.mark.asyncio
    async def test_emit_unhandled_event_no_error(self):
        bus = SignalBus()
        # No handlers registered for AgentRunComplete
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        await bus.emit(
            AgentRunComplete(
                turns=0,
                total_usage=usage,
                total_cost_usd=0.0,
                stop_reason="end_turn",
            )
        )

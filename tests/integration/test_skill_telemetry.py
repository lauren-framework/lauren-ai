"""Integration tests for Skill 50: Agent Telemetry & Performance Metrics.

Tests cover:
- AgentTelemetry subscribes to ModelCallComplete and AgentRunComplete
- Call count increments after agent run
- Token counts accumulate from events
- Run count increments after AgentRunComplete
- get_summary returns correct totals
- Per-model breakdown in summary
- Cost accumulation from events

NOTE: from __future__ import annotations is safe here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from lauren_ai import SignalBus, ModelCallComplete, AgentRunComplete
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# AgentTelemetry implementation (inline)
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    error_count: int = 0


class AgentTelemetry:
    def __init__(self, signal_bus: SignalBus):
        self._metrics: dict[str, ModelMetrics] = defaultdict(ModelMetrics)
        self._run_count = 0
        signal_bus.on(ModelCallComplete)(self._on_model_call)
        signal_bus.on(AgentRunComplete)(self._on_run_complete)

    async def _on_model_call(self, event: ModelCallComplete) -> None:
        m = self._metrics[event.model]
        m.call_count += 1
        if event.usage:
            m.total_input_tokens += event.usage.input_tokens
            m.total_output_tokens += event.usage.output_tokens
        m.total_cost_usd += event.cost_usd or 0.0
        m.total_duration_ms += event.duration_ms or 0.0

    async def _on_run_complete(self, event: AgentRunComplete) -> None:
        self._run_count += 1

    def get_summary(self) -> dict:
        total_cost = sum(m.total_cost_usd for m in self._metrics.values())
        total_calls = sum(m.call_count for m in self._metrics.values())
        return {
            "total_agent_runs": self._run_count,
            "total_llm_calls": total_calls,
            "total_cost_usd": total_cost,
            "per_model": {
                model: {
                    "calls": m.call_count,
                    "input_tokens": m.total_input_tokens,
                    "output_tokens": m.total_output_tokens,
                    "cost_usd": m.total_cost_usd,
                }
                for model, m in self._metrics.items()
            },
        }

    def clear(self) -> None:
        self._metrics.clear()
        self._run_count = 0


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model=None, system="You are helpful.")
class TelemetryAgent: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner_with_bus(bus: SignalBus, mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools={}, config=cfg, signals=bus)


# ---------------------------------------------------------------------------
# Tests: SignalBus basics
# ---------------------------------------------------------------------------


class TestSignalBus:
    async def test_signal_bus_on_decorator(self):
        bus = SignalBus()
        received = []

        @bus.on(ModelCallComplete)
        async def handler(event: ModelCallComplete) -> None:
            received.append(event)

        event = ModelCallComplete(model="test", usage=None)
        await bus.emit(event)
        assert len(received) == 1

    async def test_signal_bus_emit_correct_event(self):
        bus = SignalBus()
        models_seen = []

        @bus.on(ModelCallComplete)
        async def handler(event: ModelCallComplete) -> None:
            models_seen.append(event.model)

        await bus.emit(ModelCallComplete(model="gpt-4"))
        assert "gpt-4" in models_seen

    async def test_signal_bus_handler_count(self):
        bus = SignalBus()

        @bus.on(ModelCallComplete)
        async def h1(e: ModelCallComplete) -> None: pass

        @bus.on(ModelCallComplete)
        async def h2(e: ModelCallComplete) -> None: pass

        assert bus.handler_count(ModelCallComplete) == 2


# ---------------------------------------------------------------------------
# Tests: AgentTelemetry via direct event emission
# ---------------------------------------------------------------------------


class TestAgentTelemetryEvents:
    async def test_initial_summary_is_zero(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)
        summary = telemetry.get_summary()
        assert summary["total_agent_runs"] == 0
        assert summary["total_llm_calls"] == 0
        assert summary["total_cost_usd"] == 0.0

    async def test_model_call_increments_call_count(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)

        await bus.emit(ModelCallComplete(
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
            duration_ms=100.0,
        ))

        summary = telemetry.get_summary()
        assert summary["total_llm_calls"] == 1

    async def test_model_call_accumulates_tokens(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)

        await bus.emit(ModelCallComplete(
            model="mock-model",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            cost_usd=0.0,
        ))
        await bus.emit(ModelCallComplete(
            model="mock-model",
            usage=TokenUsage(input_tokens=200, output_tokens=100),
            cost_usd=0.0,
        ))

        summary = telemetry.get_summary()
        model_data = summary["per_model"]["mock-model"]
        assert model_data["input_tokens"] == 300
        assert model_data["output_tokens"] == 150

    async def test_agent_run_complete_increments_run_count(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)

        await bus.emit(AgentRunComplete(
            agent_id="run1",
            agent_name="TestAgent",
            turns=1,
            total_usage=TokenUsage(input_tokens=10, output_tokens=5),
            total_cost_usd=0.001,
            stop_reason="end_turn",
        ))

        summary = telemetry.get_summary()
        assert summary["total_agent_runs"] == 1

    async def test_multiple_runs_accumulate(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)

        for i in range(3):
            await bus.emit(AgentRunComplete(
                agent_id=f"run{i}",
                agent_name="TestAgent",
                turns=1,
                total_usage=TokenUsage(input_tokens=10, output_tokens=5),
                total_cost_usd=0.001,
                stop_reason="end_turn",
            ))

        summary = telemetry.get_summary()
        assert summary["total_agent_runs"] == 3

    async def test_cost_accumulates(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)

        await bus.emit(ModelCallComplete(
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.005,
        ))
        await bus.emit(ModelCallComplete(
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.003,
        ))

        summary = telemetry.get_summary()
        assert abs(summary["total_cost_usd"] - 0.008) < 1e-9

    async def test_per_model_breakdown(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)

        await bus.emit(ModelCallComplete(
            model="model-a",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
        ))
        await bus.emit(ModelCallComplete(
            model="model-b",
            usage=TokenUsage(input_tokens=20, output_tokens=10),
            cost_usd=0.002,
        ))

        summary = telemetry.get_summary()
        assert "model-a" in summary["per_model"]
        assert "model-b" in summary["per_model"]
        assert summary["per_model"]["model-a"]["calls"] == 1
        assert summary["per_model"]["model-b"]["calls"] == 1


# ---------------------------------------------------------------------------
# Tests: AgentTelemetry via TestClient + real agent runner
# ---------------------------------------------------------------------------


class TestAgentTelemetryIntegration:
    async def test_runner_emits_model_call_complete(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)
        mock = MockTransport()
        mock.queue_response(_completion("Hello!"))
        runner = _make_runner_with_bus(bus, mock)

        await runner.run(TelemetryAgent(), "Hi")

        summary = telemetry.get_summary()
        assert summary["total_llm_calls"] == 1

    async def test_runner_emits_agent_run_complete(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)
        mock = MockTransport()
        mock.queue_response(_completion("Done"))
        runner = _make_runner_with_bus(bus, mock)

        await runner.run(TelemetryAgent(), "Go")

        summary = telemetry.get_summary()
        assert summary["total_agent_runs"] == 1

    async def test_two_runs_tracked_correctly(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)
        mock = MockTransport()
        mock.queue_response(_completion("Run 1"))
        mock.queue_response(_completion("Run 2"))
        runner = _make_runner_with_bus(bus, mock)

        await runner.run(TelemetryAgent(), "First")
        await runner.run(TelemetryAgent(), "Second")

        summary = telemetry.get_summary()
        assert summary["total_agent_runs"] == 2
        assert summary["total_llm_calls"] == 2

    async def test_token_usage_captured_from_runner(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)
        mock = MockTransport()
        mock.queue_response(_completion("result"))
        runner = _make_runner_with_bus(bus, mock)

        await runner.run(TelemetryAgent(), "prompt")

        summary = telemetry.get_summary()
        model_data = summary["per_model"].get("mock-model", {})
        assert model_data.get("input_tokens", 0) > 0
        assert model_data.get("output_tokens", 0) > 0

    async def test_telemetry_with_test_client(self):
        bus = SignalBus()
        telemetry = AgentTelemetry(bus)
        mock = MockTransport()
        mock.queue_response(_completion("Hello"))
        runner = _make_runner_with_bus(bus, mock)

        client = TestClient(TelemetryAgent(), mock, runner=runner)
        await client.run_async("prompt")

        summary = telemetry.get_summary()
        assert summary["total_llm_calls"] >= 1
        assert summary["total_agent_runs"] >= 1

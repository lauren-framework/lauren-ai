"""Integration tests for the callback/tracing handler pattern (Skill 32).

Tests cover:
- SimpleTracer records ModelCallComplete spans after run
- SimpleTracer records AgentRunComplete spans with turns and cost
- Spans list is populated only after the run completes
- Multiple runs accumulate spans
- Bus with no handlers does not interfere with agent execution
"""

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._signals import AgentRunComplete, ModelCallComplete, SignalBus
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient as AgentTestClient

# ---------------------------------------------------------------------------
# SimpleTracer implementation
# ---------------------------------------------------------------------------


class SimpleTracer:
    def __init__(self, signal_bus: SignalBus):
        self._spans: list[dict] = []
        self._bus = signal_bus
        signal_bus.on(ModelCallComplete)(self._on_llm_call)
        signal_bus.on(AgentRunComplete)(self._on_run_complete)

    async def _on_llm_call(self, event: ModelCallComplete) -> None:
        self._spans.append(
            {
                "type": "llm_call",
                "model": event.model,
                "cost_usd": event.cost_usd,
                "input_tokens": event.usage.input_tokens if event.usage else 0,
                "output_tokens": event.usage.output_tokens if event.usage else 0,
            }
        )

    async def _on_run_complete(self, event: AgentRunComplete) -> None:
        self._spans.append(
            {
                "type": "run_complete",
                "turns": event.turns,
                "total_cost": event.total_cost_usd,
            }
        )

    @property
    def spans(self) -> list[dict]:
        return list(self._spans)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="Tracing test agent.")
class TracingTestAgent:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str = "OK", *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _completion_custom_tokens(input_t: int, output_t: int) -> Completion:
    return Completion(
        id="c1",
        model="mock-model",
        content="Done",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=input_t, output_tokens=output_t),
    )


def _make_client(
    *responses: str, bus: SignalBus | None = None
) -> tuple[AgentTestClient, SignalBus]:
    """Return (TestClient, SignalBus) with queued responses."""
    if bus is None:
        bus = SignalBus()
    mock = MockTransport()
    for content in responses:
        mock.queue_response(_completion(content))
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, config=cfg, signals=bus)
    client = AgentTestClient(TracingTestAgent(), mock, runner=runner)
    return client, bus


# ---------------------------------------------------------------------------
# Tests: SimpleTracer spans
# ---------------------------------------------------------------------------


class TestSimpleTracerSpans:
    async def test_llm_call_span_recorded_after_run(self):
        """A ModelCallComplete span is recorded after a successful run."""
        bus = SignalBus()
        tracer = SimpleTracer(bus)
        client, _ = _make_client("Hello", bus=bus)
        await client.run_async("Hi")
        llm_spans = [s for s in tracer.spans if s["type"] == "llm_call"]
        assert len(llm_spans) == 1
        assert llm_spans[0]["model"] == "mock-model"

    async def test_llm_call_span_has_token_counts(self):
        """The llm_call span carries input and output token counts."""
        bus = SignalBus()
        tracer = SimpleTracer(bus)
        mock = MockTransport()
        mock.queue_response(_completion_custom_tokens(42, 7))
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, config=cfg, signals=bus)
        client = AgentTestClient(TracingTestAgent(), mock, runner=runner)
        await client.run_async("Test")
        llm_spans = [s for s in tracer.spans if s["type"] == "llm_call"]
        assert len(llm_spans) == 1
        assert llm_spans[0]["input_tokens"] == 42
        assert llm_spans[0]["output_tokens"] == 7

    async def test_run_complete_span_recorded(self):
        """An AgentRunComplete span is recorded with turns and total_cost."""
        bus = SignalBus()
        tracer = SimpleTracer(bus)
        client, _ = _make_client("Finished", bus=bus)
        await client.run_async("Go")
        run_spans = [s for s in tracer.spans if s["type"] == "run_complete"]
        assert len(run_spans) == 1
        assert run_spans[0]["turns"] == 1
        assert isinstance(run_spans[0]["total_cost"], float)

    async def test_spans_empty_before_run(self):
        """The tracer has no spans before any run is executed."""
        bus = SignalBus()
        tracer = SimpleTracer(bus)
        assert len(tracer.spans) == 0

    async def test_two_runs_produce_two_sets_of_spans(self):
        """Two consecutive runs each produce their own set of spans."""
        bus = SignalBus()
        tracer = SimpleTracer(bus)
        client, _ = _make_client("First", "Second", bus=bus)
        await client.run_async("Run 1")
        await client.run_async("Run 2")
        llm_spans = [s for s in tracer.spans if s["type"] == "llm_call"]
        run_spans = [s for s in tracer.spans if s["type"] == "run_complete"]
        assert len(llm_spans) == 2
        assert len(run_spans) == 2

    async def test_run_completes_without_bus(self):
        """Agent run succeeds when no SignalBus is provided (signals=None)."""
        mock = MockTransport()
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, config=cfg, signals=None)
        mock.queue_response(_completion("No bus"))
        client = AgentTestClient(TracingTestAgent(), mock, runner=runner)
        response = await client.run_async("Hello")
        assert response.content == "No bus"

    async def test_span_order_is_llm_then_run_complete(self):
        """ModelCallComplete span is recorded before AgentRunComplete span."""
        bus = SignalBus()
        tracer = SimpleTracer(bus)
        client, _ = _make_client("OK", bus=bus)
        await client.run_async("Test")
        types = [s["type"] for s in tracer.spans]
        assert types.index("llm_call") < types.index("run_complete")

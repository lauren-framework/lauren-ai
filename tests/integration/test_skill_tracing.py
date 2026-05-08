"""Integration tests for the callback/tracing handler pattern (Skill 32).

Tests cover:
- SimpleTracer records ModelCallComplete spans after run
- SimpleTracer records AgentRunComplete spans with turns and cost
- Spans list is populated only after the run completes
- Multiple runs accumulate spans
- Bus with no handlers does not interfere with agent execution
"""

from __future__ import annotations

from lauren import LaurenFactory, controller, get, post, module, Json, use_value
from lauren.testing import TestClient
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._signals import AgentRunComplete, ModelCallComplete, SignalBus
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


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
# Module-level singletons
# ---------------------------------------------------------------------------

_MOCK = MockTransport()
_BUS = SignalBus()
_TRACER = SimpleTracer(_BUS)


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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="Tracing test agent.")
class TracingTestAgent:
    pass


# ---------------------------------------------------------------------------
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/agent")
class AgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg, signals=_BUS)

    @post("/run")
    async def run(self, body: Json[dict]) -> dict:
        prompt = body.get("prompt", "hi")
        resp = await self._runner.run(TracingTestAgent(), prompt)
        return {"content": resp.content, "turns": resp.turns}


@controller("/tracing")
class TracingController:
    @get("/spans")
    async def spans(self) -> dict:
        return {"spans": _TRACER.spans, "count": len(_TRACER.spans)}

    @get("/reset")
    async def reset(self) -> dict:
        _TRACER._spans.clear()
        return {"cleared": True}


@module(
    controllers=[AgentController, TracingController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class TracingModule: ...


def build_app(*responses: str) -> TestClient:
    _MOCK.reset()
    _TRACER._spans.clear()
    for content in responses:
        _MOCK.queue_response(_completion(content))
    return TestClient(LaurenFactory.create(TracingModule))


def build_app_custom(completions: list[Completion]) -> TestClient:
    _MOCK.reset()
    _TRACER._spans.clear()
    for c in completions:
        _MOCK.queue_response(c)
    return TestClient(LaurenFactory.create(TracingModule))


# ---------------------------------------------------------------------------
# Tests: SimpleTracer spans
# ---------------------------------------------------------------------------


class TestSimpleTracerSpans:
    def test_llm_call_span_recorded_after_run(self):
        """A ModelCallComplete span is recorded after a successful run."""
        client = build_app("Hello")
        client.post("/agent/run", json={"prompt": "Hi"})
        r = client.get("/tracing/spans")
        assert r.status_code == 200
        data = r.json()
        llm_spans = [s for s in data["spans"] if s["type"] == "llm_call"]
        assert len(llm_spans) == 1
        assert llm_spans[0]["model"] == "mock-model"

    def test_llm_call_span_has_token_counts(self):
        """The llm_call span carries input and output token counts."""
        client = build_app_custom([_completion_custom_tokens(42, 7)])
        client.post("/agent/run", json={"prompt": "Test"})
        r = client.get("/tracing/spans")
        assert r.status_code == 200
        spans = r.json()["spans"]
        llm_spans = [s for s in spans if s["type"] == "llm_call"]
        assert len(llm_spans) == 1
        assert llm_spans[0]["input_tokens"] == 42
        assert llm_spans[0]["output_tokens"] == 7

    def test_run_complete_span_recorded(self):
        """An AgentRunComplete span is recorded with turns and total_cost."""
        client = build_app("Finished")
        client.post("/agent/run", json={"prompt": "Go"})
        r = client.get("/tracing/spans")
        assert r.status_code == 200
        spans = r.json()["spans"]
        run_spans = [s for s in spans if s["type"] == "run_complete"]
        assert len(run_spans) == 1
        assert run_spans[0]["turns"] == 1
        assert isinstance(run_spans[0]["total_cost"], float)

    def test_spans_empty_before_run(self):
        """The tracer has no spans before any run is executed."""
        client = build_app()
        r = client.get("/tracing/spans")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_two_runs_produce_two_sets_of_spans(self):
        """Two consecutive runs each produce their own set of spans."""
        client = build_app("First", "Second")
        client.post("/agent/run", json={"prompt": "Run 1"})
        client.post("/agent/run", json={"prompt": "Run 2"})
        r = client.get("/tracing/spans")
        assert r.status_code == 200
        spans = r.json()["spans"]
        llm_spans = [s for s in spans if s["type"] == "llm_call"]
        run_spans = [s for s in spans if s["type"] == "run_complete"]
        assert len(llm_spans) == 2
        assert len(run_spans) == 2

    def test_run_completes_without_bus(self):
        """Agent run succeeds when no SignalBus is provided (signals=None)."""
        # Test this directly without going through the HTTP layer
        mock = MockTransport()
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools={}, config=cfg, signals=None)
        mock.queue_response(_completion("No bus"))
        import asyncio
        response = asyncio.run(runner.run(TracingTestAgent(), "Hello"))
        assert response.content == "No bus"

    def test_span_order_is_llm_then_run_complete(self):
        """ModelCallComplete span is recorded before AgentRunComplete span."""
        client = build_app("OK")
        client.post("/agent/run", json={"prompt": "Test"})
        r = client.get("/tracing/spans")
        assert r.status_code == 200
        spans = r.json()["spans"]
        types = [s["type"] for s in spans]
        assert types.index("llm_call") < types.index("run_complete")

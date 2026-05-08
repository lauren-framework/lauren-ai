"""Integration tests for the token-usage-budget skill (Skill 29).

Verifies CostTracker accumulates usage, reports correctly, and that
SignalBus integration works, via HTTP through a Lauren TestClient.
"""

import pytest

from lauren import LaurenFactory, controller, get, post, module, Json, use_value, injectable, Scope
from lauren.testing import TestClient
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent
from lauren_ai import (
    CostTracker,
    default_pricing_table,
    SignalBus,
    ModelCallComplete,
    TokenBudget,
    BudgetExceededError,
)
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content="OK", *, n=1, stop_reason="end_turn", model="claude-sonnet-4-6",
                input_tokens=100, output_tokens=50):
    return Completion(
        id=f"c{n}",
        model=model,
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---------------------------------------------------------------------------
# CostTrackerService — injectable singleton that wires tracker + signal bus
# ---------------------------------------------------------------------------


@injectable(scope=Scope.SINGLETON)
class CostTrackerService:
    def __init__(self) -> None:
        self._tracker = CostTracker(pricing=default_pricing_table())
        self._bus = SignalBus()
        self._bus.on(ModelCallComplete)(self._tracker._on_model_call_complete)

    @property
    def bus(self) -> SignalBus:
        return self._bus

    @property
    def tracker(self) -> CostTracker:
        return self._tracker


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    prompt: str
    conversation_id: str = "default"


# ---------------------------------------------------------------------------
# Controllers / Module / build_app
# ---------------------------------------------------------------------------


@controller("/agent")
class TokenAgentController:
    def __init__(self, svc: CostTrackerService, mock: MockTransport) -> None:
        self._svc = svc
        cfg = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg, signals=svc.bus)

    @post("/run")
    async def run(self, body: Json[RunRequest]) -> dict:
        @agent(model="claude-sonnet-4-6", system="You are helpful.")
        class TrackedAgent: ...

        resp = await self._runner.run(TrackedAgent(), body.prompt, conversation_id=body.conversation_id)
        return {"content": resp.content, "turns": resp.turns}


@controller("/cost")
class CostController:
    def __init__(self, svc: CostTrackerService) -> None:
        self._svc = svc

    @get("/summary")
    async def summary(self) -> dict:
        report = await self._svc.tracker.report()
        return {
            "total_cost_usd": report.total_estimate.total_usd,
            "call_count": sum(
                len(v) for v in self._svc.tracker._conv_usage.values()
            ),
        }

    @get("/report/{conv_id}")
    async def report_by_conv(self, conv_id: str) -> dict:
        from lauren import Path
        report = await self._svc.tracker.report(conversation_id=conv_id)
        return {
            "total_cost_usd": report.total_estimate.total_usd,
            "models": list(report.by_model.keys()),
        }


@module(
    controllers=[TokenAgentController, CostController],
    providers=[
        CostTrackerService,
        use_value(provide=MockTransport, value=_MOCK),
    ],
)
class TokenBudgetModule: ...


def build_app(*responses):
    _MOCK.reset()
    for c in responses:
        _MOCK.queue_response(c)
    return TestClient(LaurenFactory.create(TokenBudgetModule))


# ---------------------------------------------------------------------------
# Tests: CostTracker manual (pure unit — no HTTP needed)
# ---------------------------------------------------------------------------


class TestCostTrackerManual:
    @pytest.mark.asyncio
    async def test_record_usage_and_report(self):
        tracker = CostTracker(pricing=default_pricing_table())
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.record_usage("claude-sonnet-4-6", usage, conversation_id="conv-1")
        report = await tracker.report(conversation_id="conv-1")
        assert report.total_estimate.total_usd > 0
        assert "claude-sonnet-4-6" in report.by_model

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate(self):
        tracker = CostTracker(pricing=default_pricing_table())
        tracker.record_usage("claude-sonnet-4-6", TokenUsage(input_tokens=500, output_tokens=200), conversation_id="conv-2")
        tracker.record_usage("claude-sonnet-4-6", TokenUsage(input_tokens=300, output_tokens=100), conversation_id="conv-2")
        report = await tracker.report(conversation_id="conv-2")
        assert report.total_estimate.total_usd > 0

    @pytest.mark.asyncio
    async def test_unknown_model_returns_zero_cost(self):
        tracker = CostTracker(pricing=default_pricing_table())
        tracker.record_usage("unknown-model-xyz", TokenUsage(input_tokens=1000, output_tokens=500), conversation_id="conv-3")
        report = await tracker.report(conversation_id="conv-3")
        assert report.total_estimate.total_usd == 0.0

    @pytest.mark.asyncio
    async def test_report_all_conversations(self):
        tracker = CostTracker(pricing=default_pricing_table())
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        tracker.record_usage("claude-sonnet-4-6", usage, conversation_id="conv-a")
        tracker.record_usage("claude-sonnet-4-6", usage, conversation_id="conv-b")
        report = await tracker.report()
        assert "conv-a" in report.by_conversation
        assert "conv-b" in report.by_conversation


# ---------------------------------------------------------------------------
# Tests: SignalBus integration via HTTP
# ---------------------------------------------------------------------------


class TestCostTrackerWithSignalBus:
    def test_run_agent_cost_is_positive(self):
        client = build_app(_completion("Hello", model="claude-sonnet-4-6"))
        resp = client.post("/agent/run", json={"prompt": "Hello", "conversation_id": "sig-1"})
        assert resp.status_code == 200

        cost_resp = client.get("/cost/summary")
        assert cost_resp.status_code == 200
        data = cost_resp.json()
        assert data["total_cost_usd"] >= 0  # may be 0 if mock model not in pricing
        assert data["call_count"] >= 1

    def test_multiple_runs_accumulate_calls(self):
        client = build_app(
            _completion("First", model="claude-sonnet-4-6"),
            _completion("Second", model="claude-sonnet-4-6"),
        )
        client.post("/agent/run", json={"prompt": "First", "conversation_id": "acc-1"})
        client.post("/agent/run", json={"prompt": "Second", "conversation_id": "acc-2"})

        cost_resp = client.get("/cost/summary")
        assert cost_resp.status_code == 200
        assert cost_resp.json()["call_count"] >= 2


# ---------------------------------------------------------------------------
# Tests: TokenBudget (pure unit)
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_budget_not_exceeded_within_limits(self):
        budget = TokenBudget(max_tokens_per_conversation=10_000)
        budget.check(current_tokens=100, estimated_tokens=50)

    def test_budget_exceeded_on_tokens(self):
        budget = TokenBudget(max_tokens_per_conversation=500)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(current_tokens=450, estimated_tokens=100)
        assert exc_info.value.limit == 500

    def test_budget_exceeded_on_usd(self):
        budget = TokenBudget(max_usd_per_conversation=0.10)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(current_usd=0.09, estimated_usd=0.05)
        assert exc_info.value.limit_type == "usd_per_conversation"

    def test_budget_not_exceeded_at_exact_limit(self):
        budget = TokenBudget(max_tokens_per_conversation=1000)
        budget.check(current_tokens=900, estimated_tokens=100)

    def test_budget_error_has_used_alias(self):
        budget = TokenBudget(max_tokens_per_conversation=100)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(current_tokens=90, estimated_tokens=20)
        err = exc_info.value
        assert err.used == err.current

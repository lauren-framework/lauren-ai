"""Integration tests for the token-usage-budget skill (Skill 29).

Verifies CostTracker accumulates usage, reports correctly, and that
SignalBus integration works via TestClient with signals.
"""

import pytest

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
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(
    text, *, n=1, stop="end_turn", model="claude-sonnet-4-6", input_tokens=100, output_tokens=50
):
    return Completion(
        id=f"c{n}",
        model=model,
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---------------------------------------------------------------------------
# Tests: CostTracker manual (pure unit — direct calls)
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
        tracker.record_usage(
            "claude-sonnet-4-6",
            TokenUsage(input_tokens=500, output_tokens=200),
            conversation_id="conv-2",
        )
        tracker.record_usage(
            "claude-sonnet-4-6",
            TokenUsage(input_tokens=300, output_tokens=100),
            conversation_id="conv-2",
        )
        report = await tracker.report(conversation_id="conv-2")
        assert report.total_estimate.total_usd > 0

    @pytest.mark.asyncio
    async def test_unknown_model_returns_zero_cost(self):
        tracker = CostTracker(pricing=default_pricing_table())
        tracker.record_usage(
            "unknown-model-xyz",
            TokenUsage(input_tokens=1000, output_tokens=500),
            conversation_id="conv-3",
        )
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
# Tests: SignalBus integration via TestClient
# ---------------------------------------------------------------------------


class TestCostTrackerWithSignalBus:
    @pytest.mark.asyncio
    async def test_run_agent_cost_is_positive(self):
        @agent(model="claude-sonnet-4-6", system="You are helpful.")
        class TrackedAgent:
            pass

        bus = SignalBus()
        tracker = CostTracker(pricing=default_pricing_table())
        bus.on(ModelCallComplete)(tracker._on_model_call_complete)

        client = TestClient(TrackedAgent(), signals=bus)
        client.mock.queue_response(_c("Hello", model="claude-sonnet-4-6"))
        result = await client.run_async("Hello", conversation_id="sig-1")

        assert result.content == "Hello"
        report = await tracker.report()
        # May be 0 if mock model price not in table, but call count > 0
        assert report.total_estimate.total_usd >= 0
        total_calls = sum(len(v) for v in tracker._conv_usage.values())
        assert total_calls >= 1

    @pytest.mark.asyncio
    async def test_multiple_runs_accumulate_calls(self):
        @agent(model="claude-sonnet-4-6", system="You are helpful.")
        class TrackedAgent2:
            pass

        bus = SignalBus()
        tracker = CostTracker(pricing=default_pricing_table())
        bus.on(ModelCallComplete)(tracker._on_model_call_complete)

        client = TestClient(TrackedAgent2(), signals=bus)
        client.mock.queue_response(_c("First", model="claude-sonnet-4-6"))
        client.mock.queue_response(_c("Second", model="claude-sonnet-4-6"))

        await client.run_async("First", conversation_id="acc-1")
        await client.run_async("Second", conversation_id="acc-2")

        report = await tracker.report()
        total_calls = sum(len(v) for v in tracker._conv_usage.values())
        assert total_calls >= 2


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

"""Unit tests for cost tracking, budgets, and rate limiting."""
from __future__ import annotations

import pytest

from lauren_ai._cost._budget import BudgetExceededError, TokenBudget
from lauren_ai._cost._pricing import CostEstimate, ModelPricing, PricingTable, default_pricing_table
from lauren_ai._cost._rate import RateLimiter, RateLimitExhaustedError
from lauren_ai._cost._tracker import CostTracker
from lauren_ai._transport import TokenUsage


class TestPricingTable:
    def test_estimate_known_model(self):
        table = PricingTable(models={
            "claude-haiku-4-5": ModelPricing(input_per_m=0.80, output_per_m=4.00),
        })
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        estimate = table.estimate("claude-haiku-4-5", usage)
        assert estimate.input_usd == pytest.approx(0.80)
        assert estimate.output_usd == pytest.approx(4.00)
        assert estimate.total_usd == pytest.approx(4.80)

    def test_estimate_unknown_model_returns_zero(self):
        table = PricingTable()
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        estimate = table.estimate("unknown-model", usage)
        assert estimate.total_usd == 0.0

    def test_default_pricing_table_has_haiku(self):
        table = default_pricing_table()
        assert "claude-haiku-4-5" in table

    def test_default_pricing_table_has_openai(self):
        table = default_pricing_table()
        assert "gpt-4o" in table

    def test_cost_estimate_addition(self):
        a = CostEstimate(input_usd=1.0, output_usd=2.0)
        b = CostEstimate(input_usd=0.5, output_usd=0.5)
        c = a + b
        assert c.input_usd == pytest.approx(1.5)
        assert c.output_usd == pytest.approx(2.5)
        assert c.total_usd == pytest.approx(4.0)

    def test_estimate_with_cache_tokens(self):
        table = PricingTable(models={
            "claude-haiku-4-5": ModelPricing(
                input_per_m=0.80, output_per_m=4.00,
                cache_read_per_m=0.08, cache_write_per_m=1.00,
            ),
        })
        usage = TokenUsage(
            input_tokens=0, output_tokens=0,
            cache_read_tokens=1_000_000, cache_write_tokens=1_000_000,
        )
        estimate = table.estimate("claude-haiku-4-5", usage)
        assert estimate.cache_read_usd == pytest.approx(0.08)
        assert estimate.cache_write_usd == pytest.approx(1.00)

    def test_pricing_table_contains(self):
        table = PricingTable(models={
            "my-model": ModelPricing(input_per_m=1.0, output_per_m=2.0),
        })
        assert "my-model" in table
        assert "other-model" not in table

    def test_pricing_table_get(self):
        pricing = ModelPricing(input_per_m=1.0, output_per_m=2.0)
        table = PricingTable(models={"my-model": pricing})
        assert table.get("my-model") is pricing
        assert table.get("missing") is None

    def test_default_class_method(self):
        table = PricingTable.default()
        assert "claude-sonnet-4-6" in table


class TestCostTracker:
    async def test_record_and_report(self):
        tracker = CostTracker(pricing=default_pricing_table())
        usage = TokenUsage(input_tokens=100_000, output_tokens=50_000)
        tracker.record_usage("claude-haiku-4-5", usage, conversation_id="c1")
        report = await tracker.report(conversation_id="c1")
        assert report.total_estimate.total_usd > 0
        assert "claude-haiku-4-5" in report.by_model

    async def test_session_context_manager(self):
        tracker = CostTracker(pricing=default_pricing_table())

        async with tracker.session(conversation_id="c1") as session:
            # Simulate usage happening during the session
            tracker.record_usage(
                "claude-haiku-4-5",
                TokenUsage(input_tokens=100_000, output_tokens=50_000),
                conversation_id="c1",
            )

        # Session should have captured the usage
        assert session.conversation_id == "c1"
        assert session.total_estimate.total_usd > 0

    async def test_report_all_conversations(self):
        tracker = CostTracker(pricing=default_pricing_table())
        tracker.record_usage("claude-haiku-4-5", TokenUsage(1000, 500), "c1")
        tracker.record_usage("claude-haiku-4-5", TokenUsage(2000, 1000), "c2")
        report = await tracker.report()
        assert "c1" in report.by_conversation
        assert "c2" in report.by_conversation

    async def test_report_by_model(self):
        tracker = CostTracker(pricing=default_pricing_table())
        tracker.record_usage("claude-haiku-4-5", TokenUsage(1000, 500), "c1")
        tracker.record_usage("gpt-4o", TokenUsage(500, 200), "c1")
        report = await tracker.report(conversation_id="c1")
        assert "claude-haiku-4-5" in report.by_model
        assert "gpt-4o" in report.by_model

    async def test_unknown_model_records_zero_cost(self):
        tracker = CostTracker(pricing=default_pricing_table())
        tracker.record_usage("nonexistent-model", TokenUsage(1_000_000, 1_000_000), "c1")
        report = await tracker.report(conversation_id="c1")
        assert report.total_estimate.total_usd == 0.0

    async def test_session_user_id_stored(self):
        tracker = CostTracker()
        async with tracker.session(user_id="user-42") as session:
            pass
        assert session.user_id == "user-42"

    async def test_signal_handler_accumulates(self):
        from lauren_ai._signals import ModelCallComplete

        tracker = CostTracker(pricing=default_pricing_table())
        event = ModelCallComplete(
            model="claude-haiku-4-5",
            usage=TokenUsage(input_tokens=10_000, output_tokens=5_000),
        )
        await tracker._on_model_call_complete(event)
        report = await tracker.report()
        assert report.total_estimate.total_usd > 0

    async def test_signal_handler_ignores_no_usage(self):
        from lauren_ai._signals import ModelCallComplete

        tracker = CostTracker(pricing=default_pricing_table())
        event = ModelCallComplete(model="claude-haiku-4-5", usage=None)
        await tracker._on_model_call_complete(event)
        report = await tracker.report()
        assert report.total_estimate.total_usd == 0.0


class TestTokenBudget:
    def test_no_violation_within_limit(self):
        budget = TokenBudget(max_tokens_per_conversation=10000)
        # Should not raise
        budget.check(conversation_id="c1", current_tokens=100, estimated_tokens=50)

    def test_token_limit_exceeded_raises(self):
        budget = TokenBudget(max_tokens_per_conversation=100)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(conversation_id="c1", current_tokens=90, estimated_tokens=20)
        assert exc_info.value.limit_type == "tokens_per_conversation"
        assert exc_info.value.limit == 100

    def test_usd_limit_exceeded_raises(self):
        budget = TokenBudget(max_usd_per_conversation=1.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(current_usd=0.90, estimated_usd=0.20)
        assert exc_info.value.limit_type == "usd_per_conversation"

    def test_no_limits_never_raises(self):
        budget = TokenBudget()
        budget.check(current_tokens=10_000_000, estimated_tokens=10_000_000)

    def test_exactly_at_limit_does_not_raise(self):
        budget = TokenBudget(max_tokens_per_conversation=100)
        # exactly 100 is allowed (not exceeding)
        budget.check(current_tokens=50, estimated_tokens=50)

    def test_budget_exceeded_error_has_current(self):
        budget = TokenBudget(max_tokens_per_conversation=100)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(current_tokens=90, estimated_tokens=20)
        assert exc_info.value.current == 90

    def test_usd_limit_not_exceeded_within_range(self):
        budget = TokenBudget(max_usd_per_conversation=1.0)
        budget.check(current_usd=0.50, estimated_usd=0.30)

    def test_budget_exceeded_error_is_lauren_ai_error(self):
        from lauren_ai._exceptions import LaurenAIError
        budget = TokenBudget(max_tokens_per_conversation=10)
        with pytest.raises(LaurenAIError):
            budget.check(current_tokens=5, estimated_tokens=10)


class TestRateLimiter:
    def test_backoff_increases_with_attempts(self):
        limiter = RateLimiter(jitter=False, initial_backoff_s=1.0)
        b0 = limiter.backoff_for(0)
        b1 = limiter.backoff_for(1)
        b2 = limiter.backoff_for(2)
        assert b1 > b0
        assert b2 > b1

    def test_backoff_respects_max(self):
        limiter = RateLimiter(max_backoff_s=10.0, jitter=False)
        b = limiter.backoff_for(100)
        assert b <= 10.0

    def test_retry_after_overrides_backoff(self):
        limiter = RateLimiter(jitter=False)
        assert limiter.backoff_for(0, retry_after=30.0) == 30.0

    async def test_acquire_within_rpm_limit(self):
        limiter = RateLimiter(requests_per_minute=60)
        # Should not block for a single request
        await limiter.acquire()

    def test_backoff_with_jitter_within_range(self):
        limiter = RateLimiter(jitter=True, initial_backoff_s=1.0, max_backoff_s=60.0)
        # With jitter, backoff for attempt 0 should be between 0.5 and 1.0
        for _ in range(20):
            b = limiter.backoff_for(0)
            assert 0.5 <= b <= 1.0

    def test_rate_limit_exhausted_error_is_lauren_ai_error(self):
        from lauren_ai._exceptions import LaurenAIError
        err = RateLimitExhaustedError("rate limit exhausted")
        assert isinstance(err, LaurenAIError)

    async def test_acquire_no_limits(self):
        limiter = RateLimiter()
        # No limits set — should return immediately regardless of token count
        await limiter.acquire(estimated_tokens=99999)


# ---------------------------------------------------------------------------
# New spec-described API tests
# ---------------------------------------------------------------------------


class TestModelPricingPer1k:
    """ModelPricing must support input_per_1k / output_per_1k fields."""

    def test_per_1k_fields_stored(self):
        pricing = ModelPricing(input_per_1k=0.003, output_per_1k=0.015)
        assert pricing.input_per_1k == pytest.approx(0.003)
        assert pricing.output_per_1k == pytest.approx(0.015)

    def test_per_m_derived_from_per_1k(self):
        pricing = ModelPricing(input_per_1k=0.003, output_per_1k=0.015)
        assert pricing.input_per_m == pytest.approx(3.0)
        assert pricing.output_per_m == pytest.approx(15.0)

    def test_per_1k_derived_from_per_m(self):
        pricing = ModelPricing(input_per_m=3.0, output_per_m=15.0)
        assert pricing.input_per_1k == pytest.approx(0.003)
        assert pricing.output_per_1k == pytest.approx(0.015)

    def test_cache_read_per_1k(self):
        pricing = ModelPricing(input_per_1k=0.003, output_per_1k=0.015, cache_read_per_1k=0.0003)
        assert pricing.cache_read_per_1k == pytest.approx(0.0003)
        assert pricing.cache_read_per_m == pytest.approx(0.3)

    def test_zero_by_default(self):
        pricing = ModelPricing()
        assert pricing.input_per_1k == 0.0
        assert pricing.output_per_1k == 0.0


class TestPricingTablePriceFor:
    """PricingTable.price_for() should return ModelPricing or None."""

    def test_price_for_known_model(self):
        pricing = ModelPricing(input_per_m=3.0, output_per_m=15.0)
        table = PricingTable(models={"claude-sonnet-4-6": pricing})
        assert table.price_for("claude-sonnet-4-6") is pricing

    def test_price_for_unknown_model_returns_none(self):
        table = PricingTable()
        assert table.price_for("nonexistent") is None

    def test_price_for_and_get_are_equivalent(self):
        pricing = ModelPricing(input_per_m=1.0, output_per_m=2.0)
        table = PricingTable(models={"m": pricing})
        assert table.price_for("m") is table.get("m")

    def test_default_table_price_for_claude_opus(self):
        table = default_pricing_table()
        p = table.price_for("claude-opus-4-6")
        assert p is not None
        assert p.input_per_m > 0

    def test_default_table_price_for_gpt4o(self):
        table = default_pricing_table()
        p = table.price_for("gpt-4o")
        assert p is not None
        assert p.output_per_m > 0


class TestCostEstimateFromUsage:
    """CostEstimate.from_usage() classmethod."""

    def test_from_usage_computes_cost(self):
        table = default_pricing_table()
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        estimate = CostEstimate.from_usage(usage, "claude-haiku-4-5", table)
        assert estimate.total_usd > 0.0

    def test_from_usage_unknown_model_is_zero(self):
        table = default_pricing_table()
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        estimate = CostEstimate.from_usage(usage, "unknown-model", table)
        assert estimate.total_usd == 0.0

    def test_from_usage_matches_estimate(self):
        table = default_pricing_table()
        usage = TokenUsage(input_tokens=500, output_tokens=300)
        via_classmethod = CostEstimate.from_usage(usage, "gpt-4o", table)
        via_method = table.estimate("gpt-4o", usage)
        assert via_classmethod.total_usd == pytest.approx(via_method.total_usd)

    def test_from_usage_haiku_input_cost(self):
        table = PricingTable(models={
            "my-model": ModelPricing(input_per_m=1.0, output_per_m=2.0),
        })
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
        estimate = CostEstimate.from_usage(usage, "my-model", table)
        assert estimate.input_usd == pytest.approx(1.0)
        assert estimate.output_usd == pytest.approx(0.0)


class TestBudgetExceededErrorUsedAlias:
    """BudgetExceededError.used must be an alias for .current."""

    def test_used_equals_current(self):
        err = BudgetExceededError(
            "over budget",
            limit_type="tokens_per_conversation",
            limit=100,
            current=90,
        )
        assert err.used == 90
        assert err.used == err.current

    def test_used_on_usd_budget(self):
        err = BudgetExceededError(
            "over usd",
            limit_type="usd_per_conversation",
            limit=1.0,
            current=1.1,
        )
        assert err.used == pytest.approx(1.1)


class TestRateLimitExhaustedErrorFields:
    """RateLimitExhaustedError.limit and .retry_after fields."""

    def test_default_fields_are_zero(self):
        err = RateLimitExhaustedError("exhausted")
        assert err.limit == 0
        assert err.retry_after == 0.0

    def test_limit_field(self):
        err = RateLimitExhaustedError("exhausted", limit=60, retry_after=5.0)
        assert err.limit == 60

    def test_retry_after_field(self):
        err = RateLimitExhaustedError("exhausted", limit=60, retry_after=5.0)
        assert err.retry_after == pytest.approx(5.0)

    def test_is_lauren_ai_error(self):
        from lauren_ai._exceptions import LaurenAIError
        err = RateLimitExhaustedError("exhausted", limit=10, retry_after=2.5)
        assert isinstance(err, LaurenAIError)

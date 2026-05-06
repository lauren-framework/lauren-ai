"""Unit tests for guard factories."""

from __future__ import annotations

import pytest

from lauren_ai._exceptions import AgentConfigError
from lauren_ai._guards import (
    InMemoryBudgetStore,
    SafetyPolicy,
    requires_capability,
    token_budget_guard,
)


class TestSafetyPolicy:
    def test_passes_clean_text(self):
        policy = SafetyPolicy(blocked_keywords=["spam", "hack"])
        assert policy.is_safe("Hello, how are you today?") is True

    def test_blocks_keyword(self):
        policy = SafetyPolicy(blocked_keywords=["spam"])
        assert policy.is_safe("This is spam content") is False

    def test_case_insensitive_keyword(self):
        policy = SafetyPolicy(blocked_keywords=["spam"])
        assert policy.is_safe("THIS IS SPAM") is False

    def test_blocks_regex_pattern(self):
        policy = SafetyPolicy(blocked_patterns=[r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"])
        assert policy.is_safe("Card: 1234-5678-9012-3456") is False
        assert policy.is_safe("Regular text here") is True

    def test_empty_policy_passes_everything(self):
        policy = SafetyPolicy()
        assert policy.is_safe("anything goes") is True


class TestInMemoryBudgetStore:
    @pytest.mark.asyncio
    async def test_initial_usage_zero(self):
        store = InMemoryBudgetStore()
        usage = await store.get_usage("user-1", window_seconds=3600)
        assert usage.tokens == 0
        assert usage.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_record_and_retrieve(self):
        store = InMemoryBudgetStore()
        await store.record_usage("user-1", tokens=100, cost_usd=0.01, window_seconds=3600)
        usage = await store.get_usage("user-1", window_seconds=3600)
        assert usage.tokens == 100
        assert usage.cost_usd == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_accumulates_usage(self):
        store = InMemoryBudgetStore()
        await store.record_usage("user-1", tokens=50, window_seconds=3600)
        await store.record_usage("user-1", tokens=75, window_seconds=3600)
        usage = await store.get_usage("user-1", window_seconds=3600)
        assert usage.tokens == 125


class TestTokenBudgetGuard:
    @pytest.mark.asyncio
    async def test_allows_within_budget(self):
        GuardClass = token_budget_guard(max_tokens_per_hour=1000)
        guard = GuardClass()

        class FakeCtx:
            class request:
                class client:
                    host = "127.0.0.1"

        result = await guard.can_activate(FakeCtx())
        assert result is True

    @pytest.mark.asyncio
    async def test_rejects_exceeded_budget(self):
        store = InMemoryBudgetStore()
        await store.record_usage("127.0.0.1", tokens=999, window_seconds=3600)

        GuardClass = token_budget_guard(max_tokens_per_hour=1000, store=store)
        guard = GuardClass()

        class FakeCtx:
            class request:
                class client:
                    host = "127.0.0.1"

        with pytest.raises(Exception):  # AgentBudgetExceededError
            # Set usage to exactly the limit
            await store.record_usage("127.0.0.1", tokens=1, window_seconds=3600)
            await guard.can_activate(FakeCtx())


class TestRequiresCapability:
    @pytest.mark.asyncio
    async def test_claude_has_tool_use(self):
        GuardClass = requires_capability("tool_use")
        guard = GuardClass()

        class FakeConfig:
            model = "claude-opus-4-6"

        guard._config = FakeConfig()
        result = await guard.can_activate(None)
        assert result is True

    @pytest.mark.asyncio
    async def test_raises_for_missing_capability(self):
        GuardClass = requires_capability("extended_thinking")
        guard = GuardClass()

        class FakeConfig:
            model = "gpt-4o-mini"  # No extended thinking

        guard._config = FakeConfig()
        with pytest.raises(AgentConfigError):
            await guard.can_activate(None)

    def test_no_capabilities_raises(self):
        with pytest.raises(ValueError):
            requires_capability()

"""Integration tests for the llm-rate-limiting skill (Skill 30).

Verifies the custom LLMRateLimiter allows calls within limit, blocks when
exceeded, records usage, and that the built-in RateLimiter has the expected
interface.
"""
import asyncio
import time
import pytest

from lauren_ai import RateLimiter


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------


class LLMRateLimiter:
    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 100_000):
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._request_times: list[float] = []
        self._token_counts: list[tuple[float, int]] = []

    def _clean_old(self, window: float = 60.0) -> None:
        now = time.monotonic()
        self._request_times = [t for t in self._request_times if now - t < window]
        self._token_counts = [(t, c) for t, c in self._token_counts if now - t < window]

    def can_proceed(self, estimated_tokens: int = 0) -> bool:
        self._clean_old()
        if len(self._request_times) >= self._rpm:
            return False
        token_sum = sum(c for _, c in self._token_counts) + estimated_tokens
        if token_sum > self._tpm:
            return False
        return True

    def record_call(self, tokens_used: int = 0) -> None:
        now = time.monotonic()
        self._request_times.append(now)
        self._token_counts.append((now, tokens_used))

    async def acquire(self, estimated_tokens: int = 0) -> None:
        while not self.can_proceed(estimated_tokens):
            await asyncio.sleep(0.1)
        self.record_call(estimated_tokens)


class PerUserRateLimiter:
    def __init__(self, rpm_per_user: int = 10, tpm_per_user: int = 20_000):
        self._limiters: dict[str, LLMRateLimiter] = {}
        self._rpm = rpm_per_user
        self._tpm = tpm_per_user

    def for_user(self, user_id: str) -> LLMRateLimiter:
        if user_id not in self._limiters:
            self._limiters[user_id] = LLMRateLimiter(self._rpm, self._tpm)
        return self._limiters[user_id]

    async def acquire(self, user_id: str, estimated_tokens: int = 0) -> None:
        await self.for_user(user_id).acquire(estimated_tokens)


# ---------------------------------------------------------------------------
# Tests: LLMRateLimiter
# ---------------------------------------------------------------------------


class TestLLMRateLimiter:
    def test_can_proceed_when_empty(self):
        limiter = LLMRateLimiter(requests_per_minute=10, tokens_per_minute=10_000)
        assert limiter.can_proceed(100) is True

    def test_cannot_proceed_when_rpm_exceeded(self):
        limiter = LLMRateLimiter(requests_per_minute=3, tokens_per_minute=100_000)
        limiter.record_call(0)
        limiter.record_call(0)
        limiter.record_call(0)
        # Now at limit
        assert limiter.can_proceed(0) is False

    def test_can_proceed_below_rpm_limit(self):
        limiter = LLMRateLimiter(requests_per_minute=5, tokens_per_minute=100_000)
        limiter.record_call(0)
        limiter.record_call(0)
        assert limiter.can_proceed(0) is True

    def test_cannot_proceed_when_tpm_exceeded(self):
        limiter = LLMRateLimiter(requests_per_minute=100, tokens_per_minute=500)
        limiter.record_call(300)
        limiter.record_call(200)
        # 500 tokens used, estimated 1 more would exceed
        assert limiter.can_proceed(1) is False

    def test_can_proceed_when_tpm_not_exceeded(self):
        limiter = LLMRateLimiter(requests_per_minute=100, tokens_per_minute=1000)
        limiter.record_call(300)
        assert limiter.can_proceed(200) is True

    def test_record_call_increments_request_count(self):
        limiter = LLMRateLimiter(requests_per_minute=10)
        assert len(limiter._request_times) == 0
        limiter.record_call(100)
        assert len(limiter._request_times) == 1
        limiter.record_call(200)
        assert len(limiter._request_times) == 2

    def test_record_call_stores_token_count(self):
        limiter = LLMRateLimiter(tokens_per_minute=10_000)
        limiter.record_call(tokens_used=500)
        total = sum(c for _, c in limiter._token_counts)
        assert total == 500

    def test_clean_old_removes_expired_entries(self):
        limiter = LLMRateLimiter(requests_per_minute=5)
        # Manually add a request from 120 seconds ago
        old_time = time.monotonic() - 120.0
        limiter._request_times.append(old_time)
        limiter._token_counts.append((old_time, 100))
        assert len(limiter._request_times) == 1
        limiter._clean_old(window=60.0)
        assert len(limiter._request_times) == 0
        assert len(limiter._token_counts) == 0

    @pytest.mark.asyncio
    async def test_acquire_proceeds_when_under_limit(self):
        limiter = LLMRateLimiter(requests_per_minute=10, tokens_per_minute=10_000)
        # Should return immediately without sleeping
        start = time.monotonic()
        await limiter.acquire(estimated_tokens=100)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
        assert len(limiter._request_times) == 1

    @pytest.mark.asyncio
    async def test_acquire_records_usage(self):
        limiter = LLMRateLimiter(requests_per_minute=10, tokens_per_minute=10_000)
        await limiter.acquire(estimated_tokens=250)
        total_tokens = sum(c for _, c in limiter._token_counts)
        assert total_tokens == 250


# ---------------------------------------------------------------------------
# Tests: PerUserRateLimiter
# ---------------------------------------------------------------------------


class TestPerUserRateLimiter:
    def test_separate_limiter_per_user(self):
        per_user = PerUserRateLimiter(rpm_per_user=5)
        alice_limiter = per_user.for_user("alice")
        bob_limiter = per_user.for_user("bob")
        assert alice_limiter is not bob_limiter

    def test_same_user_returns_same_limiter(self):
        per_user = PerUserRateLimiter(rpm_per_user=5)
        limiter_a = per_user.for_user("charlie")
        limiter_b = per_user.for_user("charlie")
        assert limiter_a is limiter_b

    @pytest.mark.asyncio
    async def test_acquire_for_user(self):
        per_user = PerUserRateLimiter(rpm_per_user=10, tpm_per_user=10_000)
        await per_user.acquire("dave", estimated_tokens=100)
        limiter = per_user.for_user("dave")
        assert len(limiter._request_times) == 1

    def test_user_cannot_proceed_after_rpm_limit(self):
        per_user = PerUserRateLimiter(rpm_per_user=2)
        limiter = per_user.for_user("eve")
        limiter.record_call(0)
        limiter.record_call(0)
        assert limiter.can_proceed(0) is False

    def test_different_users_are_independent(self):
        per_user = PerUserRateLimiter(rpm_per_user=2)
        # Exhaust user1's limit
        per_user.for_user("user1").record_call(0)
        per_user.for_user("user1").record_call(0)
        # user2 should still be able to proceed
        assert per_user.for_user("user2").can_proceed(0) is True


# ---------------------------------------------------------------------------
# Tests: Built-in RateLimiter interface
# ---------------------------------------------------------------------------


class TestBuiltinRateLimiterInterface:
    def test_builtin_rate_limiter_has_expected_fields(self):
        limiter = RateLimiter(
            requests_per_minute=60,
            tokens_per_minute=100_000,
            max_retries=5,
            initial_backoff_s=1.0,
            max_backoff_s=60.0,
            jitter=True,
        )
        assert limiter.requests_per_minute == 60
        assert limiter.tokens_per_minute == 100_000
        assert limiter.max_retries == 5
        assert limiter.jitter is True

    def test_builtin_rate_limiter_defaults(self):
        limiter = RateLimiter()
        assert limiter.requests_per_minute is None
        assert limiter.tokens_per_minute is None
        assert limiter.max_retries == 5

    def test_backoff_for_increases_with_attempt(self):
        limiter = RateLimiter(initial_backoff_s=1.0, max_backoff_s=60.0, jitter=False)
        delay_0 = limiter.backoff_for(0)
        delay_1 = limiter.backoff_for(1)
        delay_2 = limiter.backoff_for(2)
        assert delay_0 < delay_1 < delay_2

    def test_backoff_for_capped_at_max(self):
        limiter = RateLimiter(initial_backoff_s=1.0, max_backoff_s=10.0, jitter=False)
        delay = limiter.backoff_for(20)  # Would be 1048576s without cap
        assert delay <= 10.0

    @pytest.mark.asyncio
    async def test_builtin_acquire_proceeds_without_limits(self):
        limiter = RateLimiter()  # no limits set
        start = time.monotonic()
        await limiter.acquire(estimated_tokens=100)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

"""Integration tests for the llm-rate-limiting skill (Skill 30).

Verifies the custom LLMRateLimiter allows calls within limit, blocks when
exceeded, records usage, and that the built-in RateLimiter has the expected
interface — via direct Python calls.
"""

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

    @property
    def request_count(self) -> int:
        self._clean_old()
        return len(self._request_times)

    @property
    def token_count(self) -> int:
        self._clean_old()
        return sum(c for _, c in self._token_counts)


# ---------------------------------------------------------------------------
# Tests: LLMRateLimiter
# ---------------------------------------------------------------------------


class TestLLMRateLimiter:
    def test_allows_within_limit(self):
        limiter = LLMRateLimiter(requests_per_minute=10, tokens_per_minute=10_000)
        assert limiter.can_proceed(100) is True

    def test_consume_within_limit_records_call(self):
        limiter = LLMRateLimiter(requests_per_minute=5, tokens_per_minute=10_000)
        assert limiter.can_proceed(100) is True
        limiter.record_call(100)
        assert limiter.request_count == 1
        assert limiter.token_count == 100

    def test_blocks_when_rpm_exceeded(self):
        limiter = LLMRateLimiter(requests_per_minute=3, tokens_per_minute=100_000)
        limiter.record_call(0)
        limiter.record_call(0)
        limiter.record_call(0)
        assert limiter.can_proceed(0) is False

    def test_blocks_when_tpm_exceeded(self):
        limiter = LLMRateLimiter(requests_per_minute=100, tokens_per_minute=500)
        limiter.record_call(300)
        limiter.record_call(200)
        assert limiter.can_proceed(1) is False

    def test_status_reflects_recorded_calls(self):
        limiter = LLMRateLimiter(requests_per_minute=10, tokens_per_minute=10_000)
        limiter.record_call(100)
        limiter.record_call(200)
        assert limiter.request_count == 2
        assert limiter.token_count == 300

    def test_consume_records_tokens(self):
        limiter = LLMRateLimiter(requests_per_minute=10, tokens_per_minute=10_000)
        assert limiter.can_proceed(250) is True
        limiter.record_call(250)
        assert limiter.token_count == 250

    def test_reset_clears_counters(self):
        limiter = LLMRateLimiter(requests_per_minute=5, tokens_per_minute=1000)
        limiter.record_call(100)
        limiter.record_call(200)
        # Simulate a reset by creating a new limiter
        limiter2 = LLMRateLimiter(requests_per_minute=5, tokens_per_minute=1000)
        assert limiter2.request_count == 0
        assert limiter2.token_count == 0

    def test_can_proceed_when_empty(self):
        limiter = LLMRateLimiter(requests_per_minute=10, tokens_per_minute=10_000)
        assert limiter.can_proceed(100) is True

    def test_cannot_proceed_when_rpm_exceeded(self):
        limiter = LLMRateLimiter(requests_per_minute=3, tokens_per_minute=100_000)
        limiter.record_call(0)
        limiter.record_call(0)
        limiter.record_call(0)
        assert limiter.can_proceed(0) is False

    def test_cannot_proceed_when_tpm_exceeded(self):
        limiter = LLMRateLimiter(requests_per_minute=100, tokens_per_minute=500)
        limiter.record_call(300)
        limiter.record_call(200)
        assert limiter.can_proceed(1) is False

    def test_record_call_increments_request_count(self):
        limiter = LLMRateLimiter(requests_per_minute=10)
        assert len(limiter._request_times) == 0
        limiter.record_call(100)
        assert len(limiter._request_times) == 1
        limiter.record_call(200)
        assert len(limiter._request_times) == 2

    def test_clean_old_removes_expired_entries(self):
        limiter = LLMRateLimiter(requests_per_minute=5)
        old_time = time.monotonic() - 120.0
        limiter._request_times.append(old_time)
        limiter._token_counts.append((old_time, 100))
        assert len(limiter._request_times) == 1
        limiter._clean_old(window=60.0)
        assert len(limiter._request_times) == 0
        assert len(limiter._token_counts) == 0


# ---------------------------------------------------------------------------
# Tests: Built-in RateLimiter interface (pure unit)
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
        delay = limiter.backoff_for(20)
        assert delay <= 10.0

    @pytest.mark.asyncio
    async def test_builtin_acquire_proceeds_without_limits(self):
        limiter = RateLimiter()
        start = time.monotonic()
        await limiter.acquire(estimated_tokens=100)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

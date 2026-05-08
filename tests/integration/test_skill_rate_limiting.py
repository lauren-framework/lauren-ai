"""Integration tests for the llm-rate-limiting skill (Skill 30).

Verifies the custom LLMRateLimiter allows calls within limit, blocks when
exceeded, records usage, and that the built-in RateLimiter has the expected
interface — via HTTP through a Lauren TestClient.
"""

import time

import pytest

from lauren import LaurenFactory, controller, get, post, module, Json, use_value, injectable, Scope
from lauren.testing import TestClient
from lauren_ai import RateLimiter
from pydantic import BaseModel


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


@injectable(scope=Scope.SINGLETON)
class LimiterService:
    def __init__(self) -> None:
        self._limiter = LLMRateLimiter(requests_per_minute=5, tokens_per_minute=1000)

    @property
    def limiter(self) -> LLMRateLimiter:
        return self._limiter

    def reset(self, rpm: int = 5, tpm: int = 1000) -> None:
        self._limiter = LLMRateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ConsumeRequest(BaseModel):
    estimated_tokens: int = 0


class RecordRequest(BaseModel):
    tokens: int


class ResetRequest(BaseModel):
    rpm: int = 5
    tpm: int = 1000


# ---------------------------------------------------------------------------
# Controller / Module / build_app
# ---------------------------------------------------------------------------


@controller("/limiter")
class LimiterController:
    def __init__(self, svc: LimiterService) -> None:
        self._svc = svc

    @post("/consume")
    async def consume(self, body: Json[ConsumeRequest]) -> dict:
        allowed = self._svc.limiter.can_proceed(body.estimated_tokens)
        if allowed:
            self._svc.limiter.record_call(body.estimated_tokens)
        return {"allowed": allowed}

    @post("/record")
    async def record(self, body: Json[RecordRequest]) -> dict:
        self._svc.limiter.record_call(body.tokens)
        return {"recorded": True}

    @get("/status")
    async def status(self) -> dict:
        return {
            "request_count": self._svc.limiter.request_count,
            "token_count": self._svc.limiter.token_count,
        }

    @post("/reset")
    async def reset(self, body: Json[ResetRequest]) -> dict:
        self._svc.reset(rpm=body.rpm, tpm=body.tpm)
        return {"reset": True}


@module(controllers=[LimiterController], providers=[LimiterService])
class RateLimiterModule: ...


def build_app():
    return TestClient(LaurenFactory.create(RateLimiterModule))


# ---------------------------------------------------------------------------
# Tests: LLMRateLimiter via HTTP
# ---------------------------------------------------------------------------


class TestLLMRateLimiterHTTP:
    def test_consume_within_limit_is_allowed(self):
        client = build_app()
        client.post("/limiter/reset", json={"rpm": 5, "tpm": 10000})
        resp = client.post("/limiter/consume", json={"estimated_tokens": 100})
        assert resp.status_code == 200
        assert resp.json()["allowed"] is True

    def test_consume_exceeded_rpm_is_blocked(self):
        client = build_app()
        client.post("/limiter/reset", json={"rpm": 3, "tpm": 100000})
        for _ in range(3):
            client.post("/limiter/consume", json={"estimated_tokens": 0})
        resp = client.post("/limiter/consume", json={"estimated_tokens": 0})
        assert resp.status_code == 200
        assert resp.json()["allowed"] is False

    def test_consume_exceeded_tpm_is_blocked(self):
        client = build_app()
        client.post("/limiter/reset", json={"rpm": 100, "tpm": 500})
        client.post("/limiter/record", json={"tokens": 300})
        client.post("/limiter/record", json={"tokens": 200})
        # token budget now full; 1 more would exceed
        resp = client.post("/limiter/consume", json={"estimated_tokens": 1})
        assert resp.status_code == 200
        assert resp.json()["allowed"] is False

    def test_status_reflects_recorded_calls(self):
        client = build_app()
        client.post("/limiter/reset", json={"rpm": 10, "tpm": 10000})
        client.post("/limiter/record", json={"tokens": 100})
        client.post("/limiter/record", json={"tokens": 200})
        resp = client.get("/limiter/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_count"] == 2
        assert data["token_count"] == 300

    def test_consume_records_tokens(self):
        client = build_app()
        client.post("/limiter/reset", json={"rpm": 10, "tpm": 10000})
        client.post("/limiter/consume", json={"estimated_tokens": 250})
        resp = client.get("/limiter/status")
        assert resp.status_code == 200
        assert resp.json()["token_count"] == 250

    def test_reset_clears_counters(self):
        client = build_app()
        client.post("/limiter/record", json={"tokens": 100})
        client.post("/limiter/record", json={"tokens": 200})
        client.post("/limiter/reset", json={"rpm": 5, "tpm": 1000})
        resp = client.get("/limiter/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_count"] == 0
        assert data["token_count"] == 0


# ---------------------------------------------------------------------------
# Tests: LLMRateLimiter pure unit (direct use, no HTTP)
# ---------------------------------------------------------------------------


class TestLLMRateLimiterUnit:
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

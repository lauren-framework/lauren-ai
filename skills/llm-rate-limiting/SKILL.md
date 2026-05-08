---
name: llm-rate-limiting
description: Enforces per-minute request and token-per-minute limits on LLM API calls with automatic retry and exponential backoff. Use when deploying multi-user applications that share an API key, when provider rate limits must not be exceeded, or when per-user quotas are needed.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Rate Limiting for LLM API Calls

## Built-in: `RateLimiter`

Lauren-AI ships `RateLimiter` in `lauren_ai._cost._rate`.  Attach it to
`LLMConfig` and the transport automatically respects the limits:

```python
# NOTE: Do NOT add `from __future__ import annotations` to tool files.
from lauren_ai import RateLimiter, LLMConfig

config = LLMConfig(
    provider="anthropic",
    model="claude-haiku-4-5",
    api_key="...",
    rate_limiter=RateLimiter(
        requests_per_minute=60,
        tokens_per_minute=100_000,
        max_retries=5,
        initial_backoff_s=1.0,
        max_backoff_s=60.0,
        jitter=True,
    ),
)
```

## `RateLimiter` fields

| Field | Default | Description |
|-------|---------|-------------|
| `requests_per_minute` | `None` | Max API calls per 60-second sliding window |
| `tokens_per_minute` | `None` | Max tokens per 60-second sliding window (token bucket) |
| `max_retries` | `5` | Retries before raising `RateLimitExhaustedError` |
| `initial_backoff_s` | `1.0` | Starting backoff in seconds |
| `max_backoff_s` | `60.0` | Ceiling for exponential backoff |
| `jitter` | `True` | Add random jitter to prevent thundering herd |

## Custom in-process rate limiter

When you need per-user limits or a limiter outside `LLMConfig`:

```python
import asyncio
import time


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
```

## Per-user limits

```python
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
```

## Handling `RateLimitExhaustedError`

```python
from lauren_ai import RateLimitExhaustedError

try:
    result = await runner.run(agent, message)
except RateLimitExhaustedError as exc:
    print(f"Rate limit hit: {exc.limit} RPM, retry after {exc.retry_after}s")
    # Return 429 to caller or queue the request
```

## Pitfalls

- The built-in `RateLimiter` uses a sliding window and token-bucket algorithm;
  it does not coordinate across multiple processes.  For multi-worker deployments,
  use a Redis-backed distributed rate limiter.
- `estimated_tokens` in `acquire()` is a pre-call estimate.  For accurate
  token-per-minute enforcement, call `record_call(tokens_used=actual_tokens)`
  after the response returns.
- The custom `LLMRateLimiter` above polls every 100ms — acceptable for low
  concurrency.  For high throughput, use `asyncio.Event` or `asyncio.Condition`
  for efficient wake-up.
- `jitter=True` in the built-in `RateLimiter` prevents thundering-herd retries
  when many concurrent requests all back off and retry simultaneously.

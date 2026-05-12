"""RateLimiter -- token-bucket rate limiter with exponential backoff on 429."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from lauren_ai._exceptions import LaurenAIError


class RateLimitExhaustedError(LaurenAIError):
    """Raised when the rate limiter's ``max_retries`` is exhausted.

    :param message: Human-readable description of the exhaustion.
    :type message: str
    :param limit: The configured requests-per-minute limit (``0`` when no
        per-minute limit is configured).
    :type limit: int
    :param retry_after: Suggested number of seconds to wait before retrying,
        if known (``0.0`` otherwise).
    :type retry_after: float
    """

    def __init__(
        self,
        message: str,
        *,
        limit: int = 0,
        retry_after: float = 0.0,
    ) -> None:
        """Initialise the error.

        :param message: Human-readable description.
        :type message: str
        :param limit: Configured RPM limit.
        :type limit: int
        :param retry_after: Suggested retry delay in seconds.
        :type retry_after: float
        """
        super().__init__(message)
        self.limit: int = limit
        self.retry_after: float = retry_after


@dataclass
class RateLimiter:
    """Token-bucket rate limiter with automatic retry on HTTP 429.

    Usage::

        config = LLMConfig(
            model="claude-haiku-4-5",
            rate_limiter=RateLimiter(
                requests_per_minute=60,
                tokens_per_minute=100_000,
                max_retries=5,
            ),
        )
    """

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    max_retries: int = 5
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    jitter: bool = True

    _request_times: list[float] = field(default_factory=list, init=False, repr=False)
    _token_count: float = field(default=0.0, init=False, repr=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False, repr=False)

    def _refill_tokens(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if self.tokens_per_minute:
            refill = elapsed * (self.tokens_per_minute / 60.0)
            self._token_count = min(float(self.tokens_per_minute), self._token_count + refill)
        self._last_refill = now

    def _clean_request_times(self) -> None:
        cutoff = time.monotonic() - 60.0
        self._request_times = [t for t in self._request_times if t > cutoff]

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Wait until a request slot is available."""
        self._refill_tokens()
        self._clean_request_times()

        if self.requests_per_minute and len(self._request_times) >= self.requests_per_minute:
            oldest = self._request_times[0]
            wait = 60.0 - (time.monotonic() - oldest)
            if wait > 0:
                await asyncio.sleep(wait)
            self._clean_request_times()

        if self.tokens_per_minute and estimated_tokens > 0:
            if self._token_count < estimated_tokens:
                wait = (estimated_tokens - self._token_count) / (self.tokens_per_minute / 60.0)
                await asyncio.sleep(wait)
                self._refill_tokens()
            self._token_count -= estimated_tokens

        self._request_times.append(time.monotonic())

    def backoff_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Compute sleep duration for a retry attempt."""
        if retry_after is not None:
            return retry_after
        base = min(self.initial_backoff_s * (2**attempt), self.max_backoff_s)
        if self.jitter:
            base *= 0.5 + random.random() * 0.5
        return base

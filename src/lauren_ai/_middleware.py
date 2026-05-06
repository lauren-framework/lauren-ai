"""Middleware factories for Lauren AI applications.

Provides middleware for conversation history management and AI endpoint rate limiting.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

__all__ = [
    "conversation_middleware",
    "ai_rate_limit",
    "RateLimitStore",
    "InMemoryRateLimitStore",
    "BudgetUsage",
]

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Conversation middleware
# ---------------------------------------------------------------------------


def conversation_middleware(
    store: Any,
    *,
    header: str = "x-conversation-id",
    cookie: str | None = "conversation_id",
    auto_create: bool = True,
) -> type:
    """Return a ``@middleware()``-decorated class that manages conversation history.

    Reads a conversation ID from a request header or cookie, loads the
    conversation history from a :class:`ConversationStore`, attaches it to
    ``request.state``, and saves the updated history after the response.

    :param store: A :class:`~lauren_ai.ConversationStore` implementation.
    :param header: HTTP header name to read the conversation ID from.
    :type header: str
    :param cookie: Cookie name to fall back to if the header is absent.
                   Set to ``None`` to disable cookie lookup.
    :type cookie: str | None
    :param auto_create: If ``True``, automatically generate a new conversation ID
                        when none is found in the request.
    :type auto_create: bool
    :returns: A middleware class suitable for ``global_middlewares=`` or
              ``@use_middlewares()``.
    :rtype: type
    """
    from lauren import Scope, injectable, middleware

    @middleware()
    @injectable(scope=Scope.REQUEST)
    class _ConversationMiddleware:
        """Middleware that attaches conversation history to each request."""

        def __init__(self, conv_store: type(store)) -> None:  # type: ignore[valid-type]
            self._store = conv_store

        async def dispatch(self, request: Any, call_next: Any) -> Any:
            """Load conversation history before the handler and save it after.

            :param request: The incoming HTTP request.
            :param call_next: Callable to invoke the next middleware/handler.
            :returns: The HTTP response.
            """
            cid: str | None = None

            # Try header first
            if hasattr(request, "headers"):
                cid = request.headers.get(header)

            # Fall back to cookie
            if cid is None and cookie and hasattr(request, "cookies"):
                cid = request.cookies.get(cookie)

            if cid:
                try:
                    history = await self._store.load(cid)
                    request.state.conversation_id = cid
                    request.state.conversation_history = history
                except Exception:
                    # Gracefully handle store errors — start fresh
                    cid = None

            if cid is None and auto_create:
                cid = uuid.uuid4().hex
                request.state.conversation_id = cid
                request.state.conversation_history = []

            response = await call_next(request)

            # Save updated conversation if modified by the handler
            if cid and hasattr(request.state, "conversation_id"):
                updated = getattr(request.state, "updated_conversation", None)
                if updated:
                    try:
                        await self._store.save(cid, updated)
                    except Exception:
                        pass  # Best-effort save; never fail the response

            return response

    return _ConversationMiddleware


# ---------------------------------------------------------------------------
# Rate limit middleware
# ---------------------------------------------------------------------------


@dataclass
class BudgetUsage:
    """Tracks token and request usage for rate limiting.

    :param tokens: Total tokens consumed in the current window.
    :param requests: Total requests made in the current window.
    :param window_start: Unix timestamp when the current window started.
    """

    tokens: int = 0
    requests: int = 0
    window_start: float = field(default_factory=time.monotonic)


class RateLimitStore:
    """Protocol for rate limit state storage.

    :method get_usage: Retrieve current usage for a key.
    :method increment: Increment usage counters and return the new usage.
    :method reset: Reset usage for a key.
    """

    async def get_usage(self, key: str, *, window_seconds: int) -> BudgetUsage:
        """Return current usage for the given key within the window.

        :param key: The rate limit key (e.g. user ID or IP address).
        :type key: str
        :param window_seconds: Window duration in seconds.
        :type window_seconds: int
        :returns: Current usage snapshot.
        :rtype: BudgetUsage
        """
        raise NotImplementedError

    async def increment(self, key: str, *, tokens: int = 0, window_seconds: int) -> BudgetUsage:
        """Increment usage and return updated snapshot.

        :param key: The rate limit key.
        :type key: str
        :param tokens: Number of tokens to add to the count.
        :type tokens: int
        :param window_seconds: Window duration in seconds.
        :type window_seconds: int
        :returns: Updated usage snapshot.
        :rtype: BudgetUsage
        """
        raise NotImplementedError


class InMemoryRateLimitStore(RateLimitStore):
    """In-memory rate limit store using sliding windows.

    Suitable for single-process deployments. State is lost on restart.
    Use a Redis-backed store for multi-process or distributed deployments.
    """

    def __init__(self) -> None:
        self._usage: dict[str, BudgetUsage] = defaultdict(BudgetUsage)

    async def get_usage(self, key: str, *, window_seconds: int) -> BudgetUsage:
        """Return usage for *key*, resetting the window if expired.

        :param key: Rate limit key.
        :type key: str
        :param window_seconds: Window size in seconds.
        :type window_seconds: int
        :returns: Current usage.
        :rtype: BudgetUsage
        """
        usage = self._usage[key]
        now = time.monotonic()
        if now - usage.window_start >= window_seconds:
            # Window has elapsed — reset
            usage.tokens = 0
            usage.requests = 0
            usage.window_start = now
        return usage

    async def increment(self, key: str, *, tokens: int = 0, window_seconds: int) -> BudgetUsage:
        """Increment usage counters for *key*.

        :param key: Rate limit key.
        :type key: str
        :param tokens: Tokens to add.
        :type tokens: int
        :param window_seconds: Window size in seconds.
        :type window_seconds: int
        :returns: Updated usage.
        :rtype: BudgetUsage
        """
        usage = await self.get_usage(key, window_seconds=window_seconds)
        usage.tokens += tokens
        usage.requests += 1
        return usage


def ai_rate_limit(
    *,
    tokens_per_minute: int | None = None,
    requests_per_minute: int | None = None,
    requests_per_day: int | None = None,
    key_fn: Callable[[Any], str] | None = None,
    store: RateLimitStore | None = None,
) -> type:
    """Return a ``@middleware()``-decorated class that rate-limits AI endpoints.

    At least one of ``tokens_per_minute``, ``requests_per_minute``, or
    ``requests_per_day`` must be provided.

    :param tokens_per_minute: Maximum tokens allowed per minute per key.
    :type tokens_per_minute: int | None
    :param requests_per_minute: Maximum requests allowed per minute per key.
    :type requests_per_minute: int | None
    :param requests_per_day: Maximum requests allowed per day per key.
    :type requests_per_day: int | None
    :param key_fn: Function that extracts the rate-limit key from a request.
                   Defaults to the client's IP address.
    :type key_fn: Callable[[Request], str] | None
    :param store: Rate limit state backend. Defaults to :class:`InMemoryRateLimitStore`.
    :type store: RateLimitStore | None
    :raises ValueError: If no limits are specified.
    :returns: A middleware class.
    :rtype: type
    """
    if not any([tokens_per_minute, requests_per_minute, requests_per_day]):
        raise ValueError(
            "At least one limit (tokens_per_minute, requests_per_minute, "
            "or requests_per_day) must be specified."
        )

    _store = store or InMemoryRateLimitStore()
    _key_fn = key_fn or _default_key_fn

    from lauren import Scope, injectable, middleware  # noqa: F401
    from lauren.types import Response

    @middleware()
    class _RateLimitMiddleware:
        """Middleware that enforces per-key token and request budgets."""

        async def dispatch(self, request: Any, call_next: Any) -> Any:
            """Check rate limits before forwarding the request.

            :param request: The incoming request.
            :param call_next: Next middleware/handler callable.
            :returns: The response, or a 429 if the limit is exceeded.
            """
            key = _key_fn(request)

            if requests_per_minute is not None:
                usage = await _store.get_usage(key, window_seconds=60)
                if usage.requests >= requests_per_minute:
                    return Response.json(
                        {"error": "rate_limit_exceeded", "limit": "requests_per_minute"},
                        status=429,
                    )

            if tokens_per_minute is not None:
                usage = await _store.get_usage(key, window_seconds=60)
                if usage.tokens >= tokens_per_minute:
                    return Response.json(
                        {"error": "rate_limit_exceeded", "limit": "tokens_per_minute"},
                        status=429,
                    )

            await _store.increment(key, window_seconds=60)
            return await call_next(request)

    return _RateLimitMiddleware


def _default_key_fn(request: Any) -> str:
    """Extract rate-limit key from request (defaults to client IP)."""
    try:
        client = getattr(request, "client", None)
        if client and hasattr(client, "host"):
            return str(client.host)
    except Exception:
        pass
    return "anonymous"

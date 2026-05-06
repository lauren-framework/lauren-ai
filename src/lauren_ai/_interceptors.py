"""Interceptor factories for ``lauren-ai`` applications.

Provides two interceptors:

* :func:`ai_metrics_interceptor` — captures token usage and timing, emitting
  :class:`~lauren_ai._signals.ModelCallComplete` signals.
* :func:`token_usage_response_interceptor` — appends ``x-token-usage`` and
  ``x-ai-cost-usd`` response headers when token usage is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "ai_metrics_interceptor",
    "token_usage_response_interceptor",
]

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# AI metrics interceptor
# ---------------------------------------------------------------------------


def ai_metrics_interceptor() -> type:
    """Return an interceptor that captures AI token usage and timing.

    The interceptor reads ``request.state.ai_token_usage`` (populated by
    the :class:`~lauren_ai._agents._runner.AgentRunner`) after each handler
    completes and emits a
    :class:`~lauren_ai._signals.ModelCallComplete` signal.

    Also attaches the following to ``request.state``:

    * ``ai_duration_ms`` — handler wall-clock time in milliseconds.

    :returns: An interceptor class suitable for ``global_interceptors=``
        or ``@use_interceptors()``.
    :rtype: type

    Example::

        @module(
            controllers=[AIController],
            global_interceptors=[ai_metrics_interceptor()],
        )
        class AppModule: ...
    """
    from lauren import interceptor

    @interceptor()
    class _AIMetricsInterceptor:
        """Interceptor that records AI token usage and emits lifecycle signals."""

        async def intercept(self, ctx: Any, call_next: Any) -> Any:
            """Wrap the handler, recording timing and token usage.

            :param ctx: Execution context.
            :param call_next: Next interceptor or handler callable.
            :returns: The handler response.
            """
            import time  # noqa: PLC0415

            start = time.monotonic()
            response = await call_next(ctx)
            elapsed_ms = (time.monotonic() - start) * 1000.0

            request = getattr(ctx, "request", None)
            if request and hasattr(request, "state"):
                request.state.ai_duration_ms = elapsed_ms

                usage = getattr(request.state, "ai_token_usage", None)
                if usage is not None:
                    try:
                        from lauren_ai._signals import ModelCallComplete  # noqa: PLC0415

                        signals = getattr(ctx, "signals", None)
                        if signals:
                            await signals.emit(
                                ModelCallComplete(
                                    agent_class=None,
                                    model=getattr(usage, "model", "unknown"),
                                    total_tokens=(
                                        getattr(usage, "input_tokens", 0)
                                        + getattr(usage, "output_tokens", 0)
                                    ),
                                    cost_usd=getattr(usage, "cost_usd", 0.0),
                                    turns=1,
                                )
                            )
                    except Exception:
                        pass

            return response

    return _AIMetricsInterceptor


# ---------------------------------------------------------------------------
# Token usage response interceptor
# ---------------------------------------------------------------------------


def token_usage_response_interceptor() -> type:
    """Return an interceptor that appends token-usage headers to responses.

    Reads ``request.state.ai_token_usage`` after the handler completes and
    appends the following headers when token usage data is available:

    * ``x-token-usage`` — total tokens used (``input + output``).
    * ``x-ai-cost-usd`` — estimated cost in USD (4 decimal places).

    :returns: An interceptor class.
    :rtype: type

    Example::

        @module(
            controllers=[AIController],
            global_interceptors=[token_usage_response_interceptor()],
        )
        class AppModule: ...
    """
    from lauren import interceptor

    @interceptor()
    class _TokenUsageResponseInterceptor:
        """Interceptor that appends AI usage metrics to response headers."""

        async def intercept(self, ctx: Any, call_next: Any) -> Any:
            """Call the handler and append token-usage headers to the response.

            :param ctx: Execution context.
            :param call_next: Next interceptor or handler callable.
            :returns: The response, possibly with added headers.
            """
            response = await call_next(ctx)

            request = getattr(ctx, "request", None)
            if request and hasattr(request, "state"):
                usage = getattr(request.state, "ai_token_usage", None)
                if usage is not None and response is not None:
                    try:
                        total = getattr(usage, "input_tokens", 0) + getattr(
                            usage, "output_tokens", 0
                        )
                        cost = getattr(usage, "cost_usd", 0.0)
                        if hasattr(response, "headers"):
                            response.headers["x-token-usage"] = str(total)
                            response.headers["x-ai-cost-usd"] = f"{cost:.4f}"
                    except Exception:
                        pass

            return response

    return _TokenUsageResponseInterceptor

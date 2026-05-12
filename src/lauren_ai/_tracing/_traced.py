"""``@traced()`` decorator for manual span instrumentation.

Wrap any async function to automatically create a :class:`~_core.Span` on
entry and finalise it (with status and timing) on exit or exception.  When a
global :class:`~_core.TraceStore` is registered via :func:`set_trace_store`
the completed trace is exported to all registered exporters after each call.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar

from lauren_ai._exceptions import DecoratorUsageError
from lauren_ai._tracing._core import Span, SpanKind, Trace, TraceStore

_P = ParamSpec("_P")
_R = TypeVar("_R")

__all__ = [
    "traced",
    "set_trace_store",
    "get_trace_store",
]

# ---------------------------------------------------------------------------
# Global trace store registry (module-level, replaces per-thread singletons)
# ---------------------------------------------------------------------------

_GLOBAL_TRACE_STORE: TraceStore | None = None

# Context variable tracking the active Trace for the current async task so
# nested @traced calls can be linked as parent/child spans.
_ACTIVE_TRACE: ContextVar[Trace | None] = ContextVar("_ACTIVE_TRACE", default=None)


def set_trace_store(store: TraceStore) -> None:
    """Register *store* as the global destination for completed traces.

    Subsequent :func:`traced`-decorated calls will export finished traces to
    all exporters attached to *store*.

    :param store: The :class:`~_core.TraceStore` instance to activate.
    :type store: TraceStore
    """
    global _GLOBAL_TRACE_STORE
    _GLOBAL_TRACE_STORE = store


def get_trace_store() -> TraceStore | None:
    """Return the currently-registered global :class:`~_core.TraceStore`, or
    ``None`` when none has been set.

    :return: The active trace store, or ``None``.
    :rtype: TraceStore | None
    """
    return _GLOBAL_TRACE_STORE


# ---------------------------------------------------------------------------
# @traced() decorator
# ---------------------------------------------------------------------------


def traced(
    fn_or_none: Any = None,
    *,
    name: str | None = None,
    kind: SpanKind = SpanKind.CUSTOM,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """Decorate an async function to create a trace span.

    The decorator **requires parentheses** — bare ``@traced`` usage raises
    :class:`~lauren_ai._exceptions.DecoratorUsageError` immediately.

    If a global :class:`~_core.TraceStore` has been set via
    :func:`set_trace_store`, the completed :class:`~_core.Trace` is recorded
    there after each invocation.  If a trace is already active in the current
    async task the new span is attached as a child of the active root span.

    Usage::

        @traced(name="fetch_user", kind=SpanKind.CUSTOM)
        async def fetch_user(user_id: str) -> dict:
            ...

    :param name: Override the span name.  Defaults to the function name.
    :type name: str | None
    :param kind: Span classification.  Defaults to :attr:`SpanKind.CUSTOM`.
    :type kind: SpanKind
    :param metadata: Extra key-value metadata to attach to the span.
    :type metadata: dict[str, Any] | None
    :return: A decorator that wraps the target async function with span
        recording.
    :rtype: Callable
    :raises DecoratorUsageError: When used bare without parentheses.
    """
    # Guard bare usage: @traced (without parentheses) passes the decorated
    # function as fn_or_none.  Detect that and raise loudly.
    if fn_or_none is not None:
        raise DecoratorUsageError(
            "Use @traced() with parentheses, not bare @traced.",
            decorator_name="traced",
        )

    def decorator(
        fn: Callable[_P, Coroutine[Any, Any, _R]],
    ) -> Callable[_P, Coroutine[Any, Any, _R]]:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # Determine whether we are inside an active trace.
            parent_trace = _ACTIVE_TRACE.get()
            parent_id: str | None = None
            if parent_trace is not None and parent_trace.root_span is not None:
                parent_id = parent_trace.root_span.span_id

            span = Span(
                name=span_name,
                kind=kind,
                metadata=dict(metadata or {}),
                parent_id=parent_id,
            )
            span.started_at = time.monotonic()

            # Build a new Trace for this invocation (root-level call) or reuse.
            if parent_trace is None:
                trace = Trace(run_id=span.span_id)
                trace.spans.append(span)
                token = _ACTIVE_TRACE.set(trace)
            else:
                trace = parent_trace
                trace.spans.append(span)
                token = None

            try:
                result = await fn(*args, **kwargs)
                span.finish(outputs={"result": str(result)[:200]})
                return result
            except Exception as exc:
                span.finish(error=str(exc))
                raise
            finally:
                if token is not None:
                    # We are at the root level — export and clean up.
                    _ACTIVE_TRACE.reset(token)
                    store = _GLOBAL_TRACE_STORE
                    if store is not None:
                        store._add(trace)
                        for exporter in getattr(store, "_exporters", []):
                            try:
                                import asyncio

                                asyncio.get_event_loop().create_task(exporter.export(trace))
                            except Exception:
                                pass

        return wrapper

    return decorator

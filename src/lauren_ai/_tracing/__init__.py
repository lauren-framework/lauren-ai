"""Tracing and observability subsystem for ``lauren-ai``.

Provides span/trace data structures, exporters, the ``@traced()`` decorator,
and the global trace-store registry.
"""

from __future__ import annotations

from lauren_ai._tracing._core import (
    Span,
    SpanKind,
    Trace,
    TraceStore,
    TracingConfig,
)
from lauren_ai._tracing._exporters import (
    ConsoleTraceExporter,
    FileTraceExporter,
    InMemoryTraceExporter,
    TraceExporter,
)
from lauren_ai._tracing._traced import get_trace_store, set_trace_store, traced

__all__ = [
    "Span",
    "Trace",
    "SpanKind",
    "TraceStore",
    "TracingConfig",
    "TraceExporter",
    "InMemoryTraceExporter",
    "ConsoleTraceExporter",
    "FileTraceExporter",
    "traced",
    "set_trace_store",
    "get_trace_store",
]

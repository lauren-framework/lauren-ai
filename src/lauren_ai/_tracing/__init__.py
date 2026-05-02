"""Tracing and observability subsystem for ``lauren-ai``.

Provides span/trace data structures, exporters, the ``@traced()`` decorator,
and the global trace-store registry.
"""

from __future__ import annotations

from lauren_ai._tracing._core import (
    Span,
    Trace,
    SpanKind,
    TraceStore,
    TracingConfig,
)
from lauren_ai._tracing._exporters import (
    TraceExporter,
    InMemoryTraceExporter,
    ConsoleTraceExporter,
    FileTraceExporter,
)
from lauren_ai._tracing._traced import traced, set_trace_store, get_trace_store

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

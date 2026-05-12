"""Built-in trace exporters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from lauren_ai._tracing._core import Span, Trace


@runtime_checkable
class TraceExporter(Protocol):
    """Protocol satisfied by all trace exporter implementations."""

    async def export(self, trace: Trace) -> None:
        """Export a completed trace.

        :param trace: The trace to export.
        """
        ...

    async def export_span(self, span: Span) -> None:
        """Export a single finished span.

        :param span: The span to export.
        """
        ...


class InMemoryTraceExporter:
    """Collect traces in memory for testing.

    Usage::

        exporter = InMemoryTraceExporter()
        config = TracingConfig(enabled=True, exporter=exporter)
        # ... run agent ...
        assert len(exporter.traces) == 1
    """

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self.spans: list[Span] = []

    async def export(self, trace: Trace) -> None:
        """Store *trace* in :attr:`traces`.

        :param trace: The trace to record.
        """
        self.traces.append(trace)

    async def export_span(self, span: Span) -> None:
        """Store *span* in :attr:`spans`.

        :param span: The span to record.
        """
        self.spans.append(span)

    def clear(self) -> None:
        """Remove all stored traces and spans."""
        self.traces.clear()
        self.spans.clear()


class ConsoleTraceExporter:
    """Pretty-print trace trees to stdout.

    Usage::

        exporter = ConsoleTraceExporter()
        config = TracingConfig(enabled=True, exporter=exporter)
    """

    async def export(self, trace: Trace) -> None:
        """Print the trace tree to stdout.

        :param trace: The trace to print.
        """
        print(f"\n[Trace {trace.trace_id[:8]}] run_id={trace.run_id}")
        print(trace.as_tree())

    async def export_span(self, span: Span) -> None:
        """Print a single span summary to stdout.

        :param span: The span to print.
        """
        dur = f" [{span.duration_ms:.0f}ms]" if span.duration_ms is not None else ""
        print(f"  span: {span.name}{dur}")


class FileTraceExporter:
    """Write traces as NDJSON to a file.

    Usage::

        exporter = FileTraceExporter("/var/log/traces.ndjson")

    :param path: Path to the output NDJSON file.  Appended to on each export.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    async def export(self, trace: Trace) -> None:
        """Append *trace* as a JSON record to the output file.

        :param trace: The trace to write.
        """
        record = {
            "trace_id": trace.trace_id,
            "run_id": trace.run_id,
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "name": s.name,
                    "kind": s.kind.value,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                    "metadata": s.metadata,
                }
                for s in trace.spans
            ],
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    async def export_span(self, span: Span) -> None:
        """No-op for file exporter (traces are exported at the trace level only).

        :param span: Ignored.
        """

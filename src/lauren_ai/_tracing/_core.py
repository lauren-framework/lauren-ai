from __future__ import annotations

"""Core tracing types: Span, Trace, TraceStore, TracingConfig."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(Enum):
    """Classification of traced operations."""

    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    CHAIN = "chain"
    CUSTOM = "custom"
    TEAM = "team"


@dataclass
class Span:
    """A single traced operation within an agent run.

    :param span_id: Unique identifier for this span (random hex).
    :param parent_id: Identifier of the parent span, or ``None`` for root spans.
    :param name: Human-readable name for the operation.
    :param kind: Classification of the operation.
    :param inputs: Input arguments recorded at span start.
    :param outputs: Output values recorded at span finish.
    :param error: Error message if the operation failed, else ``None``.
    :param started_at: Monotonic timestamp when the span started.
    :param ended_at: Monotonic timestamp when the span ended, or ``None`` if still open.
    :param metadata: Arbitrary key-value metadata.
    """

    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    name: str = ""
    kind: SpanKind = SpanKind.CUSTOM
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration in milliseconds, or ``None`` if still open.

        :return: Duration in milliseconds, or ``None``.
        """
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000

    def finish(
        self,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Mark the span as finished.

        :param outputs: Optional output values to record.
        :param error: Optional error message to record.
        """
        self.ended_at = time.monotonic()
        if outputs is not None:
            self.outputs = outputs
        if error is not None:
            self.error = error


@dataclass
class Trace:
    """A hierarchical tree of spans for one agent run.

    :param trace_id: Unique identifier for this trace (random hex).
    :param run_id: Correlation identifier linking this trace to an agent run.
    :param spans: Ordered list of all spans belonging to this trace.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = ""
    spans: list[Span] = field(default_factory=list)

    @property
    def root_span(self) -> Span | None:
        """Return the root span (the first span with no parent).

        :return: The root :class:`Span`, or ``None`` when the trace is empty.
        """
        roots = [s for s in self.spans if s.parent_id is None]
        return roots[0] if roots else None

    def as_tree(self) -> str:
        """Return a human-readable ASCII tree of the span hierarchy.

        :return: Multi-line string representing the span tree.
        """
        if not self.spans:
            return "(empty trace)"

        # Build parent -> children map
        children: dict[str | None, list[Span]] = {}
        for span in self.spans:
            if span.parent_id not in children:
                children[span.parent_id] = []
            children[span.parent_id].append(span)

        lines: list[str] = []

        def _render(span: Span, indent: str = "") -> None:
            dur = f"  [{span.duration_ms:.0f}ms]" if span.duration_ms is not None else ""
            err = f"  ERROR: {span.error}" if span.error else ""
            lines.append(f"{indent}{span.name}{dur}{err}")
            for child in children.get(span.span_id, []):
                _render(child, indent + "  ")

        for root in children.get(None, []):
            _render(root)
        return "\n".join(lines)


class TraceStore:
    """In-memory store of recent traces, injectable as a DI service.

    Usage::

        store = TraceStore()
        # ... traces are added by TracingConfig's exporter ...
        recent = await store.last(10)

    :param max_traces: Maximum number of traces to retain in memory.
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self._traces: list[Trace] = []
        self._max = max_traces

    def _add(self, trace: Trace) -> None:
        """Add a trace to the store, evicting oldest entries when at capacity.

        :param trace: The trace to add.
        """
        self._traces.append(trace)
        if len(self._traces) > self._max:
            self._traces = self._traces[-self._max:]

    async def get(self, trace_id: str) -> Trace | None:
        """Retrieve a trace by its identifier.

        :param trace_id: The trace identifier to look up.
        :return: The matching :class:`Trace`, or ``None`` when not found.
        """
        for t in self._traces:
            if t.trace_id == trace_id:
                return t
        return None

    async def list(self, run_id: str) -> list[Trace]:
        """Return all traces associated with a given run identifier.

        :param run_id: The run identifier to filter by.
        :return: List of matching traces in insertion order.
        """
        return [t for t in self._traces if t.run_id == run_id]

    async def last(self, n: int = 10) -> list[Trace]:
        """Return the *n* most recently added traces.

        :param n: Number of traces to return.
        :return: List of traces, newest first.
        """
        return list(reversed(self._traces[-n:]))

    def __len__(self) -> int:
        return len(self._traces)


@dataclass
class TracingConfig:
    """Configuration for tracing in LLMConfig.

    :param enabled: Whether tracing is active.
    :param exporter: A :class:`~lauren_ai._tracing._exporters.TraceExporter`
        instance, or ``None`` to use the default in-memory store.
    :param sample_rate: Fraction of traces to export (0.0–1.0).
    :param include_inputs: Whether to record input arguments in spans.
    :param include_outputs: Whether to record output values in spans.
    """

    enabled: bool = False
    exporter: Any = None  # TraceExporter
    sample_rate: float = 1.0
    include_inputs: bool = True
    include_outputs: bool = True

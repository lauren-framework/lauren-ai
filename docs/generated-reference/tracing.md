# Tracing

Observability spans, exporters, and the `@traced` decorator.

## Decorator

### `traced`

Decorate an async function to create a trace span.

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

## Span types

### `SpanKind`

Classification of traced operations.

### `Span`

A single traced operation within an agent run.

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

### `Trace`

A hierarchical tree of spans for one agent run.

:param trace_id: Unique identifier for this trace (random hex).
:param run_id: Correlation identifier linking this trace to an agent run.
:param spans: Ordered list of all spans belonging to this trace.

## Store & config

### `TraceStore`

In-memory store of recent traces, injectable as a DI service.

Usage::

    store = TraceStore()
    # ... traces are added by TracingConfig's exporter ...
    recent = await store.last(10)

:param max_traces: Maximum number of traces to retain in memory.

### `TracingConfig`

Configuration for tracing in LLMConfig.

:param enabled: Whether tracing is active.
:param exporter: A :class:`~lauren_ai._tracing._exporters.TraceExporter`
    instance, or ``None`` to use the default in-memory store.
:param sample_rate: Fraction of traces to export (0.0–1.0).
:param include_inputs: Whether to record input arguments in spans.
:param include_outputs: Whether to record output values in spans.

### `set_trace_store`

Register *store* as the global destination for completed traces.

Subsequent :func:`traced`-decorated calls will export finished traces to
all exporters attached to *store*.

:param store: The :class:`~_core.TraceStore` instance to activate.
:type store: TraceStore

### `get_trace_store`

Return the currently-registered global :class:`~_core.TraceStore`, or
``None`` when none has been set.

:return: The active trace store, or ``None``.
:rtype: TraceStore | None

## Exporters

### `TraceExporter`

Protocol satisfied by all trace exporter implementations.

### `InMemoryTraceExporter`

Collect traces in memory for testing.

Usage::

    exporter = InMemoryTraceExporter()
    config = TracingConfig(enabled=True, exporter=exporter)
    # ... run agent ...
    assert len(exporter.traces) == 1

### `ConsoleTraceExporter`

Pretty-print trace trees to stdout.

Usage::

    exporter = ConsoleTraceExporter()
    config = TracingConfig(enabled=True, exporter=exporter)

### `FileTraceExporter`

Write traces as NDJSON to a file.

Usage::

    exporter = FileTraceExporter("/var/log/traces.ndjson")

:param path: Path to the output NDJSON file.  Appended to on each export.


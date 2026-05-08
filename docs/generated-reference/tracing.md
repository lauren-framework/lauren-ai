# Tracing

Observability spans, exporters, and the `@traced` decorator.

## Decorator

### `traced`

Decorate an async function to create a trace span.

The decorator **requires parentheses** — bare `@traced` usage raises
`DecoratorUsageError` immediately.

If a global `TraceStore` has been set via
`set_trace_store()`, the completed `Trace` is recorded
there after each invocation.  If a trace is already active in the current
async task the new span is attached as a child of the active root span.

Usage::

    @traced(name="fetch_user", kind=SpanKind.CUSTOM)
    async def fetch_user(user_id: str) -> dict:
        ...

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str | None` | Override the span name.  Defaults to the function name. |
| `kind` | `SpanKind` | Span classification.  Defaults to `SpanKind.CUSTOM`. |
| `metadata` | `dict[str, Any] | None` | Extra key-value metadata to attach to the span. |

**Returns:** `Callable` — A decorator that wraps the target async function with span
recording.

**Raises:**

| Exception | Description |
|---|---|
| `DecoratorUsageError` | When used bare without parentheses. |

## Span types

### `SpanKind`

Classification of traced operations.

### `Span`

A single traced operation within an agent run.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `span_id` | `str` | Unique identifier for this span (random hex). |
| `parent_id` | `str | None` | Identifier of the parent span, or `None` for root spans. |
| `name` | `str` | Human-readable name for the operation. |
| `kind` | `SpanKind` | Classification of the operation. |
| `inputs` | `dict[str, Any]` | Input arguments recorded at span start. |
| `outputs` | `dict[str, Any] | None` | Output values recorded at span finish. |
| `error` | `str | None` | Error message if the operation failed, else `None`. |
| `started_at` | `float` | Monotonic timestamp when the span started. |
| `ended_at` | `float | None` | Monotonic timestamp when the span ended, or `None` if still open. |
| `metadata` | `dict[str, Any]` | Arbitrary key-value metadata. |

### `Trace`

A hierarchical tree of spans for one agent run.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `trace_id` | `str` | Unique identifier for this trace (random hex). |
| `run_id` | `str` | Correlation identifier linking this trace to an agent run. |
| `spans` | `list[Span]` | Ordered list of all spans belonging to this trace. |

## Store & config

### `TraceStore`

In-memory store of recent traces, injectable as a DI service.

Usage::

    store = TraceStore()
    # ... traces are added by TracingConfig's exporter ...
    recent = await store.last(10)

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `max_traces` | `int` | Maximum number of traces to retain in memory. |

### `TracingConfig`

Configuration for tracing in LLMConfig.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `enabled` | `bool` | Whether tracing is active. |
| `exporter` | `Any` | A `TraceExporter`
instance, or `None` to use the default in-memory store. |
| `sample_rate` | `float` | Fraction of traces to export (0.0–1.0). |
| `include_inputs` | `bool` | Whether to record input arguments in spans. |
| `include_outputs` | `bool` | Whether to record output values in spans. |

### `set_trace_store`

Register *store* as the global destination for completed traces.

Subsequent `traced()`-decorated calls will export finished traces to
all exporters attached to *store*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `store` | `TraceStore` | The `TraceStore` instance to activate. |

### `get_trace_store`

Return the currently-registered global `TraceStore`, or
`None` when none has been set.

**Returns:** `TraceStore | None` — The active trace store, or `None`.

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

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `path` | `str | Path` | Path to the output NDJSON file.  Appended to on each export. |


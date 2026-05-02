# Tracing & Observability

`lauren-ai` includes a lightweight tracing layer that records structured spans
for agent runs, tool calls, and LLM completions.  Traces can be exported to
memory (for tests), the console, or an NDJSON file.

## Core concepts

A **Span** is one traced operation: an LLM call, a tool invocation, or a custom
function.  Spans form a tree: every span except the root has a `parent_id`
pointing to the span that spawned it.

A **Trace** is the tree of spans for one agent run.

A **TraceStore** holds recent traces in memory and supports lookup by
`trace_id` or `run_id`.

## Quick start — manual instrumentation

```python
from lauren_ai._tracing import traced, SpanKind

@traced(name="fetch_document", kind=SpanKind.TOOL)
async def fetch_document(url: str) -> str:
    ...  # your logic here
```

Decorate any `async def` function; `@traced()` records the start time, calls
the function, records the end time and output, and finishes the span.
Exceptions are caught, their message recorded in `span.error`, and then
re-raised unchanged.

## Exporters

### In-memory (testing)

```python
from lauren_ai._tracing import InMemoryTraceExporter, TracingConfig

exporter = InMemoryTraceExporter()
config = TracingConfig(enabled=True, exporter=exporter)

# ... run your agent ...

assert len(exporter.traces) == 1
span_names = [s.name for s in exporter.traces[0].spans]
assert "fetch_document" in span_names
```

### Console

```python
from lauren_ai._tracing import ConsoleTraceExporter, TracingConfig

config = TracingConfig(
    enabled=True,
    exporter=ConsoleTraceExporter(),
)
```

Output looks like:

```
[Trace 3a7f1c2b] run_id=run-42
agent.run  [142ms]
  llm.complete  [130ms]
  tool.fetch  [12ms]
```

### File (NDJSON)

```python
from lauren_ai._tracing import FileTraceExporter, TracingConfig

config = TracingConfig(
    enabled=True,
    exporter=FileTraceExporter("/var/log/agent-traces.ndjson"),
)
```

Each trace is appended as a JSON line so the file is streamable with tools
like `jq`.

## `TracingConfig` reference

```python
@dataclass
class TracingConfig:
    enabled: bool = False          # master switch
    exporter: Any = None           # TraceExporter instance
    sample_rate: float = 1.0       # fraction of traces to export
    include_inputs: bool = True    # record input arguments
    include_outputs: bool = True   # record return values
```

## `Span` reference

```python
@dataclass
class Span:
    span_id: str          # 16-char hex, auto-generated
    parent_id: str | None # None for root spans
    name: str
    kind: SpanKind        # AGENT, TOOL, LLM, CHAIN, CUSTOM, TEAM
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None
    error: str | None
    started_at: float     # monotonic timestamp
    ended_at: float | None
    metadata: dict[str, Any]

    # Computed
    duration_ms: float | None
```

Call `span.finish(outputs={...})` to close a span and record its end time.

## `SpanKind` values

| Value | When to use |
|---|---|
| `AGENT` | Top-level agent run |
| `TOOL` | Tool execution |
| `LLM` | Direct LLM completion call |
| `CHAIN` | A multi-step chain |
| `TEAM` | Team orchestration |
| `CUSTOM` | Any other operation |

## `TraceStore`

```python
store = TraceStore(max_traces=1000)

# retrieve a specific trace
trace = await store.get("trace-id-hex")

# all traces for a run
traces = await store.list("run-42")

# most recent N traces
recent = await store.last(10)  # newest first
```

`TraceStore` evicts the oldest traces once `max_traces` is reached.

## Trace trees

`Trace.as_tree()` returns a human-readable hierarchy:

```python
trace = Trace(run_id="my-run")
root = Span(name="agent.run")
root.finish()
child = Span(name="llm.complete", parent_id=root.span_id)
child.finish()
trace.spans = [root, child]

print(trace.as_tree())
# agent.run  [5ms]
#   llm.complete  [3ms]
```

## Custom TraceExporter

Implement the `TraceExporter` protocol to send traces to any backend:

```python
from lauren_ai._tracing import TraceExporter, Trace, Span

class MyExporter:
    async def export(self, trace: Trace) -> None:
        # send to your backend
        ...

    async def export_span(self, span: Span) -> None:
        # optional: called after each individual span finishes
        ...
```

`TraceExporter` is a `@runtime_checkable` Protocol, so `isinstance(obj, TraceExporter)` works without explicit inheritance.

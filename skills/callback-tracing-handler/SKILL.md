---
name: callback-tracing-handler
description: Subscribe to SignalBus lifecycle events to build lightweight tracers that record LLM call spans, tool call spans, and run summaries. Use when you need observability into agent execution without adding a full tracing backend, or when building cost/latency dashboards.
---

> Use `codemap find "SignalBus"` to locate the signal bus before reading.

# Callback / Tracing Handler

Subscribe to `lauren_ai._signals` events via `SignalBus.on()` to build a
simple tracer that records spans for every agent run.

## Available signals

| Signal | When fired |
|---|---|
| `ModelCallStarted` | Before each LLM transport call |
| `ModelCallComplete` | After each LLM call (with usage + cost) |
| `ToolCallStarted` | Before each tool execution |
| `ToolCallComplete` | After each tool execution (with duration) |
| `AgentRunComplete` | When the full agent run terminates |

## Simple tracer

```python
from lauren_ai import SignalBus, ModelCallComplete, AgentRunComplete

class SimpleTracer:
    def __init__(self, signal_bus: SignalBus):
        self._spans: list[dict] = []
        self._bus = signal_bus
        signal_bus.on(ModelCallComplete)(self._on_llm_call)
        signal_bus.on(AgentRunComplete)(self._on_run_complete)

    async def _on_llm_call(self, event: ModelCallComplete) -> None:
        self._spans.append({
            "type": "llm_call",
            "model": event.model,
            "cost_usd": event.cost_usd,
            "input_tokens": event.usage.input_tokens if event.usage else 0,
            "output_tokens": event.usage.output_tokens if event.usage else 0,
        })

    async def _on_run_complete(self, event: AgentRunComplete) -> None:
        self._spans.append({
            "type": "run_complete",
            "turns": event.turns,
            "total_cost": event.total_cost_usd,
        })

    @property
    def spans(self) -> list[dict]:
        return list(self._spans)
```

## Wiring to AgentRunner

Pass the `SignalBus` instance to `AgentRunnerBase`:

```python
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig

bus = SignalBus()
tracer = SimpleTracer(bus)

mock = MockTransport()
cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
runner = AgentRunner(transport=mock, signals=bus)
```

## Tool tracing

Include tool spans for a complete picture:

```python
from lauren_ai import ToolCallStarted, ToolCallComplete
import time

class DetailedTracer(SimpleTracer):
    def __init__(self, bus: SignalBus):
        super().__init__(bus)
        self._tool_starts: dict[str, float] = {}
        bus.on(ToolCallStarted)(self._on_tool_start)
        bus.on(ToolCallComplete)(self._on_tool_done)

    async def _on_tool_start(self, event: ToolCallStarted) -> None:
        self._tool_starts[event.tool_use_id] = time.monotonic()

    async def _on_tool_done(self, event: ToolCallComplete) -> None:
        self._spans.append({
            "type": "tool_call",
            "tool_name": event.tool_name,
            "success": event.success,
            "duration_ms": event.duration_ms,
        })
```

## Cleanup

Clear handlers between tests to avoid accumulation:

```python
bus.clear(ModelCallComplete)
bus.clear(AgentRunComplete)
# or clear all:
bus.clear()
```

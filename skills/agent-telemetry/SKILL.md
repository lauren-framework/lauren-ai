---
name: agent-telemetry
description: Aggregates per-model telemetry from SignalBus events using CostTracker and a custom PerformanceMetrics collector. Use when tracking total LLM calls, token consumption, average latency, error rate, and cost per model across agent runs.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Agent Telemetry & Performance Metrics

## Overview

`AgentTelemetry` subscribes to `ModelCallComplete` and `AgentRunComplete`
events on a `SignalBus`.  For each model call it accumulates token counts,
duration, and estimated cost into per-model `ModelMetrics` buckets.

---

## Implementation

```python
# telemetry/agent_telemetry.py
from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from lauren_ai import SignalBus, ModelCallComplete, AgentRunComplete

@dataclass
class ModelMetrics:
    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    error_count: int = 0

class AgentTelemetry:
    def __init__(self, signal_bus: SignalBus):
        self._metrics: dict[str, ModelMetrics] = defaultdict(ModelMetrics)
        self._run_count = 0
        signal_bus.on(ModelCallComplete)(self._on_model_call)
        signal_bus.on(AgentRunComplete)(self._on_run_complete)

    async def _on_model_call(self, event: ModelCallComplete) -> None:
        m = self._metrics[event.model]
        m.call_count += 1
        if event.usage:
            m.total_input_tokens += event.usage.input_tokens
            m.total_output_tokens += event.usage.output_tokens
        m.total_cost_usd += event.cost_usd or 0.0
        m.total_duration_ms += event.duration_ms or 0.0

    async def _on_run_complete(self, event: AgentRunComplete) -> None:
        self._run_count += 1

    def get_summary(self) -> dict:
        total_cost = sum(m.total_cost_usd for m in self._metrics.values())
        total_calls = sum(m.call_count for m in self._metrics.values())
        return {
            "total_agent_runs": self._run_count,
            "total_llm_calls": total_calls,
            "total_cost_usd": total_cost,
            "per_model": {
                model: {
                    "calls": m.call_count,
                    "input_tokens": m.total_input_tokens,
                    "output_tokens": m.total_output_tokens,
                    "cost_usd": m.total_cost_usd,
                    "avg_duration_ms": (
                        m.total_duration_ms / m.call_count if m.call_count else 0.0
                    ),
                }
                for model, m in self._metrics.items()
            },
        }
```

---

## Wiring to an agent runner

Pass the `SignalBus` to `AgentRunnerBase` and create `AgentTelemetry` with the
same bus:

```python
from lauren_ai import SignalBus
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig

bus = SignalBus()
telemetry = AgentTelemetry(bus)

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="...")
runner = AgentRunner(transport=transport, signals=bus)
```

---

## Combining with CostTracker

`CostTracker` provides per-conversation cost tracking; `AgentTelemetry`
provides cross-conversation aggregate metrics.  Use both:

```python
from lauren_ai import CostTracker, default_pricing_table

cost_tracker = CostTracker(pricing=default_pricing_table())
# Register CostTracker's handler on the same bus
bus.on(ModelCallComplete)(cost_tracker._on_model_call_complete)
```

---

## Prometheus / OpenTelemetry export

Export the `get_summary()` dict from a Lauren health endpoint or convert it
to OTLP spans for distributed tracing:

```python
@get("/metrics")
async def metrics(self) -> dict:
    return self._telemetry.get_summary()
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_signals.py` | `SignalBus`, `ModelCallComplete`, `AgentRunComplete` |
| `src/lauren_ai/_cost/_tracker.py` | `CostTracker`, `_on_model_call_complete` |
| `src/lauren_ai/_cost/_pricing.py` | `default_pricing_table`, `PricingTable` |

---
name: token-usage-budget
description: Tracks token consumption and USD cost per conversation or user, enforces spending budgets, and surfaces usage reports. Use when billing users per-conversation, when capping agent costs in production, or when you need to monitor and alert on LLM spend.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Token Usage Tracking & Budget Enforcement

## Quick start

```python
# NOTE: future annotations are allowed, but tool signature types must resolve at import time.
from lauren_ai import CostTracker, default_pricing_table, SignalBus, ModelCallComplete

signal_bus = SignalBus()
cost_tracker = CostTracker(pricing=default_pricing_table())

@signal_bus.on(ModelCallComplete)
async def track_cost(event: ModelCallComplete) -> None:
    await cost_tracker._on_model_call_complete(event)
```

## Per-conversation cost session

```python
async with cost_tracker.session(conversation_id="conv-123", user_id="user-456") as session:
    result = await runner.run(agent, message)
    print(f"Cost: ${session.total_estimate.total_usd:.6f}")
    print(f"Tokens used: {session.total_tokens}")
```

## Manual usage recording (for testing)

```python
from lauren_ai._transport import TokenUsage

usage = TokenUsage(input_tokens=1000, output_tokens=500)
cost_tracker.record_usage("claude-sonnet-4-6", usage, conversation_id="conv-123")

report = await cost_tracker.report(conversation_id="conv-123")
print(f"Total: ${report.total_estimate.total_usd:.6f}")
print(f"By model: {report.by_model}")
```

## Per-agent cost budget

Set `max_cost_usd` on the agent config to hard-stop a run when the budget
is exceeded:

```python
from lauren_ai import agent, AgentConfig

@agent(
    model="claude-opus-4-6",
    system="You are a helpful assistant.",
    config=AgentConfig(max_cost_usd=0.10),  # $0.10 budget per run
)
class BudgetedAgent: ...
```

When the budget is exceeded mid-run, `AgentBudgetExceededError` is raised.

## `TokenBudget` for soft limits

`TokenBudget` tracks cumulative token usage and signals when a threshold is
reached without stopping the run:

```python
from lauren_ai import TokenBudget

budget = TokenBudget(max_input_tokens=50_000, max_output_tokens=10_000)
# Check after each turn:
if budget.is_exceeded(total_usage):
    # gracefully wind down the conversation
    ...
```

## Cost report

```python
report = await cost_tracker.report()  # all conversations
print(f"Total spend: ${report.total_estimate.total_usd:.4f}")
for model, estimate in report.by_model.items():
    print(f"  {model}: ${estimate.total_usd:.6f}")
```

## Pricing table

```python
from lauren_ai import default_pricing_table, PricingTable, ModelPricing

# Add a custom model to the default table
table = default_pricing_table()
# Or build from scratch:
table = PricingTable(models={
    "my-fine-tuned-model": ModelPricing(input_per_m=5.00, output_per_m=20.00),
})
tracker = CostTracker(pricing=table)
```

## Pitfalls

- `CostTracker._on_model_call_complete` is an internal method; in production,
  wire it through `SignalBus` so it fires automatically during agent runs.
- Cost estimates depend on the pricing table — models not in the table return
  `CostEstimate()` (zero cost).  Always register custom models.
- `AgentBudgetExceededError` is raised inside the agent loop; catch it in your
  controller and return a graceful error response.
- `TokenUsage.cost_usd(model)` returns 0 for unknown models — the runner uses
  it for budget checks, so always pass a recognized model name.

# Cost Tracking

`lauren-ai` ships a lightweight cost-tracking subsystem in `lauren_ai._cost`.
It provides per-model pricing, session-level cost accumulation, hard budget
limits, and a token-bucket rate limiter.

## Pricing table

`PricingTable` maps model names to `ModelPricing` objects that hold USD-per-
million-token rates for input, output, prompt-cache-read, and prompt-cache-write
tokens.

```python
from lauren_ai._cost import PricingTable, ModelPricing

table = PricingTable(models={
    "my-model": ModelPricing(input_per_m=1.00, output_per_m=5.00),
})
```

The built-in table (`default_pricing_table()`) covers current Anthropic and
OpenAI model prices and is used automatically when you construct a `CostTracker`
without supplying a custom table.

```python
from lauren_ai._cost import default_pricing_table

table = default_pricing_table()
assert "claude-haiku-4-5" in table
assert "gpt-4o" in table
```

### Estimating cost from token usage

```python
from lauren_ai._transport import TokenUsage
from lauren_ai._cost import default_pricing_table

table = default_pricing_table()
usage = TokenUsage(input_tokens=10_000, output_tokens=2_000)
estimate = table.estimate("claude-haiku-4-5", usage)
print(f"Input:  ${estimate.input_usd:.6f}")
print(f"Output: ${estimate.output_usd:.6f}")
print(f"Total:  ${estimate.total_usd:.6f}")
```

`CostEstimate` objects support addition, so you can accumulate costs across
multiple calls:

```python
total = estimate_a + estimate_b + estimate_c
print(total.total_usd)
```

## CostTracker

`CostTracker` is an injectable singleton service that accumulates token usage
over time, keyed by conversation ID.

### Manual usage recording

```python
from lauren_ai._cost import CostTracker
from lauren_ai._transport import TokenUsage

tracker = CostTracker()
tracker.record_usage("claude-haiku-4-5", usage, conversation_id="conv-1")

report = await tracker.report(conversation_id="conv-1")
print(f"Total cost: ${report.total_estimate.total_usd:.6f}")
print(f"By model:   {report.by_model}")
```

### Session context manager

`CostTracker.session()` is an async context manager that captures all token
usage recorded during its lifetime and accumulates the cost into a `CostSession`
object:

```python
async with tracker.session(conversation_id="conv-1", user_id="alice") as session:
    result = await runner.run(MyAgent, "Hello!")
    # Any record_usage() calls or signal-based accumulation during this block
    # will be reflected in session.total_estimate after the context exits.

print(f"This request cost: ${session.total_estimate.total_usd:.6f}")
```

### Signal-based accumulation

`CostTracker._on_model_call_complete` is a compatible handler for
`ModelCallComplete` signals emitted by the agent runner.  Register it on a
`SignalBus` to accumulate costs automatically:

```python
from lauren_ai._signals import SignalBus, ModelCallComplete

bus = SignalBus()
tracker = CostTracker()

@bus.on(ModelCallComplete)
async def on_complete(event: ModelCallComplete) -> None:
    await tracker._on_model_call_complete(event)
```

### Generating reports

```python
# All conversations
report = await tracker.report()
for conv_id, cost in report.by_conversation.items():
    print(f"{conv_id}: ${cost.total_usd:.4f}")

# Single conversation
report = await tracker.report(conversation_id="conv-1")
print(report.by_model)
```

## TokenBudget

`TokenBudget` enforces hard limits on token and USD spend per conversation.
Call `check()` before each LLM call; it raises `BudgetExceededError` if the
projected cumulative usage would exceed the configured limits.

```python
from lauren_ai._cost import TokenBudget, BudgetExceededError

budget = TokenBudget(
    max_tokens_per_conversation=50_000,
    max_usd_per_conversation=0.50,
)

try:
    budget.check(
        conversation_id="conv-1",
        current_tokens=45_000,
        estimated_tokens=10_000,
    )
except BudgetExceededError as exc:
    print(f"Budget exceeded: {exc.limit_type} limit={exc.limit} current={exc.current}")
```

`BudgetExceededError` attributes:

| Attribute | Description |
|---|---|
| `limit_type` | `"tokens_per_conversation"` or `"usd_per_conversation"` |
| `limit` | The configured limit value |
| `current` | The current usage at the time of violation |

## RateLimiter

`RateLimiter` implements a sliding-window request counter and token-bucket for
outbound LLM calls, plus exponential backoff calculation for 429 retries.

```python
from lauren_ai._cost import RateLimiter

limiter = RateLimiter(
    requests_per_minute=60,
    tokens_per_minute=100_000,
    max_retries=5,
    initial_backoff_s=1.0,
    max_backoff_s=60.0,
    jitter=True,
)

# Before each LLM call:
await limiter.acquire(estimated_tokens=1_500)

# On 429 / Retry-After from provider:
backoff = limiter.backoff_for(attempt=1, retry_after=30.0)
await asyncio.sleep(backoff)
```

`backoff_for(attempt)` returns exponential backoff in seconds; passing
`retry_after` bypasses the exponential calculation and returns that value
directly.

## Error types

| Error | Module | Raised when |
|---|---|---|
| `BudgetExceededError` | `_cost._budget` | `TokenBudget.check()` would exceed a limit |
| `RateLimitExhaustedError` | `_cost._rate` | Max retries exhausted waiting for rate window |

Both inherit from `LaurenAIError`.

# Cost & Rate Tracking

Token budgets, cost estimation, and rate limiting.

## Pricing

### `ModelPricing`

Per-token pricing for a specific model.

All `per_1k` fields express USD cost per **1,000 tokens**.  The
`per_m` aliases expose the same values scaled to per-million tokens
for backward compatibility.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `input_per_1k` | `float` | USD cost per 1 000 input tokens. |
| `output_per_1k` | `float` | USD cost per 1 000 output tokens. |
| `cache_read_per_1k` | `float` | USD cost per 1 000 prompt-cache read tokens. |

### `CostEstimate`

USD cost breakdown for a set of token usages.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `input_usd` | `float` | USD cost for input tokens. |
| `output_usd` | `float` | USD cost for output tokens. |
| `cache_read_usd` | `float` | USD cost for prompt-cache read tokens. |
| `cache_write_usd` | `float` | USD cost for prompt-cache write tokens. |

### `PricingTable`

Mapping of model name to `ModelPricing` for cost estimation.

Usage::

    table = PricingTable(models={
        "claude-haiku-4-5": ModelPricing(input_per_m=0.80, output_per_m=4.00),
    })
    estimate = table.estimate("claude-haiku-4-5", usage)

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `models` | `dict[str, ModelPricing] | None` | Mapping of model identifier to `ModelPricing`. |

### `default_pricing_table`

Return the built-in pricing table with current model prices.

## Cost tracker

### `CostTracker`

Injectable service that accumulates token usage from ModelCallComplete signals.

Usage::

    # Register in module
    @module(providers=[use_class(CostTracker, scope=Scope.SINGLETON)])
    class AppModule: ...

    # In a controller
    async with self.cost.session(conversation_id=cid, user_id=uid) as session:
        result = await self.runner.run(agent, message)
        print(f"Cost: ${session.total_estimate.total_usd:.6f}")

### `CostSession`

Context manager result from CostTracker.session().

### `CostReport`

Aggregated cost report for a user or conversation.

## Budgets & limits

### `TokenBudget`

Per-conversation and per-user token/cost budget limits.

Checked BEFORE each LLM call; raises BudgetExceededError if the
estimated next call would exceed the limit.

Usage::

    budget = TokenBudget(
        max_tokens_per_conversation=50_000,
        max_usd_per_conversation=0.50,
    )
    config = LLMConfig(..., budget=budget)

### `BudgetExceededError`

Raised before an LLM call that would exceed the configured budget.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the exceeded limit. |
| `limit_type` | `str` | Category of limit (e.g. `"tokens_per_conversation"`). |
| `limit` | `float` | The configured budget ceiling. |
| `current` | `float` | The actual usage at the point the budget was exceeded.
Also available as `used` for API compatibility. |

### `RateLimiter`

Token-bucket rate limiter with automatic retry on HTTP 429.

Usage::

    config = LLMConfig(
        model="claude-haiku-4-5",
        rate_limiter=RateLimiter(
            requests_per_minute=60,
            tokens_per_minute=100_000,
            max_retries=5,
        ),
    )

### `RateLimitExhaustedError`

Raised when the rate limiter's `max_retries` is exhausted.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the exhaustion. |
| `limit` | `int` | The configured requests-per-minute limit (`0` when no
per-minute limit is configured). |
| `retry_after` | `float` | Suggested number of seconds to wait before retrying,
if known (`0.0` otherwise). |


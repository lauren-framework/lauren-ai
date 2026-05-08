# Cost & Rate Tracking

Token budgets, cost estimation, and rate limiting.

## Pricing

### `ModelPricing`

Per-token pricing for a specific model.

All ``per_1k`` fields express USD cost per **1,000 tokens**.  The
``per_m`` aliases expose the same values scaled to per-million tokens
for backward compatibility.

:param input_per_1k: USD cost per 1 000 input tokens.
:type input_per_1k: float
:param output_per_1k: USD cost per 1 000 output tokens.
:type output_per_1k: float
:param cache_read_per_1k: USD cost per 1 000 prompt-cache read tokens.
:type cache_read_per_1k: float

### `CostEstimate`

USD cost breakdown for a set of token usages.

:param input_usd: USD cost for input tokens.
:type input_usd: float
:param output_usd: USD cost for output tokens.
:type output_usd: float
:param cache_read_usd: USD cost for prompt-cache read tokens.
:type cache_read_usd: float
:param cache_write_usd: USD cost for prompt-cache write tokens.
:type cache_write_usd: float

### `PricingTable`

Mapping of model name to :class:`ModelPricing` for cost estimation.

Usage::

    table = PricingTable(models={
        "claude-haiku-4-5": ModelPricing(input_per_m=0.80, output_per_m=4.00),
    })
    estimate = table.estimate("claude-haiku-4-5", usage)

:param models: Mapping of model identifier to :class:`ModelPricing`.
:type models: dict[str, ModelPricing] | None

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

:param message: Human-readable description of the exceeded limit.
:type message: str
:param limit_type: Category of limit (e.g. ``"tokens_per_conversation"``).
:type limit_type: str
:param limit: The configured budget ceiling.
:type limit: float
:param current: The actual usage at the point the budget was exceeded.
    Also available as :attr:`used` for API compatibility.
:type current: float

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

Raised when the rate limiter's ``max_retries`` is exhausted.

:param message: Human-readable description of the exhaustion.
:type message: str
:param limit: The configured requests-per-minute limit (``0`` when no
    per-minute limit is configured).
:type limit: int
:param retry_after: Suggested number of seconds to wait before retrying,
    if known (``0.0`` otherwise).
:type retry_after: float


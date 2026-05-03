# Guards Reference

Guards are HTTP-layer gatekeepers applied with `@use_guards()` or `global_guards=` on a Lauren controller or module. The factories below return guard classes that implement `async can_activate(ctx) -> bool`.

---

## `token_budget_guard()`

Returns a guard that rejects requests when a per-key token (and optionally cost) budget is exceeded within a rolling time window. On violation it raises `AgentBudgetExceededError`, which the framework maps to HTTP 429.

```python
from lauren_ai._guards import token_budget_guard

BudgetGuard = token_budget_guard(
    max_tokens_per_hour=100_000,
    max_cost_usd_per_hour=1.00,
    window_seconds=3600,
)

@use_guards(BudgetGuard)
@controller("/ai")
class AIController: ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens_per_hour` | `int` | required | Maximum tokens allowed per key per window. |
| `max_cost_usd_per_hour` | `float \| None` | `None` | Maximum USD cost per key per window. `None` disables cost tracking. |
| `key_fn` | `Callable[[ExecutionContext], str] \| None` | `None` | Extracts the budget key from the execution context. Defaults to the client's IP address. |
| `store` | `BudgetStore \| None` | `None` | Budget state backend. Defaults to `InMemoryBudgetStore`. |
| `window_seconds` | `int` | `3600` | Window duration in seconds. |

### `BudgetStore` protocol

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_usage` | `async (key, *, window_seconds) -> BudgetUsage` | Return current usage for the key within the window. |
| `record_usage` | `async (key, *, tokens, cost_usd, window_seconds) -> None` | Add to usage counters for the key. |

### `InMemoryBudgetStore`

Default in-process implementation. Resets expired windows lazily on `get_usage`. Suitable for single-process deployments; use a Redis-backed store for multi-process setups.

### `BudgetUsage`

| Attribute | Type | Description |
|-----------|------|-------------|
| `tokens` | `int` | Total tokens consumed in the current window. |
| `cost_usd` | `float` | Total cost in USD consumed in the current window. |

Methods: `reset()` — reset counters and start a new window. `is_window_expired(window_seconds)` — return `True` if the window has elapsed.

---

## `requires_capability()`

Returns a guard that rejects requests when the configured LLM model lacks one or more required capabilities. Raises `AgentConfigError` on failure.

```python
from lauren_ai._guards import requires_capability

VisionGuard = requires_capability("vision", "tool_use")

@use_guards(VisionGuard)
@controller("/vision")
class VisionController: ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `*capabilities` | `str` | One or more capability strings to require (see table below). |

Raises `ValueError` when called with no capabilities.

### Known capabilities

| Capability | Supported by |
|------------|-------------|
| `"tool_use"` | claude, claude-haiku, gpt-4o, gpt-4o-mini, o1, o3 |
| `"vision"` | claude, gpt-4o |
| `"streaming"` | claude, claude-haiku, gpt-4o, gpt-4o-mini, llama, gemma |
| `"extended_thinking"` | claude |
| `"reasoning"` | o1, o3 |

Unknown models return an empty capability set and the guard blocks all capabilities.

---

## `safety_guard()`

Returns a guard that screens incoming request content against a `SafetyPolicy`. For simple keyword/regex policies, evaluation is done locally without any LLM call.

```python
from lauren_ai._guards import safety_guard, SafetyPolicy

policy = SafetyPolicy(
    blocked_keywords=["jailbreak", "ignore previous instructions"],
    blocked_patterns=[r"(?i)act as (an? )?unrestricted"],
)

SafeGuard = safety_guard(policy=policy, on_violation="block")

@use_guards(SafeGuard)
@controller("/chat")
class ChatController: ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy` | `SafetyPolicy` | required | The safety policy to apply. |
| `model` | `str` | `"claude-haiku-4-5"` | Model for LLM-based safety screening when applicable. |
| `on_violation` | `Literal["block", "log"]` | `"block"` | `"block"` raises `PermissionError` (HTTP 403); `"log"` allows the request but logs the violation. |

### `SafetyPolicy`

| Constructor parameter | Type | Default | Description |
|----------------------|------|---------|-------------|
| `blocked_keywords` | `list[str] \| None` | `None` | Exact keywords (case-insensitive) that trigger a block. |
| `blocked_patterns` | `list[str] \| None` | `None` | Regex patterns (compiled with `re.IGNORECASE`) that trigger a block. |

#### `is_safe(text) -> bool`

Returns `True` if the text passes all keyword and pattern checks. Override this method to implement custom safety logic.

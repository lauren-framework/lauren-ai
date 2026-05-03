# Signals Reference

`lauren-ai` uses a lightweight async event bus to emit lifecycle events. Signal classes inherit from `lauren.signals.LifecycleEvent`.

---

## `SignalBus`

Standalone async event bus. Does not require the `lauren` framework. Handlers are async callables registered per event type and called concurrently via `asyncio.gather`. Individual handler exceptions are caught, printed to `stderr`, and suppressed so that one failing handler cannot block the others.

```python
from lauren_ai._signals import SignalBus, ModelCallComplete

bus = SignalBus()

@bus.on(ModelCallComplete)
async def log_cost(event: ModelCallComplete) -> None:
    print(f"Cost: ${event.cost_usd:.6f}")

await bus.emit(ModelCallComplete(model="claude-opus-4-6", cost_usd=0.001))
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `on` | `(event_type: type) -> decorator` | Register a handler for `event_type`. Can be used as a decorator. |
| `emit` | `async (event: Any) -> None` | Emit an event to all registered handlers for its type. Handlers run concurrently. |
| `off` | `(event_type: type, handler: Callable) -> None` | Unregister a previously-registered handler. No-op if not registered. |
| `clear` | `(event_type: type \| None = None) -> None` | Remove all handlers; when `event_type` is provided, only removes handlers for that type. |
| `handler_count` | `(event_type: type) -> int` | Return the number of handlers registered for `event_type`. |

---

## Signal types

### `ModelCallStarted`

Emitted immediately before invoking the LLM transport.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `""` | The model identifier that will be called. |
| `agent_id` | `str \| None` | `None` | Unique identifier for the current agent run. |
| `agent_class` | `type \| None` | `None` | The `@agent()`-decorated class. |
| `messages_count` | `int` | `0` | Number of messages in the prompt. |
| `input_tokens_estimate` | `int` | `0` | Rough token estimate for the input messages (`chars / 4`). |

### `ModelCallComplete`

Emitted after a successful LLM completion.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `""` | The model identifier that was called. |
| `agent_id` | `str \| None` | `None` | Unique identifier for the current agent run. |
| `agent_class` | `type \| None` | `None` | The `@agent()`-decorated class. |
| `usage` | `TokenUsage \| None` | `None` | Token usage statistics from the provider. |
| `duration_ms` | `float` | `0.0` | Wall-clock duration of the transport call in milliseconds. |
| `stop_reason` | `str` | `"unknown"` | The stop reason returned by the provider. |
| `cost_usd` | `float` | `0.0` | Estimated cost in USD for this completion. |

### `ToolCallStarted`

Emitted before dispatching a tool call.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_name` | `str` | `""` | Registered name of the tool being called. |
| `tool_use_id` | `str` | `""` | Provider-assigned identifier for this invocation. |
| `agent_id` | `str \| None` | `None` | Unique identifier for the current agent run. |
| `input` | `dict[str, Any]` | `{}` | The parsed input arguments passed to the tool. |
| `cache_hit` | `bool` | `False` | `True` if a cached result is being returned without executing the tool. |

### `ToolCallComplete`

Emitted after a tool call finishes (success or error).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_name` | `str` | `""` | Registered name of the tool. |
| `tool_use_id` | `str` | `""` | Provider-assigned identifier for this invocation. |
| `agent_id` | `str \| None` | `None` | Unique identifier for the current agent run. |
| `duration_ms` | `float` | `0.0` | Wall-clock duration of the tool execution in milliseconds. |
| `success` | `bool` | `True` | `True` if the tool returned a result; `False` if it raised an exception. |
| `error` | `str \| None` | `None` | Human-readable error message when `success=False`. |

### `ToolPendingApproval`

Emitted when a human-in-the-loop confirmation step is required (tool has `requires_confirmation=True`). Handlers should present the details to the user and raise `ToolConfirmationRejectedError` to reject the call.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | `""` | Unique identifier for the current agent run. |
| `agent_run_id` | `str` | `""` | Secondary correlation identifier for the run. |
| `tool_name` | `str` | `""` | The tool name awaiting approval. |
| `tool_use_id` | `str` | `""` | The provider-assigned identifier. |
| `input` | `dict[str, Any]` | `{}` | The tool call arguments awaiting approval. |

### `AgentTurnComplete`

Emitted after each agentic loop iteration (one model call plus any tool calls).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | `""` | Unique identifier for the current agent run. |
| `agent_class` | `type \| None` | `None` | The `@agent()`-decorated class. |
| `turn` | `int` | `0` | The 1-based iteration index that just completed. |
| `turn_usage` | `TokenUsage \| None` | `None` | Token usage for this single turn only. |
| `cumulative_usage` | `TokenUsage \| None` | `None` | Cumulative token usage across all turns so far. |

### `AgentRunComplete`

Emitted when an agent run terminates for any reason.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | `""` | Unique identifier for the completed agent run. |
| `agent_class` | `type \| None` | `None` | The `@agent()`-decorated class. |
| `turns` | `int` | `0` | Number of loop iterations executed. |
| `total_usage` | `TokenUsage \| None` | `None` | Cumulative token usage across the entire run. |
| `total_cost_usd` | `float` | `0.0` | Estimated total cost in USD for the run. |
| `stop_reason` | `str` | `"unknown"` | Why the agent loop terminated (e.g. `"end_turn"`, `"max_turns"`, `"budget_exceeded"`). |

### `EmbeddingGenerated`

Emitted after an embedding batch is generated.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `""` | The embedding model identifier. |
| `input_count` | `int` | `0` | Number of input strings that were embedded. |
| `duration_ms` | `float` | `0.0` | Wall-clock duration of the embedding call in milliseconds. |

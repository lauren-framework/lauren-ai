# Signals

Observable lifecycle events emitted by the agent runner.

### `SignalBus`

```python
class SignalBus()
```

Lightweight standalone async event bus.

Handlers are async callables that accept a single event argument.  They
are registered per event type and called concurrently when an event is
emitted.  Exceptions raised by individual handlers are caught, printed to
`stderr`, and suppressed so that one failing handler cannot block the
others or the caller.

This class does **not** require the `lauren` framework.

Example::

    bus = SignalBus()

    @bus.on(ModelCallComplete)
    async def log_cost(event: ModelCallComplete) -> None:
        print(f"Cost: ${event.cost_usd:.6f}")

    await bus.emit(ModelCallComplete(model="claude-opus-4-6", cost_usd=0.001))

#### `SignalBus.on`

```python
def on(self, event_type: type) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]
```

Register a handler for *event_type*.

Can be used as a decorator::

    @bus.on(ModelCallComplete)
    async def handle(event: ModelCallComplete) -> None: ...

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `event_type` | `type` | The event class to subscribe to. |

**Returns:** `Callable` — A decorator that registers the handler and returns it
unchanged.

#### `SignalBus.clear`

```python
def clear(self, event_type: type | None = None) -> None
```

Remove all handlers, optionally scoped to a specific *event_type*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `event_type` | `type | None` | When provided, only handlers for this event type
are removed.  When `None`, all handlers across all types are
cleared. |

#### `SignalBus.emit`

```python
def emit(self, event: Any) -> None
```

Emit *event* to all registered handlers for its type.

Handlers are called concurrently via `asyncio.gather()`.
Individual handler exceptions are caught, printed to `stderr`,
and suppressed.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `event` | `Any` | The event instance to emit. |

#### `SignalBus.off`

```python
def off(self, event_type: type, handler: Callable[..., Awaitable[None]]) -> None
```

Unregister a previously-registered handler.

A no-op if *handler* is not registered for *event_type*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `event_type` | `type` | The event type the handler was registered for. |
| `handler` | `Callable[..., Awaitable[None]]` | The handler to unregister. |

#### `SignalBus.handler_count`

```python
def handler_count(self, event_type: type) -> int
```

Return the number of handlers registered for *event_type*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `event_type` | `type` | The event type to query. |

**Returns:** `int` — Number of registered handlers.

## Event types

### `ModelCallStarted`

```python
class ModelCallStarted(model: str = '', agent_id: str | None = None, agent_class: type | None = None, agent_name: str = '', messages_count: int = 0, input_tokens_estimate: int = 0)
```

Emitted immediately before invoking the LLM transport.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `str` | The model identifier that will be called. |
| `agent_id` | `str | None` | Unique identifier for the current agent run, or
`None` when the call originates outside an agent context. |
| `agent_class` | `type | None` | The `@agent()`-decorated class, or `None`. |
| `agent_name` | `str` | Human-readable agent name from `AgentMeta.name`,
or the class `__name__` when not explicitly set.  Empty string when
the call originates outside an agent context. |
| `messages_count` | `int` | Number of messages in the prompt. |
| `input_tokens_estimate` | `int` | Rough token estimate for the input messages
(4 chars ≈ 1 token heuristic). |

### `ModelCallComplete`

```python
class ModelCallComplete(model: str = '', agent_id: str | None = None, agent_class: type | None = None, agent_name: str = '', usage: Any = None, duration_ms: float = 0.0, stop_reason: str = 'unknown', cost_usd: float = 0.0)
```

Emitted after a successful LLM completion.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `str` | The model identifier that was called. |
| `agent_id` | `str | None` | Unique identifier for the current agent run, or `None`. |
| `agent_class` | `type | None` | The `@agent()`-decorated class, or `None`. |
| `agent_name` | `str` | Human-readable agent name from `AgentMeta.name`,
or the class `__name__` when not explicitly set.  Empty string when
the call originates outside an agent context. |
| `usage` | `Any` | Token usage statistics from the provider. |
| `duration_ms` | `float` | Wall-clock duration of the transport call in
milliseconds. |
| `stop_reason` | `str` | The stop reason returned by the provider. |
| `cost_usd` | `float` | Estimated cost in USD for this completion. |

### `ToolCallStarted`

```python
class ToolCallStarted(tool_name: str = '', tool_use_id: str = '', agent_id: str | None = None, input: dict[str, Any] = dict(), cache_hit: bool = False)
```

Emitted before dispatching a tool call.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `tool_name` | `str` | Registered name of the tool being called. |
| `tool_use_id` | `str` | Provider-assigned identifier for this invocation. |
| `agent_id` | `str | None` | Unique identifier for the current agent run, or `None`. |
| `input` | `dict[str, Any]` | The parsed input arguments passed to the tool. |
| `cache_hit` | `bool` | `True` if a cached result is being returned without
executing the tool. |

### `ToolCallComplete`

```python
class ToolCallComplete(tool_name: str = '', tool_use_id: str = '', agent_id: str | None = None, duration_ms: float = 0.0, success: bool = True, error: str | None = None)
```

Emitted after a tool call finishes (success or error).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `tool_name` | `str` | Registered name of the tool. |
| `tool_use_id` | `str` | Provider-assigned identifier for this invocation. |
| `agent_id` | `str | None` | Unique identifier for the current agent run, or `None`. |
| `duration_ms` | `float` | Wall-clock duration of the tool execution in
milliseconds. |
| `success` | `bool` | `True` if the tool returned a result; `False` if it
raised an exception. |
| `error` | `str | None` | Human-readable error message when `success=False`.
`None` when `success=True`. |

### `AgentRunComplete`

```python
class AgentRunComplete(agent_id: str = '', agent_class: type | None = None, agent_name: str = '', turns: int = 0, total_usage: Any = None, total_cost_usd: float = 0.0, stop_reason: str = 'unknown')
```

Emitted when an agent run terminates (for any reason).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent_id` | `str` | Unique identifier for the completed agent run. |
| `agent_class` | `type | None` | The `@agent()`-decorated class. |
| `agent_name` | `str` | Human-readable agent name from `AgentMeta.name`,
or the class `__name__` when not explicitly set. |
| `turns` | `int` | Number of loop iterations that were executed. |
| `total_usage` | `Any` | Cumulative token usage across the entire run. |
| `total_cost_usd` | `float` | Estimated total cost in USD for the run. |
| `stop_reason` | `str` | Why the agent loop terminated (e.g. `"end_turn"`,
`"max_turns"`, `"budget_exceeded"`). |

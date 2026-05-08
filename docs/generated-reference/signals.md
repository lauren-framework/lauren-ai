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
``stderr``, and suppressed so that one failing handler cannot block the
others or the caller.

This class does **not** require the ``lauren`` framework.

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

:param event_type: The event class to subscribe to.
:type event_type: type
:return: A decorator that registers the handler and returns it
    unchanged.
:rtype: Callable

#### `SignalBus.clear`

```python
def clear(self, event_type: type | None = None) -> None
```

Remove all handlers, optionally scoped to a specific *event_type*.

:param event_type: When provided, only handlers for this event type
    are removed.  When ``None``, all handlers across all types are
    cleared.
:type event_type: type | None

#### `SignalBus.emit`

```python
def emit(self, event: Any) -> None
```

Emit *event* to all registered handlers for its type.

Handlers are called concurrently via :func:`asyncio.gather`.
Individual handler exceptions are caught, printed to ``stderr``,
and suppressed.

:param event: The event instance to emit.
:type event: Any

#### `SignalBus.off`

```python
def off(self, event_type: type, handler: Callable[..., Awaitable[None]]) -> None
```

Unregister a previously-registered handler.

A no-op if *handler* is not registered for *event_type*.

:param event_type: The event type the handler was registered for.
:type event_type: type
:param handler: The handler to unregister.
:type handler: Callable

#### `SignalBus.handler_count`

```python
def handler_count(self, event_type: type) -> int
```

Return the number of handlers registered for *event_type*.

:param event_type: The event type to query.
:type event_type: type
:return: Number of registered handlers.
:rtype: int

## Event types

### `ModelCallStarted`

```python
class ModelCallStarted(model: str = '', agent_id: str | None = None, agent_class: type | None = None, agent_name: str = '', messages_count: int = 0, input_tokens_estimate: int = 0)
```

Emitted immediately before invoking the LLM transport.

:param model: The model identifier that will be called.
:type model: str
:param agent_id: Unique identifier for the current agent run, or
    ``None`` when the call originates outside an agent context.
:type agent_id: str | None
:param agent_class: The ``@agent()``-decorated class, or ``None``.
:type agent_class: type | None
:param agent_name: Human-readable agent name from :attr:`AgentMeta.name`,
    or the class ``__name__`` when not explicitly set.  Empty string when
    the call originates outside an agent context.
:type agent_name: str
:param messages_count: Number of messages in the prompt.
:type messages_count: int
:param input_tokens_estimate: Rough token estimate for the input messages
    (4 chars ≈ 1 token heuristic).
:type input_tokens_estimate: int

### `ModelCallComplete`

```python
class ModelCallComplete(model: str = '', agent_id: str | None = None, agent_class: type | None = None, agent_name: str = '', usage: Any = None, duration_ms: float = 0.0, stop_reason: str = 'unknown', cost_usd: float = 0.0)
```

Emitted after a successful LLM completion.

:param model: The model identifier that was called.
:type model: str
:param agent_id: Unique identifier for the current agent run, or ``None``.
:type agent_id: str | None
:param agent_class: The ``@agent()``-decorated class, or ``None``.
:type agent_class: type | None
:param agent_name: Human-readable agent name from :attr:`AgentMeta.name`,
    or the class ``__name__`` when not explicitly set.  Empty string when
    the call originates outside an agent context.
:type agent_name: str
:param usage: Token usage statistics from the provider.
:type usage: Any
:param duration_ms: Wall-clock duration of the transport call in
    milliseconds.
:type duration_ms: float
:param stop_reason: The stop reason returned by the provider.
:type stop_reason: str
:param cost_usd: Estimated cost in USD for this completion.
:type cost_usd: float

### `ToolCallStarted`

```python
class ToolCallStarted(tool_name: str = '', tool_use_id: str = '', agent_id: str | None = None, input: dict[str, Any] = dict(), cache_hit: bool = False)
```

Emitted before dispatching a tool call.

:param tool_name: Registered name of the tool being called.
:type tool_name: str
:param tool_use_id: Provider-assigned identifier for this invocation.
:type tool_use_id: str
:param agent_id: Unique identifier for the current agent run, or ``None``.
:type agent_id: str | None
:param input: The parsed input arguments passed to the tool.
:type input: dict[str, Any]
:param cache_hit: ``True`` if a cached result is being returned without
    executing the tool.
:type cache_hit: bool

### `ToolCallComplete`

```python
class ToolCallComplete(tool_name: str = '', tool_use_id: str = '', agent_id: str | None = None, duration_ms: float = 0.0, success: bool = True, error: str | None = None)
```

Emitted after a tool call finishes (success or error).

:param tool_name: Registered name of the tool.
:type tool_name: str
:param tool_use_id: Provider-assigned identifier for this invocation.
:type tool_use_id: str
:param agent_id: Unique identifier for the current agent run, or ``None``.
:type agent_id: str | None
:param duration_ms: Wall-clock duration of the tool execution in
    milliseconds.
:type duration_ms: float
:param success: ``True`` if the tool returned a result; ``False`` if it
    raised an exception.
:type success: bool
:param error: Human-readable error message when ``success=False``.
    ``None`` when ``success=True``.
:type error: str | None

### `AgentRunComplete`

```python
class AgentRunComplete(agent_id: str = '', agent_class: type | None = None, agent_name: str = '', turns: int = 0, total_usage: Any = None, total_cost_usd: float = 0.0, stop_reason: str = 'unknown')
```

Emitted when an agent run terminates (for any reason).

:param agent_id: Unique identifier for the completed agent run.
:type agent_id: str
:param agent_class: The ``@agent()``-decorated class.
:type agent_class: type
:param agent_name: Human-readable agent name from :attr:`AgentMeta.name`,
    or the class ``__name__`` when not explicitly set.
:type agent_name: str
:param turns: Number of loop iterations that were executed.
:type turns: int
:param total_usage: Cumulative token usage across the entire run.
:type total_usage: Any
:param total_cost_usd: Estimated total cost in USD for the run.
:type total_cost_usd: float
:param stop_reason: Why the agent loop terminated (e.g. ``"end_turn"``,
    ``"max_turns"``, ``"budget_exceeded"``).
:type stop_reason: str


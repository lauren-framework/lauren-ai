# Exceptions

All exception classes raised by `lauren-ai`.

## Base

### `LaurenAIError`

```python
class LaurenAIError(message: str, cause: BaseException | None = None)
```

Base class for all ``lauren-ai`` exceptions.

:param message: Human-readable description of what went wrong.
:type message: str
:param cause: The underlying exception that caused this error, if any.
:type cause: BaseException | None

## Transport errors

### `TransportError`

```python
class TransportError(message: str, status_code: int | None = None, provider: str | None = None, cause: BaseException | None = None)
```

Raised when an LLM provider returns or raises any error.

:param message: Human-readable description of the transport failure.
:type message: str
:param status_code: HTTP status code returned by the provider, if applicable.
:type status_code: int | None
:param provider: Name of the provider (e.g. ``"anthropic"``).
:type provider: str | None
:param cause: The underlying exception from the provider SDK.
:type cause: BaseException | None

## Agent errors

### `AgentMaxTurnsError`

```python
class AgentMaxTurnsError(message: str, turns: int, agent_class: type | None = None)
```

Raised when an agent exceeds its configured ``max_turns`` limit.

:param message: Human-readable description of the limit exceeded.
:type message: str
:param turns: The number of turns that were executed before stopping.
:type turns: int
:param agent_class: The agent class that exceeded the limit.
:type agent_class: type | None

### `AgentBudgetExceededError`

```python
class AgentBudgetExceededError(message: str, budget_type: str, limit: float, used: float, agent_class: type | None = None)
```

Raised mid-run when an agent crosses its ``max_cost_usd`` or token budget.

:param message: Human-readable description of the budget exceeded.
:type message: str
:param budget_type: Either ``"cost_usd"`` or ``"tokens"``.
:type budget_type: str
:param limit: The configured budget limit.
:type limit: float
:param used: The actual amount used when the budget was exceeded.
:type used: float
:param agent_class: The agent class that exceeded the budget.
:type agent_class: type | None

### `AgentConfigError`

```python
class AgentConfigError(message: str, agent_class: type | None = None, cause: BaseException | None = None)
```

Raised at startup when an ``@agent()`` decorator is misconfigured.

:param message: Human-readable description of the configuration error.
:type message: str
:param agent_class: The offending agent class, if known.
:type agent_class: type | None
:param cause: The underlying exception.
:type cause: BaseException | None

## Tool errors

### `ToolExecutionError`

```python
class ToolExecutionError(message: str, tool_name: str, tool_use_id: str, cause: BaseException | None = None)
```

Raised when a tool raises an unexpected exception during execution.

:param message: Human-readable description of the failure.
:type message: str
:param tool_name: The registered name of the tool that failed.
:type tool_name: str
:param tool_use_id: The provider-assigned identifier for this tool call.
:type tool_use_id: str
:param cause: The original exception raised by the tool.
:type cause: BaseException | None

## Decorator errors

### `DecoratorUsageError`

```python
class DecoratorUsageError(message: str, decorator_name: str | None = None)
```

Raised when a decorator is used incorrectly, e.g. bare ``@tool`` without parentheses.

:param message: Human-readable description of the misuse.
:type message: str
:param decorator_name: The name of the decorator that was misused.
:type decorator_name: str | None

## Parser errors

### `OutputParserError`

Raised when an output parser fails to parse LLM text.

:param message: Human-readable description of the parse failure.
:type message: str
:param raw_output: The raw LLM output that could not be parsed.
:type raw_output: str | None
:param cause: The underlying exception.
:type cause: BaseException | None

## Memory errors

### `MemoryConfigError`

Raised when memory configuration is invalid or user_id is missing.

## Tracing errors

### `TracingError`

```python
class TracingError
```

Base class for tracing and observability errors.

Raised when the tracing subsystem encounters an unrecoverable condition,
such as a misconfigured exporter or a failed export operation.

:param message: Human-readable description of the failure.
:type message: str
:param cause: The underlying exception.
:type cause: BaseException | None


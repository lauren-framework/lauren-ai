# Exceptions

All exception classes raised by `lauren-ai`.

## Base

### `LaurenAIError`

```python
class LaurenAIError(message: str, cause: BaseException | None = None)
```

Base class for all `lauren-ai` exceptions.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of what went wrong. |
| `cause` | `BaseException | None` | The underlying exception that caused this error, if any. |

## Transport errors

### `TransportError`

```python
class TransportError(message: str, status_code: int | None = None, provider: str | None = None, cause: BaseException | None = None)
```

Raised when an LLM provider returns or raises any error.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the transport failure. |
| `status_code` | `int | None` | HTTP status code returned by the provider, if applicable. |
| `provider` | `str | None` | Name of the provider (e.g. `"anthropic"`). |
| `cause` | `BaseException | None` | The underlying exception from the provider SDK. |

## Agent errors

### `AgentMaxTurnsError`

```python
class AgentMaxTurnsError(message: str, turns: int, agent_class: type | None = None)
```

Raised when an agent exceeds its configured `max_turns` limit.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the limit exceeded. |
| `turns` | `int` | The number of turns that were executed before stopping. |
| `agent_class` | `type | None` | The agent class that exceeded the limit. |

### `AgentBudgetExceededError`

```python
class AgentBudgetExceededError(message: str, budget_type: str, limit: float, used: float, agent_class: type | None = None)
```

Raised mid-run when an agent crosses its `max_cost_usd` or token budget.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the budget exceeded. |
| `budget_type` | `str` | Either `"cost_usd"` or `"tokens"`. |
| `limit` | `float` | The configured budget limit. |
| `used` | `float` | The actual amount used when the budget was exceeded. |
| `agent_class` | `type | None` | The agent class that exceeded the budget. |

### `AgentConfigError`

```python
class AgentConfigError(message: str, agent_class: type | None = None, cause: BaseException | None = None)
```

Raised at startup when an `@agent()` decorator is misconfigured.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the configuration error. |
| `agent_class` | `type | None` | The offending agent class, if known. |
| `cause` | `BaseException | None` | The underlying exception. |

## Tool errors

### `ToolExecutionError`

```python
class ToolExecutionError(message: str, tool_name: str, tool_use_id: str, cause: BaseException | None = None)
```

Raised when a tool raises an unexpected exception during execution.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the failure. |
| `tool_name` | `str` | The registered name of the tool that failed. |
| `tool_use_id` | `str` | The provider-assigned identifier for this tool call. |
| `cause` | `BaseException | None` | The original exception raised by the tool. |

## Decorator errors

### `DecoratorUsageError`

```python
class DecoratorUsageError(message: str, decorator_name: str | None = None)
```

Raised when a decorator is used incorrectly, e.g. bare `@tool` without parentheses.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the misuse. |
| `decorator_name` | `str | None` | The name of the decorator that was misused. |

## Parser errors

### `OutputParserError`

```python
class OutputParserError(message: str, raw_output: str | None = None, cause: BaseException | None = None)
```

Raised when an output parser fails to parse LLM text.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the parse failure. |
| `raw_output` | `str | None` | The raw LLM output that could not be parsed. |
| `cause` | `BaseException | None` | The underlying exception. |

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

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the failure. |
| `cause` | `BaseException | None` | The underlying exception. |

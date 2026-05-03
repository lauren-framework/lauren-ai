# Exceptions Reference

All exceptions in `lauren-ai` inherit from `LaurenAIError`, which itself inherits from `Exception`.

## Exception hierarchy

```
LaurenAIError
├── TransportError
│   ├── TransientTransportError
│   └── AuthTransportError
├── ToolExecutionError
├── ToolSchemaError
├── ToolConfigError
├── AgentMaxTurnsError
├── AgentBudgetExceededError
├── AgentConfigError
├── DecoratorUsageError
├── DelegateToAgent
├── EmptyQueueError
├── ToolConfirmationRejectedError
├── KnowledgeLoadError
├── WorkflowError
├── OutputParserError
├── EvalError
└── TracingError
```

---

## Base

### `LaurenAIError`

Base class for all `lauren-ai` exceptions.

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable description of what went wrong. |
| `cause` | `BaseException \| None` | The underlying exception that caused this error, if any. |

`str()` returns `"{message} (caused by: {cause!r})"` when `cause` is set.

---

## Transport

### `TransportError(LaurenAIError)`

Raised when an LLM provider returns or raises any error.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `status_code` | `int \| None` | HTTP status code returned by the provider. |
| `provider` | `str \| None` | Name of the provider (e.g. `"anthropic"`). |

### `TransientTransportError(TransportError)`

Raised for retryable failures: rate limits (HTTP 429) and server errors (5xx). The `LLMConfig.max_retries` setting controls automatic retry behaviour before this exception is re-raised.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `retry_after` | `float \| None` | Seconds to wait before retrying, from the `Retry-After` header. |

### `AuthTransportError(TransportError)`

Raised for authentication / authorisation failures (HTTP 401 or 403). Never retried — an invalid API key will not become valid on a subsequent attempt.

---

## Tool

### `ToolExecutionError(LaurenAIError)`

Raised when a tool raises an unexpected exception during execution.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `tool_name` | `str` | The registered name of the tool that failed. |
| `tool_use_id` | `str` | The provider-assigned identifier for the failing tool call. |

`str()` returns `"Tool {name!r} (id={id!r}): {message}"`.

### `ToolSchemaError(LaurenAIError)`

Raised at startup when a tool's JSON schema cannot be generated (e.g. an unannotated parameter or an unsupported type annotation).

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `tool_name` | `str \| None` | The name of the tool with the bad schema. |
| `parameter` | `str \| None` | The name of the offending parameter. |

### `ToolConfigError(LaurenAIError)`

Raised at startup when a `@tool()` decorator is misconfigured.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `tool_name` | `str \| None` | The name of the offending tool. |

---

## Agent

### `AgentMaxTurnsError(LaurenAIError)`

Raised when an agent exceeds its configured `max_turns` limit. The `AgentResponse.stop_reason` is set to `"max_turns"` rather than this exception propagating in most cases.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `turns` | `int` | Number of turns executed before stopping. |
| `agent_class` | `type \| None` | The agent class that exceeded the limit. |

### `AgentBudgetExceededError(LaurenAIError)`

Raised mid-run when an agent crosses its `max_cost_usd` or token budget. The `AgentResponse.stop_reason` is set to `"budget_exceeded"`.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `budget_type` | `str` | Either `"cost_usd"` or `"tokens"`. |
| `limit` | `float` | The configured budget limit. |
| `used` | `float` | The actual amount consumed when the budget was exceeded. |
| `agent_class` | `type \| None` | The agent class that exceeded the budget. |

### `AgentConfigError(LaurenAIError)`

Raised at startup when an `@agent()` decorator is misconfigured.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `agent_class` | `type \| None` | The offending agent class. |

---

## Decorator

### `DecoratorUsageError(LaurenAIError)`

Raised when a decorator is used incorrectly, most commonly bare `@tool` or `@agent` without parentheses.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `decorator_name` | `str \| None` | The name of the misused decorator. |

`str()` returns `"Decorator @{name} misuse: {message}"`.

---

## Multi-agent handoff

### `DelegateToAgent(LaurenAIError)`

Internal control-flow exception used to request a multi-agent handoff. Not a fatal error — the runner catches it and performs the delegation.

Raised by `AgentContext.delegate()`. Never needs to be caught in application code.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `agent` | `Any` | The agent instance or class to delegate to. |
| `message` | `str` | The message to pass to the target agent. |

---

## Testing

### `EmptyQueueError(LaurenAIError)`

Raised by `MockTransport` when the response queue is exhausted but another `complete()` call is made.

Default message: `"MockTransport response queue is empty"`.

---

## Human-in-the-loop

### `ToolConfirmationRejectedError(LaurenAIError)`

Raised when a human-in-the-loop confirmation request is rejected (the tool call is blocked by a human reviewer).

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `tool_name` | `str` | The name of the tool whose call was rejected. |
| `tool_use_id` | `str` | The provider-assigned identifier for the rejected call. |
| `reason` | `str` | The human-provided reason for rejecting the call (may be empty). |

---

## Knowledge / Workflow

### `KnowledgeLoadError(LaurenAIError)`

Raised when a knowledge base fails to load or initialise.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `source` | `str \| None` | The knowledge source path or identifier. |

### `WorkflowError(LaurenAIError)`

Raised when a workflow step fails.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `step_name` | `str \| None` | The name of the workflow step that failed. |

---

## Output parsing

### `OutputParserError(LaurenAIError)`

Raised when an output parser fails to parse LLM text.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `raw_output` | `str \| None` | The raw LLM output that could not be parsed. |

`str()` appends `"| raw_output={snippet!r}"` (first 80 characters) when `raw_output` is set.

---

## Evaluation

### `EvalError(LaurenAIError)`

Raised when an evaluation framework operation fails.

| Extra attribute | Type | Description |
|-----------------|------|-------------|
| `eval_name` | `str \| None` | The name of the evaluation that failed. |

---

## Tracing

### `TracingError(LaurenAIError)`

Raised when the tracing subsystem encounters an unrecoverable condition, such as a misconfigured exporter or a failed export operation.

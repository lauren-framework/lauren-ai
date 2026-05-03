# Interceptors Reference

Interceptor factories from `lauren_ai._interceptors`. Each factory returns a `@interceptor()`-decorated class ready for `global_interceptors=` or `@use_interceptors()` on a Lauren controller or module.

---

## `ai_metrics_interceptor()`

Returns an interceptor that captures AI token usage and wall-clock timing for each handler invocation.

After the handler completes, the interceptor:

1. Records the elapsed time as `request.state.ai_duration_ms` (milliseconds as `float`).
2. Reads `request.state.ai_token_usage` (populated by `AgentRunner` when the handler invokes an agent) and emits a `ModelCallComplete` signal to the `SignalBus` if one is present on the execution context.

```python
from lauren_ai._interceptors import ai_metrics_interceptor

@module(
    controllers=[AIController],
    global_interceptors=[ai_metrics_interceptor()],
)
class AppModule: ...
```

No constructor parameters.

### `request.state` attributes written

| Attribute | Type | Description |
|-----------|------|-------------|
| `ai_duration_ms` | `float` | Wall-clock time of the handler in milliseconds. |

### Signal emitted

`ModelCallComplete` — emitted to the `SignalBus` attached to the execution context when `request.state.ai_token_usage` is present.

---

## `token_usage_response_interceptor()`

Returns an interceptor that appends token-usage HTTP headers to the response after the handler completes.

Reads `request.state.ai_token_usage` (a `TokenUsage`-compatible object). When usage data is available and the response object supports `headers`, two headers are appended:

| Header | Value | Example |
|--------|-------|---------|
| `x-token-usage` | Total tokens (`input_tokens + output_tokens`) as a string | `"1234"` |
| `x-ai-cost-usd` | Estimated cost formatted to 4 decimal places | `"0.0012"` |

```python
from lauren_ai._interceptors import token_usage_response_interceptor

@module(
    controllers=[AIController],
    global_interceptors=[token_usage_response_interceptor()],
)
class AppModule: ...
```

No constructor parameters.

### Combining both interceptors

```python
@module(
    controllers=[AIController],
    global_interceptors=[
        ai_metrics_interceptor(),
        token_usage_response_interceptor(),
    ],
)
class AppModule: ...
```

The interceptors are independent and can be applied in any order or combined.

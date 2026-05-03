# Middleware Reference

Middleware factories from `lauren_ai._middleware`. Each factory returns a `@middleware()`-decorated class ready for `global_middlewares=` or `@use_middlewares()` on a Lauren module.

---

## `conversation_middleware()`

Returns middleware that manages conversation history across HTTP requests.

On each request it:
1. Reads a conversation ID from a configured header (default `x-conversation-id`) or cookie (default `conversation_id`).
2. Loads the conversation history from the provided `ConversationStore`.
3. Attaches the ID and history to `request.state`.
4. After the handler completes, saves the updated history back to the store (reads `request.state.updated_conversation`).

```python
from lauren_ai._middleware import conversation_middleware
from lauren_ai import InMemoryConversationStore

store = InMemoryConversationStore()

ConvMiddleware = conversation_middleware(
    store,
    header="x-conversation-id",
    cookie="conversation_id",
    auto_create=True,
)

@module(
    controllers=[ChatController],
    global_middlewares=[ConvMiddleware],
)
class AppModule: ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `ConversationStore` | required | A `ConversationStore` implementation to load/save history. |
| `header` | `str` | `"x-conversation-id"` | HTTP header name to read the conversation ID from. |
| `cookie` | `str \| None` | `"conversation_id"` | Cookie name to fall back to when the header is absent. `None` disables cookie lookup. |
| `auto_create` | `bool` | `True` | When `True`, generates a new UUID conversation ID when none is found in the request. |

### `request.state` attributes written

| Attribute | Type | Description |
|-----------|------|-------------|
| `conversation_id` | `str` | The resolved or newly generated conversation ID. |
| `conversation_history` | `list[Message]` | The loaded conversation history (empty list for new conversations). |

To persist updated conversation history, assign the new message list to `request.state.updated_conversation` in your handler. The middleware saves it after the response.

Store errors during load cause the middleware to start a fresh conversation (fail-safe). Store errors during save are silently suppressed to avoid failing the response.

---

## `ai_rate_limit()`

Returns middleware that rate-limits AI endpoints on a per-key basis. At least one limit parameter must be specified; a `ValueError` is raised otherwise.

On limit breach the middleware returns an HTTP 429 JSON response:
```json
{"error": "rate_limit_exceeded", "limit": "requests_per_minute"}
```

```python
from lauren_ai._middleware import ai_rate_limit

RateLimitMiddleware = ai_rate_limit(
    tokens_per_minute=50_000,
    requests_per_minute=60,
    requests_per_day=1000,
)

@module(
    controllers=[AIController],
    global_middlewares=[RateLimitMiddleware],
)
class AppModule: ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokens_per_minute` | `int \| None` | `None` | Maximum tokens per minute per key. |
| `requests_per_minute` | `int \| None` | `None` | Maximum requests per minute per key. |
| `requests_per_day` | `int \| None` | `None` | Maximum requests per day per key. |
| `key_fn` | `Callable[[Request], str] \| None` | `None` | Extracts the rate-limit key from the request. Defaults to client IP address; falls back to `"anonymous"`. |
| `store` | `RateLimitStore \| None` | `None` | Rate limit state backend. Defaults to `InMemoryRateLimitStore`. |

### `RateLimitStore` protocol

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_usage` | `async (key, *, window_seconds) -> BudgetUsage` | Return current usage for the key within the window. Resets the window if expired. |
| `increment` | `async (key, *, tokens=0, window_seconds) -> BudgetUsage` | Increment usage counters and return the updated snapshot. |

### `InMemoryRateLimitStore`

Default in-process implementation using sliding windows. State is lost on restart. Use a Redis-backed store for multi-process or distributed deployments.

### `BudgetUsage` (middleware)

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | `int` | Total tokens consumed in the current window. |
| `requests` | `int` | Total requests made in the current window. |
| `window_start` | `float` | `time.monotonic()` timestamp when the current window started. |

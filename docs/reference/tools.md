# Tools Reference

## `@tool()` decorator

Marks a function or class as a tool for AI agents. Must be called **with parentheses** — bare `@tool` raises `DecoratorUsageError`.

Two forms are supported:

**Function-form** (simple, stateless tools):

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
# The @tool() decorator uses inspect.signature() at decoration time to build
# the JSON schema, and PEP 563 lazy evaluation breaks that introspection.
from lauren_ai import tool, ToolContext

@tool()
async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for current information.

    Args:
        query: The search query string.
        max_results: Maximum results to return (1-20).
    """
    ...
```

**Class-form** (stateful tools with DI constructor dependencies):

```python
@tool()
@injectable(scope=Scope.SINGLETON)
class DatabaseTool:
    """Query the application database."""

    def __init__(self, db: DatabaseService) -> None:
        self._db = db

    async def run(self, sql: str) -> list[dict]:
        """Run a SQL query.

        Args:
            sql: The SQL query to execute.
        """
        return await self._db.execute(sql)
```

Class-form tools automatically receive `@injectable(scope=Scope.SINGLETON)` unless already marked.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | `None` | Override the inferred tool name (default: function/class name in snake_case). |
| `description` | `str \| None` | `None` | Override the description extracted from the docstring. |
| `requires_confirmation` | `bool` | `False` | When `True`, emits `ToolPendingApproval` before invoking the tool (human-in-the-loop). |
| `pre_hook` | `Callable \| None` | `None` | Callable invoked before the tool runs. Receives `(tool_call, ToolContext)`. |
| `post_hook` | `Callable \| None` | `None` | Callable invoked after a successful run. Receives `(ToolResult, ToolContext)`. |
| `error_hook` | `Callable \| None` | `None` | Callable invoked when the tool raises an exception. Receives `(exception, ToolContext)`. |
| `cache_ttl` | `int \| None` | `None` | Cache successful results for this many seconds. Requires a `CacheBackend`. |
| `cache_key_fn` | `Callable \| None` | `None` | Custom factory that derives a string cache key from the input `dict`. |

---

## `ToolMeta`

Metadata stored on the decorated function or class under the `TOOL_META` attribute (`"__lauren_ai_tool__"`). Built at decoration time from the function signature and docstring.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tool name in snake_case. |
| `description` | `str` | Human-readable description extracted from the docstring first paragraph. |
| `parameters` | `ToolSchema` | JSON Schema for the tool's input parameters. |
| `is_async` | `bool` | `True` if the entry-point function/method is a coroutine. |
| `reads_context` | `bool` | `True` when the entry point declares a `ctx: ToolContext` parameter. |
| `context_param_name` | `str \| None` | Name of the `ToolContext` parameter (may be anything, e.g. `ctx`, `agent_ctx`). |
| `requires_confirmation` | `bool` | HITL gate flag. |
| `pre_hook` | `Callable \| None` | Pre-execution hook. |
| `post_hook` | `Callable \| None` | Post-execution hook. |
| `error_hook` | `Callable \| None` | Error hook. |
| `cache_ttl` | `int \| None` | Cache TTL in seconds. |
| `cache_key_fn` | `Callable \| None` | Custom cache-key factory. |

---

## `TOOL_META`

String sentinel `"__lauren_ai_tool__"` — the attribute name that `@tool()` sets on the decorated function or class to store `ToolMeta`.

---

## `ToolContext`

Injected into a tool function when its signature declares a `ctx: ToolContext` parameter. The context parameter may have any name; the `@tool()` decorator detects it by type annotation.

```python
@tool()
async def my_tool(query: str, ctx: ToolContext) -> str:
    user = ctx.get_metadata("user_id")
    ...
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `agent_context` | `AgentContext` | The owning agent's `AgentContext`. |
| `tool_use_id` | `str` | Provider-assigned tool use identifier for this invocation. |
| `turn` | `int` | Which agentic loop iteration (0-based) triggered this call. |
| `request` | `Any \| None` | Originating HTTP request, or `None` outside a web handler. |
| `execution_context` | `Any \| None` | Lauren `ExecutionContext`, or `None`. |
| `state` | `dict[str, Any]` | Mutable per-call state bag for tool-local storage. |

### Methods

#### `get_metadata(key, default=None)`

Delegates to `agent_context.get_metadata(key, default)`. Returns `default` when the agent context does not support `get_metadata`.

---

## `ToolResult`

The result of executing a single tool call, fed back into the conversation history.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_use_id` | `str` | Provider-assigned identifier echoed back to correlate with the model's request. |
| `content` | `str \| list[Any]` | Tool output. Either a plain string or a list of content blocks. |
| `is_error` | `bool` | `True` when the tool raised an exception or returned an explicit error. |

### Classmethods

#### `ToolResult.ok(content, *, tool_use_id)`

Create a successful `ToolResult`. Non-string content is serialised to JSON automatically. Lists of content blocks are passed through unchanged.

#### `ToolResult.error(message, *, tool_use_id)`

Create an error `ToolResult` with `is_error=True`.

---

## `ToolRegistry`

Collects all registered tools for a module graph at startup. Provides O(1) lookup by name.

```python
from lauren_ai._tools._registry import ToolRegistry

registry = ToolRegistry()
registry.register(search_web)
result = registry.get("search_web")       # -> (callable, ToolMeta) | None
schemas = registry.get_schemas(["search_web"])  # -> list[ToolSchema]
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `(tool_or_cls, *, instance=None) -> None` | Register a `@tool()`-decorated function or class. Raises `ValueError` on name collision. |
| `get` | `(name) -> tuple[Any, ToolMeta] \| None` | Return `(callable_or_instance, ToolMeta)` for `name`, or `None`. |
| `get_schemas` | `(tool_names) -> list[ToolSchema]` | Return `ToolSchema` list for the given names. Unknown names are silently skipped. |
| `all_names` | `() -> list[str]` | All registered tool names in registration order. |
| `all_metas` | `() -> dict[str, ToolMeta]` | Shallow copy of the name → `ToolMeta` mapping. |

Supports `len()`, `in` membership tests, and `repr()`.

---

## `ToolExecutor`

Dispatches tool calls from the agentic loop. Handles context injection, hooks, caching, and HITL gating.

```python
from lauren_ai._tools._executor import ToolExecutor, InMemoryCacheBackend

executor = ToolExecutor(
    registry=registry,
    cache_backend=InMemoryCacheBackend(),
    signals=signal_bus,
)
result = await executor.execute(tool_call, tool_context)
```

### Constructor parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `registry` | `ToolRegistry` | The populated tool registry. |
| `cache_backend` | `CacheBackend \| None` | Optional cache backend for result caching. |
| `signals` | `Any \| None` | Optional `SignalBus` for `ToolCallStarted` / `ToolCallComplete` events. |

### `async execute(tool_call, tool_context)`

Execute a single tool call. Returns a `ToolResult`.

Raises `ToolPendingApprovalSignal` when `requires_confirmation=True` on the tool's meta. Raises `ToolExecutionError` when the tool raises an unhandled exception and no `error_hook` handles it.

### `CacheBackend` protocol

| Method | Signature | Description |
|--------|-----------|-------------|
| `get` | `async (key) -> Any \| None` | Retrieve a cached value; `None` on miss. |
| `set` | `async (key, value, *, ttl) -> None` | Store a value with TTL in seconds. |
| `delete` | `async (key) -> None` | Invalidate a single cache entry. |

### `InMemoryCacheBackend`

In-memory `CacheBackend` implementation with TTL expiry using `time.monotonic()`. Suitable for development and single-process deployments. Also provides `async clear()` and `__len__()`.

# Tools Reference

## `@tool()` decorator

Marks a function or class as a tool for AI agents. Must be called **with parentheses** — bare `@tool` raises `DecoratorUsageError`.

Two forms are supported:

**Function-form** (simple, stateless tools):

```python

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
| `label` | `str` | `""` | Pretty display name for TUI, logs, and `describe()`. When empty, `display_label` title-cases `name`. |
| `requires_confirmation` | `bool` | `False` | When `True`, emits `ToolPendingApproval` before invoking the tool (human-in-the-loop). |
| `pre_hook` | `Callable \| None` | `None` | Callable invoked before the tool runs. Receives `(tool_call, ToolContext)`. |
| `post_hook` | `Callable \| None` | `None` | Callable invoked after a successful run. Receives `(ToolResult, ToolContext)`. |
| `error_hook` | `Callable \| None` | `None` | Callable invoked when the tool raises an exception. Receives `(exception, ToolContext)`. |
| `cache_ttl` | `int \| None` | `None` | Cache successful results for this many seconds. Requires a `CacheBackend`. |
| `cache_key_fn` | `Callable \| None` | `None` | Custom factory that derives a string cache key from the input `dict`. |
| `initial_state` | `Callable[[], dict] \| None` | `None` | Zero-arg factory called before **every** invocation to seed `ctx.state`. |
| `initial_tool_state` | `Callable[[], dict] \| None` | `None` | Zero-arg factory called once at `run()` start to seed `ctx.tool_state`. |
| `dependency_factory` | `Callable[[], dict] \| None` | `None` | Zero-arg factory called once at `run()` start; result exposed as `ctx.dependencies`. |

---

## `ToolMeta`

Metadata stored on the decorated function or class under the `TOOL_META` attribute (`"__lauren_ai_tool__"`). Built at decoration time from the function signature and docstring.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tool name in snake_case. |
| `description` | `str` | Human-readable description extracted from the docstring first paragraph. |
| `label` | `str` | Pretty display name. Empty string = use `display_label` property. |
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
| `initial_state` | `Callable[[], dict] \| None` | Factory seeding `ctx.state` before every call. |
| `initial_tool_state` | `Callable[[], dict] \| None` | Factory seeding `ctx.tool_state` once at run start. |
| `dependency_factory` | `Callable[[], dict] \| None` | Factory providing `ctx.dependencies` once at run start. |

### `display_label` property

Returns `label` when set; otherwise title-cases `name` (`"read_file"` → `"Read File"`).

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
| `agent_context` | `AgentContext` | The owning agent's `AgentContext`. Use `ctx.agent_context.request` and `ctx.agent_context.execution_context` to access HTTP context. |
| `tool_use_id` | `str` | Provider-assigned tool use identifier for this invocation. |
| `turn` | `int` | Which agentic loop iteration (0-based) triggered this call. |
| `state` | `dict[str, object]` | Mutable per-call state bag. Reset before each call; seeded from `initial_state()` when set. |
| `tool_state` | `dict[str, object]` | Mutable per-run state — same dict for all calls to this tool within one `run()`. Seeded from `initial_tool_state()`. |
| `dependencies` | `dict[str, object]` | Singleton deps resolved once at run start by `dependency_factory()`. |
| `extras` | `dict[str, object]` | Per-call context injected by the runner each invocation. |
| `metadata` | `dict[str, object]` | Static key-value pairs from `@set_metadata()` at decoration time. |

> **Removed fields:** `request` and `execution_context` are no longer on `ToolContext`.
> Access them via `ctx.agent_context.request` and `ctx.agent_context.execution_context`.

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

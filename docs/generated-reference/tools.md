# Tools

The `@tool()` decorator and runtime context.

### `tool`

```python
def tool(args: Any = (), name: str | None = None, description: str | None = None, requires_confirmation: bool = False, pre_hook: Callable[..., Any] | None = None, post_hook: Callable[..., Any] | None = None, error_hook: Callable[..., Any] | None = None, cache_ttl: int | None = None, cache_key_fn: Callable[..., Any] | None = None) -> Callable[[_T], _T]
```

Decorator that marks a function or class as a tool for AI agents.

Must be called **with parentheses**: `@tool()`.  Using the bare form
`@tool` (without parentheses) raises `DecoratorUsageError`.

**Function-form** (simple, stateless tools)::

    @tool()
    async def my_tool(query: str) -> list[dict]:
        """One-line description.

        Args:
            query: What to search for.
        """
        ...

**Class-form** (stateful tools with DI constructor dependencies)::

    @tool()
    @injectable(scope=Scope.SINGLETON)
    class MyTool:
        """Description used as tool description."""

        def __init__(self, dep: SomeDep) -> None:
            self._dep = dep

        async def run(self, query: str) -> list[dict]:
            ...

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str | None` | Override the inferred tool name (default: function/class name
converted to snake_case). |
| `description` | `str | None` | Override the description extracted from the docstring. |
| `requires_confirmation` | `bool` | When `True`, the executor emits a
`ToolPendingApprovalSignal` before calling the tool, enabling a
human-in-the-loop review step. |
| `pre_hook` | `Callable[..., Any] | None` | Callable invoked before the tool runs.  Receives the tool
call and `ToolContext`. |
| `post_hook` | `Callable[..., Any] | None` | Callable invoked after a successful tool run.  Receives
the `ToolResult` and `ToolContext`. |
| `error_hook` | `Callable[..., Any] | None` | Callable invoked when the tool raises an exception.
Receives the exception and `ToolContext`. |
| `cache_ttl` | `int | None` | Cache successful results for this many seconds.  Requires
a `CacheBackend` in the executor. |
| `cache_key_fn` | `Callable[..., Any] | None` | Custom factory that derives a cache key from the input
`dict`. |

**Returns:** `Callable` — The decorated function or class, with `TOOL_META` set on it.

**Raises:**

| Exception | Description |
|---|---|
| `DecoratorUsageError` | When called without parentheses (bare
`@tool`). |

### `ToolContext`

```python
class ToolContext(agent_context: Any, tool_use_id: str, turn: int, request: Any | None = None, execution_context: Any | None = None, state: dict[str, Any] = dict())
```

Context injected into a tool function when a `ctx: ToolContext` param is declared.

Carries the owning agent context, the current turn number, and a mutable
state bag for per-call data.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent_context` | `Any` | The `AgentContext` of the running agent (typed as
`Any` to avoid a circular import; at runtime it is an `AgentContext`
instance). |
| `tool_use_id` | `str` | The provider-assigned tool use identifier for this
specific invocation. |
| `turn` | `int` | Which agentic-loop iteration (0-based) triggered this call. |
| `request` | `Any | None` | The originating HTTP `Request`, if the agent was invoked
from a web handler.  `None` otherwise. |
| `state` | `dict[str, Any]` | Mutable per-call state bag for tool-local storage. |

#### `ToolContext.get_metadata`

```python
def get_metadata(self, key: str, default: Any = None) -> Any
```

Get metadata from the agent context.

Delegates to `agent_context.get_metadata(key, default)` when the
agent context supports that method, otherwise returns *default*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `key` | `str` | Metadata key to look up. |
| `default` | `Any` | Value to return when key is absent. |

**Returns:** `Any` — Metadata value or *default*.

### `ToolResult`

```python
class ToolResult(tool_use_id: str, content: str | list[Any], is_error: bool = False)
```

The result of executing a single tool call.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `tool_use_id` | `str` | The provider-assigned identifier echoed back to the
model so it can correlate the result with its own tool-use request. |
| `content` | `str | list[Any]` | The tool output.  Either a plain string or a list of
content blocks (e.g. for image responses). |
| `is_error` | `bool` | `True` when the tool raised an exception or returned an
explicit error payload. |

#### `ToolResult.ok`

```python
def ok(cls, content: Any, tool_use_id: str) -> ToolResult
```

Create a successful `ToolResult`.

Non-string content is serialised to a JSON string automatically.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `content` | `Any` | The tool return value.  Strings are passed through
unchanged; everything else is serialised via `json.dumps`. |
| `tool_use_id` | `str` | Matching provider tool-use identifier. |

**Returns:** `ToolResult` — A new `ToolResult` with `is_error=False`.

#### `ToolResult.error`

```python
def error(cls, message: str, tool_use_id: str) -> ToolResult
```

Create an error `ToolResult`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable error description. |
| `tool_use_id` | `str` | Matching provider tool-use identifier. |

**Returns:** `ToolResult` — A new `ToolResult` with `is_error=True`.


# Tools

The `@tool()` decorator and runtime context.

### `tool`

```python
def tool(args: Any = (), name: str | None = None, description: str | None = None, requires_confirmation: bool = False, confirmation_policy: Callable[..., Any] | None = None, confirmation_policy_error_default: bool = True, pre_hook: Callable[..., Any] | None = None, post_hook: Callable[..., Any] | None = None, error_hook: Callable[..., Any] | None = None, cache_ttl: int | None = None, cache_key_fn: Callable[..., Any] | None = None, label: str = '', initial_state: Callable[[], dict[str, object]] | None = None, initial_tool_state: Callable[[], dict[str, object]] | None = None, dependency_factory: Callable[[], dict[str, object]] | None = None) -> Callable[[_T], _T]
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
class ToolContext(agent_context: Any, tool_use_id: str, turn: int, metadata: dict[str, object] = dict(), state: dict[str, object] = dict(), tool_state: dict[str, object] = dict(), dependencies: dict[str, object] = dict(), extras: dict[str, object] = dict(), tool_name: str = '')
```

Context injected into a tool function when a `ctx: ToolContext` param is declared.

Carries the owning agent context, the current turn number, per-call and
per-run state bags, resolved singleton dependencies, per-call extras, and
static metadata from `@set_metadata` decorators.

State lifetime summary
----------------------
* **state** — fresh every call; seeded from `ToolMeta.initial_state()`.
  Use for within-call audit trails and hook communication.
* **tool_state** — same dict object for every call to this tool within one
  `run()` / `run_stream()` invocation; seeded once from
  `ToolMeta.initial_tool_state()`.  Use to accumulate memory across turns.
* **dependencies** — resolved once at run start by
  `ToolMeta.dependency_factory()`; same object for every call.
  Use for expensive singletons (HTTP client, DB pool).
* **extras** — injected fresh per call by the runner.
  Use for per-call runner-level context (workspace root, user ID).

`request` and `execution_context` were removed in this release.
Access them via `ctx.agent_context.request` and
`ctx.agent_context.execution_context` respectively.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent_context` | `Any` | The `AgentContext` of the running agent (typed as
`Any` to avoid a circular import; at runtime it is an `AgentContext`
instance). |
| `tool_use_id` | `str` | The provider-assigned tool use identifier for this
specific invocation. |
| `turn` | `int` | Which agentic-loop iteration (0-based) triggered this call. |
| `metadata` | `dict[str, object]` | Static metadata attached to this tool via
`@set_metadata(key, value)` at decoration time.  Readable via
`get_metadata()`. |
| `state` | `dict[str, object]` | Mutable per-call state bag.  Reset before every invocation
and pre-seeded from `ToolMeta.initial_state()` when that factory is set. |
| `tool_state` | `dict[str, object]` | Mutable per-run state bag shared across **all** calls to
this tool within one `run()` invocation.  Pre-seeded once from
`ToolMeta.initial_tool_state()` at run start. |
| `dependencies` | `dict[str, object]` | Singleton dependency dict resolved once per run by
`ToolMeta.dependency_factory()`.  Same object for every call.
Treat as read-only; mutations are visible to subsequent calls. |
| `extras` | `dict[str, object]` | Per-call context injected by the runner each invocation.
Treat as read-only; the runner re-populates it on every call. |

#### `ToolContext.get_metadata`

```python
def get_metadata(self, key: str, default: Any = None) -> Any
```

Return metadata by *key*, checking tool-level then agent-level.

Lookup order:
1. **Tool-level static metadata** — key-value pairs attached at
   decoration time via `@set_metadata(key, value)` on the tool.
2. **Agent-level runtime metadata** — the `metadata` dict supplied
   to `~lauren_ai.AgentRunnerBase.run()` (e.g.
   `runner.run(agent, prompt, metadata={"scope": "admin"})`),
   delegated through `agent_context.get_metadata`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `key` | `str` | Metadata key to look up. |
| `default` | `Any` | Value returned when the key is absent in both layers. |

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

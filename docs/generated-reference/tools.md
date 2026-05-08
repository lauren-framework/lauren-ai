# Tools

The `@tool()` decorator and runtime context.

### `tool`

```python
def tool(args: Any = (), name: str | None = None, description: str | None = None, requires_confirmation: bool = False, pre_hook: Callable[..., Any] | None = None, post_hook: Callable[..., Any] | None = None, error_hook: Callable[..., Any] | None = None, cache_ttl: int | None = None, cache_key_fn: Callable[..., Any] | None = None) -> Callable[[_T], _T]
```

Decorator that marks a function or class as a tool for AI agents.

Must be called **with parentheses**: ``@tool()``.  Using the bare form
``@tool`` (without parentheses) raises ``DecoratorUsageError``.

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

:param name: Override the inferred tool name (default: function/class name
    converted to snake_case).
:type name: str | None
:param description: Override the description extracted from the docstring.
:type description: str | None
:param requires_confirmation: When ``True``, the executor emits a
    ``ToolPendingApprovalSignal`` before calling the tool, enabling a
    human-in-the-loop review step.
:type requires_confirmation: bool
:param pre_hook: Callable invoked before the tool runs.  Receives the tool
    call and ``ToolContext``.
:type pre_hook: Callable | None
:param post_hook: Callable invoked after a successful tool run.  Receives
    the ``ToolResult`` and ``ToolContext``.
:type post_hook: Callable | None
:param error_hook: Callable invoked when the tool raises an exception.
    Receives the exception and ``ToolContext``.
:type error_hook: Callable | None
:param cache_ttl: Cache successful results for this many seconds.  Requires
    a ``CacheBackend`` in the executor.
:type cache_ttl: int | None
:param cache_key_fn: Custom factory that derives a cache key from the input
    ``dict``.
:type cache_key_fn: Callable | None
:return: The decorated function or class, with ``TOOL_META`` set on it.
:rtype: Callable
:raises DecoratorUsageError: When called without parentheses (bare
    ``@tool``).

### `ToolContext`

```python
class ToolContext(agent_context: Any, tool_use_id: str, turn: int, request: Any | None = None, execution_context: Any | None = None, state: dict[str, Any] = dict())
```

Context injected into a tool function when a ``ctx: ToolContext`` param is declared.

Carries the owning agent context, the current turn number, and a mutable
state bag for per-call data.

:param agent_context: The ``AgentContext`` of the running agent (typed as
    ``Any`` to avoid a circular import; at runtime it is an ``AgentContext``
    instance).
:type agent_context: Any
:param tool_use_id: The provider-assigned tool use identifier for this
    specific invocation.
:type tool_use_id: str
:param turn: Which agentic-loop iteration (0-based) triggered this call.
:type turn: int
:param request: The originating HTTP ``Request``, if the agent was invoked
    from a web handler.  ``None`` otherwise.
:type request: Any | None
:param state: Mutable per-call state bag for tool-local storage.
:type state: dict[str, Any]

#### `ToolContext.get_metadata`

```python
def get_metadata(self, key: str, default: Any = None) -> Any
```

Get metadata from the agent context.

Delegates to ``agent_context.get_metadata(key, default)`` when the
agent context supports that method, otherwise returns *default*.

:param key: Metadata key to look up.
:type key: str
:param default: Value to return when key is absent.
:type default: Any
:return: Metadata value or *default*.
:rtype: Any

### `ToolResult`

```python
class ToolResult(tool_use_id: str, content: str | list[Any], is_error: bool = False)
```

The result of executing a single tool call.

:param tool_use_id: The provider-assigned identifier echoed back to the
    model so it can correlate the result with its own tool-use request.
:type tool_use_id: str
:param content: The tool output.  Either a plain string or a list of
    content blocks (e.g. for image responses).
:type content: str | list[Any]
:param is_error: ``True`` when the tool raised an exception or returned an
    explicit error payload.
:type is_error: bool

#### `ToolResult.ok`

```python
def ok(cls, content: Any, tool_use_id: str) -> ToolResult
```

Create a successful ``ToolResult``.

Non-string content is serialised to a JSON string automatically.

:param content: The tool return value.  Strings are passed through
    unchanged; everything else is serialised via ``json.dumps``.
:type content: Any
:param tool_use_id: Matching provider tool-use identifier.
:type tool_use_id: str
:return: A new ``ToolResult`` with ``is_error=False``.
:rtype: ToolResult

#### `ToolResult.error`

```python
def error(cls, message: str, tool_use_id: str) -> ToolResult
```

Create an error ``ToolResult``.

:param message: Human-readable error description.
:type message: str
:param tool_use_id: Matching provider tool-use identifier.
:type tool_use_id: str
:return: A new ``ToolResult`` with ``is_error=True``.
:rtype: ToolResult


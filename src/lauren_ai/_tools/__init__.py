"""Tool system for lauren-ai.

Provides the ``@tool()`` decorator, ``ToolContext``, ``ToolResult``, and
``ToolSchema`` types used to define and execute tools within the agentic
loop.

Typical usage::

    from lauren_ai import tool, ToolContext

    @tool()
    async def search_web(query: str, max_results: int = 5) -> list[dict]:
        \"\"\"Search the web for current information.

        Args:
            query: The search query string.
            max_results: Maximum results to return (1-20).
        \"\"\"
        ...
"""

from __future__ import annotations

__all__ = [
    "TOOL_META",
    "ToolContext",
    "ToolMeta",
    "ToolResult",
    "ToolSchema",
    "tool",
]

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

# ---------------------------------------------------------------------------
# Schema type
# ---------------------------------------------------------------------------

class ToolSchema(TypedDict, total=False):
    """JSON Schema object for a tool's input parameters.

    :param name: The tool name (snake_case).
    :type name: str
    :param description: Human-readable description extracted from docstring.
    :type description: str
    :param input_schema: JSON Schema object describing input parameters.
    :type input_schema: dict[str, Any]
    """

    name: str
    description: str
    input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Context injected into a tool function when a ``ctx: ToolContext`` param is declared.

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
    """

    agent_context: Any
    tool_use_id: str
    turn: int
    request: Any | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the agent context.

        Delegates to ``agent_context.get_metadata(key, default)`` when the
        agent context supports that method, otherwise returns *default*.

        :param key: Metadata key to look up.
        :type key: str
        :param default: Value to return when key is absent.
        :type default: Any
        :return: Metadata value or *default*.
        :rtype: Any
        """
        if self.agent_context is not None and hasattr(self.agent_context, "get_metadata"):
            return self.agent_context.get_metadata(key, default)
        return default


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """The result of executing a single tool call.

    :param tool_use_id: The provider-assigned identifier echoed back to the
        model so it can correlate the result with its own tool-use request.
    :type tool_use_id: str
    :param content: The tool output.  Either a plain string or a list of
        content blocks (e.g. for image responses).
    :type content: str | list[Any]
    :param is_error: ``True`` when the tool raised an exception or returned an
        explicit error payload.
    :type is_error: bool
    """

    tool_use_id: str
    content: str | list[Any]
    is_error: bool = False

    @classmethod
    def ok(cls, content: Any, *, tool_use_id: str) -> ToolResult:
        """Create a successful ``ToolResult``.

        Non-string content is serialised to a JSON string automatically.

        :param content: The tool return value.  Strings are passed through
            unchanged; everything else is serialised via ``json.dumps``.
        :type content: Any
        :param tool_use_id: Matching provider tool-use identifier.
        :type tool_use_id: str
        :return: A new ``ToolResult`` with ``is_error=False``.
        :rtype: ToolResult
        """
        if isinstance(content, str):
            serialised: str | list[Any] = content
        elif isinstance(content, list):
            # Preserve list-of-content-blocks as-is; plain lists get serialised.
            serialised = content
        else:
            try:
                serialised = json.dumps(content, default=str)
            except (TypeError, ValueError):
                serialised = str(content)
        return cls(tool_use_id=tool_use_id, content=serialised, is_error=False)

    @classmethod
    def error(cls, message: str, *, tool_use_id: str) -> ToolResult:
        """Create an error ``ToolResult``.

        :param message: Human-readable error description.
        :type message: str
        :param tool_use_id: Matching provider tool-use identifier.
        :type tool_use_id: str
        :return: A new ``ToolResult`` with ``is_error=True``.
        :rtype: ToolResult
        """
        return cls(tool_use_id=tool_use_id, content=message, is_error=True)


# ---------------------------------------------------------------------------
# ToolMeta
# ---------------------------------------------------------------------------

@dataclass
class ToolMeta:
    """Metadata attached to a function or class decorated with ``@tool()``.

    This dataclass is stored under the ``TOOL_META`` attribute on the
    decorated callable.

    :param name: Tool name (snake_case).  Defaults to the function/class name.
    :type name: str
    :param description: Human-readable description extracted from the first
        paragraph of the docstring (or overridden via ``@tool(description=...)``.
    :type description: str
    :param parameters: JSON Schema for the tool's input parameters.
    :type parameters: ToolSchema
    :param is_async: ``True`` if the entry-point function/method is a coroutine.
    :type is_async: bool
    :param reads_context: ``True`` when the entry-point declares a ``ctx:
        ToolContext`` parameter.  The context is injected at execution time and
        never exposed in the JSON schema sent to the model.
    :type reads_context: bool
    :param requires_confirmation: When ``True`` the executor raises
        ``ToolPendingApprovalSignal`` before invoking the tool, allowing a
        human-in-the-loop review step.
    :type requires_confirmation: bool
    :param pre_hook: Optional callable invoked *before* the tool runs.
        Receives ``(tool_call, tool_context)``.
    :type pre_hook: Callable[..., Any] | None
    :param post_hook: Optional callable invoked *after* a successful tool run.
        Receives ``(tool_result, tool_context)``.
    :type post_hook: Callable[..., Any] | None
    :param error_hook: Optional callable invoked when the tool raises an
        exception.  Receives ``(exception, tool_context)``.
    :type error_hook: Callable[..., Any] | None
    :param cache_ttl: If set, successful results are cached for this many
        seconds.  Requires a ``CacheBackend`` to be registered.
    :type cache_ttl: int | None
    :param cache_key_fn: Custom cache-key factory.  Receives the tool's
        ``input`` dict and returns a string key.  When ``None`` a default
        JSON-based key is derived from ``name + sorted(input.items())``.
    :type cache_key_fn: Callable[[dict[str, Any]], str] | None
    """

    name: str
    description: str
    parameters: ToolSchema
    is_async: bool
    reads_context: bool
    requires_confirmation: bool = False
    pre_hook: Callable[..., Any] | None = None
    post_hook: Callable[..., Any] | None = None
    error_hook: Callable[..., Any] | None = None
    cache_ttl: int | None = None
    cache_key_fn: Callable[[dict[str, Any]], str] | None = None


# ---------------------------------------------------------------------------
# Sentinel attribute name
# ---------------------------------------------------------------------------

#: Attribute name set on decorated functions/classes to store ``ToolMeta``.
TOOL_META: str = "__lauren_ai_tool__"


# ---------------------------------------------------------------------------
# @tool() decorator
# ---------------------------------------------------------------------------

# Import lazily to avoid circular dependency at module initialisation time.
# _schema is a sibling module inside this package.
def _build_meta(
    fn_or_cls: Any,
    *,
    name: str | None,
    description: str | None,
    requires_confirmation: bool,
    pre_hook: Callable[..., Any] | None,
    post_hook: Callable[..., Any] | None,
    error_hook: Callable[..., Any] | None,
    cache_ttl: int | None,
    cache_key_fn: Callable[..., Any] | None,
) -> ToolMeta:
    """Build a ``ToolMeta`` for *fn_or_cls*.

    :param fn_or_cls: The decorated function or class.
    :param name: Optional name override.
    :param description: Optional description override.
    :param requires_confirmation: HITL flag.
    :param pre_hook: Pre-execution hook.
    :param post_hook: Post-execution hook.
    :param error_hook: Error hook.
    :param cache_ttl: Cache TTL in seconds.
    :param cache_key_fn: Custom cache key factory.
    :return: Populated ``ToolMeta``.
    :rtype: ToolMeta
    """
    from ._schema import generate_tool_schema  # local import — avoids cycle

    tool_name, tool_description, schema = generate_tool_schema(
        fn_or_cls,
        name=name,
        description=description,
    )

    # Determine entry-point for is_async / reads_context
    if inspect.isclass(fn_or_cls):
        entry = getattr(fn_or_cls, "run", None)
    else:
        entry = fn_or_cls

    if entry is None:
        raise ValueError(
            f"Class-form tool {fn_or_cls!r} has no 'run' method. "
            "Class-form tools must define a 'run' method as their entry point."
        )

    is_async = inspect.iscoroutinefunction(entry)

    # Check whether entry point has a `ctx: ToolContext` parameter
    reads_context = False
    try:
        sig = inspect.signature(entry)
        for param_name, param in sig.parameters.items():
            if param_name == "ctx":
                ann = param.annotation
                # Accept bare ToolContext or Optional[ToolContext]
                if ann is ToolContext:
                    reads_context = True
                    break
                # Handle Optional / X | None via string check for forward refs
                ann_str = str(ann)
                if "ToolContext" in ann_str:
                    reads_context = True
                    break
    except (ValueError, TypeError):
        pass

    return ToolMeta(
        name=tool_name,
        description=tool_description,
        parameters=schema,
        is_async=is_async,
        reads_context=reads_context,
        requires_confirmation=requires_confirmation,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        cache_ttl=cache_ttl,
        cache_key_fn=cache_key_fn,
    )


def tool(
    *args: Any,
    name: str | None = None,
    description: str | None = None,
    requires_confirmation: bool = False,
    pre_hook: Callable[..., Any] | None = None,
    post_hook: Callable[..., Any] | None = None,
    error_hook: Callable[..., Any] | None = None,
    cache_ttl: int | None = None,
    cache_key_fn: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Decorator that marks a function or class as a tool for AI agents.

    Must be called **with parentheses**: ``@tool()``.  Using the bare form
    ``@tool`` (without parentheses) raises ``DecoratorUsageError``.

    **Function-form** (simple, stateless tools)::

        @tool()
        async def my_tool(query: str) -> list[dict]:
            \"\"\"One-line description.

            Args:
                query: What to search for.
            \"\"\"
            ...

    **Class-form** (stateful tools with DI constructor dependencies)::

        @tool()
        @injectable(scope=Scope.SINGLETON)
        class MyTool:
            \"\"\"Description used as tool description.\"\"\"

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
    """
    # Detect bare usage: @tool instead of @tool()
    if args and callable(args[0]):
        # Defer import to avoid circular dependency at module load time
        try:
            from lauren.exceptions import DecoratorUsageError  # type: ignore[import]
        except ImportError:
            class DecoratorUsageError(Exception):  # type: ignore[no-redef]
                pass
        raise DecoratorUsageError(
            "@tool must be used with parentheses: @tool()"
        )

    def decorator(fn_or_cls: Any) -> Any:
        meta = _build_meta(
            fn_or_cls,
            name=name,
            description=description,
            requires_confirmation=requires_confirmation,
            pre_hook=pre_hook,
            post_hook=post_hook,
            error_hook=error_hook,
            cache_ttl=cache_ttl,
            cache_key_fn=cache_key_fn,
        )
        setattr(fn_or_cls, TOOL_META, meta)
        return fn_or_cls

    return decorator

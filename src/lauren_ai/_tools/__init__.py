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

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict, TypeVar

__all__ = [
    "TOOL_META",
    "TOOL_METADATA",
    "USE_HOOKS_META",
    "ToolContext",
    "ToolMeta",
    "ToolResult",
    "ToolSchema",
    "UnifiedToolContext",
    "ToolContextAdapter",
    "get_tool_context_from_func_args",
    "set_metadata",
    "tool",
    "use_hooks",
]

from lauren_ai._tools._unified import ToolContextAdapter, UnifiedToolContext  # noqa: E402

# Sentinel attribute used by @set_metadata to attach static key-value pairs to
# @tool()-decorated functions and classes — same pattern as lauren.set_metadata
# on route handlers.
TOOL_METADATA = "__lauren_ai_tool_metadata__"

#: Attribute name set by ``@use_hooks()`` to store hook classes on a tool.
USE_HOOKS_META: str = "__lauren_ai_tool_hook_classes__"

_T = TypeVar("_T")

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

    Carries the owning agent context, the current turn number, per-call and
    per-run state bags, resolved singleton dependencies, per-call extras, and
    static metadata from ``@set_metadata`` decorators.

    State lifetime summary
    ----------------------
    * **state** — fresh every call; seeded from ``ToolMeta.initial_state()``.
      Use for within-call audit trails and hook communication.
    * **tool_state** — same dict object for every call to this tool within one
      ``run()`` / ``run_stream()`` invocation; seeded once from
      ``ToolMeta.initial_tool_state()``.  Use to accumulate memory across turns.
    * **dependencies** — resolved once at run start by
      ``ToolMeta.dependency_factory()``; same object for every call.
      Use for expensive singletons (HTTP client, DB pool).
    * **extras** — injected fresh per call by the runner.
      Use for per-call runner-level context (workspace root, user ID).

    ``request`` and ``execution_context`` were removed in this release.
    Access them via ``ctx.agent_context.request`` and
    ``ctx.agent_context.execution_context`` respectively.

    :param agent_context: The ``AgentContext`` of the running agent (typed as
        ``Any`` to avoid a circular import; at runtime it is an ``AgentContext``
        instance).
    :type agent_context: Any
    :param tool_use_id: The provider-assigned tool use identifier for this
        specific invocation.
    :type tool_use_id: str
    :param turn: Which agentic-loop iteration (0-based) triggered this call.
    :type turn: int
    :param metadata: Static metadata attached to this tool via
        ``@set_metadata(key, value)`` at decoration time.  Readable via
        :meth:`get_metadata`.
    :type metadata: dict[str, object]
    :param state: Mutable per-call state bag.  Reset before every invocation
        and pre-seeded from ``ToolMeta.initial_state()`` when that factory is set.
    :type state: dict[str, object]
    :param tool_state: Mutable per-run state bag shared across **all** calls to
        this tool within one ``run()`` invocation.  Pre-seeded once from
        ``ToolMeta.initial_tool_state()`` at run start.
    :type tool_state: dict[str, object]
    :param dependencies: Singleton dependency dict resolved once per run by
        ``ToolMeta.dependency_factory()``.  Same object for every call.
        Treat as read-only; mutations are visible to subsequent calls.
    :type dependencies: dict[str, object]
    :param extras: Per-call context injected by the runner each invocation.
        Treat as read-only; the runner re-populates it on every call.
    :type extras: dict[str, object]
    """

    agent_context: Any
    tool_use_id: str
    turn: int
    metadata: dict[str, object] = field(default_factory=dict)
    state: dict[str, object] = field(default_factory=dict)
    tool_state: dict[str, object] = field(default_factory=dict)
    dependencies: dict[str, object] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)
    tool_name: str = ""  # populated by _execute_single_tool

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return metadata by *key*, checking tool-level then agent-level.

        Lookup order:
        1. **Tool-level static metadata** — key-value pairs attached at
           decoration time via ``@set_metadata(key, value)`` on the tool.
        2. **Agent-level runtime metadata** — the ``metadata`` dict supplied
           to :meth:`~lauren_ai.AgentRunnerBase.run` (e.g.
           ``runner.run(agent, prompt, metadata={"scope": "admin"})``),
           delegated through ``agent_context.get_metadata``.

        :param key: Metadata key to look up.
        :type key: str
        :param default: Value returned when the key is absent in both layers.
        :type default: Any
        :return: Metadata value or *default*.
        :rtype: Any
        """
        if key in self.metadata:
            return self.metadata[key]
        if self.agent_context is not None and hasattr(self.agent_context, "get_metadata"):
            return self.agent_context.get_metadata(key, default)
        return default

    @property
    def message_bus(self) -> Any | None:
        """Return the current run's message bus, if one is wired."""
        if self.agent_context is None:
            return None
        return getattr(self.agent_context, "message_bus", None)

    @property
    def runner(self) -> Any | None:
        """Return the active runner driving the current agent turn, if any."""
        if self.agent_context is None:
            return None
        return getattr(self.agent_context, "runner", None)


# ---------------------------------------------------------------------------
# Context extraction helper
# ---------------------------------------------------------------------------


def get_tool_context_from_func_args(*args: Any, **kwargs: Any) -> ToolContext | None:
    """Return the first :class:`ToolContext` found in a tool call's arguments.

    The tool executor injects :class:`ToolContext` as a named keyword argument
    (using whatever parameter name was annotated ``ToolContext`` in the tool's
    signature), but when writing a decorator that wraps a ``run()`` method or
    a function-form tool, the decorator typically receives the raw ``*args`` /
    ``**kwargs`` and does not know the parameter name ahead of time.

    This helper scans both positional and keyword arguments so that decorators
    work regardless of whether the context was passed positionally (e.g. as
    ``self, ctx, ...`` in a class-form ``run()`` method) or as a keyword
    argument::

        import functools
        from lauren_ai import tool, ToolContext, get_tool_context_from_func_args

        def require_auth(fn):
            \"\"\"Decorator: reject the tool call when no authenticated user is present.\"\"\"
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                ctx = get_tool_context_from_func_args(*args, **kwargs)
                if ctx is None:
                    raise RuntimeError("require_auth: ToolContext not found in arguments")
                exec_ctx = ctx.agent_context.execution_context if ctx.agent_context else None
                request = ctx.agent_context.request if ctx.agent_context else None
                if exec_ctx is None or request is None:
                    return {"error": "Authentication required — no request context"}
                user_id = request.state.get("user_id")
                if not user_id:
                    return {"error": "Authentication required"}
                return await fn(*args, **kwargs)
            return wrapper

        @tool()
        class MyTool:
            @require_auth
            async def run(self, ctx: ToolContext, query: str) -> dict:
                return {"result": query, "user": ctx.agent_context.execution_context.request.state.user_id}

    :param args: Positional arguments forwarded from the wrapping function.
    :param kwargs: Keyword arguments forwarded from the wrapping function.
    :returns: The :class:`ToolContext` instance if present in the call
        arguments, or ``None`` if no ``ToolContext`` was found.
    :rtype: ToolContext | None
    """
    for arg in args:
        if isinstance(arg, ToolContext):
            return arg
    for val in kwargs.values():
        if isinstance(val, ToolContext):
            return val
    return None


# ---------------------------------------------------------------------------
# set_metadata — decorator mirroring lauren.set_metadata for route handlers
# ---------------------------------------------------------------------------


def set_metadata(key: str, value: Any) -> Callable[[_T], _T]:
    """Attach static metadata to a ``@tool()`` function or class.

    Mirrors ``lauren.set_metadata`` for route handlers: the key-value pair is
    stored at decoration time and is readable via
    :meth:`ToolContext.get_metadata` inside the tool body.

    Multiple ``@set_metadata`` decorators on the same tool stack cleanly — each
    call merges into a single dict stored on the
    ``__lauren_ai_tool_metadata__`` attribute.

    Example::

        @set_metadata("required_scope", "admin")
        @tool()
        async def delete_record(record_id: str, ctx: ToolContext | None = None) -> dict:
            \"\"\"Delete a record. Args: record_id: Record ID.\"\"\"
            # tool-level: which scope this tool requires
            required = ctx.get_metadata("required_scope")   # → "admin"
            # agent-level: what scope the caller has (from runner.run(metadata={…}))
            granted  = ctx.get_metadata("caller_scope", "read")
            if granted != required:
                return {"error": f"scope_required:{required}"}
            return {"deleted": record_id}

    :param key: Metadata key.
    :type key: str
    :param value: Metadata value.
    :type value: Any
    :returns: A class/function decorator that attaches the key-value pair.
    """

    def decorator(target: _T) -> _T:
        existing: dict[str, Any] = dict(getattr(target, TOOL_METADATA, {}))
        existing[key] = value
        setattr(target, TOOL_METADATA, existing)
        return target  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


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
        human-in-the-loop review step.  Takes priority over
        ``confirmation_policy`` — the gate always fires regardless of what
        the policy returns.
    :type requires_confirmation: bool
    :param confirmation_policy: Optional callable evaluated at execution time
        (after before-hooks) to decide dynamically whether confirmation is
        needed.  Receives the ``ToolCallContext`` — which carries the
        hook-modified input and ``ctx.state`` (populated by global hooks such
        as ``AppStateInjectorHook``).  May be sync or async.  Ignored when
        ``requires_confirmation=True``.
    :type confirmation_policy: Callable[[ToolCallContext], bool | Awaitable[bool]] | None
    :param confirmation_policy_error_default: What to do when
        ``confirmation_policy`` raises an exception.  ``True`` = fail-closed
        (require confirmation — safer for write tools).  ``False`` = fail-open
        (skip confirmation — safer for read tools).  Defaults to ``True``.
    :type confirmation_policy_error_default: bool
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
    :param context_param_name: Name of the parameter annotated with
        ``ToolContext`` in the entry-point signature.  ``None`` when the
        entry point does not declare a context parameter.  Used by the
        executor to inject the context under the correct keyword name
        (the parameter may be named anything — ``ctx``, ``agent_ctx``, …).
    :type context_param_name: str | None
    """

    name: str
    description: str
    parameters: ToolSchema
    is_async: bool
    reads_context: bool
    context_param_name: str | None = None
    requires_confirmation: bool = False
    confirmation_policy: Callable[..., Any] | None = None
    confirmation_policy_error_default: bool = True
    pre_hook: Callable[..., Any] | None = None
    post_hook: Callable[..., Any] | None = None
    error_hook: Callable[..., Any] | None = None
    cache_ttl: int | None = None
    cache_key_fn: Callable[[dict[str, Any]], str] | None = None
    #: Hook classes attached via ``@use_hooks()`` at decoration time.
    hook_classes: tuple[type, ...] = field(default_factory=tuple)
    #: Resolved ``ToolHook`` instances — populated by ``AgentModule.for_root()``
    #: after DI resolution.  Empty tuple until the module is built.
    resolved_hooks: tuple[Any, ...] = field(default_factory=tuple)

    # ── decoration-time extensions ────────────────────────────────────────────
    #: Optional pretty display name shown in TUI, logs, and ``describe()``.
    #: When empty, :attr:`display_label` computes a title-cased version of
    #: :attr:`name`.
    label: str = ""

    #: Zero-arg factory called **before every tool invocation** to produce a
    #: fresh ``ctx.state`` seed.  The seed is shallow-merged into ``ctx.state``
    #: so any values already set by hooks are preserved.  ``None`` → ``ctx.state``
    #: starts as an empty dict.
    initial_state: Callable[[], dict[str, object]] | None = None

    #: Zero-arg factory called **once at run start** for this tool name to
    #: initialise ``ctx.tool_state``.  The same dict object is handed to every
    #: call within the run.  ``None`` → ``ctx.tool_state`` starts as ``{}``.
    initial_tool_state: Callable[[], dict[str, object]] | None = None

    #: Zero-arg factory called **once at run start** to resolve singleton
    #: dependencies exposed as ``ctx.dependencies``.  The same object is
    #: returned for every call within the run.  ``None`` → ``ctx.dependencies``
    #: is ``{}``.
    dependency_factory: Callable[[], dict[str, object]] | None = None

    @property
    def display_label(self) -> str:
        """Human-readable display name for this tool.

        Returns :attr:`label` when explicitly set; otherwise computes a
        title-cased version of :attr:`name` (``"read_file"`` → ``"Read File"``).

        :return: Display label string.
        :rtype: str
        """
        return self.label or self.name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Sentinel attribute name
# ---------------------------------------------------------------------------

#: Attribute name set on decorated functions/classes to store ``ToolMeta``.
TOOL_META: str = "__lauren_ai_tool__"


# ---------------------------------------------------------------------------
# Tool map helper
# ---------------------------------------------------------------------------


def _add_to_tool_map(
    tool_map: dict,
    tool_or_cls: Any,
    *,
    instance: Any | None = None,
) -> None:
    """Add a ``@tool()``-decorated function or class to a *tool_map* dict.

    *tool_map* maps ``name → (callable_or_instance, ToolMeta)``.
    Contains all validation formerly in ``ToolRegistry.register()``:

    - ``MetadataInheritanceError`` for subclasses that inherit ``TOOL_META``
      without re-applying ``@tool()``.
    - ``ValueError`` when the object has no ``TOOL_META``.
    - ``ValueError`` on name collision.

    :param tool_map: Mutable dict to add the tool into.
    :param tool_or_cls: A ``@tool()``-decorated function or class.
    :param instance: Pre-resolved instance for class-form tools.
    :raises MetadataInheritanceError: When a subclass inherits ``TOOL_META``
        without re-applying ``@tool()``.
    :raises ValueError: When the object has no ``TOOL_META`` or a name
        collision is detected.
    """
    _cls = tool_or_cls if inspect.isclass(tool_or_cls) else (type(instance) if instance is not None else None)
    if _cls is not None and TOOL_META not in _cls.__dict__:
        _base = next((b for b in _cls.__mro__[1:] if TOOL_META in b.__dict__), None)
        if _base is not None:
            from lauren.exceptions import MetadataInheritanceError  # noqa: PLC0415

            raise MetadataInheritanceError(
                f"{_cls.__name__} inherits @tool() metadata from {_base.__name__} "
                f"but is not itself decorated with @tool(). Subclasses of "
                f"@tool()-decorated classes must re-apply @tool() to be usable as tools."
            )

    meta: ToolMeta | None = getattr(tool_or_cls, TOOL_META, None)
    if meta is None:
        if instance is not None:
            meta = getattr(type(instance), TOOL_META, None) or getattr(instance, TOOL_META, None)
        if meta is None:
            raise ValueError(
                f"Object {tool_or_cls!r} does not have a {TOOL_META!r} attribute. "
                "Only objects decorated with @tool() can be registered."
            )

    name = meta.name
    if name in tool_map:
        existing_meta = tool_map[name][1]
        raise ValueError(
            f"Tool name collision: '{name}' is already in the tool map "
            f"(existing={existing_meta!r}, new={meta!r}). "
            "Each tool must have a unique name."
        )

    tool_map[name] = (instance if instance is not None else tool_or_cls, meta)


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
    confirmation_policy: Callable[..., Any] | None = None,
    confirmation_policy_error_default: bool = True,
    pre_hook: Callable[..., Any] | None,
    post_hook: Callable[..., Any] | None,
    error_hook: Callable[..., Any] | None,
    cache_ttl: int | None,
    cache_key_fn: Callable[..., Any] | None,
    label: str = "",
    initial_state: Callable[[], dict[str, object]] | None = None,
    initial_tool_state: Callable[[], dict[str, object]] | None = None,
    dependency_factory: Callable[[], dict[str, object]] | None = None,
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
    entry = getattr(fn_or_cls, "run", None) if inspect.isclass(fn_or_cls) else fn_or_cls

    if entry is None:
        raise ValueError(
            f"Class-form tool {fn_or_cls!r} has no 'run' method. "
            "Class-form tools must define a 'run' method as their entry point."
        )

    is_async = inspect.iscoroutinefunction(entry)

    # Check whether entry point has a `ctx: ToolContext` parameter.
    # Use typing.get_type_hints() so that `from __future__ import annotations`
    # (PEP 563) in the user's file doesn't turn the annotation into a string.
    reads_context = False
    context_param_name: str | None = None
    try:
        sig = inspect.signature(entry)
        import sys as _sys
        import typing as _typing

        _entry_module = _sys.modules.get(entry.__module__)
        _globalns = vars(_entry_module) if _entry_module is not None else {}
        try:
            _hints = _typing.get_type_hints(entry, globalns=_globalns, include_extras=False)
        except Exception:
            _hints = {}
        for param_name, param in sig.parameters.items():
            ann = _hints.get(param_name, param.annotation)
            # Accept bare ToolContext or Optional[ToolContext] / ToolContext | None,
            # regardless of the parameter name.
            if ann is ToolContext:
                reads_context = True
                context_param_name = param_name
                break
            # Handle Optional / X | None via string check for forward refs
            ann_str = str(ann)
            if "ToolContext" in ann_str:
                reads_context = True
                context_param_name = param_name
                break
    except (ValueError, TypeError):
        pass

    hook_classes = tuple(getattr(fn_or_cls, USE_HOOKS_META, ()))

    return ToolMeta(
        name=tool_name,
        description=tool_description,
        parameters=schema,
        is_async=is_async,
        reads_context=reads_context,
        context_param_name=context_param_name,
        requires_confirmation=requires_confirmation,
        confirmation_policy=confirmation_policy,
        confirmation_policy_error_default=confirmation_policy_error_default,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        cache_ttl=cache_ttl,
        cache_key_fn=cache_key_fn,
        hook_classes=hook_classes,
        label=label,
        initial_state=initial_state,
        initial_tool_state=initial_tool_state,
        dependency_factory=dependency_factory,
    )


def tool(
    *args: Any,
    name: str | None = None,
    description: str | None = None,
    requires_confirmation: bool = False,
    confirmation_policy: Callable[..., Any] | None = None,
    confirmation_policy_error_default: bool = True,
    pre_hook: Callable[..., Any] | None = None,
    post_hook: Callable[..., Any] | None = None,
    error_hook: Callable[..., Any] | None = None,
    cache_ttl: int | None = None,
    cache_key_fn: Callable[..., Any] | None = None,
    label: str = "",
    initial_state: Callable[[], dict[str, object]] | None = None,
    initial_tool_state: Callable[[], dict[str, object]] | None = None,
    dependency_factory: Callable[[], dict[str, object]] | None = None,
) -> Callable[[_T], _T]:
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
        from lauren.exceptions import DecoratorUsageError  # noqa: PLC0415

        raise DecoratorUsageError("@tool must be used with parentheses: @tool()")

    def decorator(fn_or_cls: _T) -> _T:
        meta = _build_meta(
            fn_or_cls,
            name=name,
            description=description,
            requires_confirmation=requires_confirmation,
            confirmation_policy=confirmation_policy,
            confirmation_policy_error_default=confirmation_policy_error_default,
            pre_hook=pre_hook,
            post_hook=post_hook,
            error_hook=error_hook,
            cache_ttl=cache_ttl,
            cache_key_fn=cache_key_fn,
            label=label,
            initial_state=initial_state,
            initial_tool_state=initial_tool_state,
            dependency_factory=dependency_factory,
        )

        # For class-form tools: auto-apply @injectable(scope=SINGLETON) so the
        # class can participate in the Lauren DI container and receive injected
        # constructor dependencies (database, HTTP clients, other services, …).
        # This mirrors what @agent() does for agent classes.
        # NOTE: assign back — injectable() may return a new object.
        if inspect.isclass(fn_or_cls):
            try:
                from lauren import Scope, injectable  # noqa: PLC0415

                # Only apply if not already marked injectable (check __dict__
                # to avoid detecting inherited __lauren_injectable__).
                if "__lauren_injectable__" not in fn_or_cls.__dict__:  # type: ignore[union-attr]
                    fn_or_cls = injectable(scope=Scope.SINGLETON)(fn_or_cls)  # type: ignore[assignment]
            except ImportError:
                pass  # lauren not installed — DI unavailable, skip silently

        # Set TOOL_META after the injectable block so it lands on the final object.
        setattr(fn_or_cls, TOOL_META, meta)

        return fn_or_cls

    return decorator


# ---------------------------------------------------------------------------
# @use_hooks() decorator
# ---------------------------------------------------------------------------


def use_hooks(*hook_classes: type) -> Callable[[_T], _T]:
    """Attach injectable :class:`~lauren_ai.ToolHook` classes to a ``@tool()``.

    Must be stacked **above** ``@tool()``::

        @use_hooks(AuditHook, RateLimitHook)
        @tool()
        async def my_tool(query: str) -> list[dict]:
            ...

        @use_hooks(AuditHook)
        @tool()
        class MyTool:
            def __init__(self, db: Database) -> None: ...
            async def run(self, query: str) -> dict: ...

    Each hook class must be a subclass of :class:`~lauren_ai.ToolHook`.
    Hooks are resolved as ``SINGLETON`` injectables; they are auto-decorated
    with ``@injectable(scope=Scope.SINGLETON)`` if not already marked.

    Hook execution order for each tool call:

    * **before** — global hooks first, then per-tool hooks (leftmost first).
    * **after** — per-tool hooks first (leftmost first), then global hooks.
    * **error** — same order as after.

    :param hook_classes: Subclasses of :class:`~lauren_ai.ToolHook`.
    :type hook_classes: type
    :raises DecoratorUsageError: When a hook class is not a
        :class:`~lauren_ai.ToolHook` subclass, or when called without
        parentheses.
    """
    # Bare-usage guard: @use_hooks without parentheses passes the decorated
    # object as the first positional arg.
    if hook_classes and callable(hook_classes[0]) and not inspect.isclass(hook_classes[0]):
        from lauren_ai._exceptions import DecoratorUsageError  # noqa: PLC0415

        raise DecoratorUsageError(
            "@use_hooks must be called with parentheses and hook classes: @use_hooks(MyHook, ...)"
        )

    from lauren_ai._exceptions import DecoratorUsageError  # noqa: PLC0415

    for hcls in hook_classes:
        if not (inspect.isclass(hcls)):
            raise DecoratorUsageError(f"@use_hooks: {hcls!r} is not a class. Pass ToolHook subclasses, not instances.")

    def decorator(fn_or_cls: _T) -> _T:
        # Validate hook classes are ToolHook subclasses (imported lazily to
        # avoid circular import at module-definition time).
        try:
            from lauren_ai._tools._hooks import ToolHook as _ToolHook  # noqa: PLC0415

            for hcls in hook_classes:
                if not issubclass(hcls, _ToolHook):
                    raise DecoratorUsageError(f"@use_hooks: {hcls.__name__!r} must be a subclass of ToolHook.")
        except ImportError:
            pass

        # Auto-apply @injectable(scope=SINGLETON) to each hook class.
        try:
            from lauren import Scope, injectable  # noqa: PLC0415

            for hcls in hook_classes:
                if inspect.isclass(hcls) and "__lauren_injectable__" not in hcls.__dict__:
                    injectable(scope=Scope.SINGLETON)(hcls)
        except ImportError:
            pass

        # Merge with any hooks already set (stacking multiple @use_hooks allowed).
        existing: tuple[type, ...] = getattr(fn_or_cls, USE_HOOKS_META, ())
        merged = existing + tuple(hook_classes)
        setattr(fn_or_cls, USE_HOOKS_META, merged)
        # Sync ToolMeta.hook_classes when @use_hooks is stacked above @tool()
        # (i.e. @tool() already ran and set TOOL_META on fn_or_cls).
        _tool_meta = getattr(fn_or_cls, TOOL_META, None)
        if _tool_meta is not None:
            _tool_meta.hook_classes = merged
        return fn_or_cls

    return decorator

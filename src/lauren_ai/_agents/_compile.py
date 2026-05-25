"""Startup validation for ``@agent()`` and ``@tool()`` decorated classes.

Analogous to :func:`~lauren._asgi._compile_handler_signature` in the
lauren framework: all validation happens once at startup, before any requests
are served, so misconfigured agents and tools fail loudly at boot time rather
than at runtime.

Key functions
-------------

* :func:`validate_agent_class` — validates an ``@agent()``-decorated class.
* :func:`validate_tool` — validates a ``@tool()``-decorated function or class.
"""

from __future__ import annotations

__all__ = [
    "validate_agent_class",
    "validate_tool",
]

import inspect
from typing import Any

from lauren_ai._agents import AGENT_META, AgentMeta
from lauren_ai._exceptions import AgentConfigError, ToolConfigError
from lauren_ai._tools import TOOL_META, ToolMeta

# ---------------------------------------------------------------------------
# Agent validation
# ---------------------------------------------------------------------------


def validate_agent_class(cls: type) -> AgentMeta:
    """Validate an ``@agent()``-decorated class at startup.

    Checks:

    1. The class has the :data:`~lauren_ai._agents.AGENT_META` attribute.
    2. Optional lifecycle hooks have correct signatures (must accept ``self``
       plus the expected positional arguments).
    3. Tool entries in ``USE_TOOLS_META`` are valid ``@tool()``-decorated
       objects (have ``TOOL_META`` set).

    :param cls: The class to validate.
    :type cls: type
    :return: The :class:`~lauren_ai._agents.AgentMeta` for the class.
    :rtype: AgentMeta
    :raises AgentConfigError: When any check fails.
    """
    meta: AgentMeta | None = getattr(cls, AGENT_META, None)
    if meta is None:
        raise AgentConfigError(
            f"{cls.__name__!r} is not decorated with @agent().  "
            "Only classes decorated with @agent() can be validated as agents.",
            agent_class=cls,
        )

    # Validate lifecycle hooks
    _validate_agent_hooks(cls)

    # Validate attached tools
    from lauren_ai._agents import USE_TOOLS_META  # noqa: PLC0415

    tool_classes = getattr(cls, USE_TOOLS_META, ())
    for tool_item in tool_classes:
        if tool_item is None:
            continue  # None entries are silently dropped by use_tools()
        try:
            validate_tool(tool_item)
        except ToolConfigError as exc:
            raise AgentConfigError(
                f"Agent {cls.__name__!r} has an invalid tool {tool_item!r}: {exc}",
                agent_class=cls,
                cause=exc,
            ) from exc

    return meta


def _validate_agent_hooks(cls: type) -> None:
    """Validate the optional lifecycle hook methods on an agent class.

    All hooks are optional; when present they must be instance methods with
    the correct number of positional parameters.

    Expected signatures:

    * ``on_start(self, ctx: AgentContext) -> None``
    * ``on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult``
    * ``on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None``
    * ``on_finish(self, response: AgentResponse, ctx: AgentContext) -> None``

    :param cls: The agent class to validate.
    :type cls: type
    :raises AgentConfigError: When a hook has an unexpected signature.
    """
    # (hook_name, min_positional_params_excluding_self)
    hook_specs: list[tuple[str, int, int]] = [
        # (name, min_extra_params, max_extra_params)
        ("on_start", 1, 1),  # ctx
        ("on_tool_result", 2, 2),  # result, ctx
        ("on_turn_complete", 2, 2),  # completion, ctx
        ("on_finish", 2, 2),  # response, ctx
    ]

    for hook_name, min_params, max_params in hook_specs:
        hook = cls.__dict__.get(hook_name)  # Only check own dict, not inherited
        if hook is None:
            continue  # Hook is optional

        if not callable(hook):
            raise AgentConfigError(
                f"Agent {cls.__name__!r}: lifecycle hook '{hook_name}' must be a callable, got {type(hook)!r}.",
                agent_class=cls,
            )

        try:
            sig = inspect.signature(hook)
        except (TypeError, ValueError) as exc:
            raise AgentConfigError(
                f"Agent {cls.__name__!r}: could not inspect signature of '{hook_name}': {exc}",
                agent_class=cls,
                cause=exc,
            ) from exc

        params = list(sig.parameters.values())
        # Filter out 'self' and **kwargs; count positional-capable params
        positional_params = [
            p
            for p in params
            if p.name != "self"
            and p.kind
            not in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        n = len(positional_params)
        if n < min_params or n > max_params:
            raise AgentConfigError(
                f"Agent {cls.__name__!r}: lifecycle hook '{hook_name}' must "
                f"accept between {min_params} and {max_params} positional "
                f"parameter(s) (excluding 'self'); got {n}.  "
                f"Expected signature: {hook_name}(self, " + _expected_hook_signature(hook_name) + ")",
                agent_class=cls,
            )


def _expected_hook_signature(hook_name: str) -> str:
    """Return the expected parameters portion of a hook signature string.

    :param hook_name: Hook method name.
    :type hook_name: str
    :return: Human-readable parameter list string.
    :rtype: str
    """
    sigs = {
        "on_start": "ctx: AgentContext",
        "on_tool_result": "result: ToolResult, ctx: AgentContext",
        "on_turn_complete": "completion: Completion, ctx: AgentContext",
        "on_finish": "response: AgentResponse, ctx: AgentContext",
    }
    return sigs.get(hook_name, "...")


# ---------------------------------------------------------------------------
# Tool validation
# ---------------------------------------------------------------------------


def validate_tool(tool_func_or_cls: Any) -> ToolMeta:
    """Validate a ``@tool()``-decorated function or class at startup.

    Checks:

    1. The object has the :data:`~lauren_ai._tools.TOOL_META` attribute.
    2. Class-form tools have a ``run()`` method.
    3. The entry-point function/method's parameters all have type annotations
       (unannotated parameters prevent schema generation).
    4. Schema generation succeeds (i.e. no TypeError from
       ``generate_tool_schema``).

    :param tool_func_or_cls: The object to validate.
    :type tool_func_or_cls: Any
    :return: The :class:`~lauren_ai._tools.ToolMeta` for the object.
    :rtype: ToolMeta
    :raises ToolConfigError: When any check fails.
    """
    meta: ToolMeta | None = getattr(tool_func_or_cls, TOOL_META, None)
    if meta is None:
        raise ToolConfigError(
            f"{tool_func_or_cls!r} does not have a {TOOL_META!r} attribute.  "
            "Only objects decorated with @tool() can be used as agent tools.",
            tool_name=getattr(tool_func_or_cls, "__name__", repr(tool_func_or_cls)),
        )

    tool_name = meta.name

    if inspect.isclass(tool_func_or_cls):
        # Class-form: must have a run() method
        run_method = getattr(tool_func_or_cls, "run", None)
        if run_method is None:
            raise ToolConfigError(
                f"Class-form tool {tool_func_or_cls.__name__!r} must define a 'run()' method as its entry point.",
                tool_name=tool_name,
            )
        if not callable(run_method):
            raise ToolConfigError(
                f"Class-form tool {tool_func_or_cls.__name__!r}: 'run' must be callable, got {type(run_method)!r}.",
                tool_name=tool_name,
            )
        entry = run_method
    else:
        entry = tool_func_or_cls

    # Validate that parameters have type annotations (required for schema)
    try:
        sig = inspect.signature(entry)
    except (TypeError, ValueError) as exc:
        raise ToolConfigError(
            f"Could not inspect signature of tool {tool_name!r}: {exc}",
            tool_name=tool_name,
            cause=exc,
        ) from exc

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.annotation is inspect.Parameter.empty:
            raise ToolConfigError(
                f"Tool {tool_name!r}: parameter '{param_name}' has no type "
                "annotation.  All tool parameters must be annotated for "
                "JSON schema generation.",
                tool_name=tool_name,
            )

    return meta

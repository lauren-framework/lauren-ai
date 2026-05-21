"""Tool hook types for ``@use_hooks()``-based injectable lifecycle.

Each hook is an injectable class implementing any subset of:

* :meth:`ToolHook.before_tool_call` — runs before dispatch; can abort or
  modify the tool's input dict.
* :meth:`ToolHook.after_tool_call` — runs after a successful dispatch; can
  replace the raw result.
* :meth:`ToolHook.on_tool_error` — runs when the tool raises; can suppress
  the error and return a fallback result.

All three methods have default no-op implementations so concrete hooks only
need to override what they care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lauren_ai._tools import ToolContext

__all__ = [
    "AfterToolHookDecision",
    "BeforeToolHookDecision",
    "ErrorToolHookDecision",
    "ToolCallContext",
    "ToolHook",
]

# ---------------------------------------------------------------------------
# Internal sentinel — distinguishes "no replacement" from None
# ---------------------------------------------------------------------------

_NO_REPLACE: object = object()


# ---------------------------------------------------------------------------
# ToolCallContext
# ---------------------------------------------------------------------------


@dataclass
class ToolCallContext(ToolContext):
    """Extended context passed to :class:`ToolHook` methods.

    Adds the tool name and the current input dict to the base
    :class:`~lauren_ai._tools.ToolContext` so hooks can inspect (or, via
    :meth:`BeforeToolHookDecision.modify`, replace) the parameters the model
    sent.

    :param tool_name: Snake-case name of the tool being invoked.
    :type tool_name: str
    :param tool_input: Input dict as supplied by the model.  The
        ``before_tool_call`` hook may return a modified copy via
        :meth:`BeforeToolHookDecision.modify`; subsequent hooks and the tool
        itself receive the modified version.
    :type tool_input: dict[str, Any]
    """

    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeforeToolHookDecision:
    """Return value of :meth:`ToolHook.before_tool_call`.

    Use the factory class-methods — do **not** construct directly.

    :param _aborted: Internal flag; ``True`` when the tool should be skipped.
    :param _abort_result: Result to return when aborted.
    :param _modified_input: Replacement input dict, or ``None`` when the input
        is unchanged.
    """

    _aborted: bool = False
    _abort_result: Any = None
    _modified_input: dict[str, Any] | None = None

    @classmethod
    def proceed(cls) -> BeforeToolHookDecision:
        """Continue to the next hook / tool dispatch unchanged."""
        return cls()

    @classmethod
    def abort(cls, result: Any) -> BeforeToolHookDecision:
        """Skip the tool entirely and return *result* as the tool result.

        Subsequent before hooks, the tool itself, and all after hooks are
        **not** invoked.  Error hooks are also skipped.
        """
        return cls(_aborted=True, _abort_result=result)

    @classmethod
    def modify(cls, new_input: dict[str, Any]) -> BeforeToolHookDecision:
        """Continue to the next hook / tool dispatch with a replaced input dict.

        The replacement is forwarded to every subsequent before hook and to the
        tool itself.  The :attr:`~ToolCallContext.tool_input` field on
        :class:`ToolCallContext` is updated so later hooks see the new value.
        """
        return cls(_modified_input=new_input)


@dataclass(frozen=True)
class AfterToolHookDecision:
    """Return value of :meth:`ToolHook.after_tool_call`.

    Use the factory class-methods — do **not** construct directly.

    :param _replacement: Replacement raw result, or :data:`_NO_REPLACE` when
        the result is left unchanged.
    """

    _replacement: Any = _NO_REPLACE

    @classmethod
    def proceed(cls) -> AfterToolHookDecision:
        """Pass the result through unchanged."""
        return cls()

    @classmethod
    def replace(cls, result: Any) -> AfterToolHookDecision:
        """Replace the tool result seen by subsequent hooks and the caller."""
        return cls(_replacement=result)


@dataclass(frozen=True)
class ErrorToolHookDecision:
    """Return value of :meth:`ToolHook.on_tool_error`.

    Use the factory class-methods — do **not** construct directly.

    :param _suppressed: ``True`` when the error should be swallowed.
    :param _fallback: Raw result to return when the error is suppressed.
    """

    _suppressed: bool = False
    _fallback: Any = None

    @classmethod
    def reraise(cls) -> ErrorToolHookDecision:
        """Propagate the error to subsequent hooks and finally the runner."""
        return cls()

    @classmethod
    def suppress_with(cls, result: Any) -> ErrorToolHookDecision:
        """Swallow the error and return *result* as the tool result.

        Subsequent error hooks are **not** invoked.  After hooks are also
        skipped.
        """
        return cls(_suppressed=True, _fallback=result)


# ---------------------------------------------------------------------------
# ToolHook base class
# ---------------------------------------------------------------------------


class ToolHook:
    """Base class for injectable tool lifecycle hooks.

    Subclass and override any combination of the three methods.  Unoverridden
    methods are no-ops (proceed / reraise).

    Hooks are resolved as DI singletons and shared across all tool calls that
    use them.  Do not store per-call mutable state on ``self``; use
    :attr:`ToolCallContext.state` instead.

    Example — audit logger hook::

        @injectable()
        class AuditHook(ToolHook):
            def __init__(self, audit_log: AuditLogService) -> None:
                self._log = audit_log

            async def before_tool_call(
                self, ctx: ToolCallContext
            ) -> BeforeToolHookDecision:
                self._log.record(ctx.tool_name, ctx.tool_input)
                return BeforeToolHookDecision.proceed()

            async def after_tool_call(
                self, result: Any, ctx: ToolCallContext
            ) -> AfterToolHookDecision:
                self._log.record_result(ctx.tool_name, result)
                return AfterToolHookDecision.proceed()
    """

    async def before_tool_call(
        self,
        ctx: ToolCallContext,
    ) -> BeforeToolHookDecision:
        """Called before the tool is dispatched.

        :param ctx: Extended tool context including name and input.
        :return: Decision controlling whether to proceed, abort, or modify
            the input.
        """
        return BeforeToolHookDecision.proceed()

    async def after_tool_call(
        self,
        result: Any,
        ctx: ToolCallContext,
    ) -> AfterToolHookDecision:
        """Called after a successful tool dispatch.

        :param result: The raw return value of the tool.
        :param ctx: Extended tool context.
        :return: Decision controlling whether to pass the result through or
            replace it.
        """
        return AfterToolHookDecision.proceed()

    async def on_tool_error(
        self,
        exc: Exception,
        ctx: ToolCallContext,
    ) -> ErrorToolHookDecision:
        """Called when the tool raises an exception.

        :param exc: The exception raised by the tool.
        :param ctx: Extended tool context.
        :return: Decision controlling whether to reraise or suppress the error.
        """
        return ErrorToolHookDecision.reraise()

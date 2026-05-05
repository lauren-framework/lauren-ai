"""Agent system for ``lauren-ai``.

Provides the ``@agent()`` decorator, ``use_tools()`` decorator, and the core
agent data types: :class:`AgentMeta`, :class:`AgentContext`, and
:class:`AgentResponse`.

Typical usage::

    from lauren_ai import agent, use_tools, AgentContext, AgentResponse
    from lauren_ai.skills import WebSearchTool

    @use_tools(WebSearchTool)
    @agent(model="claude-opus-4-6", system="You are a research assistant.")
    class ResearchAgent:
        async def on_start(self, ctx: AgentContext) -> None:
            ...

        async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
            ...
"""

from __future__ import annotations

__all__ = [
    "AGENT_META",
    "USE_TOOLS_META",
    "AgentMeta",
    "AgentContext",
    "AgentResponse",
    "agent",
    "use_tools",
]

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from lauren_ai._config import AgentConfig
    from lauren_ai._memory import ShortTermMemory
    from lauren_ai._signals import SignalBus as _SignalBus
    from lauren_ai._transport import TokenUsage, ToolCall

C = TypeVar("C", bound=type)

# ---------------------------------------------------------------------------
# Sentinel attribute names
# ---------------------------------------------------------------------------

#: Attribute name set on ``@agent()``-decorated classes to store ``AgentMeta``.
AGENT_META: str = "__lauren_ai_agent__"

#: Attribute name set by ``@use_tools()`` to store attached tool classes.
USE_TOOLS_META: str = "__lauren_ai_tools__"


# ---------------------------------------------------------------------------
# AgentMeta
# ---------------------------------------------------------------------------


@dataclass
class AgentMeta:
    """Metadata attached to a class decorated with ``@agent()``.

    Stored under the :data:`AGENT_META` attribute on the decorated class.

    :param model: LLM model identifier.  ``None`` means "inherit from
        :class:`~lauren_ai._config.LLMConfig` at runtime".
    :type model: str | None
    :param system: System prompt for this agent.  ``None`` falls back to the
        class docstring (if any) or the ``AgentConfig.system_prompt`` default.
    :type system: str | None
    :param config: Runtime behaviour configuration.
    :type config: AgentConfig
    :param tool_classes: Tool classes/functions attached via ``@use_tools()``.
        Resolved at compile time; ``None`` entries are already removed.
    :type tool_classes: tuple[Any, ...]
    :param name: Human-readable agent name.  Defaults to the decorated class
        name when not supplied explicitly to ``@agent()``.
    :type name: str
    """

    model: str | None
    system: str | None
    config: AgentConfig
    tool_classes: tuple[Any, ...] = field(default_factory=tuple)
    name: str = ""


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Runtime context for a single agent run.

    One :class:`AgentContext` is created at the start of every
    :meth:`~lauren_ai._agents._runner.AgentRunner.run` call and passed to
    all lifecycle hooks.

    :param agent_id: Unique identifier for this agent instance (random hex).
    :type agent_id: str
    :param agent_run_id: Unique identifier for *this specific run* (random hex).
        Distinct from ``agent_id`` — the same agent instance may be run
        multiple times.
    :type agent_run_id: str
    :param agent_class: The ``@agent()``-decorated class.
    :type agent_class: type
    :param config: Effective :class:`~lauren_ai._config.AgentConfig` for this
        run (merged from module-level defaults and per-agent overrides).
    :type config: AgentConfig
    :param memory: Short-term conversation memory for this run.
    :type memory: ShortTermMemory
    :param turn: Current agentic loop iteration (0-based).
    :type turn: int
    :param metadata: Key-value metadata bag.  Populated from
        ``@set_metadata()`` decorators and caller-supplied ``metadata=``
        arguments.
    :type metadata: dict[str, Any]
    :param request: Originating HTTP :class:`~lauren.types.Request`, or
        ``None`` when the agent is not invoked from a web handler.
    :type request: Any | None
    :param execution_context: The lauren :class:`~lauren.types.ExecutionContext`
        (carries ``request``, ``handler_class``, ``handler_func``,
        ``route_template``, and ``metadata``) when the agent is invoked
        from a route handler.  ``None`` otherwise.
    :type execution_context: Any | None
    :param signals: Signal bus for emitting lifecycle events.  ``None`` in
        environments where no :class:`SignalBus` is registered.
    :type signals: Any | None
    """

    agent_id: str
    agent_run_id: str
    agent_class: type
    config: AgentConfig
    memory: Any  # ShortTermMemory — Any avoids circular import at parse time
    turn: int
    metadata: dict[str, Any]
    request: Any | None = None
    execution_context: Any | None = None  # lauren ExecutionContext, or None
    signals: Any | None = None

    @property
    def agent_name(self) -> str:
        """Human-readable name for this agent.

        Returns :attr:`AgentMeta.name` when the class was decorated with
        ``@agent(name=...)``, otherwise falls back to the class ``__name__``.

        :return: The agent's name string.
        :rtype: str
        """
        meta: AgentMeta | None = getattr(self.agent_class, AGENT_META, None)
        if meta and meta.name:
            return meta.name
        return self.agent_class.__name__

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return metadata value for *key*, or *default* if absent.

        :param key: Metadata key to look up.
        :type key: str
        :param default: Fallback value when the key is not present.
        :type default: Any
        :return: The metadata value or *default*.
        :rtype: Any
        """
        return self.metadata.get(key, default)

    async def delegate(
        self,
        agent: Any,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Hand off execution to another agent.

        Raises :class:`~lauren_ai._exceptions.DelegateToAgent` which the
        :class:`~lauren_ai._agents._runner.AgentRunner` catches and handles by
        calling :meth:`~lauren_ai._agents._runner.AgentRunner.run` recursively
        on the target agent.

        :param agent: The ``@agent()``-decorated class or instance to delegate
            execution to.
        :type agent: Any
        :param message: The message to pass to the delegated agent.
        :type message: str
        :param metadata: Optional additional metadata for the delegated run.
        :type metadata: dict[str, Any] | None
        :return: The delegated agent's :class:`AgentResponse` (raised and caught
            by the runner, which then returns it).
        :rtype: AgentResponse
        :raises DelegateToAgent: Always — the runner intercepts this exception.
        """
        from lauren_ai._exceptions import DelegateToAgent  # noqa: PLC0415

        raise DelegateToAgent(agent=agent, message=message)


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """The result of a completed agent run.

    Returned by :meth:`~lauren_ai._agents._runner.AgentRunner.run` after the
    agentic loop terminates.

    :param content: Final text output from the agent (the last assistant
        message's text content).
    :type content: str
    :param turns: Number of agentic loop iterations executed.
    :type turns: int
    :param total_usage: Cumulative :class:`~lauren_ai._transport.TokenUsage`
        across all turns.
    :type total_usage: TokenUsage
    :param tool_calls_made: All :class:`~lauren_ai._transport.ToolCall`
        instances that were executed during the run (across all turns).
    :type tool_calls_made: list[ToolCall]
    :param stop_reason: Why the agent loop terminated:

        * ``"end_turn"`` — the model indicated a natural end.
        * ``"max_turns"`` — the ``max_turns`` limit was reached.
        * ``"budget_exceeded"`` — cost or token budget was crossed.
        * ``"delegated"`` — execution was handed off to another agent.
        * ``"error"`` — an unrecoverable error occurred.
    :type stop_reason: Literal["end_turn", "max_turns", "budget_exceeded", "delegated", "error"]
    :param metadata: Arbitrary metadata attached to the response.
    :type metadata: dict[str, Any]
    :param reasoning_traces: Extended-thinking / reasoning traces collected
        across all turns (Anthropic only).
    :type reasoning_traces: list[str]
    """

    content: str
    turns: int
    total_usage: Any  # TokenUsage — typed as Any to avoid circular import
    tool_calls_made: list[Any]  # list[ToolCall]
    stop_reason: Literal["end_turn", "max_turns", "budget_exceeded", "delegated", "error"]
    metadata: dict[str, Any] = field(default_factory=dict)
    reasoning_traces: list[str] = field(default_factory=list)

    async def as_stream(self) -> AsyncIterator[str]:
        """Wrap the response content as a single-item async iterator.

        Useful for handlers that expect an async generator regardless of
        whether the response was produced via streaming or not.

        :return: An async iterator yielding the single content string.
        :rtype: AsyncIterator[str]
        """
        yield self.content


# ---------------------------------------------------------------------------
# @agent() decorator
# ---------------------------------------------------------------------------


def agent(
    *args: Any,
    name: str | None = None,
    model: str | None = None,
    system: str | None = None,
    max_turns: int | None = None,
    temperature: float | None = None,
    **config_kwargs: Any,
) -> Callable[[type[C]], type[C]]:
    """Decorator that marks a class as an AI agent.

    Must be called **with parentheses**: ``@agent()``.  Using the bare form
    ``@agent`` (without parentheses) raises
    :class:`~lauren_ai._exceptions.DecoratorUsageError`.

    The decorated class:

    * Gets the :data:`AGENT_META` attribute set to an :class:`AgentMeta`
      instance.
    * Is automatically registered as ``@injectable(scope=Scope.SINGLETON)``
      unless ``@injectable`` is already applied.
    * Can define optional lifecycle hooks: ``on_start``, ``on_tool_result``,
      ``on_turn_complete``, ``on_finish``.

    Example::

        @use_tools(WebSearchTool, CitationTool)
        @agent(
            name="Research Agent",
            model="claude-opus-4-6",
            system="You are a research assistant.",
            max_turns=10,
            temperature=0.7,
        )
        class ResearchAgent:
            async def on_start(self, ctx: AgentContext) -> None: ...
            async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None: ...

    :param name: Human-readable agent name exposed via
        :attr:`AgentMeta.name` and :attr:`AgentContext.agent_name`.  When
        ``None`` the decorated class ``__name__`` is used.
    :type name: str | None
    :param model: LLM model identifier override.  When ``None`` the model is
        taken from :class:`~lauren_ai._config.LLMConfig` at runtime.
    :type model: str | None
    :param system: System prompt.  When ``None`` the class docstring is used,
        falling back to ``AgentConfig.system_prompt``.
    :type system: str | None
    :param max_turns: Maximum agentic loop iterations.  Forwarded to
        :class:`~lauren_ai._config.AgentConfig`.
    :type max_turns: int | None
    :param temperature: Sampling temperature override.  Forwarded to
        :class:`~lauren_ai._config.AgentConfig`.
    :type temperature: float | None
    :param config_kwargs: Additional keyword arguments forwarded to
        :class:`~lauren_ai._config.AgentConfig`.
    :return: A class decorator.
    :rtype: Callable[[type], type]
    :raises DecoratorUsageError: When called without parentheses (bare
        ``@agent``).
    """
    # Detect bare usage: @agent instead of @agent()
    if args and callable(args[0]):
        from lauren_ai._exceptions import DecoratorUsageError  # noqa: PLC0415

        raise DecoratorUsageError(
            "@agent must be used with parentheses: @agent()",
            decorator_name="agent",
        )

    def decorator(cls: type[C]) -> type[C]:
        from lauren_ai._config import AgentConfig  # noqa: PLC0415

        # Build the effective AgentConfig by merging explicit overrides into
        # an AgentConfig instance.  We start from defaults and only override
        # fields that were explicitly passed.
        agent_config_kwargs: dict[str, Any] = dict(config_kwargs)
        if max_turns is not None:
            agent_config_kwargs["max_turns"] = max_turns
        if temperature is not None:
            agent_config_kwargs["temperature"] = temperature

        # Use class docstring as fallback system prompt.
        effective_system = system
        if effective_system is None and cls.__doc__:
            effective_system = cls.__doc__.strip() or None

        # Resolve agent name: explicit > class __name__.
        effective_name: str = name if name is not None else cls.__name__

        cfg = AgentConfig(**agent_config_kwargs)

        # Gather tool classes registered by @use_tools() on this class.
        raw_tools: tuple[Any, ...] = getattr(cls, USE_TOOLS_META, ())

        meta = AgentMeta(
            model=model,
            system=effective_system,
            config=cfg,
            tool_classes=raw_tools,
            name=effective_name,
        )
        setattr(cls, AGENT_META, meta)

        # Auto-apply @injectable(scope=Scope.SINGLETON) unless already applied.
        _INJECTABLE_META = "__lauren_injectable__"
        if _INJECTABLE_META not in cls.__dict__:
            from lauren import Scope, injectable  # noqa: PLC0415

            cls = injectable(scope=Scope.SINGLETON)(cls)

        return cls  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# @use_tools() decorator
# ---------------------------------------------------------------------------


def use_tools(*tools: Any) -> Callable[[type[C]], type[C]]:
    """Attach tool classes or functions to an ``@agent()``-decorated class.

    ``None`` entries are silently dropped (consistent with ``use_guards``
    behaviour in the lauren framework).

    Typically stacked above ``@agent()``::

        @use_tools(WebSearchTool, get_weather, None)  # None is dropped
        @agent(model="claude-opus-4-6")
        class ResearchAgent: ...

    At compile time, :func:`validate_agent_class` resolves each entry from the
    DI container and builds the :class:`~lauren_ai._transport.ToolSchema` list.

    :param tools: Tool classes or callables decorated with ``@tool()``.
        ``None`` entries are ignored.
    :type tools: Any
    :return: A class decorator that attaches the tools to the class.
    :rtype: Callable[[type], type]
    """

    def decorator(cls: type[C]) -> type[C]:
        # Filter out None entries.
        filtered: tuple[Any, ...] = tuple(t for t in tools if t is not None)
        # Merge with any tools already set (stacking multiple @use_tools is allowed).
        existing: tuple[Any, ...] = getattr(cls, USE_TOOLS_META, ())
        setattr(cls, USE_TOOLS_META, existing + filtered)
        return cls

    return decorator  # type: ignore[return-value]

"""Lifecycle signals emitted by the ``lauren-ai`` agent and transport layers.

All signal classes inherit from :class:`lauren.signals.LifecycleEvent`.

Signal classes
--------------

* :class:`ModelCallStarted` — emitted before calling the LLM transport.
* :class:`ModelCallComplete` — emitted after a successful LLM completion.
* :class:`ToolCallStarted` — emitted before dispatching a tool call.
* :class:`ToolCallComplete` — emitted after a tool call finishes.
* :class:`ToolPendingApproval` — emitted when HITL confirmation is required.
* :class:`AgentTurnComplete` — emitted after each agentic loop turn.
* :class:`AgentRunComplete` — emitted when an agent run terminates.
* :class:`EmbeddingGenerated` — emitted after embedding generation.

:class:`SignalBus`
------------------

A lightweight standalone async event bus.  Handlers are registered per event
type and called concurrently when an event is emitted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

__all__ = [
    # Base
    "LifecycleEvent",
    # Signal types
    "ModelCallStarted",
    "ModelCallComplete",
    "ToolCallStarted",
    "ToolCallComplete",
    "ToolPendingApproval",
    "AgentTurnComplete",
    "AgentRunComplete",
    "EmbeddingGenerated",
    "AgentMessageSent",
    "AgentMessageRequestCompleted",
    "SubagentStarted",
    "SubagentCompleted",
    # Bus
    "SignalBus",
]

import contextlib

from lauren.signals import LifecycleEvent

from lauren_ai._messaging import AgentMessageType

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Model call signals
# ---------------------------------------------------------------------------


@dataclass
class ModelCallStarted(LifecycleEvent):  # type: ignore[misc]
    """Emitted immediately before invoking the LLM transport.

    :param model: The model identifier that will be called.
    :type model: str
    :param agent_id: Unique identifier for the current agent run, or
        ``None`` when the call originates outside an agent context.
    :type agent_id: str | None
    :param agent_class: The ``@agent()``-decorated class, or ``None``.
    :type agent_class: type | None
    :param agent_name: Human-readable agent name from :attr:`AgentMeta.name`,
        or the class ``__name__`` when not explicitly set.  Empty string when
        the call originates outside an agent context.
    :type agent_name: str
    :param messages_count: Number of messages in the prompt.
    :type messages_count: int
    :param input_tokens_estimate: Rough token estimate for the input messages
        (4 chars ≈ 1 token heuristic).
    :type input_tokens_estimate: int
    """

    model: str = ""
    agent_id: str | None = None
    agent_class: type | None = None
    agent_name: str = ""
    messages_count: int = 0
    input_tokens_estimate: int = 0


@dataclass
class ModelCallComplete(LifecycleEvent):  # type: ignore[misc]
    """Emitted after a successful LLM completion.

    :param model: The model identifier that was called.
    :type model: str
    :param agent_id: Unique identifier for the current agent run, or ``None``.
    :type agent_id: str | None
    :param agent_class: The ``@agent()``-decorated class, or ``None``.
    :type agent_class: type | None
    :param agent_name: Human-readable agent name from :attr:`AgentMeta.name`,
        or the class ``__name__`` when not explicitly set.  Empty string when
        the call originates outside an agent context.
    :type agent_name: str
    :param usage: Token usage statistics from the provider.
    :type usage: Any
    :param duration_ms: Wall-clock duration of the transport call in
        milliseconds.
    :type duration_ms: float
    :param stop_reason: The stop reason returned by the provider.
    :type stop_reason: str
    :param cost_usd: Estimated cost in USD for this completion.
    :type cost_usd: float
    """

    model: str = ""
    agent_id: str | None = None
    agent_class: type | None = None
    agent_name: str = ""
    usage: Any = None  # TokenUsage — kept as Any to avoid circular imports
    duration_ms: float = 0.0
    stop_reason: str = "unknown"
    cost_usd: float = 0.0
    total_tokens: int = 0
    turns: int = 1


# ---------------------------------------------------------------------------
# Tool call signals
# ---------------------------------------------------------------------------


@dataclass
class ToolCallStarted(LifecycleEvent):  # type: ignore[misc]
    """Emitted before dispatching a tool call.

    :param tool_name: Registered name of the tool being called.
    :type tool_name: str
    :param tool_use_id: Provider-assigned identifier for this invocation.
    :type tool_use_id: str
    :param agent_id: Unique identifier for the current agent run, or ``None``.
    :type agent_id: str | None
    :param input: The parsed input arguments passed to the tool.
    :type input: dict[str, Any]
    :param cache_hit: ``True`` if a cached result is being returned without
        executing the tool.
    :type cache_hit: bool
    """

    tool_name: str = ""
    tool_use_id: str = ""
    agent_id: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False


@dataclass
class ToolCallComplete(LifecycleEvent):  # type: ignore[misc]
    """Emitted after a tool call finishes (success or error).

    :param tool_name: Registered name of the tool.
    :type tool_name: str
    :param tool_use_id: Provider-assigned identifier for this invocation.
    :type tool_use_id: str
    :param agent_id: Unique identifier for the current agent run, or ``None``.
    :type agent_id: str | None
    :param duration_ms: Wall-clock duration of the tool execution in
        milliseconds.
    :type duration_ms: float
    :param success: ``True`` if the tool returned a result; ``False`` if it
        raised an exception.
    :type success: bool
    :param error: Human-readable error message when ``success=False``.
        ``None`` when ``success=True``.
    :type error: str | None
    """

    tool_name: str = ""
    tool_use_id: str = ""
    agent_id: str | None = None
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class ToolPendingApproval(LifecycleEvent):  # type: ignore[misc]
    """Emitted when a human-in-the-loop confirmation step is required.

    Handlers should present the tool call details to the user and accept or
    reject the call.  On rejection, raise
    :class:`~lauren_ai._exceptions.ToolConfirmationRejectedError`.

    :param agent_id: Unique identifier for the current agent run.
    :type agent_id: str
    :param agent_run_id: A secondary correlation identifier for the run.
    :type agent_run_id: str
    :param tool_name: The tool name awaiting approval.
    :type tool_name: str
    :param tool_use_id: The provider-assigned identifier.
    :type tool_use_id: str
    :param input: The tool call arguments awaiting approval.
    :type input: dict[str, Any]
    """

    agent_id: str = ""
    agent_run_id: str = ""
    tool_name: str = ""
    tool_use_id: str = ""
    input: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent lifecycle signals
# ---------------------------------------------------------------------------


@dataclass
class AgentTurnComplete(LifecycleEvent):  # type: ignore[misc]
    """Emitted after each agentic loop iteration (one model call + tool calls).

    :param agent_id: Unique identifier for the current agent run.
    :type agent_id: str
    :param agent_class: The ``@agent()``-decorated class.
    :type agent_class: type
    :param turn: The 1-based iteration index that just completed.
    :type turn: int
    :param turn_usage: Token usage for this single turn only.
    :type turn_usage: Any
    :param cumulative_usage: Cumulative token usage across all turns so far.
    :type cumulative_usage: Any
    """

    agent_id: str = ""
    agent_class: type | None = None
    turn: int = 0
    turn_usage: Any = None  # TokenUsage
    cumulative_usage: Any = None  # TokenUsage


@dataclass
class AgentRunComplete(LifecycleEvent):  # type: ignore[misc]
    """Emitted when an agent run terminates (for any reason).

    :param agent_id: Unique identifier for the completed agent run.
    :type agent_id: str
    :param agent_class: The ``@agent()``-decorated class.
    :type agent_class: type
    :param agent_name: Human-readable agent name from :attr:`AgentMeta.name`,
        or the class ``__name__`` when not explicitly set.
    :type agent_name: str
    :param turns: Number of loop iterations that were executed.
    :type turns: int
    :param total_usage: Cumulative token usage across the entire run.
    :type total_usage: Any
    :param total_cost_usd: Estimated total cost in USD for the run.
    :type total_cost_usd: float
    :param stop_reason: Why the agent loop terminated (e.g. ``"end_turn"``,
        ``"max_turns"``, ``"budget_exceeded"``).
    :type stop_reason: str
    """

    agent_id: str = ""
    agent_class: type | None = None
    agent_name: str = ""
    turns: int = 0
    total_usage: Any = None  # TokenUsage
    total_cost_usd: float = 0.0
    stop_reason: str = "unknown"


@dataclass
class SubagentStarted(LifecycleEvent):  # type: ignore[misc]
    """Emitted when a subagent tool starts an isolated child-agent run."""

    parent_agent_name: str = ""
    subagent_name: str = ""
    parent_run_id: str = ""
    conversation_id: str | None = None
    brief_length_chars: int = 0


@dataclass
class SubagentCompleted(LifecycleEvent):  # type: ignore[misc]
    """Emitted when a subagent tool completes, whether successful or not."""

    parent_agent_name: str = ""
    subagent_name: str = ""
    parent_run_id: str = ""
    conversation_id: str | None = None
    elapsed_ms: float = 0.0
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Embedding signal
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingGenerated(LifecycleEvent):  # type: ignore[misc]
    """Emitted after an embedding batch is generated.

    :param model: The embedding model identifier.
    :type model: str
    :param input_count: Number of input strings that were embedded.
    :type input_count: int
    :param duration_ms: Wall-clock duration of the embedding call in
        milliseconds.
    :type duration_ms: float
    """

    model: str = ""
    input_count: int = 0
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Agent messaging signals
# ---------------------------------------------------------------------------


@dataclass
class AgentMessageSent(LifecycleEvent):  # type: ignore[misc]
    """Emitted after an inter-agent message is routed."""

    message_id: Any = None
    from_agent: str = ""
    to: str | None = None
    topic: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    correlation_id: Any = None
    message_type: AgentMessageType = AgentMessageType.NOTIFICATION
    receiver_count: int = 0
    dropped_count: int = 0
    attempt: int = 1


@dataclass
class AgentMessageRequestCompleted(LifecycleEvent):  # type: ignore[misc]
    """Emitted when a request/response exchange completes or times out."""

    request_id: Any = None
    from_agent: str = ""
    target: str = ""
    session_id: str | None = None
    task_id: str | None = None
    elapsed_ms: float = 0.0
    attempts: int = 1
    timed_out: bool = False


# ---------------------------------------------------------------------------
# SignalBus — standalone, does not require lauren
# ---------------------------------------------------------------------------


class SignalBus:
    """Lightweight standalone async event bus.

    Handlers are async callables that accept a single event argument.  They
    are registered per event type and called concurrently when an event is
    emitted.  Exceptions raised by individual handlers are caught, printed to
    ``stderr``, and suppressed so that one failing handler cannot block the
    others or the caller.

    This class does **not** require the ``lauren`` framework.

    Example::

        bus = SignalBus()

        @bus.on(ModelCallComplete)
        async def log_cost(event: ModelCallComplete) -> None:
            print(f"Cost: ${event.cost_usd:.6f}")

        await bus.emit(ModelCallComplete(model="claude-opus-4-6", cost_usd=0.001))
    """

    def __init__(self) -> None:
        """Initialise the signal bus with an empty handler registry."""
        self._handlers: dict[type, list[Callable[..., Awaitable[None]]]] = {}

    def on(
        self,
        event_type: type,
    ) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
        """Register a handler for *event_type*.

        Can be used as a decorator::

            @bus.on(ModelCallComplete)
            async def handle(event: ModelCallComplete) -> None: ...

        :param event_type: The event class to subscribe to.
        :type event_type: type
        :return: A decorator that registers the handler and returns it
            unchanged.
        :rtype: Callable
        """

        def decorator(handler: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler

        return decorator

    async def emit(self, event: Any) -> None:
        """Emit *event* to all registered handlers for its type.

        Handlers are called concurrently via :func:`asyncio.gather`.
        Individual handler exceptions are caught, printed to ``stderr``,
        and suppressed.

        :param event: The event instance to emit.
        :type event: Any
        """
        handlers = self._handlers.get(type(event), [])
        if not handlers:
            return
        coros = [handler(event) for handler in handlers]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                import sys  # noqa: PLC0415

                print(
                    f"SignalBus: handler raised {type(result).__name__}: {result}",
                    file=sys.stderr,
                )

    def off(
        self,
        event_type: type,
        handler: Callable[..., Awaitable[None]],
    ) -> None:
        """Unregister a previously-registered handler.

        A no-op if *handler* is not registered for *event_type*.

        :param event_type: The event type the handler was registered for.
        :type event_type: type
        :param handler: The handler to unregister.
        :type handler: Callable
        """
        handlers = self._handlers.get(event_type)
        if handlers is None:
            return
        with contextlib.suppress(ValueError):
            handlers.remove(handler)
        if not handlers:
            del self._handlers[event_type]

    def clear(self, event_type: type | None = None) -> None:
        """Remove all handlers, optionally scoped to a specific *event_type*.

        :param event_type: When provided, only handlers for this event type
            are removed.  When ``None``, all handlers across all types are
            cleared.
        :type event_type: type | None
        """
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)

    def handler_count(self, event_type: type) -> int:
        """Return the number of handlers registered for *event_type*.

        :param event_type: The event type to query.
        :type event_type: type
        :return: Number of registered handlers.
        :rtype: int
        """
        return len(self._handlers.get(event_type, []))

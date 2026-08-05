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
from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = [
    # Base
    "LifecycleEvent",
    # Signal types
    "ModelCallStarted",
    "ModelCallComplete",
    "ToolCallStarted",
    "ToolCallComplete",
    "ToolExchangeStarted",
    "ToolExchangeResultRecorded",
    "ToolExchangeCommitted",
    "ToolExchangeAborted",
    "ToolExchangeRepaired",
    "ToolConversationInvariantViolation",
    "ToolSerializationBlocked",
    "ToolPendingApproval",
    "ToolApprovalResolved",
    "AgentTurnComplete",
    "AgentRunComplete",
    "EmbeddingGenerated",
    "AgentMessageSent",
    "AgentMessageRequestCompleted",
    "SubagentStarted",
    "SubagentCompleted",
    # MCP-specific signals
    "ToolProgressEvent",
    "McpToolsRefreshed",
    # Bus
    "SignalBus",
    # Event sink (PRD: pluggable-event-sink)
    "EventSink",
    "serialize",
]

import contextlib
import dataclasses

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
class ToolExchangeStarted(LifecycleEvent):  # type: ignore[misc]
    """Emitted when a complete assistant tool-call batch begins execution."""

    exchange_id: str = ""
    run_id: str | None = None
    agent_id: str | None = None
    call_count: int = 0


@dataclass
class ToolExchangeResultRecorded(LifecycleEvent):  # type: ignore[misc]
    """Emitted after one result is durably associated with an exchange."""

    exchange_id: str = ""
    run_id: str | None = None
    agent_id: str | None = None
    tool_use_id: str = ""
    status: str = ""
    synthetic: bool = False


@dataclass
class ToolExchangeCommitted(LifecycleEvent):  # type: ignore[misc]
    """Emitted after every requested call has a canonical result."""

    exchange_id: str = ""
    run_id: str | None = None
    agent_id: str | None = None
    call_count: int = 0
    completed_count: int = 0
    synthetic_count: int = 0


@dataclass
class ToolExchangeAborted(LifecycleEvent):  # type: ignore[misc]
    """Emitted when an interrupted exchange is closed with recovery results."""

    exchange_id: str = ""
    run_id: str | None = None
    agent_id: str | None = None
    call_count: int = 0
    repaired: bool = False


@dataclass
class ToolExchangeRepaired(LifecycleEvent):  # type: ignore[misc]
    """Emitted after deterministic recovery changes canonical memory."""

    exchange_id: str = ""
    run_id: str | None = None
    agent_id: str | None = None
    call_count: int = 0


@dataclass
class ToolConversationInvariantViolation(LifecycleEvent):  # type: ignore[misc]
    """Emitted when malformed canonical history blocks provider I/O."""

    code: str = ""
    expected_count: int = 0
    observed_count: int = 0
    assistant_index: int | None = None
    repairable: bool = False
    provider: str = ""
    agent_id: str | None = None


@dataclass
class ToolSerializationBlocked(LifecycleEvent):  # type: ignore[misc]
    """Emitted when a provider serializer rejects invalid tool history."""

    provider: str = ""
    code: str = ""
    expected_count: int = 0
    observed_count: int = 0


@dataclass
class ToolProgressEvent(LifecycleEvent):  # type: ignore[misc]
    """Emitted when an MCP tool sends a ``notifications/progress`` message.

    Subscribers can forward live progress updates to a client-facing transport
    (SSE, WebSocket) without polling.

    :param tool_name: Namespaced tool name in ``{alias}__{name}`` format.
    :type tool_name: str
    :param tool_use_id: Provider-assigned identifier correlating to the active
        tool call.  Empty string when the progress token is not a tool-use ID.
    :type tool_use_id: str
    :param agent_id: The active agent run identifier, or ``None``.
    :type agent_id: str | None
    :param agent_name: Human-readable agent name, or empty string.
    :type agent_name: str
    :param progress: Current progress value reported by the server.
    :type progress: float
    :param total: Optional upper bound.  ``None`` means indeterminate.
    :type total: float | None
    :param message: Optional human-readable status string from the server.
    :type message: str | None
    :param alias: The MCP server alias (e.g. ``"code_runner"``).
    :type alias: str
    """

    tool_name: str = ""
    tool_use_id: str = ""
    agent_id: str | None = None
    agent_name: str = ""
    progress: float = 0.0
    total: float | None = None
    message: str | None = None
    alias: str = ""


@dataclass
class McpToolsRefreshed(LifecycleEvent):  # type: ignore[misc]
    """Emitted when an MCP server's tool catalogue changes at runtime.

    Fired by the dynamic tool discovery bridge after it has atomically updated
    ``AgentMeta.tools`` in response to a ``notifications/tools/list_changed``
    notification (or a poll-based refresh).

    :param alias: The MCP server alias whose tools changed.
    :type alias: str
    :param added: Tool names added to the catalogue.
    :type added: list[str]
    :param removed: Tool names removed from the catalogue.
    :type removed: list[str]
    :param total: Total number of tools after the refresh.
    :type total: int
    """

    alias: str = ""
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    total: int = 0


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


@dataclass
class ToolApprovalResolved(LifecycleEvent):  # type: ignore[misc]
    """Emitted when a pending HITL approval is resolved (approved or denied).

    Fired immediately after :class:`ToolPendingApproval` is resolved — either
    by a call to ``approve_tool()``, ``reject_tool()``, or by timeout.

    :param agent_id: Unique identifier for the current agent run.
    :type agent_id: str
    :param agent_run_id: A secondary correlation identifier for the run.
    :type agent_run_id: str
    :param tool_name: The tool whose approval was resolved.
    :type tool_name: str
    :param tool_use_id: The provider-assigned identifier.
    :type tool_use_id: str
    :param approved: ``True`` when the tool was approved; ``False`` when
        denied or timed out.
    :type approved: bool
    :param reason: Human-readable reason for denial.  ``""`` when approved.
        ``"timeout"`` when the approval window expired.
        ``"denied by user"`` when explicitly rejected via ``reject_tool()``.
    :type reason: str
    """

    agent_id: str = ""
    agent_run_id: str = ""
    tool_name: str = ""
    tool_use_id: str = ""
    approved: bool = False
    reason: str = ""


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
# EventSink protocol (PRD: pluggable-event-sink)
# ---------------------------------------------------------------------------


@runtime_checkable
class EventSink(Protocol):
    """Receives every lifecycle signal emitted by an :class:`AgentRunnerBase`.

    Implementations must be exception-safe: anything raised by
    :meth:`on_signal` is logged at ``WARNING`` and swallowed so it can
    never interrupt the agentic loop.

    Sinks are awaited **sequentially, in registration order**, before the
    :class:`SignalBus` fan-out — within a single run a sink observes
    signals in exactly the order they were emitted.

    Example::

        class KernelBridge:
            def __init__(self, processor: EventProcessor) -> None:
                self._processor = processor

            async def on_signal(self, signal: Any) -> None:
                await self._processor.emit(Event.create(
                    event_type=f"lauren.{type(signal).__name__}",
                    payload=serialize(signal),
                ))
    """

    async def on_signal(self, signal: Any) -> None: ...


# ---------------------------------------------------------------------------
# serialize() — JSON-safe dict from any LifecycleEvent dataclass
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively make *value* JSON-serialisable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {f.name: _json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)}
        result["signal_type"] = type(value).__name__
        return result
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted([_json_safe(v) for v in value], key=str)
    # Enum → .value
    try:
        return value.value  # type: ignore[attr-defined]
    except AttributeError:
        pass
    # datetime / date → ISO-8601
    try:
        return value.isoformat()  # type: ignore[attr-defined]
    except AttributeError:
        pass
    # UUID → str
    try:
        return str(value.hex)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    # Final fallback
    try:
        return repr(value)
    except Exception:  # noqa: BLE001
        return "<unserializable>"


def serialize(event: Any) -> dict[str, Any]:
    """Return a JSON-safe ``dict`` representation of a lifecycle signal.

    Works for any dataclass-based :class:`LifecycleEvent` subclass — present
    or future — without per-class knowledge:

    - the result always contains ``"signal_type": type(event).__name__``
    - nested dataclasses (e.g. ``TokenUsage``) are recursed
    - ``type`` values (e.g. ``agent_class``) become ``"module.QualName"`` strings
    - ``Enum`` → ``.value``; ``datetime`` → ISO-8601; ``set`` / ``frozenset`` →
      sorted ``list``
    - anything else not JSON-encodable falls back to ``repr()``

    The output is designed for append-only event logs (e.g. Agenthicc's kernel
    log); it is not intended for round-trip reconstruction of the dataclass.

    :param event: A signal instance (any dataclass).
    :type event: Any
    :return: JSON-safe dictionary with ``"signal_type"`` discriminator key.
    :rtype: dict[str, Any]
    :raises TypeError: When *event* is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(event) or isinstance(event, type):
        raise TypeError(f"serialize() requires a dataclass instance; got {type(event)!r}")
    result: dict[str, Any] = {"signal_type": type(event).__name__}
    for f in dataclasses.fields(event):
        result[f.name] = _json_safe(getattr(event, f.name))
    return result


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
        self._any_handlers: list[Callable[..., Awaitable[None]]] = []

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
        """Emit *event* to all registered handlers for its type, then to
        wildcard handlers registered via :meth:`on_any`.

        Handlers are called concurrently via :func:`asyncio.gather`.
        Typed handlers precede wildcard handlers in argument order so
        invocation *start* order is deterministic.  Individual handler
        exceptions are caught, printed to ``stderr``, and suppressed.

        :param event: The event instance to emit.
        :type event: Any
        """
        handlers = [*self._handlers.get(type(event), []), *self._any_handlers]
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

    def on_any(
        self,
        handler: Callable[..., Awaitable[None]],
    ) -> Callable[..., Awaitable[None]]:
        """Register *handler* for **every** emitted event, regardless of type.

        Usable directly or as a bare decorator::

            @bus.on_any
            async def audit(event: Any) -> None:
                log.info("signal %s", type(event).__name__)

        Wildcard handlers are called via the same :func:`asyncio.gather` as
        typed handlers for the event, **after** them in argument order, so
        invocation start order is typed → wildcard.  Returns *handler*
        unchanged.

        :param handler: Async callable accepting a single event argument.
        :type handler: Callable
        :return: *handler* unchanged (enables use as a decorator).
        :rtype: Callable
        """
        self._any_handlers.append(handler)
        return handler

    def off_any(self, handler: Callable[..., Awaitable[None]]) -> None:
        """Unregister a wildcard handler.  No-op when not registered.

        :param handler: The handler to remove.
        :type handler: Callable
        """
        with contextlib.suppress(ValueError):
            self._any_handlers.remove(handler)

    def any_handler_count(self) -> int:
        """Return the number of registered wildcard handlers.

        :return: Wildcard handler count.
        :rtype: int
        """
        return len(self._any_handlers)

    def clear(self, event_type: type | None = None) -> None:
        """Remove all handlers, optionally scoped to a specific *event_type*.

        When *event_type* is ``None``, both typed and wildcard handlers are
        cleared.  When a type is provided, only handlers for that type are
        removed (wildcard handlers are untouched).

        :param event_type: When provided, only handlers for this event type
            are removed.  When ``None``, all handlers across all types and
            all wildcard handlers are cleared.
        :type event_type: type | None
        """
        if event_type is None:
            self._handlers.clear()
            self._any_handlers.clear()
        else:
            self._handlers.pop(event_type, None)

    def handler_count(self, event_type: type) -> int:
        """Return the number of handlers registered for *event_type*.

        Does **not** include wildcard handlers.  See :meth:`any_handler_count`.

        :param event_type: The event type to query.
        :type event_type: type
        :return: Number of registered handlers.
        :rtype: int
        """
        return len(self._handlers.get(event_type, []))

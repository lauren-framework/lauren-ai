"""Agent runner — the main agentic loop execution engine.

:class:`AgentRunner` is a ``@runtime_checkable`` Protocol that defines the
injection interface.  :class:`AgentRunnerBase` is the concrete implementation
that owns the observe → think → act → observe loop.

:class:`AgentModule` generates a unique :class:`AgentRunnerBase` subclass per
``for_root()`` call and registers it within its module.  Any service or tool
in that module can inject ``runner: AgentRunner`` to receive the module's
runner without naming it explicitly.

Streaming mode (``run_stream``) follows the same loop but yields
:class:`~lauren_ai._transport.CompletionChunk` items from the transport as
they arrive; tool calls are executed silently between turns.
"""

from __future__ import annotations

__all__ = [
    "AgentRunner",
    "AgentRunnerBase",
]

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from lauren_ai._agents import AGENT_META, AgentContext, AgentMeta, AgentResponse
from lauren_ai._config import AgentConfig
from lauren_ai._exceptions import (
    AgentBudgetExceededError,
    AgentConfigError,
)
from lauren_ai._memory import ShortTermMemory
from lauren_ai._tools import TOOL_METADATA, ToolContext, ToolResult
from lauren_ai._tools._executor import CacheBackend, ToolExecutor
from lauren_ai._tools._executor import ToolCall as ExecutorToolCall
from lauren_ai._transport import Completion, CompletionChunk, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context-window summarisation helpers
# ---------------------------------------------------------------------------

_SUMMARISATION_PROMPT_TEMPLATE = (
    "The following is a portion of an ongoing conversation. "
    "Please write a concise summary that captures all important facts, "
    "decisions, and context needed to continue the conversation. "
    "Focus on information the assistant will need to answer future questions.\n\n"
    "Conversation to summarise:\n\n{turns}"
)


def _build_system_prompt(base: str, memory: ShortTermMemory) -> str:
    """Return *base* with the memory summary appended when one exists."""
    summary = memory.summary
    if not summary:
        return base
    return f"{base}\n\n[Earlier conversation summary]\n{summary}"


def _should_summarize(
    memory: ShortTermMemory,
    config: AgentConfig,
) -> bool:
    """Return ``True`` when a summarisation call should be triggered."""
    if config.summarize_at is None:
        return False
    threshold = config.memory_window_tokens * config.summarize_at
    return memory.token_estimate >= threshold


async def _summarize_memory(
    memory: ShortTermMemory,
    transport: Any,
    *,
    model: str,
    keep_recent: int = 6,
) -> None:
    """Compress older conversation turns into a summary stored on *memory*.

    The ``keep_recent`` most-recent non-system messages are preserved verbatim;
    everything older is distilled into a single text block via a cheap LLM call.
    The summary is injected into the system prompt by ``_build_system_prompt``
    so it is always visible to the model and cannot be dropped by the sliding
    window.

    :param memory: The short-term memory buffer to compress.
    :param transport: The LLM transport used to generate the summary.
    :param model: Model identifier for the summarisation call.
    :param keep_recent: Number of most-recent non-system messages to keep
        verbatim.  Defaults to 6 (≈ 3 user/assistant pairs).
    """
    to_compress = memory.messages_to_summarize(keep_recent)
    if not to_compress:
        return

    # Format the turns into a readable transcript for the summarisation prompt.
    transcript_parts: list[str] = []
    for msg in to_compress:
        role = (
            msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
        )
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, list):
            # Extract text portions from structured content blocks
            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(block.get("text") or block.get("content") or "")
                else:
                    texts.append(str(block))
            content = " ".join(t for t in texts if t)
        transcript_parts.append(f"{role.upper()}: {content}")

    transcript = "\n\n".join(transcript_parts)
    summary_prompt = _SUMMARISATION_PROMPT_TEMPLATE.format(turns=transcript)

    try:
        result = await transport.complete(
            [Message.user(summary_prompt)],
            model=model,
            system="You are a conversation summariser. Be concise and factual.",
            max_tokens=512,
            temperature=0.0,
            stream=False,
        )
        summary_text = result.content or ""
    except Exception as exc:  # noqa: BLE001
        # Summarisation failing must never break the agent run — log and skip.
        logger.warning("lauren_ai: context-window summarisation failed: %s", exc)
        return

    if summary_text:
        memory.trim_to_recent(keep_recent)
        memory.set_summary(summary_text)


# Module-level cache for ``AgentRunner[X]`` parameterizations.  Hoisted out
# of the Protocol class body so non-method members don't break
# ``issubclass()`` against the Protocol (which the structural-Protocol
# fallback in lauren-framework's DI uses).
_AGENT_RUNNER_PARAM_CACHE: dict[type, type] = {}


class _AgentRunnerParam:
    """Base for ``AgentRunner[X]`` parameterizations.

    Deliberately **not** a subclass of :class:`AgentRunner` (the Protocol).
    If it were, every ``AgentRunner[X]`` token registered as a DI provider
    would also satisfy ``issubclass(cls, AgentRunner)`` — and the
    structural-Protocol fallback in lauren-framework would pick all of
    them up alongside the real runner, raising
    ``ProtocolAmbiguityError`` for plain ``runner: AgentRunner``.

    These classes are used purely as DI tokens — the resolver finds an
    explicit provider for each ``AgentRunner[X]`` and never hits the
    structural fallback.  The actual instance handed to the consumer is a
    concrete ``AgentRunnerBase`` subclass (the module's runner singleton).
    """

    _agent_param: type | None = None


def _agent_runner_class_getitem(cls: type, agent_cls: type) -> type:
    """Return a cached real class bound to *agent_cls*, suitable as a DI token.

    Mirrors the :class:`HandoffTo` / :class:`HandoffBackTo` precedent —
    parameterized form is a real class (not ``_GenericAlias``), so
    lauren-framework's DI compiler accepts it as a constructor annotation.

    :raises TypeError: When *agent_cls* is not a class or is not
        ``@agent()``-decorated.
    """
    if not isinstance(agent_cls, type):
        raise TypeError(f"AgentRunner[X] requires X to be a class; got {agent_cls!r}")
    if not hasattr(agent_cls, AGENT_META):
        raise TypeError(
            f"AgentRunner[{agent_cls.__name__}] — {agent_cls.__name__} "
            f"is not @agent-decorated.  Decorate it with @agent(...) first."
        )
    cached = _AGENT_RUNNER_PARAM_CACHE.get(agent_cls)
    if cached is not None:
        return cached
    new_cls = type(
        f"AgentRunner[{agent_cls.__name__}]",
        (_AgentRunnerParam,),
        {"_agent_param": agent_cls},
    )
    _AGENT_RUNNER_PARAM_CACHE[agent_cls] = new_cls
    return new_cls


@runtime_checkable
class AgentRunner(Protocol):
    """Structural interface for agent runner implementations.

    In-module DI
    ------------
    Declare ``runner: AgentRunner`` in any service or tool inside the same
    :class:`~lauren_ai._module.AgentModule` and the DI container injects
    that module's runner automatically.

    Cross-module DI — ``AgentRunner[AgentX]``
    -----------------------------------------
    For controllers in other modules that need a *specific* agent's
    runner, subscript ``AgentRunner`` with the agent class:

        class MyController:
            def __init__(
                self,
                unauth_runner: AgentRunner[UnauthenticatedCRMAgent],
                auth_runner:   AgentRunner[AuthenticatedCRMAgent],
            ): ...

    ``AgentRunner[X]`` returns a fresh, **cached** real subclass — so
    ``AgentRunner[X] is AgentRunner[X]`` and the parameterized form is a
    valid DI token.  ``AgentModule.for_root`` registers
    ``use_existing(provide=AgentRunner[agent_cls],
    existing=<module's runner>)`` for every agent in ``agents=``, so the
    container can resolve cross-module references by agent class.

    The mechanism mirrors :class:`HandoffTo` / :class:`HandoffBackTo`'s
    ``__class_getitem__`` precedent — the parameterized form is a real
    class (not ``_GenericAlias``), so the framework's ``_looks_injectable``
    check accepts it as a constructor annotation.

    Static-typing note
    ------------------
    Because subscript returns a real subclass via ``__class_getitem__``,
    static type-checkers (mypy, pyright) see ``AgentRunner[X]`` as bare
    ``AgentRunner`` — the type parameter ``X`` is *not* preserved for
    static analysis.  This is the same limitation as ``HandoffTo[X, Y]``.
    Runtime DI resolution is unaffected.
    """

    # Method-only Protocol body so ``issubclass(p.cls, AgentRunner)`` keeps
    # working (the structural-Protocol fallback in lauren-framework's DI
    # uses it).  ``__class_getitem__`` is bound below, after class creation.

    async def run(
        self,
        agent: Any,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        request: Any | None = None,
        execution_context: Any | None = None,
        run_id: str | None = None,
    ) -> AgentResponse: ...

    async def run_stream(
        self,
        agent: Any,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        request: Any | None = None,
        execution_context: Any | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[CompletionChunk]: ...

    async def approve_tool(self, agent_run_id: str, tool_use_id: str) -> None: ...

    async def reject_tool(
        self, agent_run_id: str, tool_use_id: str, *, reason: str = ""
    ) -> None: ...


# Attach ``__class_getitem__`` after the Protocol body so it doesn't appear in
# the typing.Protocol member scan (which would treat it as a non-method member
# and break ``issubclass()`` matching).  Type-ignored because mypy doesn't
# know about the class object's attribute slot.
AgentRunner.__class_getitem__ = classmethod(_agent_runner_class_getitem)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Guardrail helpers
# ---------------------------------------------------------------------------
#
# ``@use_guardrails()`` stores ``UseGuardrailsMeta`` on the agent class at
# ``__lauren_ai_use_guardrails__``.  The runner reads it here and invokes the
# guardrail checks at the appropriate points in the agentic loop.


def _get_input_guardrails(agent: Any) -> list[Any]:
    from lauren_ai._guardrails._decorator import USE_GUARDRAILS_META  # noqa: PLC0415

    cls = type(agent) if not isinstance(agent, type) else agent
    meta = getattr(cls, USE_GUARDRAILS_META, None)
    return meta.input_guardrails if meta else []


def _get_output_guardrails(agent: Any) -> list[Any]:
    from lauren_ai._guardrails._decorator import USE_GUARDRAILS_META  # noqa: PLC0415

    cls = type(agent) if not isinstance(agent, type) else agent
    meta = getattr(cls, USE_GUARDRAILS_META, None)
    return meta.output_guardrails if meta else []


async def _run_input_guardrails(
    guardrails: list[Any],
    message: str,
    agent_name: str,
) -> Any | None:
    """Run input guardrails in order; return the first non-pass decision."""
    from lauren_ai._guardrails._base import GuardrailContext  # noqa: PLC0415

    ctx = GuardrailContext(agent_name=agent_name)
    for guard in guardrails:
        decision = await guard.check(message, ctx)
        if decision.action != "pass":
            return decision
    return None


async def _run_output_guardrails(
    guardrails: list[Any],
    response_text: str,
    agent_name: str,
) -> Any | None:
    """Run output guardrails in order; return the first non-pass decision."""
    from lauren_ai._guardrails._base import GuardrailContext  # noqa: PLC0415

    ctx = GuardrailContext(agent_name=agent_name)
    for guard in guardrails:
        decision = await guard.check(response_text, ctx)
        if decision.action != "pass":
            return decision
    return None


class AgentRunnerBase(AgentRunner):
    """Concrete implementation of the :class:`AgentRunner` Protocol.

    Owns the observe → think → act → observe loop.  Resolves agent meta from
    the decorated class, creates per-run state (:class:`~lauren_ai._agents.AgentContext`
    and :class:`~lauren_ai._memory.ShortTermMemory`), calls the LLM transport,
    dispatches tool calls, and aggregates results into an
    :class:`~lauren_ai._agents.AgentResponse`.

    :param transport: Provider-agnostic LLM transport.
    :type transport: Any
    :param signals: Optional signal bus for emitting lifecycle events.
    :type signals: Any | None
    :param cache_backend: Optional cache backend for tool result caching.
    :type cache_backend: CacheBackend | None
    """

    def __init__(
        self,
        transport: Any,
        signals: Any | None = None,
        cache_backend: CacheBackend | None = None,
        global_hooks: list[Any] | None = None,
    ) -> None:
        self._transport = transport
        self._signals = signals
        self._executor = ToolExecutor(
            tools={},
            cache_backend=cache_backend,
            signals=signals,
            global_hooks=global_hooks,
        )
        # Pending HITL approvals: agent_run_id -> tool_use_id -> Future
        self._pending_approvals: dict[str, dict[str, asyncio.Future[bool]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        agent: Any,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        request: Any | None = None,
        execution_context: Any | None = None,
        run_id: str | None = None,
        conversation_store: Any | None = None,
        memory: Any | None = None,
    ) -> AgentResponse:
        """Run an ``@agent()``-decorated instance through the agentic loop.

        Returns once the loop terminates (end of turn, max turns, budget
        exceeded, or delegation).

        :param agent: A resolved ``@agent()``-decorated class instance (from
            the DI container) or the class itself (auto-resolved when a
            container is set).
        :type agent: Any
        :param message: The initial user message to seed the conversation.
        :type message: str
        :param conversation_id: Optional conversation session identifier.
            When provided, initial history is loaded from the effective
            conversation store.
        :type conversation_id: str | None
        :param metadata: Additional key-value metadata injected into
            :class:`~lauren_ai._agents.AgentContext`.
        :type metadata: dict[str, Any] | None
        :param request: Originating HTTP request, if any.
        :type request: Any | None
        :param execution_context: The lauren ``ExecutionContext`` (route
            metadata, handler class/func, authenticated user via
            ``request.state``) when invoked from a route handler.
        :type execution_context: Any | None
        :param run_id: Optional explicit run identifier.  A random hex string
            is generated when ``None``.
        :type run_id: str | None
        :param conversation_store: Per-request override of the agent's
            conversation store.  Wins over ``meta.conversation_store``.
        :type conversation_store: Any | None
        :param memory: Per-request override of the agent's memory instance.
            Wins over ``meta.memory``.  When neither is supplied, a fresh
            :class:`~lauren_ai._memory.ShortTermMemory` is constructed for
            this turn.
        :type memory: Any | None
        :return: The aggregated result of the agentic run.
        :rtype: AgentResponse
        :raises AgentConfigError: When *agent* is not decorated with
            ``@agent()``.
        :raises AgentMaxTurnsError: When the ``max_turns`` limit is reached
            and ``tool_error_policy`` is ``"raise"``.
        :raises AgentBudgetExceededError: When the cost / token budget is
            crossed mid-run.
        """
        meta = self._get_meta(agent)
        effective_config = self._merge_config(meta)
        agent_run_id = run_id or uuid.uuid4().hex
        agent_id = uuid.uuid4().hex

        # Resolve per-run state from AgentMeta with per-request overrides.
        # The store / memory live with the agent class (set via @agent(...))
        # so two agents in the same module never share state by default.
        # Use explicit ``is None`` checks — ShortTermMemory has ``__len__``
        # so an empty memory is falsy under ``or``.
        effective_store = (
            conversation_store if conversation_store is not None else meta.conversation_store
        )
        if memory is None:
            memory = (
                meta.memory
                if meta.memory is not None
                else ShortTermMemory(max_tokens=effective_config.memory_window_tokens)
            )
        if conversation_id and effective_store is not None:
            prior = await effective_store.load(conversation_id)
            if prior:
                memory.restore(prior)
        # ── Input guardrails ─────────────────────────────────────────────────
        _input_guards = _get_input_guardrails(agent)
        if _input_guards:
            try:
                _inp_decision = await _run_input_guardrails(_input_guards, message, meta.name)
            except Exception as _exc:  # noqa: BLE001
                logger.warning("lauren_ai: input guardrail check failed — failing open: %s", _exc)
                _inp_decision = None
            if _inp_decision is not None:
                _redirect = (
                    _inp_decision.modified_content
                    if _inp_decision.action == "modify"
                    else (_inp_decision.violation or "Message blocked by guardrail.")
                )
                if _inp_decision.action == "block":
                    return AgentResponse(
                        content=_redirect,
                        turns=0,
                        total_usage=TokenUsage(input_tokens=0, output_tokens=0),
                        tool_calls_made=[],
                        stop_reason="end_turn",
                    )
                # modify — replace the seeded user message with the safe version
                message = _redirect
                memory.restore([])
                memory.add_user(message)
            else:
                memory.add_user(message)
        else:
            memory.add_user(message)

        # Agent context
        ctx = AgentContext(
            agent_id=agent_id,
            agent_run_id=agent_run_id,
            agent_class=type(agent) if not isinstance(agent, type) else agent,
            config=effective_config,
            memory=memory,
            turn=0,
            metadata=dict(metadata or {}),
            request=request,
            execution_context=execution_context,
            signals=self._signals,
        )

        # Determine model to use
        model = meta.model
        if model is None:
            raise AgentConfigError(
                f"Agent '{ctx.agent_name}' has no configured model. Set one via @agent(...)."
            )
        system_prompt = _build_system_prompt(meta.system or effective_config.system_prompt, memory)

        # Gather tool schemas for attached tools
        tool_schemas = self._get_tool_schemas(meta)

        total_usage = TokenUsage(input_tokens=0, output_tokens=0)
        all_tool_calls: list[ToolCall] = []
        last_completion: Completion | None = None
        stop_reason: str = "max_turns"
        _output_guards = _get_output_guardrails(agent)

        try:
            # Lifecycle hook: on_start
            await self._call_hook(agent, "on_start", ctx)

            # Signal: ModelCallStarted
            await self._emit(
                "ModelCallStarted",
                model=model,
                agent_id=agent_run_id,
                agent_class=ctx.agent_class,
                agent_name=ctx.agent_name,
                messages_count=len(memory.messages()),
                input_tokens_estimate=memory.token_estimate,
            )

            for _turn in range(effective_config.max_turns):
                ctx.turn = _turn

                # ── Context-window summarisation (opt-in) ─────────────────
                if _should_summarize(memory, effective_config):
                    await _summarize_memory(
                        memory=memory,
                        transport=self._transport,
                        model=effective_config.summary_model or model,
                    )
                    # Rebuild system prompt with new summary
                    system_prompt = _build_system_prompt(
                        meta.system or effective_config.system_prompt, memory
                    )

                messages = memory.messages()

                t0 = time.monotonic()
                completion = await self._transport.complete(
                    messages,
                    model=model,
                    system=system_prompt,
                    tools=tool_schemas if tool_schemas else None,
                    max_tokens=effective_config.max_tokens_per_turn,
                    temperature=effective_config.temperature,
                    stream=False,
                )
                duration_ms = (time.monotonic() - t0) * 1000

                # Accumulate usage
                turn_usage = completion.usage
                total_usage = total_usage + turn_usage
                last_completion = completion

                # Signal: ModelCallComplete
                await self._emit(
                    "ModelCallComplete",
                    model=model,
                    agent_id=agent_run_id,
                    agent_class=ctx.agent_class,
                    agent_name=ctx.agent_name,
                    usage=turn_usage,
                    duration_ms=duration_ms,
                    stop_reason=completion.stop_reason,
                    cost_usd=turn_usage.cost_usd(model),
                )

                # Record assistant message
                memory.add_assistant(completion)

                # ── Output guardrails ─────────────────────────────────────
                if _output_guards and completion.content:
                    try:
                        _out_decision = await _run_output_guardrails(
                            _output_guards, completion.content, meta.name
                        )
                    except Exception as _exc:  # noqa: BLE001
                        logger.warning(
                            "lauren_ai: output guardrail check failed — failing open: %s", _exc
                        )
                        _out_decision = None
                    if _out_decision is not None and _out_decision.action != "pass":
                        _override = (
                            _out_decision.modified_content
                            if _out_decision.action == "modify"
                            else (_out_decision.violation or "Response blocked by guardrail.")
                        )
                        # Swap the assistant message in memory with the safe content
                        memory._messages.pop()
                        _safe = Completion(
                            id=completion.id,
                            model=completion.model,
                            content=_override,
                            tool_calls=[],
                            stop_reason="end_turn",
                            usage=completion.usage,
                        )
                        memory.add_assistant(_safe)
                        last_completion = _safe
                        stop_reason = "end_turn"
                        break
                # ─────────────────────────────────────────────────────────

                # Lifecycle hook: on_turn_complete
                await self._call_hook(agent, "on_turn_complete", completion, ctx)

                # Signal: AgentTurnComplete
                await self._emit(
                    "AgentTurnComplete",
                    agent_id=agent_run_id,
                    agent_class=ctx.agent_class,
                    turn=_turn,
                    turn_usage=turn_usage,
                    cumulative_usage=total_usage,
                )

                # Check budget
                if effective_config.max_cost_usd is not None:
                    cumulative_cost = total_usage.cost_usd(model)
                    if cumulative_cost > effective_config.max_cost_usd:
                        stop_reason = "budget_exceeded"
                        raise AgentBudgetExceededError(
                            f"Agent exceeded cost budget of ${effective_config.max_cost_usd:.4f} "
                            f"(used ${cumulative_cost:.4f})",
                            budget_type="cost_usd",
                            limit=effective_config.max_cost_usd,
                            used=cumulative_cost,
                            agent_class=ctx.agent_class,
                        )

                if (
                    completion.stop_reason == "end_turn"
                    or completion.stop_reason == "stop_sequence"
                ):
                    stop_reason = "end_turn"
                    break

                if completion.stop_reason == "tool_use" and completion.tool_calls:
                    # Execute all tool calls (serial or parallel)
                    results = await self._execute_tools(
                        completion.tool_calls,
                        ctx=ctx,
                        agent=agent,
                        model=model,
                    )
                    all_tool_calls.extend(completion.tool_calls)

                    # Record tool results in memory
                    for result in results:
                        memory.add_tool_result(result)

                    # Continue loop for next turn
                    continue

                if completion.stop_reason == "max_tokens":
                    stop_reason = "max_turns"
                    break

                # Unrecognised stop reason — treat as end_turn
                stop_reason = "end_turn"
                break

        except AgentBudgetExceededError:
            stop_reason = "budget_exceeded"
            # Build partial response and re-raise so callers can handle it.
            # (The on_finish hook is still called.)

        # Build final response
        final_content = last_completion.content if last_completion else ""
        reasoning_traces: list[str] = []
        if last_completion and hasattr(last_completion, "thinking_blocks"):
            for tb in last_completion.thinking_blocks:
                thinking_text = getattr(tb, "thinking", None)
                if thinking_text:
                    reasoning_traces.append(thinking_text)

        response = AgentResponse(
            content=final_content,
            turns=ctx.turn + 1,
            total_usage=total_usage,
            tool_calls_made=all_tool_calls,
            stop_reason=stop_reason,  # type: ignore[arg-type]
            reasoning_traces=reasoning_traces,
        )

        # Lifecycle hook: on_finish
        await self._call_hook(agent, "on_finish", response, ctx)

        # Persist conversation history for the next turn
        if conversation_id and effective_store is not None:
            await effective_store.save(conversation_id, memory.snapshot())

        # Signal: AgentRunComplete
        await self._emit(
            "AgentRunComplete",
            agent_id=agent_run_id,
            agent_class=ctx.agent_class,
            agent_name=ctx.agent_name,
            turns=ctx.turn + 1,
            total_usage=total_usage,
            total_cost_usd=total_usage.cost_usd(model),
            stop_reason=stop_reason,
        )

        return response

    async def run_stream(
        self,
        agent: Any,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        request: Any | None = None,
        execution_context: Any | None = None,
        run_id: str | None = None,
        conversation_store: Any | None = None,
        memory: Any | None = None,
    ) -> AsyncIterator[CompletionChunk]:
        """Run an agent with streaming output.

        Yields :class:`~lauren_ai._transport.CompletionChunk` items as they
        arrive from the transport.  Tool calls are executed silently between
        turns (their results are **not** yielded to the caller).

        Functionally at parity with :meth:`run` — fires the same lifecycle
        hooks (``on_start`` / ``on_turn_complete`` / ``on_finish``), emits the
        same signals (``ModelCallStarted``, ``ModelCallComplete``,
        ``AgentTurnComplete``, ``ToolCall*``, ``AgentRunComplete``), enforces
        ``max_cost_usd``, and loads / saves conversation history through the
        agent's ``meta.conversation_store`` (with per-request override).

        Usage::

            async for chunk in await runner.run_stream(agent, "Hello"):
                print(chunk.delta, end="", flush=True)

        :param agent: A resolved ``@agent()``-decorated instance.
        :type agent: Any
        :param message: The initial user message.
        :type message: str
        :param conversation_id: Optional conversation session identifier.
        :type conversation_id: str | None
        :param metadata: Additional key-value metadata for the context.
        :type metadata: dict[str, Any] | None
        :param request: Originating HTTP request, if any.
        :type request: Any | None
        :param execution_context: The lauren ``ExecutionContext`` (route
            metadata, handler class/func, authenticated user via
            ``request.state``) when invoked from a route handler.  Threaded
            into ``ToolContext.execution_context`` for every tool call.
        :type execution_context: Any | None
        :param run_id: Optional explicit run identifier.
        :type run_id: str | None
        :param conversation_store: Per-request override of the agent's
            conversation store.  Wins over ``meta.conversation_store``.
        :type conversation_store: Any | None
        :param memory: Per-request override of the agent's memory instance.
            Wins over ``meta.memory``.
        :type memory: Any | None
        :return: An async iterator of completion chunks.
        :rtype: AsyncIterator[CompletionChunk]
        """
        meta = self._get_meta(agent)
        effective_config = self._merge_config(meta)
        agent_run_id = run_id or uuid.uuid4().hex
        agent_id = uuid.uuid4().hex

        # Resolve per-run state from AgentMeta with per-request overrides.
        # Use explicit ``is None`` checks — ShortTermMemory has ``__len__``
        # so an empty memory is falsy under ``or``.
        effective_store = (
            conversation_store if conversation_store is not None else meta.conversation_store
        )
        if memory is None:
            memory = (
                meta.memory
                if meta.memory is not None
                else ShortTermMemory(max_tokens=effective_config.memory_window_tokens)
            )
        if conversation_id and effective_store is not None:
            prior = await effective_store.load(conversation_id)
            if prior:
                memory.restore(prior)

        # ── Input guardrails (before first LLM call) ─────────────────────
        _stream_input_guards = _get_input_guardrails(agent)
        if _stream_input_guards:
            try:
                _inp_decision = await _run_input_guardrails(
                    _stream_input_guards, message, meta.name
                )
            except Exception as _exc:  # noqa: BLE001
                logger.warning("lauren_ai: input guardrail check failed — failing open: %s", _exc)
                _inp_decision = None
            if _inp_decision is not None:
                _redirect = (
                    _inp_decision.modified_content
                    if _inp_decision.action == "modify"
                    else (_inp_decision.violation or "Message blocked by guardrail.")
                )
                if _inp_decision.action == "block":
                    # Yield the redirect as a single guardrail_override sentinel
                    # so the frontend can display it via the normal SSE flow.
                    async def _blocked_stream() -> AsyncIterator[CompletionChunk]:
                        yield CompletionChunk(guardrail_override=_redirect)
                        yield CompletionChunk(stop_reason="end_turn")

                    return _blocked_stream()
                # modify — replace the message before seeding memory
                message = _redirect
        # ─────────────────────────────────────────────────────────────────

        memory.add_user(message)

        ctx = AgentContext(
            agent_id=agent_id,
            agent_run_id=agent_run_id,
            agent_class=type(agent) if not isinstance(agent, type) else agent,
            config=effective_config,
            memory=memory,
            turn=0,
            metadata=dict(metadata or {}),
            request=request,
            execution_context=execution_context,
            signals=self._signals,
        )

        model = meta.model
        if model is None:
            raise AgentConfigError(
                f"Agent '{ctx.agent_name}' has no configured model. Set one via @agent(...)."
            )
        system_prompt = meta.system or effective_config.system_prompt
        tool_schemas = self._get_tool_schemas(meta)

        await self._call_hook(agent, "on_start", ctx)

        # Mirrors run(): ModelCallStarted fires once before the loop.
        await self._emit(
            "ModelCallStarted",
            model=model,
            agent_id=agent_run_id,
            agent_class=ctx.agent_class,
            agent_name=ctx.agent_name,
            messages_count=len(memory.messages()),
            input_tokens_estimate=memory.token_estimate,
        )

        return self._stream_loop(
            agent=agent,
            ctx=ctx,
            memory=memory,
            model=model,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            effective_config=effective_config,
            agent_run_id=agent_run_id,
            conversation_id=conversation_id,
            conversation_store=effective_store,
            output_guards=_get_output_guardrails(agent),
        )

    async def approve_tool(self, agent_run_id: str, tool_use_id: str) -> None:
        """Approve a pending HITL tool call.

        :param agent_run_id: The run identifier returned by ``run()``.
        :type agent_run_id: str
        :param tool_use_id: The provider-assigned tool call identifier to
            approve.
        :type tool_use_id: str
        """
        futures = self._pending_approvals.get(agent_run_id, {})
        fut = futures.get(tool_use_id)
        if fut is not None and not fut.done():
            fut.set_result(True)

    async def reject_tool(
        self,
        agent_run_id: str,
        tool_use_id: str,
        *,
        reason: str = "",
    ) -> None:
        """Reject a pending HITL tool call.

        :param agent_run_id: The run identifier.
        :type agent_run_id: str
        :param tool_use_id: The tool call identifier to reject.
        :type tool_use_id: str
        :param reason: Optional human-readable rejection reason.
        :type reason: str
        """
        from lauren_ai._exceptions import ToolConfirmationRejectedError  # noqa: PLC0415

        futures = self._pending_approvals.get(agent_run_id, {})
        fut = futures.get(tool_use_id)
        if fut is not None and not fut.done():
            fut.set_exception(
                ToolConfirmationRejectedError(
                    reason or "Tool call rejected by operator",
                    tool_name="",
                    tool_use_id=tool_use_id,
                    reason=reason,
                )
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _stream_loop(
        self,
        agent: Any,
        ctx: AgentContext,
        memory: ShortTermMemory,
        model: str,
        system_prompt: str,
        tool_schemas: list[Any],
        effective_config: AgentConfig,
        agent_run_id: str,
        conversation_id: str | None = None,
        conversation_store: Any | None = None,
        output_guards: list[Any] | None = None,
    ) -> AsyncIterator[CompletionChunk]:
        """Internal generator that drives the streaming agentic loop.

        Mirrors the per-turn signal / hook / budget machinery of
        :meth:`run`.  When the loop ends naturally, fires ``on_finish``,
        persists conversation history, and emits ``AgentRunComplete``.
        Caller-cancelled streams (``aclose()`` / ``GeneratorExit``) skip
        the post-loop cleanup.

        :param agent: The agent instance.
        :param ctx: The agent context.
        :param memory: The short-term memory buffer.
        :param model: The effective model identifier.
        :param system_prompt: The effective system prompt.
        :param tool_schemas: List of tool schemas.
        :param effective_config: The merged agent config.
        :param agent_run_id: The unique run identifier.
        :param conversation_id: When set with a configured store, history
            is saved when the loop ends naturally.
        :return: An async generator of completion chunks.
        :rtype: AsyncIterator[CompletionChunk]
        """
        total_usage = TokenUsage(input_tokens=0, output_tokens=0)
        all_tool_calls: list[ToolCall] = []
        last_synthetic_completion: Completion | None = None
        last_thinking_text = ""
        final_stop_reason: str = "max_turns"

        # Resolve the base system prompt for the stream loop (the caller
        # already built it once; rebuild here in case memory was loaded from
        # a prior conversation store).
        meta_for_stream = self._get_meta(agent)
        system_prompt = _build_system_prompt(
            meta_for_stream.system or effective_config.system_prompt, memory
        )

        # Pre-resolve static loop flags — these are determined before the first
        # turn and never change mid-run, so there is no point re-evaluating
        # them on every iteration.
        _may_summarize: bool = effective_config.summarize_at is not None
        _has_guardrails: bool = bool(output_guards)
        _has_budget_cap: bool = effective_config.max_cost_usd is not None
        _tool_schemas_arg = tool_schemas or None

        try:
            for _turn in range(effective_config.max_turns):
                ctx.turn = _turn

                # ── Context-window summarisation (opt-in) ─────────────────
                if _may_summarize and _should_summarize(memory, effective_config):
                    await _summarize_memory(
                        memory=memory,
                        transport=self._transport,
                        model=effective_config.summary_model or model,
                    )
                    system_prompt = _build_system_prompt(
                        meta_for_stream.system or effective_config.system_prompt, memory
                    )

                messages = memory.messages()

                t0 = time.monotonic()
                stream = await self._transport.complete(
                    messages,
                    model=model,
                    system=system_prompt,
                    tools=_tool_schemas_arg,
                    max_tokens=effective_config.max_tokens_per_turn,
                    temperature=effective_config.temperature,
                    stream=True,
                )

                # Accumulate the full completion while yielding chunks
                accumulated_text = ""
                accumulated_thinking = ""
                accumulated_stop_reason: str | None = None
                accumulated_usage: TokenUsage | None = None
                accumulated_tool_calls: list[ToolCall] = []
                partial_tool_inputs: dict[str, str] = {}  # tool_use_id -> input_json
                partial_tool_names: dict[str, str] = {}  # tool_use_id -> name

                async for chunk in stream:
                    if chunk.delta:
                        accumulated_text += chunk.delta

                    if chunk.thinking_delta:
                        accumulated_thinking += chunk.thinking_delta

                    if chunk.tool_call_delta is not None:
                        tcd = chunk.tool_call_delta
                        tid = tcd.tool_use_id
                        if tcd.name:
                            partial_tool_names[tid] = tcd.name
                        partial_tool_inputs.setdefault(tid, "")
                        partial_tool_inputs[tid] += tcd.input_delta

                    if chunk.stop_reason is not None:
                        accumulated_stop_reason = chunk.stop_reason

                    if chunk.usage is not None:
                        accumulated_usage = chunk.usage

                    yield chunk

                duration_ms = (time.monotonic() - t0) * 1000

                # ── Output guardrails (post-stream, real-time yield preserved) ──
                # Chunks have already been yielded to the caller; the guardrail
                # now judges the fully-assembled text.  If it fires, emit a
                # sentinel CompletionChunk(guardrail_override=...) that tells
                # the controller / frontend to replace the displayed content.
                if _has_guardrails and accumulated_text:
                    try:
                        _out_decision = await _run_output_guardrails(
                            output_guards,  # type: ignore[arg-type]
                            accumulated_text,
                            ctx.agent_name,
                        )
                    except Exception as _exc:  # noqa: BLE001
                        logger.warning(
                            "lauren_ai: output guardrail check failed — failing open: %s", _exc
                        )
                        _out_decision = None
                    if _out_decision is not None and _out_decision.action != "pass":
                        _override = (
                            _out_decision.modified_content
                            if _out_decision.action == "modify"
                            else (_out_decision.violation or "Response blocked by guardrail.")
                        )
                        yield CompletionChunk(guardrail_override=_override)
                        # Rewrite the accumulated text so memory + housekeeping
                        # reflect the safe content, not the hallucination.
                        accumulated_text = _override
                        accumulated_tool_calls = []
                        # Clear the raw tool-delta dicts so the tool-rebuild
                        # loop below cannot re-populate accumulated_tool_calls
                        # from stale deltas.
                        partial_tool_inputs = {}
                        partial_tool_names = {}
                        accumulated_stop_reason = "end_turn"
                # ─────────────────────────────────────────────────────────────

                # Build tool calls from accumulated deltas
                import json  # noqa: PLC0415

                for tid, input_json in partial_tool_inputs.items():
                    try:
                        parsed_input = json.loads(input_json)
                    except (json.JSONDecodeError, ValueError):
                        parsed_input = {}
                    accumulated_tool_calls.append(
                        ToolCall(
                            tool_use_id=tid,
                            name=partial_tool_names.get(tid, ""),
                            input=parsed_input,
                        )
                    )

                turn_usage = accumulated_usage or TokenUsage(input_tokens=0, output_tokens=0)
                total_usage = total_usage + turn_usage

                synthetic_completion = Completion(
                    id=uuid.uuid4().hex,
                    model=model,
                    content=accumulated_text,
                    tool_calls=accumulated_tool_calls,
                    stop_reason=accumulated_stop_reason or "end_turn",  # type: ignore[arg-type]
                    usage=turn_usage,
                )
                memory.add_assistant(synthetic_completion)
                last_synthetic_completion = synthetic_completion
                last_thinking_text = accumulated_thinking

                # Per-turn signals + hook (mirrors run() 260–286)
                await self._emit(
                    "ModelCallComplete",
                    model=model,
                    agent_id=agent_run_id,
                    agent_class=ctx.agent_class,
                    agent_name=ctx.agent_name,
                    usage=turn_usage,
                    duration_ms=duration_ms,
                    stop_reason=accumulated_stop_reason,
                    cost_usd=turn_usage.cost_usd(model),
                )

                await self._call_hook(agent, "on_turn_complete", synthetic_completion, ctx)

                await self._emit(
                    "AgentTurnComplete",
                    agent_id=agent_run_id,
                    agent_class=ctx.agent_class,
                    turn=_turn,
                    turn_usage=turn_usage,
                    cumulative_usage=total_usage,
                )

                # Budget check (mirrors run() 289–300)
                if _has_budget_cap:
                    cumulative_cost = total_usage.cost_usd(model)
                    if cumulative_cost > effective_config.max_cost_usd:  # type: ignore[operator]
                        raise AgentBudgetExceededError(
                            f"Agent exceeded cost budget of ${effective_config.max_cost_usd:.4f} "
                            f"(used ${cumulative_cost:.4f})",
                            budget_type="cost_usd",
                            limit=effective_config.max_cost_usd or 0.0,
                            used=cumulative_cost,
                            agent_class=ctx.agent_class,
                        )

                if accumulated_stop_reason in ("end_turn", "stop_sequence", None):
                    final_stop_reason = "end_turn"
                    break

                if accumulated_stop_reason == "tool_use" and accumulated_tool_calls:
                    # Execute tools silently (do not yield tool results)
                    results = await self._execute_tools(
                        accumulated_tool_calls,
                        ctx=ctx,
                        agent=agent,
                        model=model,
                    )
                    all_tool_calls.extend(accumulated_tool_calls)
                    for result in results:
                        memory.add_tool_result(result)
                    continue

                if accumulated_stop_reason == "max_tokens":
                    final_stop_reason = "max_turns"
                    break

                final_stop_reason = "end_turn"
                break

        except AgentBudgetExceededError:
            final_stop_reason = "budget_exceeded"
            # Swallowed — matches run() 372–376; consumers introspect
            # AgentRunComplete.stop_reason="budget_exceeded".

        # Post-loop cleanup (mirrors run() 378–412).  Skipped on caller
        # cancellation (GeneratorExit propagates without running this).
        final_content = last_synthetic_completion.content if last_synthetic_completion else ""
        reasoning_traces: list[str] = [last_thinking_text] if last_thinking_text else []
        response = AgentResponse(
            content=final_content,
            turns=ctx.turn + 1,
            total_usage=total_usage,
            tool_calls_made=all_tool_calls,
            stop_reason=final_stop_reason,  # type: ignore[arg-type]
            reasoning_traces=reasoning_traces,
        )

        await self._call_hook(agent, "on_finish", response, ctx)

        if conversation_id and conversation_store is not None:
            await conversation_store.save(conversation_id, memory.snapshot())

        await self._emit(
            "AgentRunComplete",
            agent_id=agent_run_id,
            agent_class=ctx.agent_class,
            agent_name=ctx.agent_name,
            turns=ctx.turn + 1,
            total_usage=total_usage,
            total_cost_usd=total_usage.cost_usd(model),
            stop_reason=final_stop_reason,
        )

    def _get_meta(self, agent: Any) -> AgentMeta:
        """Extract :class:`AgentMeta` from an agent instance or class.

        :param agent: An ``@agent()``-decorated instance or class.
        :type agent: Any
        :return: The attached ``AgentMeta``.
        :rtype: AgentMeta
        :raises AgentConfigError: When the object lacks ``AGENT_META``.
        """
        cls = type(agent) if not isinstance(agent, type) else agent
        meta: AgentMeta | None = getattr(cls, AGENT_META, None)
        if meta is None:
            raise AgentConfigError(
                f"{cls!r} is not decorated with @agent().  "
                "Only classes decorated with @agent() can be passed to AgentRunner.run().",
                agent_class=cls,
            )
        return meta

    def _merge_config(self, meta: AgentMeta) -> AgentConfig:
        """Merge agent-specific config with the application-level LLM config.

        Agent config fields take precedence.  Currently the merge is trivial —
        the per-agent config is used as-is.  Future versions may allow
        module-level config to supply defaults that per-agent config can
        override on a per-field basis.

        :param meta: The agent's ``AgentMeta``.
        :type meta: AgentMeta
        :return: The effective ``AgentConfig`` for this run.
        :rtype: AgentConfig
        """
        return meta.config

    def _get_tool_schemas(self, meta: AgentMeta) -> list[Any]:
        """Build the list of tool schemas for the agent's attached tools.

        Reads directly from ``meta.tools`` — the per-agent map built once at
        startup by ``AgentModule.for_root``.

        :param meta: The agent's ``AgentMeta``.
        :type meta: AgentMeta
        :return: List of JSON schema dicts suitable for passing to the transport.
        :rtype: list[Any]
        """
        return [tm.parameters for _, tm in meta.tools.values()]

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        *,
        ctx: AgentContext,
        agent: Any,
        model: str,
    ) -> list[ToolResult]:
        """Execute a batch of tool calls, respecting the parallel_tool_calls config.

        :param tool_calls: Tool calls to execute.
        :type tool_calls: list[ToolCall]
        :param ctx: Current agent context.
        :type ctx: AgentContext
        :param agent: The agent instance (for on_tool_result hook).
        :type agent: Any
        :param model: The current model identifier (used for budget checks).
        :type model: str
        :return: Ordered list of tool results.
        :rtype: list[ToolResult]
        """
        if ctx.config.parallel_tool_calls and len(tool_calls) > 1:
            coros = [self._execute_single_tool(tc, ctx=ctx, agent=agent) for tc in tool_calls]
            results = list(await asyncio.gather(*coros, return_exceptions=False))
        else:
            results = []
            for tc in tool_calls:
                result = await self._execute_single_tool(tc, ctx=ctx, agent=agent)
                results.append(result)
        return results

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        *,
        ctx: AgentContext,
        agent: Any,
    ) -> ToolResult:
        """Execute one tool call, emit signals, and invoke the agent hook.

        Handles tool_error_policy: "raise", "return_error", or "skip".

        :param tool_call: The tool call request.
        :type tool_call: ToolCall
        :param ctx: Current agent context.
        :type ctx: AgentContext
        :param agent: The agent instance.
        :type agent: Any
        :return: The tool result.
        :rtype: ToolResult
        """
        # Resolve the per-agent tool map from the agent's AgentMeta, then
        # pull static @set_metadata from the callable on the matching entry.
        _tool_map: dict = getattr(agent, AGENT_META).tools
        _tool_entry = _tool_map.get(tool_call.name)
        _tool_static_meta: dict[str, Any] = {}
        if _tool_entry is not None:
            _callable, _ = _tool_entry
            _tool_static_meta = dict(getattr(_callable, TOOL_METADATA, {}))

        tool_context = ToolContext(
            agent_context=ctx,
            tool_use_id=tool_call.tool_use_id,
            turn=ctx.turn,
            request=ctx.request,
            execution_context=ctx.execution_context,
            metadata=_tool_static_meta,
            state={},
        )

        await self._emit(
            "ToolCallStarted",
            tool_name=tool_call.name,
            tool_use_id=tool_call.tool_use_id,
            agent_id=ctx.agent_run_id,
            input=tool_call.input,
        )

        t0 = time.monotonic()
        try:
            result = await self._executor.execute(
                ExecutorToolCall(
                    tool_use_id=tool_call.tool_use_id,
                    name=tool_call.name,
                    input=tool_call.input,
                ),
                tool_context,
                tool_map=_tool_map,
            )
            duration_ms = (time.monotonic() - t0) * 1000
            await self._emit(
                "ToolCallComplete",
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
                agent_id=ctx.agent_run_id,
                duration_ms=duration_ms,
                success=True,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.monotonic() - t0) * 1000
            error_msg = str(exc)
            await self._emit(
                "ToolCallComplete",
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
                agent_id=ctx.agent_run_id,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )

            policy = ctx.config.tool_error_policy
            if policy == "raise":
                raise
            if policy == "skip":
                logger.warning(
                    "lauren_ai.AgentRunner: tool '%s' failed (policy=skip): %s",
                    tool_call.name,
                    exc,
                )
                # Return a synthetic empty result so the loop can continue
                return ToolResult(
                    tool_use_id=tool_call.tool_use_id,
                    content="",
                    is_error=False,
                )
            # policy == "return_error" (default)
            result = ToolResult.error(
                message=f"Tool execution failed: {error_msg}",
                tool_use_id=tool_call.tool_use_id,
            )

        # Lifecycle hook: on_tool_result
        hook_result = await self._call_hook_with_return(agent, "on_tool_result", result, ctx)
        if hook_result is not None and isinstance(hook_result, ToolResult):
            result = hook_result

        return result

    async def _call_hook(self, agent: Any, hook_name: str, *args: Any) -> None:
        """Invoke an optional lifecycle hook on the agent instance.

        No-op when the hook is not defined.  Handles both sync and async hooks.

        When *agent* is a class (not an instance) and the hook attribute is an
        unbound instance method, a temporary no-arg instance is created so the
        method can be bound and called normally.  If instantiation requires
        arguments (DI-injected deps), the hook is silently skipped.

        :param agent: The agent instance or class.
        :type agent: Any
        :param hook_name: Name of the hook method.
        :type hook_name: str
        :param args: Positional arguments forwarded to the hook.
        """
        import inspect  # noqa: PLC0415

        # When a class is passed instead of an instance (e.g. runner.run(MyAgent, ...))
        # getattr returns an unbound function.  Bind it by creating a throwaway
        # instance so lifecycle hooks defined as normal instance methods work correctly.
        if isinstance(agent, type) and inspect.isfunction(getattr(agent, hook_name, None)):
            try:
                agent = agent()
            except Exception:  # noqa: BLE001
                return  # Requires DI args — skip hook rather than crash

        hook = getattr(agent, hook_name, None)
        if hook is None:
            return

        try:
            result = hook(*args)
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001
            logger.warning(
                "lauren_ai.AgentRunner: hook '%s' on %r raised an exception",
                hook_name,
                type(agent).__name__,
                exc_info=True,
            )

    async def _call_hook_with_return(self, agent: Any, hook_name: str, *args: Any) -> Any:
        """Invoke an optional lifecycle hook and return its result.

        Same class-vs-instance handling as :meth:`_call_hook`.

        :param agent: The agent instance or class.
        :type agent: Any
        :param hook_name: Name of the hook method.
        :type hook_name: str
        :param args: Positional arguments forwarded to the hook.
        :return: The hook's return value, or ``None`` if not defined or on error.
        :rtype: Any
        """
        import inspect  # noqa: PLC0415

        if isinstance(agent, type) and inspect.isfunction(getattr(agent, hook_name, None)):
            try:
                agent = agent()
            except Exception:  # noqa: BLE001
                return None

        hook = getattr(agent, hook_name, None)
        if hook is None:
            return None

        try:
            result = hook(*args)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception:  # noqa: BLE001
            logger.warning(
                "lauren_ai.AgentRunner: hook '%s' on %r raised an exception",
                hook_name,
                type(agent).__name__,
                exc_info=True,
            )
            return None

    async def _emit(self, signal_name: str, **kwargs: Any) -> None:
        """Emit a named signal via the signal bus, if available.

        Looks up the signal class in ``lauren_ai._signals`` and emits it.
        Failures are logged and swallowed so they never interrupt the loop.

        :param signal_name: Class name of the signal to emit.
        :type signal_name: str
        :param kwargs: Fields for the signal dataclass.
        """
        if self._signals is None:
            return
        try:
            from lauren_ai import _signals  # noqa: PLC0415

            signal_cls = getattr(_signals, signal_name, None)
            if signal_cls is None:
                return
            event = signal_cls(**kwargs)
            await self._signals.emit(event)
        except Exception:  # noqa: BLE001
            logger.debug(
                "lauren_ai.AgentRunner: failed to emit signal '%s'",
                signal_name,
                exc_info=True,
            )

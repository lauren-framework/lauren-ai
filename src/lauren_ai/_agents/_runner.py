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
from lauren_ai._config import AgentConfig, LLMConfig
from lauren_ai._exceptions import (
    AgentBudgetExceededError,
    AgentConfigError,
)
from lauren_ai._memory import ShortTermMemory
from lauren_ai._tools import TOOL_META, ToolContext, ToolMeta, ToolResult
from lauren_ai._tools._executor import CacheBackend, ToolExecutor
from lauren_ai._transport import Completion, CompletionChunk, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentRunner(Protocol):
    """Structural interface for agent runner implementations.

    Declare ``runner: AgentRunner`` in any service or tool and the DI
    container will inject the module's runner automatically — no concrete
    class name required.  Each :meth:`~lauren_ai._module.AgentModule.for_root`
    call generates a unique :class:`AgentRunnerBase` subclass that satisfies
    this Protocol.
    """

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


class AgentRunnerBase(AgentRunner):
    """Concrete implementation of the :class:`AgentRunner` Protocol.

    Owns the observe → think → act → observe loop.  Resolves agent meta from
    the decorated class, creates per-run state (:class:`~lauren_ai._agents.AgentContext`
    and :class:`~lauren_ai._memory.ShortTermMemory`), calls the LLM transport,
    dispatches tool calls, and aggregates results into an
    :class:`~lauren_ai._agents.AgentResponse`.

    :param transport: Provider-agnostic LLM transport.
    :type transport: Any
    :param tools: Mapping of tool name to ``(callable_or_instance, ToolMeta)``.
        Built by ``AgentModule.for_root()`` or ``AgentTestClient``.
    :type tools: dict[str, tuple[Any, ToolMeta]]
    :param config: Application-level LLM configuration (model, max_tokens, etc.).
    :type config: LLMConfig
    :param signals: Optional signal bus for emitting lifecycle events.
    :type signals: Any | None
    :param cache_backend: Optional cache backend for tool result caching.
    :type cache_backend: CacheBackend | None
    """

    def __init__(
        self,
        transport: Any,
        tools: dict[str, tuple[Any, ToolMeta]],
        config: LLMConfig,
        signals: Any | None = None,
        cache_backend: CacheBackend | None = None,
        conversation_store: Any | None = None,
    ) -> None:
        self._transport = transport
        self._tools = tools
        self._config = config
        self._signals = signals
        self._conversation_store = conversation_store
        self._executor = ToolExecutor(
            tools=tools,
            cache_backend=cache_backend,
            signals=signals,
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
            When provided, initial history may be loaded from a
            ``ConversationStore`` (if configured on the agent).
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

        # Short-term memory for this run — seeded with prior history when a
        # conversation_store is configured and a conversation_id is provided.
        memory = ShortTermMemory(max_tokens=effective_config.memory_window_tokens)
        if conversation_id and self._conversation_store is not None:
            prior = await self._conversation_store.load(conversation_id)
            if prior:
                memory.restore(prior)
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
        model = meta.model or self._config.model
        system_prompt = meta.system or effective_config.system_prompt

        # Gather tool schemas for attached tools
        tool_schemas = self._get_tool_schemas(meta)

        total_usage = TokenUsage(input_tokens=0, output_tokens=0)
        all_tool_calls: list[ToolCall] = []
        last_completion: Completion | None = None
        stop_reason: str = "max_turns"

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
        if conversation_id and self._conversation_store is not None:
            await self._conversation_store.save(conversation_id, memory.snapshot())

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
    ) -> AsyncIterator[CompletionChunk]:
        """Run an agent with streaming output.

        Yields :class:`~lauren_ai._transport.CompletionChunk` items as they
        arrive from the transport.  Tool calls are executed silently between
        turns (their results are **not** yielded to the caller).

        Functionally at parity with :meth:`run` — fires the same lifecycle
        hooks (``on_start`` / ``on_turn_complete`` / ``on_finish``), emits the
        same signals (``ModelCallStarted``, ``ModelCallComplete``,
        ``AgentTurnComplete``, ``ToolCall*``, ``AgentRunComplete``), enforces
        ``max_cost_usd``, and loads / saves conversation history.

        Usage::

            async for chunk in await runner.run_stream(agent, "Hello"):
                print(chunk.delta, end="", flush=True)

        :param agent: A resolved ``@agent()``-decorated instance.
        :type agent: Any
        :param message: The initial user message.
        :type message: str
        :param conversation_id: Optional conversation session identifier.
            When provided, history is loaded from the configured
            ``ConversationStore`` and saved back when the loop ends naturally.
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
        :return: An async iterator of completion chunks.
        :rtype: AsyncIterator[CompletionChunk]
        """
        meta = self._get_meta(agent)
        effective_config = self._merge_config(meta)
        agent_run_id = run_id or uuid.uuid4().hex
        agent_id = uuid.uuid4().hex

        # Short-term memory — seeded with prior history when a conversation
        # store is configured and a conversation_id is provided.
        memory = ShortTermMemory(max_tokens=effective_config.memory_window_tokens)
        if conversation_id and self._conversation_store is not None:
            prior = await self._conversation_store.load(conversation_id)
            if prior:
                memory.restore(prior)
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

        model = meta.model or self._config.model
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

        try:
            for _turn in range(effective_config.max_turns):
                ctx.turn = _turn
                messages = memory.messages()

                t0 = time.monotonic()
                stream = await self._transport.complete(
                    messages,
                    model=model,
                    system=system_prompt,
                    tools=tool_schemas if tool_schemas else None,
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
                if effective_config.max_cost_usd is not None:
                    cumulative_cost = total_usage.cost_usd(model)
                    if cumulative_cost > effective_config.max_cost_usd:
                        raise AgentBudgetExceededError(
                            f"Agent exceeded cost budget of ${effective_config.max_cost_usd:.4f} "
                            f"(used ${cumulative_cost:.4f})",
                            budget_type="cost_usd",
                            limit=effective_config.max_cost_usd,
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
        final_content = (
            last_synthetic_completion.content if last_synthetic_completion else ""
        )
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

        if conversation_id and self._conversation_store is not None:
            await self._conversation_store.save(conversation_id, memory.snapshot())

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

        :param meta: The agent's ``AgentMeta``.
        :type meta: AgentMeta
        :return: List of JSON schema dicts suitable for passing to the transport.
        :rtype: list[Any]
        """
        if not meta.tool_classes:
            return []
        schemas: list[Any] = []
        for tool_item in meta.tool_classes:
            tool_meta: ToolMeta | None = getattr(tool_item, TOOL_META, None)
            if tool_meta is None:
                continue
            entry = self._tools.get(tool_meta.name)
            if entry is None:
                logger.warning(
                    "lauren_ai.AgentRunner: tool '%s' not found in tool map — skipping",
                    tool_meta.name,
                )
                continue
            schemas.append(tool_meta.parameters)
        return schemas

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
        tool_context = ToolContext(
            agent_context=ctx,
            tool_use_id=tool_call.tool_use_id,
            turn=ctx.turn,
            request=ctx.request,
            execution_context=ctx.execution_context,
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
            result = await self._executor.execute(tool_call, tool_context)
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

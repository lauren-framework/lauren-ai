"""Test utilities for ``lauren-ai`` applications.

:class:`AgentTestClient` wraps an agent instance and a
:class:`~lauren_ai._transport._mock.MockTransport` to provide a simple
synchronous/asynchronous interface for testing agents without making real
network calls.

Example::

    from lauren_ai import LLMConfig, AgentModule, LLMModule
    from lauren_ai._transport import Completion, TokenUsage
    from lauren_ai.testing import AgentTestClient


    @pytest.fixture()
    def client():
        cfg, mock = LLMConfig.for_testing()
        LLMProviderModule = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[TravelAgent], tools=[get_weather])

        @module(imports=[LLMProviderModule, AIModule])
        class TestModule: ...

        app = LaurenFactory.create(TestModule)
        agent = await app.container.resolve(TravelAgent)
        runner = await app.container.resolve(AgentRunner)
        return AgentTestClient(agent, mock, runner=runner)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "AgentTestClient",
    "TestClient",
]

if TYPE_CHECKING:
    from lauren_ai._agents import AgentResponse


class AgentTestClient:
    """Synchronous + asynchronous test client for ``@agent()``-decorated classes.

    Wraps an agent instance with a
    :class:`~lauren_ai._transport._mock.MockTransport` so tests can queue
    canned responses and make assertions without network calls.

    :param agent_instance: The ``@agent()``-decorated class instance to test.
    :type agent_instance: Any
    :param mock_transport: The mock transport to queue responses on.
    :type mock_transport: MockTransport
    :param runner: Pre-built :class:`~lauren_ai._agents._runner.AgentRunner`.
        When ``None``, a minimal runner is built from the mock transport.
    :type runner: Any | None
    """

    def __init__(
        self,
        agent_instance: Any,
        mock_transport: Any,
        *,
        runner: Any = None,
    ) -> None:
        self._agent = agent_instance
        self._mock = mock_transport
        self._runner = runner or self._build_runner(mock_transport)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Run the agent synchronously (blocks the calling thread).

        :param message: The user message to process.
        :type message: str
        :param conversation_id: Optional conversation ID.
        :type conversation_id: str | None
        :param metadata: Optional metadata dict.
        :type metadata: dict[str, Any] | None
        :return: The :class:`~lauren_ai._agents.AgentResponse`.
        :rtype: AgentResponse
        """
        return asyncio.run(
            self.run_async(
                message,
                conversation_id=conversation_id,
                metadata=metadata,
            )
        )

    async def run_async(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Run the agent asynchronously.

        :param message: The user message to process.
        :type message: str
        :param conversation_id: Optional conversation ID.
        :type conversation_id: str | None
        :param metadata: Optional metadata dict.
        :type metadata: dict[str, Any] | None
        :return: The :class:`~lauren_ai._agents.AgentResponse`.
        :rtype: AgentResponse
        """
        return await self._runner.run(
            self._agent,
            message,
            conversation_id=conversation_id,
            metadata=metadata,
        )

    @property
    def mock(self) -> Any:
        """The underlying :class:`~lauren_ai._transport._mock.MockTransport`.

        Use this to queue canned responses before calling :meth:`run`::

            client.mock.queue_response(Completion(...))
        """
        return self._mock

    @property
    def calls(self) -> list[Any]:
        """All :class:`~lauren_ai._transport.CompletionCall` objects recorded
        by the mock transport since the last :meth:`reset`.
        """
        return self._mock.calls

    def reset(self) -> None:
        """Reset the mock transport's call history and response queue.

        Useful for isolating consecutive test scenarios within the same
        fixture.
        """
        self._mock.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_runner(self, mock_transport: Any) -> Any:
        """Build a :class:`~lauren_ai._agents._runner.AgentRunnerBase` for testing.

        :param mock_transport: The mock transport.
        :type mock_transport: MockTransport
        :return: A configured :class:`~lauren_ai._agents._runner.AgentRunnerBase`.
        :rtype: AgentRunnerBase
        """
        return _build_runner_for_agent(self._agent, mock_transport)


# ---------------------------------------------------------------------------
# Shared runner factory
# ---------------------------------------------------------------------------


def _build_runner_for_agent(
    agent_instance: Any,
    mock_transport: Any,
    *,
    signals: Any = None,
    cache_backend: Any = None,
    knowledge_tool_names: set[str] | None = None,
) -> Any:
    """Build an :class:`~lauren_ai._agents._runner.AgentRunnerBase` for *agent_instance*.

    Identical to what :class:`~lauren_ai._module.AgentModule` wires up in
    production: tools are resolved from the ``@use_tools()`` metadata, and all
    optional runtime features (signals, cache, knowledge sources) are forwarded.
    """
    from lauren_ai._agents import AGENT_META  # noqa: PLC0415
    from lauren_ai._agents._runner import AgentRunnerBase  # noqa: PLC0415
    from lauren_ai._config import LLMConfig  # noqa: PLC0415
    from lauren_ai._tools import TOOL_META, _add_to_tool_map  # noqa: PLC0415

    agent_cls = type(agent_instance)
    meta = getattr(agent_cls, AGENT_META, None)

    if meta is not None:
        tools: dict = {}
        for tool_ref in meta.tool_classes or []:
            if tool_ref is None:
                continue
            if getattr(tool_ref, TOOL_META, None) is not None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    _add_to_tool_map(tools, tool_ref)
        meta.tools = tools

    config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunnerBase(
        transport=mock_transport,
        config=config,
        signals=signals,
        cache_backend=cache_backend,
    )


# ---------------------------------------------------------------------------
# TestClient — full-featured, the recommended client for skill tests
# ---------------------------------------------------------------------------


class TestClient:
    """Full-featured test client for ``@agent()``-decorated classes.

    Runner behaviour is **identical** to :class:`~lauren_ai._agents._runner.AgentRunnerBase`
    in production: all tools are resolved from ``@use_tools()`` metadata,
    guardrails from ``@use_guardrails()``, memory and conversation stores
    from ``@agent(memory=…, conversation_store=…)``, and signals are
    forwarded if provided.

    A :class:`~lauren_ai._transport._mock.MockTransport` is created
    automatically when *mock* is omitted.

    Usage::

        from lauren_ai.testing import TestClient
        from lauren_ai._transport import Completion, TokenUsage

        def _c(text):
            return Completion(
                id="c1", model="mock", content=text, tool_calls=[],
                stop_reason="end_turn", usage=TokenUsage(10, 5),
            )

        client = TestClient(MyAgent())
        client.mock.queue_response(_c("Hello!"))
        result = client.run("Hi")
        assert result.content == "Hello!"

        # Async form (use in async tests with asyncio_mode="auto"):
        result = await client.run_async("Hi")

        # Streaming:
        async for chunk in await client.run_stream_async("Hi"):
            if chunk.delta:
                print(chunk.delta, end="")

    :param agent_instance: The ``@agent()``-decorated class instance to test.
    :type agent_instance: Any
    :param mock: :class:`~lauren_ai._transport._mock.MockTransport` to use.
        When ``None`` a fresh transport is created automatically.
    :type mock: Any | None
    :param signals: Optional :class:`~lauren_ai.SignalBus` for lifecycle events.
    :type signals: Any | None
    :param runner: Pre-built runner.  When ``None`` a runner is built from the
        agent metadata with full feature parity to production.
    :type runner: Any | None
    :param cache_backend: Optional cache backend forwarded to the runner.
    :type cache_backend: Any | None
    :param knowledge_tool_names: Optional set of knowledge-source tool names.
    :type knowledge_tool_names: set[str] | None
    """

    def __init__(
        self,
        agent_instance: Any,
        mock: Any = None,
        *,
        signals: Any = None,
        runner: Any = None,
        cache_backend: Any = None,
        knowledge_tool_names: set[str] | None = None,
    ) -> None:
        from lauren_ai._transport._mock import MockTransport  # noqa: PLC0415

        self._agent = agent_instance
        self._mock: Any = mock if mock is not None else MockTransport()
        self._runner = runner or _build_runner_for_agent(
            agent_instance,
            self._mock,
            signals=signals,
            cache_backend=cache_backend,
            knowledge_tool_names=knowledge_tool_names,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_context: Any = None,
        run_id: str | None = None,
    ) -> Any:
        """Run the agent synchronously (blocks the calling thread).

        Safe to call from sync test functions even when ``asyncio_mode="auto"``
        is configured — if an event loop is already running the coroutine is
        dispatched to a background thread to avoid nesting.

        :param message: The user message to process.
        :type message: str
        :returns: The :class:`~lauren_ai._agents.AgentResponse`.
        """
        coro = self.run_async(
            message,
            conversation_id=conversation_id,
            metadata=metadata,
            execution_context=execution_context,
            run_id=run_id,
        )
        try:
            asyncio.get_running_loop()
            # A loop is already running (e.g. inside an async test or Lauren
            # request handler).  Run the coroutine in a dedicated thread so
            # we do not nest event loops.
            import concurrent.futures  # noqa: PLC0415

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            # No running loop — standard path.
            return asyncio.run(coro)

    async def run_async(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_context: Any = None,
        run_id: str | None = None,
    ) -> Any:
        """Run the agent asynchronously.

        :param message: The user message to process.
        :type message: str
        :returns: The :class:`~lauren_ai._agents.AgentResponse`.
        """
        return await self._runner.run(
            self._agent,
            message,
            conversation_id=conversation_id,
            metadata=metadata,
            execution_context=execution_context,
            run_id=run_id,
        )

    async def run_stream_async(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_context: Any = None,
        run_id: str | None = None,
    ) -> Any:
        """Stream the agent's response asynchronously.

        :param message: The user message to process.
        :type message: str
        :returns: An :class:`~collections.abc.AsyncIterator` of
            :class:`~lauren_ai._transport.CompletionChunk` objects.
        """
        return await self._runner.run_stream(
            self._agent,
            message,
            conversation_id=conversation_id,
            metadata=metadata,
            execution_context=execution_context,
            run_id=run_id,
        )

    @property
    def mock(self) -> Any:
        """The underlying :class:`~lauren_ai._transport._mock.MockTransport`.

        Use this to queue canned responses before calling :meth:`run`::

            client.mock.queue_response(Completion(...))
        """
        return self._mock

    @property
    def calls(self) -> list:
        """All :class:`~lauren_ai._transport.CompletionCall` records since
        the last :meth:`reset`.
        """
        return self._mock.calls

    def reset(self) -> None:
        """Clear the mock transport's call history and response queue."""
        self._mock.reset()

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
from typing import TYPE_CHECKING, Any

__all__ = [
    "AgentTestClient",
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
        """Build a minimal :class:`~lauren_ai._agents._runner.AgentRunner`.

        :param mock_transport: The mock transport.
        :type mock_transport: MockTransport
        :return: A configured :class:`~lauren_ai._agents._runner.AgentRunner`.
        :rtype: AgentRunner
        """
        from lauren_ai._agents import AGENT_META  # noqa: PLC0415
        from lauren_ai._agents._runner import AgentRunner  # noqa: PLC0415
        from lauren_ai._config import LLMConfig  # noqa: PLC0415
        from lauren_ai._tools import TOOL_META, _add_to_tool_map  # noqa: PLC0415

        agent_cls = type(self._agent)
        meta = getattr(agent_cls, AGENT_META, None)

        tools: dict = {}
        if meta is not None:
            for tool_ref in meta.tool_classes:
                if tool_ref is None:
                    continue
                if getattr(tool_ref, TOOL_META, None) is not None:
                    try:
                        _add_to_tool_map(tools, tool_ref)
                    except Exception:  # noqa: BLE001
                        pass

        config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        return AgentRunner(
            transport=mock_transport,
            tools=tools,
            config=config,
        )

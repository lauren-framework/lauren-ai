"""PRD 9 — Team Coordination via MCP.

Provides :class:`McpAgentTeam` which coordinates remote MCP agent servers
(deployed via :class:`~lauren_ai.mcp._agent_server.AgentMcpServer`) as
first-class team workers.  The coordinator is a local ``@agent`` whose LLM
calls worker tools via the standard MCP tool-call mechanism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TeamWorkerResult:
    """Final result from a single worker.

    :param worker_name: The alias of the worker.
    :param content: Full assembled response text.
    :param input_tokens: Token usage (input side).
    :param output_tokens: Token usage (output side).
    """

    worker_name: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class McpTeamResult:
    """Aggregated result of an :class:`McpAgentTeam` run.

    :param final_answer: The coordinator's synthesised final answer.
    :param worker_results: Per-worker results keyed by alias.
    :param rounds: Number of coordinator→worker rounds completed.
    :param total_input_tokens: Sum across coordinator + all workers.
    :param total_output_tokens: Sum across coordinator + all workers.
    """

    final_answer: str
    worker_results: dict[str, TeamWorkerResult] = field(default_factory=dict)
    rounds: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class McpAgentTeam:
    """Coordinator that treats remote MCP agent servers as team workers.

    Each worker is an independently deployed Lauren agent reachable via MCP
    (deployed using :class:`~lauren_ai.mcp._agent_server.AgentMcpServer`).
    The coordinator is a local ``@agent`` class whose LLM calls worker tools
    via the standard MCP tool-call mechanism.

    Example::

        from lauren_ai.mcp import McpAgentTeam
        from lauren_mcp import McpServer

        team = McpAgentTeam(
            coordinator=CoordinatorAgent,
            workers={
                "researcher": McpServer.streamable_http("http://researcher.example.com/"),
                "writer":     McpServer.streamable_http("http://writer.example.com/"),
            },
            llm_module=LLMProvider,
            max_rounds=5,
        )
        result = await team.run("Write a research report on AI trends in 2026")
        print(result.final_answer)

    :param coordinator: An ``@agent()``-decorated class.
    :param workers: Dict mapping alias → ``McpClientProtocol`` instance.
    :param llm_module: Result of ``LLMModule.for_root(...)`` for the coordinator.
    :param max_rounds: Maximum agentic loop turns for the coordinator.
    :param health_check_timeout: Seconds to wait per worker health-check ping.
    """

    def __init__(
        self,
        coordinator: type,
        workers: dict[str, Any],
        llm_module: type,
        *,
        max_rounds: int = 5,
        health_check_timeout: float = 5.0,
    ) -> None:
        from lauren_ai._agents import AGENT_META  # noqa: PLC0415

        meta = getattr(coordinator, AGENT_META, None)
        if meta is None:
            raise ValueError(
                f"{coordinator.__name__!r} is not decorated with @agent().  "
                "Only @agent()-decorated classes can be used as a coordinator."
            )
        if not workers:
            raise ValueError("McpAgentTeam requires at least one worker.")

        self._coordinator = coordinator
        self._workers = dict(workers)
        self._llm_module = llm_module
        self._max_rounds = max_rounds
        self._health_check_timeout = health_check_timeout
        self._runner: Any | None = None
        self._app: Any | None = None

    def _build_app(self) -> Any:
        """Build the Lauren app backing this team."""
        from lauren import LaurenFactory, module  # noqa: PLC0415

        from lauren_ai._module import AgentModule  # noqa: PLC0415

        try:
            from lauren_mcp._bridge import McpServerConfig as _McpServerConfig  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError("McpAgentTeam requires lauren-mcp") from exc

        mcp_configs = [_McpServerConfig(alias=alias, client=client) for alias, client in self._workers.items()]

        # Restrict coordinator to only worker aliases via @use_mcp_servers
        from lauren_ai._agents import (  # noqa: PLC0415
            AGENT_META,
            use_mcp_servers,
        )

        worker_aliases = tuple(self._workers.keys())
        # Apply @use_mcp_servers if not already restricted
        coordinator_meta = getattr(self._coordinator, AGENT_META)
        if coordinator_meta.allowed_mcp_aliases is None:
            self._coordinator = use_mcp_servers(*worker_aliases)(self._coordinator)

        agent_module = AgentModule.for_root(
            agents=[self._coordinator],
            imports=self._llm_module,
            mcp_servers=mcp_configs,
        )

        @module(imports=[agent_module])
        class _TeamApp:
            pass

        return LaurenFactory.create(_TeamApp)

    async def _ensure_ready(self) -> Any:
        """Lazily build and boot the app, return the runner."""
        if self._runner is not None:
            return self._runner
        from lauren.testing import TestClient  # noqa: PLC0415

        from lauren_ai._agents._runner import AgentRunner  # noqa: PLC0415

        self._app = self._build_app()
        TestClient(self._app)
        # Get the runner from the DI container
        from lauren_ai._agents import AGENT_META  # noqa: PLC0415

        coordinator_meta = getattr(self._coordinator, AGENT_META)
        runner_cls = coordinator_meta.runner_class
        if runner_cls is not None:
            try:
                self._runner = self._app.get(runner_cls)
            except Exception:  # noqa: BLE001
                self._runner = self._app.get(AgentRunner)
        else:
            self._runner = self._app.get(AgentRunner)
        return self._runner

    async def run(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> McpTeamResult:
        """Run the team and return the aggregated result.

        :param message: The task to delegate to the coordinator.
        :param conversation_id: Optional conversation session ID.
        :return: :class:`McpTeamResult` with the final answer.
        """
        runner = await self._ensure_ready()
        app = self._app
        if app is None:  # pragma: no cover — _ensure_ready always sets _app
            raise RuntimeError("McpAgentTeam app not initialised")
        coordinator_instance = app.get(self._coordinator)
        response = await runner.run(coordinator_instance, message, conversation_id=conversation_id)
        usage = getattr(response, "total_usage", None)
        return McpTeamResult(
            final_answer=getattr(response, "content", str(response)),
            rounds=getattr(response, "turns", 0),
            total_input_tokens=getattr(usage, "input_tokens", 0) if usage is not None else 0,
            total_output_tokens=getattr(usage, "output_tokens", 0) if usage is not None else 0,
        )

    async def close(self) -> None:
        """Disconnect all worker clients and release resources."""
        import contextlib  # noqa: PLC0415

        for client in self._workers.values():
            with contextlib.suppress(Exception):
                await client.close()

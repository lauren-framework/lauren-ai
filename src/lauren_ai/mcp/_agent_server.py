"""PRD 7 — Expose @agent classes as MCP servers.

Provides :class:`AgentMcpServer` which wraps any ``@agent``-decorated class
and produces a Lauren module ready for ``LaurenFactory.create()``, exposing
the agent as a first-class MCP server with ``run``, ``stream``, ``memory``
(resource), and ``system_prompt`` (prompt) endpoints.
"""

from __future__ import annotations

from typing import Any


def _generate_mcp_server_class(agent_class: type, path: str) -> type:
    """Dynamically build an ``@mcp_server`` class wrapping *agent_class*.

    The generated class exposes:
    - ``@mcp_tool("run")`` — synchronous single-turn invocation.
    - ``@mcp_tool("stream")`` — streaming invocation; reports progress per chunk.
    - ``@mcp_resource("memory://{conversation_id}")`` — conversation history.
    - ``@mcp_prompt("system_prompt")`` — returns the agent's system prompt.
    - ``@mcp_lifespan`` — wires runner + agent at startup.
    """
    from lauren_mcp._server._context import McpToolContext  # noqa: PLC0415
    from lauren_mcp.server._decorators import (  # noqa: PLC0415
        mcp_lifespan,
        mcp_prompt,
        mcp_resource,
        mcp_server,
        mcp_tool,
    )

    from lauren_ai._agents import AGENT_META  # noqa: PLC0415

    agent_meta = getattr(agent_class, AGENT_META)
    agent_name: str = agent_meta.name or agent_class.__name__
    system_prompt_text: str = agent_meta.system or ""

    @mcp_server(path)
    class _GeneratedAgentMcpServer:
        """Auto-generated MCP server wrapping an @agent class."""

        def __init__(self, runner: Any, agent_instance: agent_class) -> None:  # type: ignore[valid-type]
            self._runner = runner
            self._agent = agent_instance

        @mcp_lifespan
        async def _lifespan(self) -> Any:
            try:
                yield {"runner": self._runner, "agent": self._agent}
            finally:
                pass

        @mcp_tool(
            name="run",
            description=(f"Run the {agent_name} agent with a message and return the final response."),
        )
        async def run(
            self,
            message: str,
            conversation_id: str | None = None,
            ctx: McpToolContext | None = None,
        ) -> dict[str, Any]:
            """Invoke the agent and return its final response.

            Args:
                message: The user message to send to the agent.
                conversation_id: Optional session ID for multi-turn memory.
            """
            response = await self._runner.run(
                self._agent,
                message,
                conversation_id=conversation_id,
            )
            result: dict[str, Any] = {
                "content": response.content,
                "turns": response.turns,
                "stop_reason": response.stop_reason,
                "conversation_id": conversation_id,
            }
            if hasattr(response, "total_usage") and response.total_usage is not None:
                result["input_tokens"] = getattr(response.total_usage, "input_tokens", 0)
                result["output_tokens"] = getattr(response.total_usage, "output_tokens", 0)
            return result

        @mcp_tool(
            name="stream",
            description=(
                f"Stream the {agent_name} agent response token-by-token. Progress notifications carry each text chunk."
            ),
        )
        async def stream(
            self,
            message: str,
            conversation_id: str | None = None,
            ctx: McpToolContext | None = None,
        ) -> str:
            """Stream agent tokens as they arrive.

            Args:
                message: The user message to send to the agent.
                conversation_id: Optional session ID for multi-turn memory.
            """
            parts: list[str] = []
            stream_iter = await self._runner.run_stream(
                self._agent,
                message,
                conversation_id=conversation_id,
            )
            async for chunk in stream_iter:
                delta = getattr(chunk, "delta", None) or ""
                if delta:
                    parts.append(delta)
                    if ctx is not None:
                        await ctx.report_progress(
                            progress=float(len(parts)),
                            message=delta,
                        )
            return "".join(parts)

        @mcp_resource(
            "memory://{conversation_id}",
            mime_type="application/json",
            description=f"Conversation memory for {agent_name}",
        )
        async def memory(self, conversation_id: str) -> str:
            """Return serialised conversation history for *conversation_id*."""
            import json  # noqa: PLC0415

            store = agent_meta.conversation_store
            if store is None:
                return json.dumps([])
            try:
                history = await store.load(conversation_id)
                if history is None:
                    return json.dumps([])
                if hasattr(history, "messages"):
                    return json.dumps(
                        [{"role": m.get("role", ""), "content": m.get("content", "")} for m in history.messages]
                    )
                return json.dumps(str(history))
            except Exception:  # noqa: BLE001
                return json.dumps([])

        @mcp_prompt(
            name="system_prompt",
            description=f"System prompt for the {agent_name} agent.",
        )
        async def system_prompt(self) -> str:
            """Return the agent's system prompt."""
            return system_prompt_text

    _GeneratedAgentMcpServer.__name__ = f"_AgentMcpServer[{agent_name}]"
    _GeneratedAgentMcpServer.__qualname__ = _GeneratedAgentMcpServer.__name__
    return _GeneratedAgentMcpServer


class AgentMcpServer:
    """Wraps an ``@agent``-decorated class as an MCP server.

    Example::

        from lauren_ai import agent
        from lauren_ai.mcp import AgentMcpServer

        @agent(model="claude-sonnet-4-6", system="You are a researcher.")
        class ResearchAgent: ...

        # Build a Lauren module that serves ResearchAgent over MCP:
        server_module = AgentMcpServer(ResearchAgent, path="/research").build_module(llm_module)

        @module(imports=[server_module])
        class AppModule: pass

        app = LaurenFactory.create(AppModule)
        # → Streamable HTTP: POST /research/
        # → WebSocket:        WS  /research/ws

    :param agent_class: An ``@agent()``-decorated class.
    :param path: Mount path for the MCP server (default: ``"/agent"``).
    :param transport: Transport type: ``"ws"``, ``"sse"``, ``"streamable"``, or
        ``"all"`` (default).
    """

    def __init__(
        self,
        agent_class: type,
        path: str = "/agent",
        transport: str = "all",
    ) -> None:
        from lauren_ai._agents import AGENT_META  # noqa: PLC0415

        meta = getattr(agent_class, AGENT_META, None)
        if meta is None:
            raise ValueError(
                f"{agent_class.__name__!r} is not decorated with @agent().  "
                "Only @agent()-decorated classes can be wrapped as MCP servers."
            )
        self._agent_class = agent_class
        self._path = path
        self._transport = transport

    def build_server_class(self) -> type:
        """Return only the generated ``@mcp_server`` class (for direct use with
        :meth:`McpServerModule.for_root`)."""
        return _generate_mcp_server_class(self._agent_class, self._path)

    def build_module(self, llm_module: type, **runner_kwargs: Any) -> type:
        """Return a Lauren ``@module`` that exposes the agent as an MCP server.

        The generated module imports *llm_module* so the ``AgentRunner`` can
        resolve ``Transport`` and ``LLMConfig`` from the DI graph.

        :param llm_module: Result of ``LLMModule.for_root(...)``.
        :param runner_kwargs: Additional keyword arguments forwarded to
            ``AgentModule.for_root()``.
        :return: A ``@module``-decorated class.
        """
        from lauren_mcp import McpServerModule  # noqa: PLC0415

        from lauren_ai._module import AgentModule  # noqa: PLC0415

        agent_module = AgentModule.for_root(
            agents=[self._agent_class],
            imports=llm_module,
            **runner_kwargs,
        )
        server_cls = _generate_mcp_server_class(self._agent_class, self._path)
        mcp_module = McpServerModule.for_root(
            server_cls,
            transport=self._transport,
            imports=[agent_module],
        )
        return mcp_module

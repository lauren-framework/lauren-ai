"""PRD 11 — AgentHttpModule.for_agent() factory.

Creates a Lauren module with standard HTTP endpoints for deploying
a ``lauren-ai`` agent as a streaming HTTP service.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


class AgentHttpModule:
    """Factory for creating Lauren modules with agent HTTP endpoints.

    Calling :meth:`for_agent` returns a ``@module``-decorated class with:

    - ``POST /chat`` — single-turn JSON response
    - ``POST /stream`` — SSE streaming response (NDJSON line-delimited
      ``AgentEvent`` objects)

    Example::

        from lauren_ai._http import AgentHttpModule

        app_module = AgentHttpModule.for_agent(
            MyAgent,
            llm_module=LLMModule.for_root(...),
            path_prefix="/api",
        )

        @module(imports=[app_module])
        class AppModule: pass

        app = LaurenFactory.create(AppModule)
    """

    @classmethod
    def for_agent(
        cls,
        agent_class: type,
        *,
        llm_module: type,
        path_prefix: str = "",
        signals: Any | None = None,
    ) -> type:
        """Create a Lauren ``@module`` with agent HTTP endpoints.

        :param agent_class: An ``@agent()``-decorated class.
        :param llm_module: Result of ``LLMModule.for_root(...)``.
        :param path_prefix: Optional URL prefix for the mounted controller.
        :param signals: Optional :class:`~lauren_ai._signals.SignalBus`.
        :return: A ``@module``-decorated class.
        """
        from lauren import module  # noqa: PLC0415

        from lauren_ai._agents import AGENT_META  # noqa: PLC0415
        from lauren_ai._module import AgentModule  # noqa: PLC0415

        meta = getattr(agent_class, AGENT_META, None)
        if meta is None:
            raise ValueError(f"{agent_class.__name__!r} is not decorated with @agent().")

        agent_module = AgentModule.for_root(
            agents=[agent_class],
            imports=llm_module,
            signals=signals,
        )

        controller_cls = _build_agent_controller(agent_class, path_prefix)

        @module(
            imports=[agent_module],
            controllers=[controller_cls],
        )
        class _AgentHttpModule:
            pass

        _AgentHttpModule.__name__ = f"_AgentHttpModule[{agent_class.__name__}]"
        _AgentHttpModule.__qualname__ = _AgentHttpModule.__name__
        return _AgentHttpModule


def _build_agent_controller(agent_class: type, path_prefix: str) -> type:
    """Build the @controller class for the agent."""
    from typing import cast  # noqa: PLC0415

    from lauren import controller, post  # noqa: PLC0415
    from lauren.extractors import Json  # noqa: PLC0415
    from lauren.sse import EventStream, ServerSentEvent  # noqa: PLC0415

    from lauren_ai._agents import AGENT_META  # noqa: PLC0415
    from lauren_ai._agents._runner import AgentRunner  # noqa: PLC0415
    from lauren_ai._http._events import (  # noqa: PLC0415
        AgentDoneEvent,
        AgentHttpTokenEvent,
    )

    prefix = path_prefix.rstrip("/")
    agent_meta = getattr(agent_class, AGENT_META)
    agent_name = agent_meta.name or agent_class.__name__

    @controller(f"{prefix}/{agent_name.lower().replace(' ', '_')}")
    class _AgentController:
        def __init__(self, runner: AgentRunner, agent_instance: agent_class) -> None:  # type: ignore[valid-type]
            self._runner = runner
            self._agent = agent_instance

        @post("/chat")
        async def chat(
            self,
            body: Json,  # type: ignore[type-arg]
        ) -> dict[str, Any]:
            """Single-turn JSON chat endpoint."""
            data = cast(dict[str, Any], body)
            message: str = data.get("message") or ""
            conversation_id: str | None = data.get("conversation_id")
            if not message:
                return {"error": "message is required"}
            response = await self._runner.run(self._agent, message, conversation_id=conversation_id)
            return {
                "content": getattr(response, "content", str(response)),
                "conversation_id": conversation_id,
                "turns": getattr(response, "turns", 0),
                "stop_reason": getattr(response, "stop_reason", "end_turn"),
            }

        @post("/stream")
        async def stream(
            self,
            body: Json,  # type: ignore[type-arg]
        ) -> EventStream:
            """SSE streaming endpoint — yields ``AgentEvent`` as SSE data."""
            data2 = cast(dict[str, Any], body)
            message: str = data2.get("message") or ""
            conversation_id: str | None = data2.get("conversation_id")

            async def _generate() -> AsyncIterator[ServerSentEvent]:
                if not message:
                    yield ServerSentEvent(data=AgentDoneEvent(content="", stop_reason="no_message").to_json())
                    return
                try:
                    parts: list[str] = []
                    stream_iter = await self._runner.run_stream(self._agent, message, conversation_id=conversation_id)
                    async for chunk in stream_iter:
                        delta = getattr(chunk, "delta", None) or ""
                        if delta:
                            parts.append(delta)
                            yield ServerSentEvent(data=AgentHttpTokenEvent(content=delta).to_json())
                    full_content = "".join(parts)
                    yield ServerSentEvent(
                        data=AgentDoneEvent(
                            content=full_content,
                            stop_reason="end_turn",
                        ).to_json()
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("AgentController.stream: %s", exc)
                    yield ServerSentEvent(
                        data=AgentDoneEvent(
                            content=f"Error: {exc}",
                            stop_reason="error",
                        ).to_json()
                    )

            return EventStream(_generate())

    _AgentController.__name__ = f"_AgentController[{agent_name}]"
    _AgentController.__qualname__ = _AgentController.__name__
    return _AgentController

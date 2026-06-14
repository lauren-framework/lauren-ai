"""PRD — MCP Sampling → AgentRunner Bridge.

Provides :class:`AgentSamplingHandler` which routes MCP
``sampling/createMessage`` requests to an ``AgentRunnerBase`` so that MCP
server tools can leverage the host LLM for sub-tasks (summarisation,
classification, code generation) without managing their own LLM credentials.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lauren_ai._agents._runner import AgentRunnerBase
    from lauren_ai._config import AgentConfig

logger = logging.getLogger(__name__)

_STOP_REASON_END_TURN = "endTurn"
_STOP_REASON_MAX_TOKENS = "maxTokens"


class AgentSamplingHandler:
    """Routes MCP ``sampling/createMessage`` requests to a ``lauren-ai`` runner.

    Use as the ``sampling_handler`` when creating an MCP client::

        from lauren_ai.mcp import AgentSamplingHandler

        handler = AgentSamplingHandler(runner=runner, agent_class=MyAgent)
        client = McpServer.streamable_http(url, sampling_handler=handler)

    :param runner: The ``AgentRunnerBase`` instance to invoke.
    :param agent_class: The ``@agent()``-decorated class. A fresh instance is
        created per sampling request.
    :param config_override: Override ``AgentConfig`` applied to every sampling
        call.  Defaults to ``AgentConfig(max_turns=1)`` to prevent loops.
    :param model_override: Optional model identifier for sampling calls.
    """

    def __init__(
        self,
        runner: AgentRunnerBase,
        agent_class: type,
        *,
        config_override: AgentConfig | None = None,
        model_override: str | None = None,
    ) -> None:
        self._runner = runner
        self._agent_class = agent_class
        self._config_override = config_override
        self._model_override = model_override

    async def __call__(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``sampling/createMessage`` request.

        :param params: Raw params dict with ``messages``, ``maxTokens``,
            ``systemPrompt``, ``temperature``, ``modelPreferences`` fields.
        :return: ``CreateMessageResult``-shaped dict.
        """
        prompt = _extract_prompt(params)
        model = _resolve_model(params, self._model_override, self._agent_class)

        config = self._config_override
        if config is None:
            from lauren_ai._config import AgentConfig  # noqa: PLC0415

            config = AgentConfig(max_turns=1)

        try:
            agent_instance = self._agent_class()
            run_kwargs: dict[str, Any] = {}
            if model:
                run_kwargs["model_override"] = model
            response = await self._runner.run(agent_instance, prompt, **run_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("AgentSamplingHandler: run failed: %s", exc)
            return {
                "role": "assistant",
                "content": {"type": "text", "text": f"Sampling error: {exc}"},
                "model": model or "unknown",
                "stopReason": "error",
            }

        text = getattr(response, "content", "") or ""
        stop_reason = _STOP_REASON_MAX_TOKENS if _looks_truncated(response) else _STOP_REASON_END_TURN
        inferred_model = model or _infer_model(self._agent_class)
        return {
            "role": "assistant",
            "content": {"type": "text", "text": text},
            "model": inferred_model,
            "stopReason": stop_reason,
        }


def _extract_prompt(params: dict[str, Any]) -> str:
    """Concatenate user-role text messages from sampling params."""
    messages = params.get("messages") or []
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", None)
            if str(role) == "user" and content is not None:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
            continue
        role = msg.get("role", "")
        if role != "user":
            continue
        content = msg.get("content") or {}
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, dict) and content.get("type") == "text":
            text = content.get("text", "")
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _resolve_model(
    params: dict[str, Any],
    override: str | None,
    agent_class: type,
) -> str | None:
    if override:
        return override
    prefs = params.get("modelPreferences") or {}
    if isinstance(prefs, dict):
        hints = prefs.get("hints") or []
        for h in hints:
            name = (h.get("name") if isinstance(h, dict) else getattr(h, "name", None)) or ""
            if name:
                return name
    return _infer_model(agent_class)


def _infer_model(agent_class: type) -> str:
    from lauren_ai._agents import AGENT_META  # noqa: PLC0415

    meta = getattr(agent_class, AGENT_META, None)
    return (meta.model if meta and meta.model else "") or ""


def _looks_truncated(response: Any) -> bool:
    stop = getattr(response, "stop_reason", "")
    return str(stop).lower() in {"max_tokens", "maxtokens", "length"}

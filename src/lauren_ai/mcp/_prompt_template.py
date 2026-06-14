"""PRD — MCP Prompts as Agent Prompt Templates.

Provides :class:`McpPromptTemplate` and :func:`list_mcp_prompts` for using
MCP server prompts as prompt templates in ``lauren-ai`` agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lauren_mcp import McpClientProtocol

logger = logging.getLogger(__name__)


class McpPromptTemplate:
    """Adapter that wraps an MCP prompt as a ``lauren-ai`` prompt template.

    Fetches prompt content from an MCP server at render time via
    ``prompts/get``.  Returns either a plain ``str`` (single user-role text
    message) or a list of message dicts (multi-turn template).

    Can be passed directly to ``@agent(system=McpPromptTemplate(...))``
    because it implements ``async __call__(self, **kwargs) -> str``.

    Example::

        from lauren_ai.mcp import McpPromptTemplate

        template = McpPromptTemplate(client, "system_prompt", alias="myserver")

        @agent(model="claude-sonnet-4-6", system=template)
        class MyAgent: ...

    :param client: Connected MCP client.
    :param prompt_name: Name of the prompt as returned by ``prompts/list``.
    :param alias: Short label used in logging and :attr:`name` property.
    """

    def __init__(
        self,
        client: McpClientProtocol,
        prompt_name: str,
        alias: str = "",
    ) -> None:
        self._client = client
        self._prompt_name = prompt_name
        self._alias = alias or prompt_name

    @property
    def name(self) -> str:
        return f"mcp:{self._alias}:{self._prompt_name}"

    @property
    def prompt_name(self) -> str:
        return self._prompt_name

    async def argument_names(self) -> list[str]:
        """Return declared argument names by consulting ``prompts/list``."""
        schemas = await self._client.list_prompts()
        for schema in schemas:
            if schema.name == self._prompt_name:
                return [arg.name for arg in (schema.arguments or [])]
        raise ValueError(f"McpPromptTemplate: prompt '{self._prompt_name}' not found in server catalogue")

    async def render(self, **arguments: str) -> str | list[Any]:
        """Fetch the prompt from the MCP server and convert to text."""
        result = await self._client.get_prompt(
            self._prompt_name,
            arguments if arguments else None,
        )
        return _convert_get_prompt_result(result)

    async def __call__(self, **arguments: str) -> str | list[Any]:
        return await self.render(**arguments)

    async def invoke(self, input: Any) -> Any:
        """Invoke as a Runnable in a Chain."""
        kwargs: dict[str, str] = input if isinstance(input, dict) else {}
        return await self.render(**kwargs)

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])


def _convert_get_prompt_result(result: Any) -> str | list[Any]:
    """Convert a ``GetPromptResult`` to ``str | list``."""
    if result is None:
        return ""

    messages: list[Any] = []
    if hasattr(result, "messages"):
        messages = list(result.messages or [])
    elif isinstance(result, dict) and "messages" in result:
        messages = list(result["messages"] or [])

    if not messages:
        return ""

    if len(messages) == 1:
        msg = messages[0]
        role = getattr(msg, "role", None) or msg.get("role", "")
        if str(role) == "user":
            content = getattr(msg, "content", None) or msg.get("content", {})
            if isinstance(content, str):
                return content
            if hasattr(content, "text"):
                return content.text
            if isinstance(content, dict):
                return content.get("text", "")

    # Multi-message: convert to list of dicts
    result_messages: list[Any] = []
    for msg in messages:
        role = str(getattr(msg, "role", None) or (msg.get("role", "") if isinstance(msg, dict) else ""))
        content = getattr(msg, "content", None) or (msg.get("content", {}) if isinstance(msg, dict) else {})
        if isinstance(content, str):
            text = content
        elif hasattr(content, "text"):
            text = content.text
        elif isinstance(content, dict):
            text = content.get("text", "")
        else:
            text = str(content)
        result_messages.append({"role": role, "content": text})
    return result_messages


async def list_mcp_prompts(client: McpClientProtocol) -> list[McpPromptTemplate]:
    """List all prompts on *client* and return ``McpPromptTemplate`` wrappers.

    :param client: A connected MCP client.
    :return: One :class:`McpPromptTemplate` per prompt.
    """
    schemas = await client.list_prompts()
    return [McpPromptTemplate(client, s.name) for s in schemas]


class McpSystemPromptBuilder:
    """Lazily builds a system prompt by fetching it from an MCP server at agent startup.

    Pass to ``AgentModule.for_root(system_prompt_builder=...)`` (conceptual;
    the actual integration point is the agent's ``system`` parameter which
    accepts an async callable returning ``str``).

    :param client: Connected MCP client.
    :param prompt_name: Name of the prompt.
    :param arguments: Static keyword arguments passed to ``prompts/get``.
    """

    def __init__(
        self,
        client: McpClientProtocol,
        prompt_name: str,
        **arguments: str,
    ) -> None:
        self._template = McpPromptTemplate(client, prompt_name)
        self._arguments = arguments

    async def __call__(self) -> str:
        result = await self._template.render(**self._arguments)
        if isinstance(result, list):
            return "\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in result)
        return str(result)

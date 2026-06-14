"""PRD 10 — Agent Memory via MCP Resources.

Provides MCP-backed implementations of the three memory protocols:
- :class:`McpConversationStore` — ``ConversationStore`` over MCP
- :class:`McpUserMemoryStore` — ``UserMemoryStore`` over MCP

These allow agents to share memory across deployments, scale horizontally,
and use any MCP-compatible memory service.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _extract_resource_text(raw: Any) -> str:
    """Extract text from a ReadResourceResult (or raw dict)."""
    if raw is None:
        return ""
    contents: list[Any] = []
    if hasattr(raw, "contents"):
        contents = list(raw.contents or [])
    elif isinstance(raw, dict):
        contents = list(raw.get("contents") or [])

    for item in contents:
        if hasattr(item, "text") and item.text:
            return item.text
        if isinstance(item, dict) and item.get("text"):
            return item["text"]
    return ""


class McpConversationStore:
    """``ConversationStore`` backed by a remote MCP server.

    Uses ``resources/read`` for loading and ``tools/call`` for saving /
    deleting.  Works with ``LaurenMcpMemoryServer`` (the deployable server in
    ``lauren-mcp``) or any MCP server that exposes the expected tools.

    :param client: A connected MCP client instance.
    :param alias: URI scheme used for resource URIs.  Defaults to ``"memory"``.
    :param tool_prefix: Prefix for tool names on the remote server.
    """

    def __init__(
        self,
        client: Any,
        *,
        alias: str = "memory",
        tool_prefix: str = "",
    ) -> None:
        self._client = client
        self._alias = alias
        self._prefix = tool_prefix

    async def load(self, conversation_id: str) -> Any:
        """Load a conversation snapshot.

        Returns ``[]`` when the conversation is not found.
        """
        uri = f"{self._alias}://conversations/{conversation_id}"
        try:
            raw = await self._client.read_resource(uri)
        except Exception as exc:  # noqa: BLE001
            err_code = getattr(exc, "code", None)
            if err_code in (-32002, -32600, -32601):
                return []
            logger.debug("McpConversationStore.load: %r → %s", conversation_id, exc)
            return []

        text = _extract_resource_text(raw)
        if not text:
            return []
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

    async def save(self, conversation_id: str, snapshot: Any) -> None:
        """Save a conversation snapshot."""
        if not conversation_id:
            return
        if isinstance(snapshot, list):
            snapshot = {"messages": snapshot, "summary": None}
        tool_name = f"{self._prefix}save_conversation"
        try:
            await self._client.call_tool(
                tool_name,
                {
                    "conversation_id": conversation_id,
                    "snapshot": json.dumps(snapshot),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("McpConversationStore.save: %r → %s", conversation_id, exc)

    async def delete(self, conversation_id: str) -> None:
        """Delete a conversation."""
        tool_name = f"{self._prefix}delete_conversation"
        try:
            await self._client.call_tool(
                tool_name,
                {"conversation_id": conversation_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("McpConversationStore.delete: %r → %s", conversation_id, exc)


@dataclass
class McpMemoryFact:
    """A discrete user memory fact returned by :class:`McpUserMemoryStore`."""

    key: str
    value: str
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class McpUserMemoryStore:
    """``UserMemoryStore`` backed by a remote MCP server.

    Expects the server to expose:
    - ``save_user_fact(user_id, key, value)`` tool
    - ``get_user_facts(user_id)`` tool → JSON array of ``{key, value}``
    - ``delete_user_fact(user_id, key)`` tool

    :param client: A connected MCP client instance.
    :param tool_prefix: Prefix for tool names on the remote server.
    """

    def __init__(
        self,
        client: Any,
        *,
        tool_prefix: str = "",
    ) -> None:
        self._client = client
        self._prefix = tool_prefix

    async def save(self, user_id: str, key: str, value: str) -> None:
        """Persist a user fact."""
        tool_name = f"{self._prefix}save_user_fact"
        try:
            await self._client.call_tool(
                tool_name,
                {"user_id": user_id, "key": key, "value": value},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("McpUserMemoryStore.save: %r/%r → %s", user_id, key, exc)

    async def get_all(self, user_id: str) -> list[McpMemoryFact]:
        """Retrieve all facts for a user."""
        tool_name = f"{self._prefix}get_user_facts"
        try:
            raw = await self._client.call_tool(tool_name, {"user_id": user_id})
        except Exception as exc:  # noqa: BLE001
            logger.debug("McpUserMemoryStore.get_all: %r → %s", user_id, exc)
            return []

        # Unwrap text content
        text = ""
        if isinstance(raw, list) and raw:
            item = raw[0]
            text = (item.get("text", "") or "") if isinstance(item, dict) else (getattr(item, "text", "") or "")
        elif isinstance(raw, str):
            text = raw

        try:
            facts_data = json.loads(text) if text else []
        except (json.JSONDecodeError, ValueError):
            return []

        return [
            McpMemoryFact(
                key=f.get("key", "") if isinstance(f, dict) else getattr(f, "key", ""),
                value=f.get("value", "") if isinstance(f, dict) else getattr(f, "value", ""),
                user_id=user_id,
            )
            for f in (facts_data if isinstance(facts_data, list) else [])
        ]

    async def delete(self, user_id: str, key: str) -> None:
        """Delete a user fact."""
        tool_name = f"{self._prefix}delete_user_fact"
        try:
            await self._client.call_tool(tool_name, {"user_id": user_id, "key": key})
        except Exception as exc:  # noqa: BLE001
            logger.debug("McpUserMemoryStore.delete: %r/%r → %s", user_id, key, exc)

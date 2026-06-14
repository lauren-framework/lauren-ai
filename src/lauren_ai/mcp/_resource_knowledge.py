"""PRD — MCP Resources as Agent Knowledge Sources.

Provides :class:`McpResourceKnowledgeSource` that bridges MCP resource
listings into the ``lauren-ai`` knowledge-source protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lauren_mcp import McpClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeChunk:
    """A single retrieved knowledge fragment from an MCP resource.

    :param content: Text content of the chunk.
    :param source: Human-readable provenance label (URI, file path, etc.).
    :param metadata: Arbitrary key-value metadata (uri, mimeType, score, ...).
    :param score: Optional relevance score in [0.0, 1.0].
    """

    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class McpResourceKnowledgeSource:
    """Adapter that exposes MCP resources as a ``lauren-ai`` knowledge source.

    Lists resources from an MCP server, reads their text content, and returns
    up to *k* :class:`KnowledgeChunk` objects per ``search(query, k)`` call.

    Implements the duck-typed knowledge-source protocol:
    ``async search(query: str, k: int) -> list[KnowledgeChunk]``.

    Example::

        from lauren_ai.mcp import McpResourceKnowledgeSource

        docs_source = McpResourceKnowledgeSource(
            client=mcp_client,
            alias="docs",
            filter_fn=lambda r: r.uri.endswith(".md"),
        )
        chunks = await docs_source.search("authentication", k=5)

    :param client: Connected MCP client.
    :param alias: Short, URL-safe alias — used to namespace the generated tool
        name: ``search_<alias>``.
    :param filter_fn: Optional predicate on ``ResourceSchema`` objects.  Resources
        for which it returns ``False`` are skipped.
    :param tool_name: Override the generated tool name (default: ``search_{alias}``).
    """

    def __init__(
        self,
        client: McpClientProtocol,
        alias: str,
        *,
        filter_fn: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        self._client = client
        self._alias = alias
        self._filter_fn = filter_fn
        self.tool_name: str = tool_name or f"search_{alias}"

    @property
    def name(self) -> str:
        return f"mcp:{self._alias}"

    async def search(self, query: str, k: int = 5) -> list[KnowledgeChunk]:
        """List resources, read text content, return up to *k* chunks.

        Resources are returned in listing order (no semantic ranking in this
        version).  Binary-only resources are skipped silently.

        :param query: Search query (currently unused for ranking; reserved for
            future BM25/embedding re-rank support).
        :param k: Maximum number of chunks to return.
        :return: Up to *k* :class:`KnowledgeChunk` objects.
        """
        try:
            resources = await self._client.list_resources()
        except Exception as exc:  # noqa: BLE001
            logger.error("McpResourceKnowledgeSource: list_resources failed: %s", exc)
            return []

        if self._filter_fn is not None:
            resources = [r for r in resources if self._filter_fn(r)]

        chunks: list[KnowledgeChunk] = []
        for resource in resources:
            if len(chunks) >= k:
                break
            uri = getattr(resource, "uri", "") or ""
            try:
                result = await self._client.read_resource(uri)
                text = _extract_text(result)
                if not text:
                    continue
                chunks.append(
                    KnowledgeChunk(
                        content=text,
                        source=uri,
                        metadata={
                            "uri": uri,
                            "name": getattr(resource, "name", ""),
                            "mimeType": getattr(resource, "mimeType", "text/plain"),
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("McpResourceKnowledgeSource: skip '%s': %s", uri, exc)
                continue

        return chunks


def _extract_text(result: Any) -> str:
    """Extract text from a ``ReadResourceResult`` (or raw dict)."""
    if result is None:
        return ""
    # ReadResourceResult has .contents
    contents: list[Any] = []
    if hasattr(result, "contents"):
        contents = list(result.contents or [])
    elif isinstance(result, dict):
        contents = list(result.get("contents") or [])

    parts: list[str] = []
    for item in contents:
        if hasattr(item, "text") and item.text:
            parts.append(item.text)
        elif isinstance(item, dict) and item.get("text"):
            parts.append(item["text"])
    return "\n".join(parts)

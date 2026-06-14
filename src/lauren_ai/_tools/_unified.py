"""UnifiedToolContext protocol and ToolContextAdapter (PRD 12).

Provides a minimal interface shared by both ``ToolContext`` (native lauren-ai)
and ``McpToolContext`` (lauren-mcp), enabling tool authors to write once and
deploy in both contexts.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class UnifiedToolContext(Protocol):
    """Minimal protocol shared by both ``ToolContext`` and ``McpToolContext``.

    Both context types satisfy this protocol without any adapter.  Tool authors
    can annotate a parameter as ``UnifiedToolContext`` to write a function that
    works as both a native ``@tool()`` and an ``@mcp_tool()``::

        @dual_tool(name="search")
        async def search(query: str, ctx: UnifiedToolContext) -> str:
            role = ctx.get_metadata("required_role", "user")
            return await do_search(query, role)

    When called natively by ``AgentRunnerBase``, ``ctx`` is a ``ToolContext``.
    When called remotely via MCP, ``ctx`` is a ``McpToolContext`` or a
    ``ToolContextAdapter`` wrapping one.
    """

    tool_name: str
    metadata: dict[str, Any]
    state: dict[str, Any]

    def get_metadata(self, key: str, default: Any = None) -> Any: ...


class ToolContextAdapter:
    """Wraps ``McpToolContext`` to present the ``ToolContext``-compatible interface.

    Use this when an ``@mcp_tool()`` function wants to access the calling
    agent's ``AgentContext`` (e.g. for conversation_id-based caching or
    per-agent metrics).

    The adapter is created by calling :meth:`McpToolContext.as_tool_context`
    (added to ``McpToolContext`` via a monkey-patch on import of this module).
    Agent context identity is populated from ``extras`` that the ``lauren-ai``
    executor forwards via the ``_meta`` field in ``call_tool``.

    :param mcp_ctx: The ``McpToolContext`` to wrap.
    :param agent_context: Optional ``AgentContext`` reconstructed from
        ``mcp_ctx.extras`` or passed explicitly.
    """

    def __init__(
        self,
        mcp_ctx: Any,
        agent_context: Any | None = None,
    ) -> None:
        self._mcp = mcp_ctx
        self._agent_context = agent_context

    # ------------------------------------------------------------------
    # UnifiedToolContext fields
    # ------------------------------------------------------------------

    @property
    def tool_name(self) -> str:
        return str(getattr(self._mcp, "tool_name", ""))  # type: ignore[return-value]

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(getattr(self._mcp, "metadata", {}))

    @property
    def state(self) -> dict[str, Any]:
        return dict(getattr(self._mcp, "state", {}))

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._mcp.get_metadata(key, default)

    # ------------------------------------------------------------------
    # ToolContext-compatible fields
    # ------------------------------------------------------------------

    @property
    def tool_use_id(self) -> str | None:
        raw = getattr(self._mcp, "tool_use_id", None)
        return str(raw) if raw is not None else None

    @property
    def execution_context(self) -> Any:
        return getattr(self._mcp, "execution_context", None)

    @property
    def request(self) -> None:
        return None

    @property
    def turn(self) -> int:
        return int(getattr(self._mcp, "extras", {}).get("turn", 0))

    @property
    def agent_context(self) -> Any:
        return self._agent_context

    # ------------------------------------------------------------------
    # MCP-specific pass-through
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        return getattr(self._mcp, "session_id", None)  # type: ignore[return-value]

    @property
    def extras(self) -> dict[str, Any]:
        return dict(getattr(self._mcp, "extras", {}))

    @property
    def lifespan_context(self) -> dict[str, Any]:
        return dict(getattr(self._mcp, "lifespan_context", {}))

    @property
    def calling_agent(self) -> str | None:
        """Agent name from extras when called from a lauren-ai agent loop."""
        return self.extras.get("agent_name")

    # Delegate MCP-only methods so ToolContextAdapter can be used in both branches:
    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        if hasattr(self._mcp, "report_progress"):
            await self._mcp.report_progress(progress, total, message)

    async def log(self, level: str, msg: str, data: Any = None) -> None:
        if hasattr(self._mcp, "log"):
            await self._mcp.log(level, msg, data)

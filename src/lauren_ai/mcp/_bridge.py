"""Phase 6 — MCP ↔ AgentModule integration bridge.

Provides:

* :class:`McpServerConfig` — pairs an alias string with an MCP client.
* :func:`_make_mcp_bridge_class` — factory that returns a unique
  ``@injectable(SINGLETON)`` class whose ``@post_construct`` connects every
  configured MCP server and injects the resulting tools into each agent's
  ``AgentMeta.tools`` map.

The returned class is generated fresh per ``AgentModule.for_root()`` call so
each module has its own isolated bridge singleton — the same pattern used by
``_make_kb_loader_class`` for knowledge bases.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Re-export McpServerConfig from lauren_mcp so users only need one import.
try:
    from lauren_mcp._bridge import McpServerConfig
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "MCP support requires the 'lauren-mcp' package.  Install it with: pip install 'lauren-ai[mcp]'"
    ) from _exc

__all__ = ["McpServerConfig"]


# ---------------------------------------------------------------------------
# Executor factory
# ---------------------------------------------------------------------------


def _make_mcp_executor(client: Any, tool_name: str) -> Any:
    """Return an async callable that routes **kwargs to client.call_tool."""

    async def executor(**kwargs: Any) -> Any:
        content = await client.call_tool(tool_name, kwargs)
        # Unwrap a single TextContent item to a plain string — matches
        # what native @tool() functions return so the runner serialises it
        # consistently.
        if isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get("type") == "text":
            return content[0]["text"]
        return json.dumps(content)

    return executor


# ---------------------------------------------------------------------------
# Bridge injectable factory
# ---------------------------------------------------------------------------


def _make_mcp_bridge_class(
    configs: list[McpServerConfig],
    agents: list[type],
) -> type:
    """Return a fresh ``@injectable(SINGLETON)`` class for the given servers.

    The generated class's ``@post_construct`` hook:

    1. Calls ``client.connect()`` for each server (performs the MCP
       ``initialize`` handshake).
    2. Calls ``client.list_tools()`` to retrieve the server's tool catalogue.
    3. For each tool, injects a ``(executor, ToolMeta)`` entry into every
       agent's ``AgentMeta.tools`` dict under the namespaced key
       ``{alias}__{tool_name}``.
    4. Logs a startup summary at INFO level.

    A ``@pre_destruct`` hook closes all clients during application teardown.
    """
    from lauren import Scope, injectable, post_construct, pre_destruct  # noqa: PLC0415

    from lauren_ai._agents import AGENT_META  # noqa: PLC0415
    from lauren_ai._tools import ToolMeta  # noqa: PLC0415

    @injectable(scope=Scope.SINGLETON)
    class _McpBridge:
        """Auto-generated MCP bridge singleton."""

        @post_construct
        async def _connect_all(self) -> None:
            for cfg in configs:
                try:
                    await cfg.client.connect()
                    tools = await cfg.client.list_tools()
                    for tool in tools:
                        ns_name = f"{cfg.alias}__{tool.name}"
                        executor = _make_mcp_executor(cfg.client, tool.name)
                        tool_meta = ToolMeta(
                            name=ns_name,
                            description=tool.description,
                            parameters={
                                "name": ns_name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            },
                            is_async=True,
                            reads_context=False,
                        )
                        for agent_cls in agents:
                            meta = getattr(agent_cls, AGENT_META)
                            meta.tools[ns_name] = (executor, tool_meta)
                    logger.info(
                        "MCP bridge: loaded %d tool(s) from '%s'",
                        len(tools),
                        cfg.alias,
                    )
                    for tool in tools:
                        logger.info("MCP bridge:   %s__%s", cfg.alias, tool.name)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "MCP bridge: failed to connect '%s': %s",
                        cfg.alias,
                        exc,
                    )

        @pre_destruct
        async def _disconnect_all(self) -> None:
            for cfg in configs:
                try:  # noqa: SIM105
                    await cfg.client.close()
                except Exception:  # noqa: BLE001
                    pass

    # Give the class a meaningful name for DI diagnostics.
    aliases = "_".join(c.alias for c in configs) or "none"
    _McpBridge.__name__ = f"_McpBridge[{aliases}]"
    _McpBridge.__qualname__ = _McpBridge.__name__
    return _McpBridge

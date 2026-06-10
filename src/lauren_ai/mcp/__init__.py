"""Model Context Protocol integration for lauren-ai.

Re-exports the MCP client API from ``lauren_mcp`` and provides the
Phase 6 integration bridge that connects MCP servers into the agent tool
registry via ``AgentModule.for_root(mcp_servers=[...])``.

Typical usage::

    from lauren_ai.mcp import McpServer, McpServerConfig

    AgentModule.for_root(
        agents=[MyAgent],
        mcp_servers=[
            McpServerConfig(
                alias="fs",
                client=McpServer.stdio(["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
            ),
        ],
    )
"""

from __future__ import annotations

from ._bridge import McpServerConfig

try:
    from lauren_mcp import McpClientProtocol, McpServer
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "MCP support requires the 'lauren-mcp' package.  Install it with: pip install 'lauren-ai[mcp]'"
    ) from _exc

__all__ = [
    "McpServer",
    "McpClientProtocol",
    "McpServerConfig",
]

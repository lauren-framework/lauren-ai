"""Phase 6 — MCP ↔ AgentModule integration bridge.

Provides:

* :class:`McpServerConfig` — pairs an alias string with an MCP client.
* :func:`_make_mcp_bridge_class` — factory returning a unique SINGLETON class
  whose ``@post_construct`` connects every configured MCP server and injects
  namespaced tools into each agent's ``AgentMeta.tools`` map.
* :func:`_make_dynamic_mcp_bridge_class` — variant that subscribes to
  ``notifications/tools/list_changed`` and atomically updates the tool map at
  runtime (PRD 5).

PRD features implemented here:
- PRD 4: streaming progress — ``_make_mcp_executor`` with per-call progress
  handler + ``ToolProgressEvent`` emission.
- PRD 5: dynamic discovery — list_changed subscription + atomic catalogue diff.
- PRD 6: unified signals — ``ToolProgressEvent`` / ``McpToolsRefreshed`` on bus.
- PRD 8: per-agent access control — ``allowed_mcp_aliases`` filtering.
"""

from __future__ import annotations

import asyncio
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
# Executor factory (PRDs 4, 6, 8)
# ---------------------------------------------------------------------------


def _make_mcp_executor(
    client: Any,
    tool_name: str,
    *,
    alias: str = "",
    signals: Any | None = None,
) -> Any:
    """Return an async callable that routes **kwargs to client.call_tool.

    When *signals* is provided the executor:

    - Registers a per-call progress handler before calling ``call_tool``.
    - Emits :class:`~lauren_ai._signals.ToolProgressEvent` for each
      ``notifications/progress`` message.
    - Unregisters the handler once the call completes.

    The ``_tool_use_id`` kwarg (injected by the runner, not the LLM) is used as
    the progress-correlation token and stripped from the call arguments.
    """
    namespaced_name = f"{alias}__{tool_name}" if alias else tool_name

    async def executor(**kwargs: Any) -> Any:
        tool_use_id: str = str(kwargs.pop("_tool_use_id", "") or "")

        unsubscribe: Any | None = None
        if signals is not None and hasattr(client, "on_progress"):

            def _on_progress(params: dict[str, Any]) -> None:
                token = str(params.get("progressToken", ""))
                if tool_use_id and token and token != tool_use_id:
                    return

                from lauren_ai._signals import ToolProgressEvent  # noqa: PLC0415

                asyncio.ensure_future(
                    signals.emit(
                        ToolProgressEvent(
                            tool_name=namespaced_name,
                            tool_use_id=tool_use_id,
                            progress=float(params.get("progress", 0)),
                            total=float(params["total"]) if "total" in params else None,
                            message=params.get("message"),
                            alias=alias,
                        )
                    )
                )

            unsubscribe = client.on_progress(_on_progress)

        try:
            content = await client.call_tool(tool_name, kwargs)
        finally:
            if unsubscribe is not None:
                import contextlib  # noqa: PLC0415

                with contextlib.suppress(Exception):
                    unsubscribe()

        if isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get("type") == "text":
            return content[0]["text"]
        return json.dumps(content)

    return executor


# ---------------------------------------------------------------------------
# Static bridge (PRDs 6, 8)
# ---------------------------------------------------------------------------


def _make_mcp_bridge_class(
    configs: list[McpServerConfig],
    agents: list[type],
    *,
    signals: Any | None = None,
) -> type:
    """Return a fresh ``@injectable(SINGLETON)`` class.

    At ``@post_construct`` time:
    1. Connects every MCP client.
    2. Fetches each server's tool list.
    3. Injects ``{alias}__{tool_name}`` entries into every **allowed** agent's
       ``AgentMeta.tools`` dict (respects ``allowed_mcp_aliases`` — PRD 8).
    4. Emits signal events when *signals* is provided (PRD 6).
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
                        executor = _make_mcp_executor(cfg.client, tool.name, alias=cfg.alias, signals=signals)
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
                            agent_meta = getattr(agent_cls, AGENT_META)
                            # PRD 8: skip aliases not in agent's allowed set.
                            if (
                                agent_meta.allowed_mcp_aliases is not None
                                and cfg.alias not in agent_meta.allowed_mcp_aliases
                            ):
                                logger.debug(
                                    "MCP bridge: skipping alias %r for agent %r (not in allowed_mcp_aliases=%r)",
                                    cfg.alias,
                                    agent_cls.__name__,
                                    agent_meta.allowed_mcp_aliases,
                                )
                                continue
                            agent_meta.tools[ns_name] = (executor, tool_meta)
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

    aliases = "_".join(c.alias for c in configs) or "none"
    _McpBridge.__name__ = f"_McpBridge[{aliases}]"
    _McpBridge.__qualname__ = _McpBridge.__name__
    return _McpBridge


# ---------------------------------------------------------------------------
# Dynamic bridge (PRDs 5, 6, 8)
# ---------------------------------------------------------------------------


def _make_dynamic_mcp_bridge_class(
    configs: list[McpServerConfig],
    agents: list[type],
    *,
    signals: Any | None = None,
) -> type:
    """Return a fresh SINGLETON bridge that reacts to ``tools/list_changed``.

    Differences from :func:`_make_mcp_bridge_class`:

    - Subscribes to ``notifications/tools/list_changed`` on each client at
      startup (if the client supports ``on_list_changed``).
    - When notified, atomically diffs the new catalogue against the old one and
      applies adds/removes to all agents' tool maps.
    - Emits :class:`~lauren_ai._signals.McpToolsRefreshed` signals (PRD 6).
    - Respects ``allowed_mcp_aliases`` per-agent filtering (PRD 8).
    """
    from lauren import Scope, injectable, post_construct, pre_destruct  # noqa: PLC0415

    from lauren_ai._agents import AGENT_META  # noqa: PLC0415
    from lauren_ai._tools import ToolMeta  # noqa: PLC0415

    @injectable(scope=Scope.SINGLETON)
    class _DynamicMcpBridge:
        """Auto-generated dynamic MCP bridge singleton."""

        def __init__(self) -> None:
            self._locks: dict[str, asyncio.Lock] = {cfg.alias: asyncio.Lock() for cfg in configs}
            # name → ToolMeta per alias — used for diffing.
            self._current_tool_metas: dict[str, dict[str, Any]] = {cfg.alias: {} for cfg in configs}
            self._unsubscribes: list[Any] = []

        @post_construct
        async def _connect_all(self) -> None:
            for cfg in configs:
                try:
                    await cfg.client.connect()
                    tools = await cfg.client.list_tools()
                    await self._apply_catalogue(cfg, tools, is_initial=True)
                    logger.info(
                        "DynamicMcpBridge: loaded %d tool(s) from '%s'",
                        len(tools),
                        cfg.alias,
                    )
                    if hasattr(cfg.client, "on_list_changed"):
                        unsub = cfg.client.on_list_changed(self._make_list_changed_handler(cfg))
                        self._unsubscribes.append(unsub)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "DynamicMcpBridge: failed to connect '%s': %s",
                        cfg.alias,
                        exc,
                    )

        @pre_destruct
        async def _disconnect_all(self) -> None:
            import contextlib  # noqa: PLC0415

            for unsub in self._unsubscribes:
                with contextlib.suppress(Exception):
                    unsub()
            for cfg in configs:
                with contextlib.suppress(Exception):
                    await cfg.client.close()

        def _make_list_changed_handler(self, cfg: McpServerConfig) -> Any:
            def handler(category: str) -> None:
                if category != "tools":
                    return
                asyncio.ensure_future(self._refresh_catalogue(cfg))

            return handler

        async def _refresh_catalogue(self, cfg: McpServerConfig) -> None:
            async with self._locks[cfg.alias]:
                try:
                    tools = await cfg.client.list_tools()
                    await self._apply_catalogue(cfg, tools, is_initial=False)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "DynamicMcpBridge: failed to refresh '%s': %s",
                        cfg.alias,
                        exc,
                    )

        async def _apply_catalogue(
            self,
            cfg: McpServerConfig,
            tools: list[Any],
            *,
            is_initial: bool,
        ) -> None:
            """Diff *tools* against current catalogue and update AgentMeta."""
            new_names: set[str] = {t.name for t in tools}
            old_names: set[str] = set(self._current_tool_metas.get(cfg.alias, {}).keys())

            added_raw = new_names - old_names
            removed_raw = old_names - new_names

            added_ns: list[str] = []
            removed_ns: list[str] = []

            # Build new entries for added tools.
            new_entries: dict[str, tuple[Any, Any]] = {}
            for tool in tools:
                if tool.name in added_raw:
                    ns_name = f"{cfg.alias}__{tool.name}"
                    executor = _make_mcp_executor(cfg.client, tool.name, alias=cfg.alias, signals=signals)
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
                    new_entries[ns_name] = (executor, tool_meta)
                    added_ns.append(ns_name)
                    self._current_tool_metas[cfg.alias][tool.name] = tool_meta

            for name in removed_raw:
                removed_ns.append(f"{cfg.alias}__{name}")
                self._current_tool_metas[cfg.alias].pop(name, None)

            # Apply atomically to all agents.
            for agent_cls in agents:
                agent_meta = getattr(agent_cls, AGENT_META)
                # PRD 8: respect allowed_mcp_aliases.
                if agent_meta.allowed_mcp_aliases is not None and cfg.alias not in agent_meta.allowed_mcp_aliases:
                    continue
                for ns_name in removed_ns:
                    agent_meta.tools.pop(ns_name, None)
                for ns_name, entry in new_entries.items():
                    agent_meta.tools[ns_name] = entry

            if not is_initial and (added_ns or removed_ns) and signals is not None:
                from lauren_ai._signals import McpToolsRefreshed  # noqa: PLC0415

                await signals.emit(
                    McpToolsRefreshed(
                        alias=cfg.alias,
                        added=added_ns,
                        removed=removed_ns,
                        total=len(new_names),
                    )
                )
            if added_ns or removed_ns:
                logger.info(
                    "DynamicMcpBridge: '%s' catalogue updated — added=%r removed=%r",
                    cfg.alias,
                    added_ns,
                    removed_ns,
                )

    aliases = "_".join(c.alias for c in configs) or "none"
    _DynamicMcpBridge.__name__ = f"_DynamicMcpBridge[{aliases}]"
    _DynamicMcpBridge.__qualname__ = _DynamicMcpBridge.__name__
    return _DynamicMcpBridge

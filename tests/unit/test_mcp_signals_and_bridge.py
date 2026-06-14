"""Unit tests for PRDs 4, 5, 6, 8:
- PRD 4: ToolProgressEvent emitted from _make_mcp_executor
- PRD 5: McpToolsRefreshed emitted by _make_dynamic_mcp_bridge_class
- PRD 6: New signal types present and correct
- PRD 8: @use_mcp_servers / allowed_mcp_aliases filtering
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai import agent, use_mcp_servers
from lauren_ai._agents import AGENT_META, USE_MCP_SERVERS_META, AgentMeta
from lauren_ai._signals import McpToolsRefreshed, SignalBus, ToolProgressEvent

# ---------------------------------------------------------------------------
# PRD 6 — signal dataclasses
# ---------------------------------------------------------------------------


class TestNewSignalTypes:
    def test_tool_progress_event_defaults(self):
        ev = ToolProgressEvent()
        assert ev.tool_name == ""
        assert ev.tool_use_id == ""
        assert ev.agent_id is None
        assert ev.progress == 0.0
        assert ev.total is None
        assert ev.message is None
        assert ev.alias == ""

    def test_tool_progress_event_values(self):
        ev = ToolProgressEvent(
            tool_name="fs__read",
            tool_use_id="tid-1",
            progress=0.5,
            total=1.0,
            message="halfway",
            alias="fs",
        )
        assert ev.progress == 0.5
        assert ev.total == 1.0
        assert ev.message == "halfway"

    def test_mcp_tools_refreshed_defaults(self):
        ev = McpToolsRefreshed()
        assert ev.alias == ""
        assert ev.added == []
        assert ev.removed == []
        assert ev.total == 0

    def test_mcp_tools_refreshed_values(self):
        ev = McpToolsRefreshed(alias="fs", added=["fs__new"], removed=["fs__old"], total=5)
        assert ev.alias == "fs"
        assert "fs__new" in ev.added
        assert "fs__old" in ev.removed

    async def test_signal_bus_emits_tool_progress_event(self):
        bus = SignalBus()
        seen: list[ToolProgressEvent] = []

        @bus.on(ToolProgressEvent)
        async def handler(e: ToolProgressEvent) -> None:
            seen.append(e)

        await bus.emit(ToolProgressEvent(tool_name="x__y", progress=0.3))
        assert len(seen) == 1
        assert seen[0].progress == 0.3


# ---------------------------------------------------------------------------
# PRD 4 — _make_mcp_executor with progress
# ---------------------------------------------------------------------------


class TestMcpExecutorProgress:
    async def test_progress_events_emitted_to_bus(self):
        from lauren_ai.mcp._bridge import _make_mcp_executor

        progress_events: list[ToolProgressEvent] = []
        bus = SignalBus()

        @bus.on(ToolProgressEvent)
        async def handle(ev: ToolProgressEvent) -> None:
            progress_events.append(ev)

        # Mock client
        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": "done"}]

        # Capture the registered progress handler
        registered_handler: Any = None

        def on_progress(h: Any) -> Any:
            nonlocal registered_handler
            registered_handler = h
            return lambda: None  # unsubscribe

        client.on_progress = on_progress

        executor = _make_mcp_executor(client, "read_file", alias="fs", signals=bus)

        # Start executor in background and fire progress before awaiting
        task = asyncio.create_task(executor())

        # Wait a tick so executor registers the handler
        await asyncio.sleep(0)

        # Fire progress notification
        assert registered_handler is not None
        registered_handler({"progressToken": "", "progress": 0.5, "total": 1.0, "message": "half"})

        # Give the event loop several ticks to process ensure_future + gather
        await task
        for _ in range(5):
            await asyncio.sleep(0)

        assert len(progress_events) == 1
        assert progress_events[0].progress == 0.5
        assert progress_events[0].total == 1.0
        assert progress_events[0].alias == "fs"
        assert progress_events[0].tool_name == "fs__read_file"

    async def test_progress_event_not_emitted_without_signals(self):
        from lauren_ai.mcp._bridge import _make_mcp_executor

        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": "ok"}]
        # No on_progress registration expected when signals=None
        client.on_progress = MagicMock(side_effect=AssertionError("should not be called"))

        executor = _make_mcp_executor(client, "tool", alias="x", signals=None)
        result = await executor()
        assert result == "ok"

    async def test_cross_call_token_filtering(self):
        """Progress notifications with wrong token are ignored."""
        from lauren_ai.mcp._bridge import _make_mcp_executor

        progress_events: list[ToolProgressEvent] = []
        bus = SignalBus()

        @bus.on(ToolProgressEvent)
        async def handle(ev: ToolProgressEvent) -> None:
            progress_events.append(ev)

        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": "ok"}]
        registered: Any = None

        def on_progress(h: Any) -> Any:
            nonlocal registered
            registered = h
            return lambda: None

        client.on_progress = on_progress

        executor = _make_mcp_executor(client, "tool", alias="a", signals=bus)
        task = asyncio.create_task(executor(_tool_use_id="my-id"))
        await asyncio.sleep(0)

        # Wrong token — should be filtered out
        registered({"progressToken": "other-id", "progress": 0.9})
        await asyncio.sleep(0)
        await task

        assert len(progress_events) == 0

    async def test_tool_use_id_stripped_from_kwargs(self):
        from lauren_ai.mcp._bridge import _make_mcp_executor

        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": "x"}]

        executor = _make_mcp_executor(client, "tool")
        await executor(_tool_use_id="tid-xyz", arg1="val1")

        # _tool_use_id must NOT be forwarded to call_tool
        call_kwargs = client.call_tool.call_args[0][1]  # kwargs arg
        assert "_tool_use_id" not in call_kwargs
        assert call_kwargs.get("arg1") == "val1"

    async def test_json_content_returned_as_string(self):
        from lauren_ai.mcp._bridge import _make_mcp_executor

        client = AsyncMock()
        client.call_tool.return_value = [{"type": "image", "url": "https://example.com/img"}]

        executor = _make_mcp_executor(client, "tool")
        result = await executor()
        import json

        assert json.loads(result)[0]["type"] == "image"


# ---------------------------------------------------------------------------
# PRD 8 — @use_mcp_servers / allowed_mcp_aliases
# ---------------------------------------------------------------------------


class TestUseMcpServers:
    def test_sets_frozenset_on_class(self):
        @use_mcp_servers("filesystem", "weather")
        @agent(model="claude-sonnet-4-6")
        class AgentA:
            pass

        aliases = getattr(AgentA, USE_MCP_SERVERS_META)
        assert aliases == frozenset({"filesystem", "weather"})

    def test_sets_allowed_aliases_on_meta(self):
        @use_mcp_servers("database")
        @agent(model="claude-sonnet-4-6")
        class AgentB:
            pass

        meta: AgentMeta = getattr(AgentB, AGENT_META)
        assert meta.allowed_mcp_aliases == frozenset({"database"})

    def test_no_decorator_means_none(self):
        @agent(model="claude-sonnet-4-6")
        class AgentC:
            pass

        meta: AgentMeta = getattr(AgentC, AGENT_META)
        assert meta.allowed_mcp_aliases is None

    def test_empty_means_no_tools(self):
        @use_mcp_servers()
        @agent(model="claude-sonnet-4-6")
        class AgentD:
            pass

        meta: AgentMeta = getattr(AgentD, AGENT_META)
        assert meta.allowed_mcp_aliases == frozenset()

    def test_duplicate_alias_raises(self):
        from lauren_ai._exceptions import DecoratorUsageError

        with pytest.raises(DecoratorUsageError, match="duplicate alias"):
            use_mcp_servers("fs", "fs")

    def test_stacking_accumulates(self):
        @use_mcp_servers("weather")
        @use_mcp_servers("filesystem")
        @agent(model="claude-sonnet-4-6")
        class AgentE:
            pass

        meta: AgentMeta = getattr(AgentE, AGENT_META)
        assert meta.allowed_mcp_aliases == frozenset({"filesystem", "weather"})


class TestBridgeAliasFiltering:
    async def test_bridge_skips_disallowed_alias(self):
        from lauren_ai.mcp._bridge import McpServerConfig, _make_mcp_bridge_class

        @use_mcp_servers("weather")
        @agent(model="claude-sonnet-4-6")
        class FilteredAgent:
            pass

        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        fs_tool.description = "Read a file"
        fs_tool.inputSchema = {"type": "object"}

        weather_tool = MagicMock()
        weather_tool.name = "current"
        weather_tool.description = "Get weather"
        weather_tool.inputSchema = {"type": "object"}

        fs_client = AsyncMock()
        fs_client.list_tools.return_value = [fs_tool]

        weather_client = AsyncMock()
        weather_client.list_tools.return_value = [weather_tool]

        configs = [
            McpServerConfig(alias="filesystem", client=fs_client),
            McpServerConfig(alias="weather", client=weather_client),
        ]

        bridge_cls = _make_mcp_bridge_class(configs, [FilteredAgent])
        bridge = bridge_cls()
        await bridge._connect_all()

        meta: AgentMeta = getattr(FilteredAgent, AGENT_META)
        assert "filesystem__read_file" not in meta.tools
        assert "weather__current" in meta.tools

    async def test_bridge_allows_all_when_no_decorator(self):
        from lauren_ai.mcp._bridge import McpServerConfig, _make_mcp_bridge_class

        @agent(model="claude-sonnet-4-6")
        class UnfilteredAgent:
            pass

        tool = MagicMock()
        tool.name = "do_thing"
        tool.description = "Do it"
        tool.inputSchema = {"type": "object"}

        client = AsyncMock()
        client.list_tools.return_value = [tool]

        configs = [McpServerConfig(alias="svc", client=client)]
        bridge_cls = _make_mcp_bridge_class(configs, [UnfilteredAgent])
        bridge = bridge_cls()
        await bridge._connect_all()

        meta: AgentMeta = getattr(UnfilteredAgent, AGENT_META)
        assert "svc__do_thing" in meta.tools


# ---------------------------------------------------------------------------
# PRD 5 — _make_dynamic_mcp_bridge_class
# ---------------------------------------------------------------------------


class TestDynamicMcpBridge:
    async def test_initial_catalogue_loaded(self):
        from lauren_ai.mcp._bridge import McpServerConfig, _make_dynamic_mcp_bridge_class

        @agent(model="claude-sonnet-4-6")
        class DynAgent:
            pass

        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search"
        tool.inputSchema = {"type": "object"}

        client = AsyncMock()
        client.list_tools.return_value = [tool]

        configs = [McpServerConfig(alias="search_svc", client=client)]
        bridge_cls = _make_dynamic_mcp_bridge_class(configs, [DynAgent])
        bridge = bridge_cls()
        await bridge._connect_all()

        meta: AgentMeta = getattr(DynAgent, AGENT_META)
        assert "search_svc__search" in meta.tools

    async def test_list_changed_adds_new_tool(self):
        from lauren_ai.mcp._bridge import McpServerConfig, _make_dynamic_mcp_bridge_class

        @agent(model="claude-sonnet-4-6")
        class DynAgent2:
            pass

        tool1 = MagicMock(name="t1")
        tool1.name = "existing"
        tool1.description = "Existing"
        tool1.inputSchema = {"type": "object"}

        tool2 = MagicMock(name="t2")
        tool2.name = "new_tool"
        tool2.description = "New"
        tool2.inputSchema = {"type": "object"}

        client = AsyncMock()
        # First call returns only tool1; second returns both
        client.list_tools.side_effect = [[tool1], [tool1, tool2]]

        subscribed_handler: Any = None

        def on_list_changed(h: Any) -> Any:
            nonlocal subscribed_handler
            subscribed_handler = h
            return lambda: None

        client.on_list_changed = on_list_changed

        configs = [McpServerConfig(alias="svc", client=client)]
        bus = SignalBus()
        bridge_cls = _make_dynamic_mcp_bridge_class(configs, [DynAgent2], signals=bus)
        bridge = bridge_cls()
        await bridge._connect_all()

        meta: AgentMeta = getattr(DynAgent2, AGENT_META)
        assert "svc__existing" in meta.tools
        assert "svc__new_tool" not in meta.tools

        # Trigger list_changed
        refreshed_events: list[McpToolsRefreshed] = []

        @bus.on(McpToolsRefreshed)
        async def handle(ev: McpToolsRefreshed) -> None:
            refreshed_events.append(ev)

        subscribed_handler("tools")
        await asyncio.sleep(0.05)

        assert "svc__new_tool" in meta.tools
        assert len(refreshed_events) == 1
        assert "svc__new_tool" in refreshed_events[0].added

    async def test_list_changed_removes_old_tool(self):
        from lauren_ai.mcp._bridge import McpServerConfig, _make_dynamic_mcp_bridge_class

        @agent(model="claude-sonnet-4-6")
        class DynAgent3:
            pass

        tool1 = MagicMock()
        tool1.name = "old"
        tool1.description = "Old"
        tool1.inputSchema = {"type": "object"}

        subscribed_handler: Any = None

        def on_list_changed(h: Any) -> Any:
            nonlocal subscribed_handler
            subscribed_handler = h
            return lambda: None

        client = AsyncMock()
        # First call has tool, second removes it
        client.list_tools.side_effect = [[tool1], []]
        client.on_list_changed = on_list_changed

        configs = [McpServerConfig(alias="svc2", client=client)]
        bridge_cls = _make_dynamic_mcp_bridge_class(configs, [DynAgent3])
        bridge = bridge_cls()
        await bridge._connect_all()

        meta: AgentMeta = getattr(DynAgent3, AGENT_META)
        assert "svc2__old" in meta.tools

        subscribed_handler("tools")
        await asyncio.sleep(0.05)

        assert "svc2__old" not in meta.tools

    async def test_non_tools_category_ignored(self):
        from lauren_ai.mcp._bridge import McpServerConfig, _make_dynamic_mcp_bridge_class

        @agent(model="claude-sonnet-4-6")
        class DynAgent4:
            pass

        client = AsyncMock()
        client.list_tools.return_value = []
        call_count: list[int] = [0]

        subscribed_handler: Any = None

        def on_list_changed(h: Any) -> Any:
            nonlocal subscribed_handler
            subscribed_handler = h
            return lambda: None

        client.on_list_changed = on_list_changed

        async def counting_list_tools() -> list:
            call_count[0] += 1
            return []

        client.list_tools = counting_list_tools

        configs = [McpServerConfig(alias="s", client=client)]
        bridge_cls = _make_dynamic_mcp_bridge_class(configs, [DynAgent4])
        bridge = bridge_cls()
        await bridge._connect_all()
        initial_count = call_count[0]

        # Fire non-tools category — should not trigger refresh
        subscribed_handler("resources")
        await asyncio.sleep(0.02)

        assert call_count[0] == initial_count  # no additional list_tools call

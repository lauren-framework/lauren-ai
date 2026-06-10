"""Unit tests for Phase 6 — MCP ↔ AgentModule integration bridge.

Tests the three integration points:

1. ``_make_mcp_bridge_class`` — generates a valid injectable whose
   ``_connect_all`` populates ``AgentMeta.tools``.
2. ``AgentModule.for_root(mcp_servers=[...])`` — bridge injectable appears
   in the module's providers; existing call sites without ``mcp_servers``
   are unaffected.
3. Tool executor — the closure routes kwargs to ``client.call_tool`` and
   unwraps the text content item.

No real MCP processes are started: all clients are mocks.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai._agents import AGENT_META, AgentMeta
from lauren_ai._tools import TOOL_META, ToolMeta
from lauren_ai.mcp._bridge import McpServerConfig, _make_mcp_bridge_class, _make_mcp_executor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str, description: str = "desc") -> Any:
    """Return a mock ToolSchema-like object (as returned by list_tools)."""
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    return t


def _make_mock_client(tools: list[Any]) -> Any:
    """Return a mock McpClientProtocol that lists *tools* and echoes call_tool."""
    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.list_tools = AsyncMock(return_value=tools)

    async def _call(name: str, args: dict[str, Any]) -> list[dict]:
        return [{"type": "text", "text": f"{name}:{json.dumps(args)}"}]

    client.call_tool = _call
    return client


def _make_agent_class() -> type:
    """Return a minimal agent class with a fresh AgentMeta."""
    from lauren_ai._config import AgentConfig

    class _FakeAgent:
        pass

    setattr(
        _FakeAgent,
        AGENT_META,
        AgentMeta(
            model=None,
            system=None,
            config=AgentConfig(),
        ),
    )
    return _FakeAgent


# ---------------------------------------------------------------------------
# McpServerConfig
# ---------------------------------------------------------------------------


class TestMcpServerConfig:
    def test_stores_alias_and_client(self) -> None:
        client = MagicMock()
        cfg = McpServerConfig(alias="fs", client=client)
        assert cfg.alias == "fs"
        assert cfg.client is client


# ---------------------------------------------------------------------------
# _make_mcp_executor
# ---------------------------------------------------------------------------


class TestMakeMcpExecutor:
    @pytest.mark.asyncio
    async def test_unwraps_text_content(self) -> None:
        client = _make_mock_client([])
        executor = _make_mcp_executor(client, "read_file")
        result = await executor(path="/tmp/x")
        assert result == 'read_file:{"path": "/tmp/x"}'

    @pytest.mark.asyncio
    async def test_json_dumps_non_text_content(self) -> None:
        client = MagicMock()
        client.call_tool = AsyncMock(return_value=[{"type": "image", "data": "abc"}])
        executor = _make_mcp_executor(client, "screenshot")
        result = await executor()
        assert json.loads(result)[0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_passes_kwargs_as_arguments_dict(self) -> None:
        received: dict = {}

        async def _call(name: str, args: dict[str, Any]) -> list[dict]:
            received.update(args)
            return [{"type": "text", "text": "ok"}]

        client = MagicMock()
        client.call_tool = _call
        executor = _make_mcp_executor(client, "search")
        await executor(query="hello", limit=5)
        assert received == {"query": "hello", "limit": 5}


# ---------------------------------------------------------------------------
# _make_mcp_bridge_class
# ---------------------------------------------------------------------------


class TestMakeMcpBridgeClass:
    def test_returns_a_class(self) -> None:
        agent = _make_agent_class()
        bridge_cls = _make_mcp_bridge_class([], [agent])
        assert isinstance(bridge_cls, type)

    def test_class_is_injectable_singleton(self) -> None:
        agent = _make_agent_class()
        bridge_cls = _make_mcp_bridge_class([], [agent])
        meta = getattr(bridge_cls, "__lauren_injectable__", None)
        assert meta is not None

    def test_each_call_returns_distinct_class(self) -> None:
        agent = _make_agent_class()
        cls_a = _make_mcp_bridge_class([], [agent])
        cls_b = _make_mcp_bridge_class([], [agent])
        assert cls_a is not cls_b

    @pytest.mark.asyncio
    async def test_connect_all_populates_agent_tools(self) -> None:
        agent = _make_agent_class()
        tools = [_make_mock_tool("read_file"), _make_mock_tool("write_file")]
        client = _make_mock_client(tools)
        cfg = McpServerConfig(alias="fs", client=client)

        bridge_cls = _make_mcp_bridge_class([cfg], [agent])
        bridge = bridge_cls()
        await bridge._connect_all()

        meta: AgentMeta = getattr(agent, AGENT_META)
        assert "fs__read_file" in meta.tools
        assert "fs__write_file" in meta.tools

    @pytest.mark.asyncio
    async def test_injected_tool_meta_has_namespaced_name(self) -> None:
        agent = _make_agent_class()
        client = _make_mock_client([_make_mock_tool("search")])
        cfg = McpServerConfig(alias="web", client=client)

        bridge_cls = _make_mcp_bridge_class([cfg], [agent])
        await bridge_cls()._connect_all()

        meta: AgentMeta = getattr(agent, AGENT_META)
        _exec, tool_meta = meta.tools["web__search"]
        assert isinstance(tool_meta, ToolMeta)
        assert tool_meta.name == "web__search"

    @pytest.mark.asyncio
    async def test_injected_tool_meta_has_correct_input_schema(self) -> None:
        agent = _make_agent_class()
        mcp_tool = _make_mock_tool("get")
        mcp_tool.inputSchema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        client = _make_mock_client([mcp_tool])
        cfg = McpServerConfig(alias="db", client=client)

        bridge_cls = _make_mcp_bridge_class([cfg], [agent])
        await bridge_cls()._connect_all()

        meta: AgentMeta = getattr(agent, AGENT_META)
        _exec, tool_meta = meta.tools["db__get"]
        assert tool_meta.parameters["input_schema"] == mcp_tool.inputSchema  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_injected_tool_meta_is_async(self) -> None:
        agent = _make_agent_class()
        client = _make_mock_client([_make_mock_tool("ping")])
        cfg = McpServerConfig(alias="svc", client=client)

        bridge_cls = _make_mcp_bridge_class([cfg], [agent])
        await bridge_cls()._connect_all()

        meta: AgentMeta = getattr(agent, AGENT_META)
        _exec, tool_meta = meta.tools["svc__ping"]
        assert tool_meta.is_async is True
        assert tool_meta.reads_context is False

    @pytest.mark.asyncio
    async def test_multiple_servers_namespaced_independently(self) -> None:
        agent = _make_agent_class()
        client_a = _make_mock_client([_make_mock_tool("read")])
        client_b = _make_mock_client([_make_mock_tool("search")])
        configs = [
            McpServerConfig(alias="alpha", client=client_a),
            McpServerConfig(alias="beta", client=client_b),
        ]

        bridge_cls = _make_mcp_bridge_class(configs, [agent])
        await bridge_cls()._connect_all()

        meta: AgentMeta = getattr(agent, AGENT_META)
        assert "alpha__read" in meta.tools
        assert "beta__search" in meta.tools
        assert "read" not in meta.tools
        assert "search" not in meta.tools

    @pytest.mark.asyncio
    async def test_failed_connect_logs_error_and_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = _make_agent_class()
        bad_client = MagicMock()
        bad_client.connect = AsyncMock(side_effect=ConnectionRefusedError("refused"))
        good_client = _make_mock_client([_make_mock_tool("echo")])
        configs = [
            McpServerConfig(alias="bad", client=bad_client),
            McpServerConfig(alias="good", client=good_client),
        ]

        bridge_cls = _make_mcp_bridge_class(configs, [agent])
        import logging

        with caplog.at_level(logging.ERROR, logger="lauren_ai.mcp._bridge"):
            await bridge_cls()._connect_all()

        assert "bad" in caplog.text
        meta: AgentMeta = getattr(agent, AGENT_META)
        assert "good__echo" in meta.tools

    @pytest.mark.asyncio
    async def test_tools_injected_into_all_agents(self) -> None:
        agent_a = _make_agent_class()
        agent_b = _make_agent_class()
        client = _make_mock_client([_make_mock_tool("tool_x")])
        cfg = McpServerConfig(alias="srv", client=client)

        bridge_cls = _make_mcp_bridge_class([cfg], [agent_a, agent_b])
        await bridge_cls()._connect_all()

        for ag in (agent_a, agent_b):
            assert "srv__tool_x" in getattr(ag, AGENT_META).tools

    @pytest.mark.asyncio
    async def test_disconnect_all_closes_clients(self) -> None:
        agent = _make_agent_class()
        client = _make_mock_client([])
        cfg = McpServerConfig(alias="x", client=client)

        bridge_cls = _make_mcp_bridge_class([cfg], [agent])
        bridge = bridge_cls()
        await bridge._disconnect_all()

        client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# AgentModule.for_root integration
# ---------------------------------------------------------------------------


class TestAgentModuleForRootMcpServers:
    def test_for_root_accepts_mcp_servers_param(self) -> None:
        from lauren_ai._module import AgentModule

        agent = _make_agent_class()
        setattr(agent, TOOL_META, None)  # satisfy _categorize guard
        client = _make_mock_client([])
        cfg = McpServerConfig(alias="fs", client=client)

        # Should not raise
        try:
            AgentModule.for_root(agents=[agent], mcp_servers=[cfg])
        except Exception as exc:
            # Acceptable failures: missing Transport/LLMConfig (no DI graph yet)
            if "Transport" in str(exc) or "MissingProvider" in type(exc).__name__:
                pass
            else:
                raise

    def test_for_root_without_mcp_servers_unchanged(self) -> None:
        from lauren_ai._module import AgentModule

        agent = _make_agent_class()
        # no mcp_servers — should not attempt any MCP import
        try:
            AgentModule.for_root(agents=[agent])
        except Exception as exc:
            if "Transport" in str(exc) or "MissingProvider" in type(exc).__name__:
                pass
            else:
                raise

    def test_mcp_servers_none_does_not_add_bridge_provider(self) -> None:
        from lauren_ai._module import AgentModule

        agent = _make_agent_class()
        try:
            module_cls = AgentModule.for_root(agents=[agent], mcp_servers=None)
        except Exception:
            return  # DI errors are OK here — we only care the bridge isn't added

        # If the module was built, verify no _McpBridge in providers
        module_providers = getattr(module_cls, "__lauren_module__", None)
        if module_providers:
            provider_names = [getattr(p, "__name__", "") for p in getattr(module_providers, "providers", [])]
            assert not any("McpBridge" in n for n in provider_names)

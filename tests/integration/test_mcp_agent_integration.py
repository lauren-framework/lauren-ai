"""Integration tests: MCP bridge + AgentRunnerBase.

Every agent class is defined inside its test method so each test gets a
fresh ``AgentMeta`` with no cross-test state.  All MCP clients are mocks
— no subprocess is started.

Coverage:
  - Bridge injects tools into AgentMeta and runner schema
  - Runner executes MCP tool via bridge executor
  - Tool result text is forwarded to the next LLM call
  - Multi-turn: tool call → result → final response
  - Multiple MCP servers coexist under different aliases
  - Tool namespacing: same bare tool name, different aliases → different entries
  - Failed connect is logged, agent still starts without those tools
  - Executor passes kwargs dict to client.call_tool
  - Non-text tool content is JSON-serialised
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.mcp._bridge import McpServerConfig, _make_mcp_bridge_class

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(text: str = "OK", *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock: MockTransport | None = None) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    return AgentRunner(transport=mock), mock


def _mcp_tool(name: str, *, props: dict | None = None, required: list | None = None) -> Any:
    """Return a mock McpToolSchema-like object."""
    t = MagicMock()
    t.name = name
    t.description = f"Tool {name}."
    t.inputSchema = {
        "type": "object",
        "properties": props or {"text": {"type": "string"}},
        "required": required if required is not None else list(props or {"text": None}),
    }
    return t


def _mcp_client(*tool_names: str) -> Any:
    """Return a mock client that exposes the named tools and echoes call_tool."""
    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.list_tools = AsyncMock(return_value=[_mcp_tool(n) for n in tool_names])

    async def _call(name: str, args: dict) -> list[dict]:
        return [{"type": "text", "text": f"{name}:{json.dumps(args, sort_keys=True)}"}]

    client.call_tool = _call
    return client


# ---------------------------------------------------------------------------
# Schema injection
# ---------------------------------------------------------------------------


class TestBridgeInjectsSchemas:
    @pytest.mark.asyncio
    async def test_tool_appears_in_agent_meta_tools(self):
        @agent(model="mock-model")
        class A: ...

        client = _mcp_client("echo")
        bridge = _make_mcp_bridge_class([McpServerConfig("srv", client)], [A])()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        meta = getattr(A, AGENT_META)
        assert "srv__echo" in meta.tools

    @pytest.mark.asyncio
    async def test_tool_schema_has_correct_name_and_input_schema(self):
        @agent(model="mock-model")
        class A: ...

        tool_mock = _mcp_tool("search", props={"query": {"type": "string"}})
        client = MagicMock()
        client.connect = AsyncMock()
        client.close = AsyncMock()
        client.list_tools = AsyncMock(return_value=[tool_mock])
        client.call_tool = AsyncMock(return_value=[{"type": "text", "text": ""}])

        bridge = _make_mcp_bridge_class([McpServerConfig("db", client)], [A])()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        _exec, meta = getattr(A, AGENT_META).tools["db__search"]
        assert meta.name == "db__search"
        assert meta.parameters["name"] == "db__search"  # type: ignore[index]
        assert meta.parameters["input_schema"] == tool_mock.inputSchema  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_multiple_tools_from_one_server(self):
        @agent(model="mock-model")
        class A: ...

        client = _mcp_client("read_file", "write_file", "list_dir")
        bridge = _make_mcp_bridge_class([McpServerConfig("fs", client)], [A])()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        tools = getattr(A, AGENT_META).tools
        assert "fs__read_file" in tools
        assert "fs__write_file" in tools
        assert "fs__list_dir" in tools

    @pytest.mark.asyncio
    async def test_runner_schema_list_includes_mcp_tool(self):
        @agent(model="mock-model")
        class A: ...

        client = _mcp_client("greet")
        bridge = _make_mcp_bridge_class([McpServerConfig("hello", client)], [A])()
        await bridge._connect_all()

        runner, mock = _make_runner()
        mock.queue_response(_completion("Done"))
        await runner.run(A(), "say hello")

        assert mock.calls, "No LLM calls recorded"
        tool_schemas = mock.calls[0].tools or []
        names = [s.get("name") for s in tool_schemas]
        assert "hello__greet" in names


# ---------------------------------------------------------------------------
# Tool execution via runner
# ---------------------------------------------------------------------------


class TestRunnerExecutesMcpTool:
    @pytest.mark.asyncio
    async def test_tool_call_routed_to_mcp_executor(self):
        @agent(model="mock-model")
        class A: ...

        called_with: list = []

        async def _call(name: str, args: dict) -> list[dict]:
            called_with.append((name, args))
            return [{"type": "text", "text": "echo_result"}]

        client = MagicMock()
        client.connect = AsyncMock()
        client.close = AsyncMock()
        client.list_tools = AsyncMock(return_value=[_mcp_tool("echo")])
        client.call_tool = _call

        bridge = _make_mcp_bridge_class([McpServerConfig("srv", client)], [A])()
        await bridge._connect_all()

        runner, mock = _make_runner()
        mock.queue_tool_use("srv__echo", {"text": "hello"})
        mock.queue_response(_completion("agent says: echo_result"))
        await runner.run(A(), "please echo")

        assert called_with == [("echo", {"text": "hello"})]

    @pytest.mark.asyncio
    async def test_tool_result_text_forwarded_to_llm(self):
        @agent(model="mock-model")
        class A: ...

        async def _call(name: str, args: dict) -> list[dict]:
            return [{"type": "text", "text": "TOOL_OUTPUT_42"}]

        client = MagicMock()
        client.connect = AsyncMock()
        client.close = AsyncMock()
        client.list_tools = AsyncMock(return_value=[_mcp_tool("compute")])
        client.call_tool = _call

        bridge = _make_mcp_bridge_class([McpServerConfig("calc", client)], [A])()
        await bridge._connect_all()

        runner, mock = _make_runner()
        mock.queue_tool_use("calc__compute", {"text": "x"})
        mock.queue_response(_completion("Final"))
        await runner.run(A(), "compute x")

        # Second call should carry the tool result in conversation history.
        # Tool results arrive as role="user" messages with a "tool_result" content block.
        assert len(mock.calls) == 2
        second_messages = mock.calls[1].messages
        tool_result_texts: list[str] = []
        for m in second_messages:
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_texts.append(str(block.get("content", "")))
        assert any("TOOL_OUTPUT_42" in t for t in tool_result_texts)

    @pytest.mark.asyncio
    async def test_final_response_returned_after_tool_call(self):
        @agent(model="mock-model")
        class A: ...

        client = _mcp_client("ping")
        bridge = _make_mcp_bridge_class([McpServerConfig("svc", client)], [A])()
        await bridge._connect_all()

        runner, mock = _make_runner()
        mock.queue_tool_use("svc__ping", {})
        mock.queue_response(_completion("pong received"))

        resp = await runner.run(A(), "ping please")
        assert resp.content == "pong received"

    @pytest.mark.asyncio
    async def test_non_text_content_is_json_serialised(self):
        @agent(model="mock-model")
        class A: ...

        async def _call(name: str, args: dict) -> list[dict]:
            return [{"type": "image", "data": "abc123", "mimeType": "image/png"}]

        client = MagicMock()
        client.connect = AsyncMock()
        client.close = AsyncMock()
        client.list_tools = AsyncMock(return_value=[_mcp_tool("screenshot")])
        client.call_tool = _call

        bridge = _make_mcp_bridge_class([McpServerConfig("ui", client)], [A])()
        await bridge._connect_all()

        runner, mock = _make_runner()
        mock.queue_tool_use("ui__screenshot", {})
        mock.queue_response(_completion("seen image"))

        resp = await runner.run(A(), "take screenshot")
        assert resp.content == "seen image"
        # Tool result in history should be JSON-serialised image content.
        second_messages = mock.calls[1].messages
        tool_result_texts: list[str] = []
        for m in second_messages:
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_texts.append(str(block.get("content", "")))
        assert any("image" in t for t in tool_result_texts)

    @pytest.mark.asyncio
    async def test_kwargs_passed_correctly_to_call_tool(self):
        @agent(model="mock-model")
        class A: ...

        received: dict = {}

        async def _call(name: str, args: dict) -> list[dict]:
            received.update(args)
            return [{"type": "text", "text": "ok"}]

        tool = _mcp_tool("query", props={"sql": {"type": "string"}, "limit": {"type": "integer"}})
        client = MagicMock()
        client.connect = AsyncMock()
        client.close = AsyncMock()
        client.list_tools = AsyncMock(return_value=[tool])
        client.call_tool = _call

        bridge = _make_mcp_bridge_class([McpServerConfig("db", client)], [A])()
        await bridge._connect_all()

        runner, mock = _make_runner()
        mock.queue_tool_use("db__query", {"sql": "SELECT 1", "limit": 10})
        mock.queue_response(_completion("done"))
        await runner.run(A(), "run query")

        assert received == {"sql": "SELECT 1", "limit": 10}


# ---------------------------------------------------------------------------
# Multi-server scenarios
# ---------------------------------------------------------------------------


class TestMultipleServerCoexistence:
    @pytest.mark.asyncio
    async def test_tools_from_two_servers_both_available(self):
        @agent(model="mock-model")
        class A: ...

        client_a = _mcp_client("read")
        client_b = _mcp_client("search")
        configs = [McpServerConfig("fs", client_a), McpServerConfig("web", client_b)]
        bridge = _make_mcp_bridge_class(configs, [A])()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        tools = getattr(A, AGENT_META).tools
        assert "fs__read" in tools
        assert "web__search" in tools

    @pytest.mark.asyncio
    async def test_same_bare_name_from_two_servers_both_kept(self):
        @agent(model="mock-model")
        class A: ...

        client_a = _mcp_client("get")
        client_b = _mcp_client("get")
        configs = [McpServerConfig("alpha", client_a), McpServerConfig("beta", client_b)]
        bridge = _make_mcp_bridge_class(configs, [A])()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        tools = getattr(A, AGENT_META).tools
        assert "alpha__get" in tools
        assert "beta__get" in tools
        assert "get" not in tools

    @pytest.mark.asyncio
    async def test_tool_call_routes_to_correct_server(self):
        @agent(model="mock-model")
        class A: ...

        calls_a: list = []
        calls_b: list = []

        async def call_a(name: str, args: dict) -> list[dict]:
            calls_a.append(args)
            return [{"type": "text", "text": "from_a"}]

        async def call_b(name: str, args: dict) -> list[dict]:
            calls_b.append(args)
            return [{"type": "text", "text": "from_b"}]

        client_a, client_b = MagicMock(), MagicMock()
        for c, fn in [(client_a, call_a), (client_b, call_b)]:
            c.connect = AsyncMock()
            c.close = AsyncMock()
            c.list_tools = AsyncMock(return_value=[_mcp_tool("op")])
            c.call_tool = fn

        bridge = _make_mcp_bridge_class([McpServerConfig("srv_a", client_a), McpServerConfig("srv_b", client_b)], [A])()
        await bridge._connect_all()

        runner, mock = _make_runner()
        mock.queue_tool_use("srv_b__op", {"text": "x"})
        mock.queue_response(_completion("ok"))
        await runner.run(A(), "run op on srv_b")

        assert calls_b and not calls_a

    @pytest.mark.asyncio
    async def test_tools_injected_into_all_agents_in_bridge(self):
        @agent(model="mock-model")
        class Ag1: ...

        @agent(model="mock-model")
        class Ag2: ...

        client = _mcp_client("tool_x")
        bridge = _make_mcp_bridge_class([McpServerConfig("srv", client)], [Ag1, Ag2])()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        assert "srv__tool_x" in getattr(Ag1, AGENT_META).tools
        assert "srv__tool_x" in getattr(Ag2, AGENT_META).tools


# ---------------------------------------------------------------------------
# Failure resilience
# ---------------------------------------------------------------------------


class TestBridgeFailureResilience:
    @pytest.mark.asyncio
    async def test_failed_connect_does_not_raise(self):
        @agent(model="mock-model")
        class A: ...

        bad_client = MagicMock()
        bad_client.connect = AsyncMock(side_effect=ConnectionRefusedError("refused"))

        bridge = _make_mcp_bridge_class([McpServerConfig("bad", bad_client)], [A])()
        # Must not raise
        await bridge._connect_all()

    @pytest.mark.asyncio
    async def test_healthy_server_loads_after_failed_server(self):
        @agent(model="mock-model")
        class A: ...

        bad_client = MagicMock()
        bad_client.connect = AsyncMock(side_effect=OSError("gone"))
        good_client = _mcp_client("ok_tool")

        bridge = _make_mcp_bridge_class(
            [McpServerConfig("bad", bad_client), McpServerConfig("good", good_client)], [A]
        )()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        assert "good__ok_tool" in getattr(A, AGENT_META).tools

    @pytest.mark.asyncio
    async def test_failed_server_tools_absent_from_agent(self):
        @agent(model="mock-model")
        class A: ...

        bad_client = MagicMock()
        bad_client.connect = AsyncMock(side_effect=TimeoutError)

        bridge = _make_mcp_bridge_class([McpServerConfig("bad", bad_client)], [A])()
        await bridge._connect_all()

        from lauren_ai._agents import AGENT_META

        tools = getattr(A, AGENT_META).tools
        assert not any(k.startswith("bad__") for k in tools)

    @pytest.mark.asyncio
    async def test_disconnect_called_on_all_clients(self):
        @agent(model="mock-model")
        class A: ...

        clients = [_mcp_client("t") for _ in range(3)]
        configs = [McpServerConfig(f"s{i}", c) for i, c in enumerate(clients)]
        bridge = _make_mcp_bridge_class(configs, [A])()
        await bridge._connect_all()
        await bridge._disconnect_all()

        for c in clients:
            c.close.assert_awaited_once()

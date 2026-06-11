"""End-to-end tests: AgentRunnerBase with a real stdio MCP subprocess.

An inline Python script serves as the MCP server over stdin/stdout.  The
LLM is provided by ``MockTransport`` so no API keys or network are needed.

Coverage:
  - Agent calls a real MCP tool via stdio subprocess → correct result
  - Two sequential MCP tool calls in one run
  - Agent with both a native @tool() and an MCP tool — both resolvable
  - AgentRunner receives tool result from subprocess and produces final response
  - Bridge disconnect shuts the subprocess down cleanly
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import textwrap

import pytest
from lauren_mcp import McpServer

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.mcp._bridge import McpServerConfig, _make_mcp_bridge_class

# ---------------------------------------------------------------------------
# Echo MCP server script (inline Python subprocess)
# ---------------------------------------------------------------------------

_ECHO_SERVER = textwrap.dedent("""\
    import sys, json

    def respond(id_, result):
        print(json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method = msg.get("method")
        id_ = msg.get("id")
        if method == "initialize":
            respond(id_, {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo", "version": "1.0.0"},
            })
        elif method == "tools/list":
            respond(id_, {"tools": [
                {"name": "echo", "description": "Echo text back.",
                 "inputSchema": {"type": "object",
                                 "properties": {"text": {"type": "string"}},
                                 "required": ["text"]}},
                {"name": "upper", "description": "Uppercase text.",
                 "inputSchema": {"type": "object",
                                 "properties": {"text": {"type": "string"}},
                                 "required": ["text"]}},
            ]})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            text = args.get("text", "")
            if name == "upper":
                respond(id_, {"content": [{"type": "text", "text": text.upper()}], "isError": False})
            else:
                respond(id_, {"content": [{"type": "text", "text": text}], "isError": False})
        elif method == "ping":
            respond(id_, {})
        elif method in ("resources/list", "prompts/list"):
            respond(id_, {"resources": [], "prompts": []}.get(method.split("/")[0] + "s", {}))
        elif method == "resources/list":
            respond(id_, {"resources": []})
        elif method == "prompts/list":
            respond(id_, {"prompts": []})
        sys.stdout.flush()
""")

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


@pytest.fixture
def echo_server_argv():
    """Write the echo server to a temp file and return its argv."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_ECHO_SERVER)
        path = f.name
    yield [sys.executable, path]
    os.unlink(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentRunnerWithRealMcpSubprocess:
    @pytest.mark.asyncio
    async def test_agent_calls_echo_tool_and_receives_text(self, echo_server_argv):
        @agent(model="mock-model")
        class A: ...

        client = McpServer.stdio(echo_server_argv, startup_timeout=10.0)
        cfg = McpServerConfig(alias="echo_srv", client=client)
        bridge = _make_mcp_bridge_class([cfg], [A])()
        await bridge._connect_all()

        mock = MockTransport()
        mock.queue_tool_use("echo_srv__echo", {"text": "hello from agent"})
        mock.queue_response(_completion("I echoed: hello from agent"))

        runner = AgentRunner(transport=mock)
        resp = await asyncio.wait_for(runner.run(A(), "echo hello from agent"), timeout=15.0)
        assert resp.content == "I echoed: hello from agent"
        await bridge._disconnect_all()

    @pytest.mark.asyncio
    async def test_tool_result_text_matches_subprocess_output(self, echo_server_argv):
        @agent(model="mock-model")
        class A: ...

        client = McpServer.stdio(echo_server_argv, startup_timeout=10.0)
        cfg = McpServerConfig(alias="srv", client=client)
        bridge = _make_mcp_bridge_class([cfg], [A])()
        await bridge._connect_all()

        tool_results: list[str] = []

        mock = MockTransport()
        mock.queue_tool_use("srv__echo", {"text": "PING"})
        mock.queue_response(_completion("Done"))

        runner = AgentRunner(transport=mock)
        await asyncio.wait_for(runner.run(A(), "ping"), timeout=15.0)

        # Tool results arrive as role="user" with a "tool_result" content block.
        assert len(mock.calls) >= 2
        second_messages = mock.calls[1].messages
        for m in second_messages:
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_results.append(str(block.get("content", "")))
        assert any("PING" in t for t in tool_results), "Tool result 'PING' not found in second LLM call messages"
        await bridge._disconnect_all()

    @pytest.mark.asyncio
    async def test_two_sequential_mcp_tool_calls(self, echo_server_argv):
        @agent(model="mock-model")
        class A: ...

        client = McpServer.stdio(echo_server_argv, startup_timeout=10.0)
        bridge = _make_mcp_bridge_class([McpServerConfig("s", client)], [A])()
        await bridge._connect_all()

        mock = MockTransport()
        mock.queue_tool_use("s__echo", {"text": "first"})
        mock.queue_tool_use("s__upper", {"text": "second"})
        mock.queue_response(_completion("both done"))

        runner = AgentRunner(transport=mock)
        resp = await asyncio.wait_for(runner.run(A(), "two calls"), timeout=15.0)
        assert resp.content == "both done"
        await bridge._disconnect_all()

    @pytest.mark.asyncio
    async def test_agent_with_native_and_mcp_tools(self, echo_server_argv):
        @tool()
        async def native_greet(name: str) -> str:
            """Greet someone by name.

            Args:
                name: Person's name.
            """
            return f"Hello, {name}!"

        @agent(model="mock-model")
        class A: ...

        # Inject native tool
        from lauren_ai._agents import AGENT_META
        from lauren_ai._tools import _add_to_tool_map

        tools: dict = {}
        _add_to_tool_map(tools, native_greet)
        getattr(A, AGENT_META).tools.update(tools)

        # Inject MCP tools via bridge
        client = McpServer.stdio(echo_server_argv, startup_timeout=10.0)
        bridge = _make_mcp_bridge_class([McpServerConfig("mcp", client)], [A])()
        await bridge._connect_all()

        meta = getattr(A, AGENT_META)
        assert "native_greet" in meta.tools
        assert "mcp__echo" in meta.tools

        mock = MockTransport()
        mock.queue_response(_completion("both present"))
        runner = AgentRunner(transport=mock)
        resp = await asyncio.wait_for(runner.run(A(), "list tools"), timeout=15.0)
        assert resp.content == "both present"

        # Schema should include both
        schemas = mock.calls[0].tools or []
        names = {s.get("name") for s in schemas}
        assert "native_greet" in names
        assert "mcp__echo" in names
        await bridge._disconnect_all()

    @pytest.mark.asyncio
    async def test_upper_tool_converts_text(self, echo_server_argv):
        @agent(model="mock-model")
        class A: ...

        client = McpServer.stdio(echo_server_argv, startup_timeout=10.0)
        bridge = _make_mcp_bridge_class([McpServerConfig("srv", client)], [A])()
        await bridge._connect_all()

        mock = MockTransport()
        mock.queue_tool_use("srv__upper", {"text": "hello"})
        mock.queue_response(_completion("uppercased"))

        runner = AgentRunner(transport=mock)
        await asyncio.wait_for(runner.run(A(), "uppercase hello"), timeout=15.0)

        # Tool result should contain the uppercased text "HELLO".
        second_messages = mock.calls[1].messages
        uppercase_found = False
        for m in second_messages:
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and "HELLO" in str(block.get("content", ""))
                    ):
                        uppercase_found = True
        assert uppercase_found, "Expected HELLO in tool result messages"
        await bridge._disconnect_all()

    @pytest.mark.asyncio
    async def test_bridge_disconnect_closes_subprocess(self, echo_server_argv):
        @agent(model="mock-model")
        class A: ...

        client = McpServer.stdio(echo_server_argv, startup_timeout=10.0)
        bridge = _make_mcp_bridge_class([McpServerConfig("srv", client)], [A])()
        await bridge._connect_all()
        await bridge._disconnect_all()

        # After disconnect, calling list_tools should fail
        from lauren_mcp import McpCallError  # noqa: PLC0415

        with pytest.raises(McpCallError):
            await asyncio.wait_for(client.list_tools(), timeout=3.0)

"""Unit tests for PRDs 7, 10, 11, 12 and MCP prompt/resource/sampling PRDs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai import agent
from lauren_ai._tools import ToolContextAdapter, UnifiedToolContext
from lauren_ai.mcp import (
    AgentMcpServer,
    AgentSamplingHandler,
    KnowledgeChunk,
    McpConversationStore,
    McpPromptTemplate,
    McpResourceKnowledgeSource,
    McpUserMemoryStore,
)

# ---------------------------------------------------------------------------
# PRD 12 — UnifiedToolContext protocol + ToolContextAdapter
# ---------------------------------------------------------------------------


class TestUnifiedToolContext:
    def test_protocol_has_expected_fields(self):
        assert hasattr(UnifiedToolContext, "__protocol_attrs__") or hasattr(UnifiedToolContext, "_is_protocol")

    def test_tool_context_adapter_wraps_mcp_ctx(self):
        mcp_ctx = MagicMock()
        mcp_ctx.tool_name = "fs__read"
        mcp_ctx.metadata = {"required_role": "admin"}
        mcp_ctx.state = {}
        mcp_ctx.get_metadata = lambda key, default=None: mcp_ctx.metadata.get(key, default)
        mcp_ctx.tool_use_id = "tid-1"
        mcp_ctx.session_id = "sess-abc"
        mcp_ctx.extras = {"turn": 3, "agent_name": "ResearchAgent"}
        mcp_ctx.lifespan_context = {"db": "handle"}
        mcp_ctx.execution_context = None

        adapter = ToolContextAdapter(mcp_ctx)

        assert adapter.tool_name == "fs__read"
        assert adapter.metadata == {"required_role": "admin"}
        assert adapter.get_metadata("required_role") == "admin"
        assert adapter.get_metadata("missing", "default") == "default"
        assert adapter.tool_use_id == "tid-1"
        assert adapter.session_id == "sess-abc"
        assert adapter.turn == 3
        assert adapter.calling_agent == "ResearchAgent"  # from extras["agent_name"]

    def test_adapter_satisfies_unified_protocol(self):
        mcp_ctx = MagicMock()
        mcp_ctx.tool_name = "t"
        mcp_ctx.metadata = {}
        mcp_ctx.state = {}
        mcp_ctx.get_metadata = lambda k, d=None: d
        mcp_ctx.extras = {}

        adapter = ToolContextAdapter(mcp_ctx)
        assert isinstance(adapter, UnifiedToolContext)

    async def test_adapter_report_progress_delegates(self):
        mcp_ctx = MagicMock()
        mcp_ctx.tool_name = "t"
        mcp_ctx.metadata = {}
        mcp_ctx.state = {}
        mcp_ctx.get_metadata = lambda k, d=None: d
        mcp_ctx.extras = {}
        mcp_ctx.report_progress = AsyncMock()

        adapter = ToolContextAdapter(mcp_ctx)
        await adapter.report_progress(0.5, 1.0, "halfway")
        mcp_ctx.report_progress.assert_awaited_once_with(0.5, 1.0, "halfway")


# ---------------------------------------------------------------------------
# PRD 7 — AgentMcpServer
# ---------------------------------------------------------------------------


class TestAgentMcpServer:
    def test_requires_agent_class(self):
        class NotAnAgent:
            pass

        with pytest.raises(ValueError, match="not decorated with @agent"):
            AgentMcpServer(NotAnAgent)

    def test_accepts_valid_agent(self):
        @agent(model="claude-sonnet-4-6")
        class ValidAgent:
            pass

        server = AgentMcpServer(ValidAgent, path="/va", transport="ws")
        assert server._path == "/va"
        assert server._transport == "ws"

    def test_build_server_class_returns_mcp_server(self):
        from lauren_mcp.server._meta import MCP_SERVER_META

        @agent(model="claude-sonnet-4-6", system="You are helpful.")
        class MyAgent:
            pass

        server = AgentMcpServer(MyAgent, path="/my-agent")
        cls = server.build_server_class()
        assert hasattr(cls, MCP_SERVER_META)


# ---------------------------------------------------------------------------
# PRD — McpPromptTemplate
# ---------------------------------------------------------------------------


class TestMcpPromptTemplate:
    def test_name_property(self):
        client = MagicMock()
        tmpl = McpPromptTemplate(client, "greeting", alias="myserver")
        assert tmpl.name == "mcp:myserver:greeting"
        assert tmpl.prompt_name == "greeting"

    async def test_render_single_user_message_returns_string(self):
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.text = "Hello, world!"
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "greeting")
        result = await tmpl.render()
        assert result == "Hello, world!"

    async def test_render_multi_message_returns_list(self):
        client = AsyncMock()
        mock_result = MagicMock()
        msg1 = MagicMock()
        msg1.role = "system"
        msg1.content = MagicMock()
        msg1.content.text = "Be helpful."
        msg2 = MagicMock()
        msg2.role = "user"
        msg2.content = MagicMock()
        msg2.content.text = "Hello?"
        mock_result.messages = [msg1, msg2]
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "chat_start")
        result = await tmpl.render()
        assert isinstance(result, list)
        assert len(result) == 2

    async def test_argument_names_fetches_from_list(self):
        client = AsyncMock()
        schema = MagicMock()
        schema.name = "my_prompt"
        arg1 = MagicMock()
        arg1.name = "topic"
        arg2 = MagicMock()
        arg2.name = "language"
        schema.arguments = [arg1, arg2]
        client.list_prompts.return_value = [schema]

        tmpl = McpPromptTemplate(client, "my_prompt")
        names = await tmpl.argument_names()
        assert names == ["topic", "language"]

    async def test_callable_interface(self):
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.text = "Callable result"
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "p")
        result = await tmpl()
        assert result == "Callable result"


# ---------------------------------------------------------------------------
# PRD — McpResourceKnowledgeSource
# ---------------------------------------------------------------------------


class TestMcpResourceKnowledgeSource:
    def test_tool_name_default(self):
        client = MagicMock()
        src = McpResourceKnowledgeSource(client, "docs")
        assert src.tool_name == "search_docs"
        assert src.name == "mcp:docs"

    def test_tool_name_override(self):
        client = MagicMock()
        src = McpResourceKnowledgeSource(client, "docs", tool_name="search_all_docs")
        assert src.tool_name == "search_all_docs"

    async def test_search_returns_chunks(self):
        client = AsyncMock()
        resource = MagicMock()
        resource.uri = "file:///docs/faq.md"
        resource.name = "FAQ"
        resource.mimeType = "text/markdown"
        client.list_resources.return_value = [resource]

        mock_result = MagicMock()
        content_item = MagicMock()
        content_item.text = "Frequently asked questions"
        mock_result.contents = [content_item]
        client.read_resource.return_value = mock_result

        src = McpResourceKnowledgeSource(client, "docs")
        chunks = await src.search("faq", k=5)

        assert len(chunks) == 1
        assert chunks[0].content == "Frequently asked questions"
        assert chunks[0].source == "file:///docs/faq.md"

    async def test_filter_fn_applied(self):
        client = AsyncMock()
        r1 = MagicMock()
        r1.uri = "file:///a.md"
        r1.name = "A"
        r1.mimeType = "text/markdown"
        r2 = MagicMock()
        r2.uri = "file:///b.pdf"
        r2.name = "B"
        r2.mimeType = "application/pdf"

        client.list_resources.return_value = [r1, r2]

        content_item = MagicMock()
        content_item.text = "content"
        mock_result = MagicMock()
        mock_result.contents = [content_item]
        client.read_resource.return_value = mock_result

        # Only allow .md files
        src = McpResourceKnowledgeSource(client, "docs", filter_fn=lambda r: r.uri.endswith(".md"))
        chunks = await src.search("q", k=10)
        assert len(chunks) == 1
        assert chunks[0].source == "file:///a.md"

    def test_knowledge_chunk_dataclass(self):
        chunk = KnowledgeChunk(content="hello", source="file:///x.txt", score=0.9)
        assert chunk.content == "hello"
        assert chunk.score == 0.9


# ---------------------------------------------------------------------------
# PRD — McpConversationStore
# ---------------------------------------------------------------------------


class TestMcpConversationStore:
    async def test_load_returns_empty_on_not_found(self):
        client = AsyncMock()
        from lauren_mcp._client._stdio import McpCallError

        exc = McpCallError("not found", code=-32002)
        client.read_resource.side_effect = exc

        store = McpConversationStore(client)
        result = await store.load("missing-conv")
        assert result == []

    async def test_load_returns_parsed_json(self):
        import json

        client = AsyncMock()
        content_item = MagicMock()
        content_item.text = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
        mock_result = MagicMock()
        mock_result.contents = [content_item]
        client.read_resource.return_value = mock_result

        store = McpConversationStore(client)
        result = await store.load("conv-1")
        assert isinstance(result, dict)
        assert result["messages"][0]["role"] == "user"

    async def test_save_calls_mcp_tool(self):
        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": "ok"}]

        store = McpConversationStore(client)
        await store.save("conv-1", {"messages": [], "summary": None})

        client.call_tool.assert_awaited_once()
        call_args = client.call_tool.call_args[0]
        assert call_args[0] == "save_conversation"
        assert "conv-1" in call_args[1]["conversation_id"]

    async def test_delete_calls_mcp_tool(self):
        client = AsyncMock()
        store = McpConversationStore(client)
        await store.delete("conv-1")
        client.call_tool.assert_awaited_once()
        assert client.call_tool.call_args[0][0] == "delete_conversation"


# ---------------------------------------------------------------------------
# PRD — McpUserMemoryStore
# ---------------------------------------------------------------------------


class TestMcpUserMemoryStore:
    async def test_save_calls_tool(self):
        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": "ok"}]

        store = McpUserMemoryStore(client)
        await store.save("user-1", "language", "Python")

        client.call_tool.assert_awaited_once()
        args = client.call_tool.call_args[0]
        assert args[0] == "save_user_fact"
        assert args[1]["key"] == "language"

    async def test_get_all_returns_facts(self):
        import json

        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": json.dumps([{"key": "lang", "value": "Python"}])}]

        store = McpUserMemoryStore(client)
        facts = await store.get_all("user-1")

        assert len(facts) == 1
        assert facts[0].key == "lang"
        assert facts[0].value == "Python"

    async def test_delete_calls_tool(self):
        client = AsyncMock()
        store = McpUserMemoryStore(client)
        await store.delete("user-1", "language")
        client.call_tool.assert_awaited_once()
        assert client.call_tool.call_args[0][0] == "delete_user_fact"


# ---------------------------------------------------------------------------
# PRD — AgentSamplingHandler
# ---------------------------------------------------------------------------


class TestAgentSamplingHandler:
    async def test_basic_sampling_call(self):
        @agent(model="claude-haiku-4-5")
        class SamplingAgent:
            pass

        mock_runner = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Sampled answer"
        mock_response.stop_reason = "end_turn"
        mock_runner.run = AsyncMock(return_value=mock_response)

        handler = AgentSamplingHandler(mock_runner, SamplingAgent)
        result = await handler(
            {
                "messages": [{"role": "user", "content": {"type": "text", "text": "What is 2+2?"}}],
                "maxTokens": 100,
            }
        )

        assert result["role"] == "assistant"
        assert result["content"]["text"] == "Sampled answer"
        assert result["stopReason"] == "endTurn"

    async def test_sampling_error_returns_error_dict(self):
        @agent(model="claude-haiku-4-5")
        class ErrorAgent:
            pass

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=RuntimeError("LLM error"))

        handler = AgentSamplingHandler(mock_runner, ErrorAgent)
        result = await handler({"messages": [], "maxTokens": 10})

        assert result["role"] == "assistant"
        assert "error" in result["content"]["text"].lower() or "LLM error" in result["content"]["text"]
        assert result["stopReason"] == "error"

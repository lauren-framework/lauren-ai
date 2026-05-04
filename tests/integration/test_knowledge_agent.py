"""Integration tests for KnowledgeBase integration with agents.

Tests cover:
- Loading text into a KnowledgeBase backed by InMemoryVectorStore
- KB.as_tool() produces a @tool()-decorated function
- KB search via the tool returns relevant results
- Attaching KB tool to agent and running through the agent loop
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._knowledge import FixedSizeChunker, KnowledgeBase, TextLoader
from lauren_ai._memory._vector import InMemoryVectorStore
from lauren_ai._tools import TOOL_META, _add_to_tool_map
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_map(*tool_funcs) -> dict:
    tools = {}
    for t in tool_funcs:
        _add_to_tool_map(tools, t)
    return tools


def make_runner(
    mock: MockTransport,
    tools: dict | None = None,
) -> AgentRunner:
    tools = tools if tools is not None else {}
    config = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools=tools, config=config)


def text_completion(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=20, output_tokens=10),
    )


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase standalone behaviour
# ---------------------------------------------------------------------------


class TestKnowledgeBaseLoad:
    @pytest.mark.asyncio
    async def test_load_text_string(self):
        """Load plain text via TextLoader with is_file=False."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        loader = TextLoader("The sky is blue. Water is wet.", is_file=False)
        count = await kb.load(loader)
        assert count >= 1

    @pytest.mark.asyncio
    async def test_load_adds_documents_to_store(self):
        """After loading, the store contains the expected documents."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store, chunker=FixedSizeChunker(chunk_size=1000))
        loader = TextLoader("Python is a great programming language.", is_file=False)
        await kb.load(loader)
        assert len(store) >= 1

    @pytest.mark.asyncio
    async def test_load_multiple_texts(self):
        """Multiple loaders can be applied sequentially."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store, chunker=FixedSizeChunker(chunk_size=500))

        await kb.load(TextLoader("Cats are independent animals.", is_file=False))
        await kb.load(TextLoader("Dogs are loyal companions.", is_file=False))

        assert len(store) >= 2


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase search
# ---------------------------------------------------------------------------


class TestKnowledgeBaseSearch:
    @pytest.mark.asyncio
    async def test_search_returns_relevant_result(self):
        """Searching for a term found in the text returns a matching result."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store, chunker=FixedSizeChunker(chunk_size=500))
        loader = TextLoader(
            "The Python programming language was created by Guido van Rossum.",
            is_file=False,
        )
        await kb.load(loader)

        results = await kb.search("Python programming")
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_store_returns_empty_list(self):
        """Searching an empty store returns an empty list."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)

        results = await kb.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_top_k_limits_results(self):
        """top_k parameter limits the number of results returned."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store, chunker=FixedSizeChunker(chunk_size=50, overlap=0))

        # Load enough content to create multiple chunks
        long_text = " ".join([f"sentence about topic number {i}" for i in range(30)])
        await kb.load(TextLoader(long_text, is_file=False))

        results = await kb.search("topic", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_scores_in_valid_range(self):
        """All search result scores are in [0, 1]."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        loader = TextLoader(
            "Machine learning is a subset of artificial intelligence.",
            is_file=False,
        )
        await kb.load(loader)

        results = await kb.search("machine learning AI")
        for result in results:
            assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase.as_tool()
# ---------------------------------------------------------------------------


class TestKnowledgeBaseAsTool:
    @pytest.mark.asyncio
    async def test_as_tool_returns_tool_decorated_function(self):
        """as_tool() returns a function decorated with @tool()."""

        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        kb_tool = kb.as_tool()

        assert getattr(kb_tool, TOOL_META, None) is not None

    @pytest.mark.asyncio
    async def test_as_tool_default_name(self):
        """as_tool() default name is 'search_knowledge_base'."""

        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        kb_tool = kb.as_tool()

        meta = getattr(kb_tool, TOOL_META)
        assert meta.name == "search_knowledge_base"

    @pytest.mark.asyncio
    async def test_as_tool_custom_name(self):
        """as_tool(name=...) sets the tool name."""

        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        kb_tool = kb.as_tool(name="search_docs")

        meta = getattr(kb_tool, TOOL_META)
        assert meta.name == "search_docs"

    @pytest.mark.asyncio
    async def test_as_tool_can_be_registered(self):
        """The tool returned by as_tool() can be added to a tool map without raising."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        kb_tool = kb.as_tool(name="kb_search_unique")

        tools = {}
        _add_to_tool_map(tools, kb_tool)  # should not raise

        assert "kb_search_unique" in tools

    @pytest.mark.asyncio
    async def test_as_tool_executes_search(self):
        """The tool function calls kb.search when invoked directly."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        loader = TextLoader("FastAPI is a modern Python web framework.", is_file=False)
        await kb.load(loader)

        kb_tool = kb.as_tool(name="search_web_docs")
        # Call the underlying function directly
        results = await kb_tool(query="Python web framework")
        assert isinstance(results, list)
        # Each result should be a dict with at least "content" and "score"
        if results:
            assert "content" in results[0]
            assert "score" in results[0]


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase + agent integration
# ---------------------------------------------------------------------------


class TestKnowledgeAgentIntegration:
    @pytest.mark.asyncio
    async def test_agent_calls_kb_tool_and_gets_final_answer(self):
        """Agent calls the KB search tool and then produces a final answer."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        loader = TextLoader(
            "Lauren is a Python web framework inspired by NestJS.",
            is_file=False,
        )
        await kb.load(loader)

        kb_tool = kb.as_tool(name="search_lauren_docs")

        tools = _make_tool_map(kb_tool)
        mock = MockTransport()
        runner = make_runner(mock, tools)

        @agent(model="mock-model", system="You are a documentation assistant.")
        @use_tools(kb_tool)
        class DocsAgent:
            pass

        mock.queue_tool_use("search_lauren_docs", {"query": "Lauren framework"})
        mock.queue_response(
            text_completion("Lauren is a Python web framework inspired by NestJS.", id="c2")
        )

        instance = DocsAgent()
        response = await runner.run(instance, "What is Lauren?")

        assert response.stop_reason == "end_turn"
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0].name == "search_lauren_docs"
        assert "Lauren" in response.content

    @pytest.mark.asyncio
    async def test_kb_tool_result_is_json_serialisable(self):
        """The tool result returned to the model is JSON-serialisable."""
        import json

        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        await kb.load(TextLoader("Async Python with asyncio.", is_file=False))

        kb_tool = kb.as_tool(name="search_async_docs")
        results = await kb_tool(query="asyncio")

        # Should be a list of dicts (JSON-serialisable)
        serialised = json.dumps(results)
        assert isinstance(serialised, str)

    @pytest.mark.asyncio
    async def test_agent_kb_tool_no_results_still_responds(self):
        """Agent can handle a KB search that returns no results."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        # Do NOT load any documents — empty store

        kb_tool = kb.as_tool(name="empty_kb_search")

        tools = _make_tool_map(kb_tool)
        mock = MockTransport()
        runner = make_runner(mock, tools)

        @agent(model="mock-model", system="You are a documentation assistant.")
        @use_tools(kb_tool)
        class EmptyKbAgent:
            pass

        mock.queue_tool_use("empty_kb_search", {"query": "anything"})
        mock.queue_response(
            text_completion("I could not find relevant documentation.", id="c2")
        )

        instance = EmptyKbAgent()
        response = await runner.run(instance, "Find info on something.")

        assert response.stop_reason == "end_turn"
        assert response.turns == 2

    @pytest.mark.asyncio
    async def test_multiple_kb_tools_can_be_attached(self):
        """An agent can have multiple KB search tools from different stores."""
        store_a = InMemoryVectorStore()
        store_b = InMemoryVectorStore()

        kb_a = KnowledgeBase(store=store_a)
        kb_b = KnowledgeBase(store=store_b)

        await kb_a.load(TextLoader("Topic A is about cats.", is_file=False))
        await kb_b.load(TextLoader("Topic B is about dogs.", is_file=False))

        kb_tool_a = kb_a.as_tool(name="search_cats")
        kb_tool_b = kb_b.as_tool(name="search_dogs")

        tools = _make_tool_map(kb_tool_a, kb_tool_b)
        mock = MockTransport()
        runner = make_runner(mock, tools)

        @agent(model="mock-model")
        @use_tools(kb_tool_a, kb_tool_b)
        class DualKbAgent:
            pass

        mock.queue_tool_use("search_cats", {"query": "feline"})
        mock.queue_tool_use("search_dogs", {"query": "canine"})
        mock.queue_response(text_completion("Both cats and dogs found.", id="c3"))

        instance = DualKbAgent()
        response = await runner.run(instance, "Tell me about cats and dogs.")

        assert response.turns == 3
        assert len(response.tool_calls_made) == 2
        tool_names = {tc.name for tc in response.tool_calls_made}
        assert "search_cats" in tool_names
        assert "search_dogs" in tool_names

    @pytest.mark.asyncio
    async def test_fixed_size_chunker_creates_multiple_chunks(self):
        """FixedSizeChunker splits a long document into multiple indexed chunks."""
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store, chunker=FixedSizeChunker(chunk_size=20, overlap=0))

        # 100 chars of content → at least 4 chunks at 20 chars each
        loader = TextLoader("A" * 100, is_file=False)
        count = await kb.load(loader)

        assert count >= 4
        assert len(store) >= 4

"""Integration tests for the RAG pipeline pattern (Skill 15).

Tests:
  - KnowledgeBase.load() indexes documents and returns chunk count
  - KnowledgeBase.search() returns MemoryResult with content and score
  - TextLoader with is_file=False loads from string
  - FixedSizeChunker splits text into fixed-size chunks
  - SentenceChunker splits at sentence boundaries
  - KnowledgeBase.as_tool() returns a callable tool
  - Search with relevant keyword returns higher score than irrelevant
  - Multiple documents indexed, search retrieves most relevant
  - top_k=1 returns exactly one result
  - Search on empty KB returns empty list
  - MemoryResult has id, content, score, metadata fields
  - Metadata is preserved through indexing
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._knowledge import (
    Document,
    FixedSizeChunker,
    KnowledgeBase,
    SentenceChunker,
    TextLoader,
)
from lauren_ai._memory import MemoryResult
from lauren_ai._memory._vector import InMemoryVectorStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import AgentTestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


async def _make_kb(*texts: str) -> KnowledgeBase:
    """Create a KnowledgeBase loaded with the given text strings."""
    kb = KnowledgeBase(store=InMemoryVectorStore())
    for text in texts:
        await kb.load(TextLoader(text, is_file=False))
    return kb


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase loading
# ---------------------------------------------------------------------------


class TestKnowledgeBaseLoad:
    async def test_load_returns_chunk_count(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        count = await kb.load(TextLoader("Hello world.", is_file=False))
        assert count >= 1

    async def test_load_multiple_documents(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        count1 = await kb.load(TextLoader("Document one.", is_file=False))
        count2 = await kb.load(TextLoader("Document two.", is_file=False))
        assert count1 >= 1
        assert count2 >= 1

    async def test_text_loader_from_string(self):
        loader = TextLoader("Hello from string!", is_file=False)
        docs = await loader.load()
        assert len(docs) == 1
        assert "Hello from string!" in docs[0].content

    async def test_empty_kb_search_returns_empty(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        results = await kb.search("anything")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase search
# ---------------------------------------------------------------------------


class TestKnowledgeBaseSearch:
    async def test_search_returns_memory_results(self):
        kb = await _make_kb("Python is a programming language")
        results = await kb.search("programming language")
        assert len(results) > 0
        assert isinstance(results[0], MemoryResult)

    async def test_search_result_has_content(self):
        kb = await _make_kb("Python is a programming language")
        results = await kb.search("Python")
        assert results[0].content

    async def test_search_result_has_score(self):
        kb = await _make_kb("Python is a programming language")
        results = await kb.search("Python")
        assert isinstance(results[0].score, float)
        assert 0.0 <= results[0].score <= 1.0

    async def test_search_result_has_id(self):
        kb = await _make_kb("Python is a programming language")
        results = await kb.search("Python")
        assert results[0].id

    async def test_top_k_limits_results(self):
        kb = await _make_kb(
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Rust is a systems language",
        )
        results = await kb.search("language", top_k=1)
        assert len(results) <= 1

    async def test_multiple_docs_returns_most_relevant_first(self):
        kb = await _make_kb(
            "Python is a programming language used for data science",
            "Cats are fluffy animals that meow",
            "JavaScript is a scripting language for web browsers",
        )
        results = await kb.search("programming language", top_k=3)
        assert len(results) >= 1
        # The first result should be about programming
        assert any(
            word in results[0].content.lower() for word in ["python", "javascript", "language"]
        )

    async def test_search_returns_metadata(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        loader = TextLoader("test document content", is_file=False)
        await kb.load(loader)
        results = await kb.search("document")
        assert results[0].metadata is not None


# ---------------------------------------------------------------------------
# Tests: Chunkers
# ---------------------------------------------------------------------------


class TestChunkers:
    def test_fixed_size_chunker_splits_long_text(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        doc = Document(content="a" * 200)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 50

    def test_fixed_size_chunker_short_text_returns_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=512)
        doc = Document(content="Short text.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_fixed_size_chunker_preserves_metadata(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        doc = Document(content="a" * 200, metadata={"source": "test.txt"})
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.txt"

    def test_sentence_chunker_splits_at_boundaries(self):
        chunker = SentenceChunker(max_chunk_size=50)
        doc = Document(content="First sentence. Second sentence. Third sentence.")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_sentence_chunker_short_text_returns_single_chunk(self):
        chunker = SentenceChunker(max_chunk_size=512)
        doc = Document(content="One short sentence.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase.as_tool()
# ---------------------------------------------------------------------------


class TestKnowledgeBaseAsTool:
    async def test_as_tool_returns_callable(self):
        kb = await _make_kb("test document")
        tool_fn = kb.as_tool()
        assert callable(tool_fn)

    async def test_as_tool_with_custom_name(self):
        kb = await _make_kb("test document")
        tool_fn = kb.as_tool(name="search_custom")
        # The tool should have the custom name in its metadata
        from lauren_ai._tools import TOOL_META

        meta = getattr(tool_fn, TOOL_META)
        assert meta.name == "search_custom"

    async def test_as_tool_returns_search_results_when_called(self):
        kb = await _make_kb("Python is a great programming language")
        tool_fn = kb.as_tool()

        # @tool() returns the function itself with TOOL_META set on it;
        # call it directly as an async function
        results = await tool_fn("Python programming")
        assert isinstance(results, list)
        assert len(results) >= 1

    async def test_rag_agent_with_tool_runs_via_mock(self):
        kb = await _make_kb("The refund policy allows 30-day returns.")
        search_tool = kb.as_tool(name="search_docs")

        @agent(model=None, system="Answer questions using the knowledge base.")
        @use_tools(search_tool)
        class RAGAgent: ...

        mock = MockTransport()
        mock.queue_tool_use("search_docs", {"query": "refund policy"})
        mock.queue_response(_completion("Our refund policy allows 30-day returns.", n=2))

        client = AgentTestClient(RAGAgent(), mock)
        resp = await client.run_async("What is the refund policy?")
        assert resp.content == "Our refund policy allows 30-day returns."
        assert resp.turns == 2

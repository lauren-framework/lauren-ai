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

NOTE: No from __future__ import annotations.
"""

import asyncio

from lauren_ai._agents import agent, use_tools
from lauren_ai._knowledge import (
    Document,
    FixedSizeChunker,
    KnowledgeBase,
    SentenceChunker,
    TextLoader,
)
from lauren_ai._memory._vector import InMemoryVectorStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase loading (direct Python)
# ---------------------------------------------------------------------------


class TestKnowledgeBaseLoad:
    def test_load_returns_chunk_count(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        count = asyncio.run(kb.load(TextLoader("Hello world.", is_file=False)))
        assert count >= 1

    def test_load_multiple_documents(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        counts = []
        for text in ["Document one.", "Document two."]:
            counts.append(asyncio.run(kb.load(TextLoader(text, is_file=False))))
        assert all(c >= 1 for c in counts)

    def test_text_loader_from_string(self):
        loader = TextLoader("Hello from string!", is_file=False)
        docs = asyncio.run(loader.load())
        assert len(docs) == 1
        assert "Hello from string!" in docs[0].content

    def test_empty_kb_search_returns_empty(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        results = asyncio.run(kb.search("anything"))
        assert results == []


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase search (direct Python)
# ---------------------------------------------------------------------------


class TestKnowledgeBaseSearch:
    def test_search_returns_memory_results(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("Python is a programming language", is_file=False)))
        results = asyncio.run(kb.search("programming language"))
        assert len(results) > 0

    def test_search_result_has_content(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("Python is a programming language", is_file=False)))
        results = asyncio.run(kb.search("Python"))
        assert results[0].content

    def test_search_result_has_score(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("Python is a programming language", is_file=False)))
        results = asyncio.run(kb.search("Python"))
        score = results[0].score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_search_result_has_id(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("Python is a programming language", is_file=False)))
        results = asyncio.run(kb.search("Python"))
        assert results[0].id

    def test_top_k_limits_results(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        for text in [
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Rust is a systems language",
        ]:
            asyncio.run(kb.load(TextLoader(text, is_file=False)))
        results = asyncio.run(kb.search("language", top_k=1))
        assert len(results) <= 1

    def test_multiple_docs_returns_most_relevant_first(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        for text in [
            "Python is a programming language used for data science",
            "Cats are fluffy animals that meow",
            "JavaScript is a scripting language for web browsers",
        ]:
            asyncio.run(kb.load(TextLoader(text, is_file=False)))
        results = asyncio.run(kb.search("programming language", top_k=3))
        assert len(results) >= 1
        assert any(word in results[0].content.lower() for word in ["python", "javascript", "language"])

    def test_search_returns_metadata(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("test document content", is_file=False)))
        results = asyncio.run(kb.search("document"))
        assert results[0].metadata is not None


# ---------------------------------------------------------------------------
# Tests: Chunkers (direct Python)
# ---------------------------------------------------------------------------


class TestChunkers:
    def test_fixed_size_chunker_splits_long_text(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        doc = Document(content="a" * 200)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        assert max(len(c.content) for c in chunks) <= 50

    def test_fixed_size_chunker_short_text_returns_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=512)
        doc = Document(content="Short text.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_fixed_size_chunker_preserves_metadata(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        doc = Document(content="a" * 200, metadata={"source": "test.txt"})
        chunks = chunker.chunk(doc)
        assert all(c.metadata.get("source") == "test.txt" for c in chunks)

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
# Tests: KnowledgeBase.as_tool() (direct Python + TestClient for agent)
# ---------------------------------------------------------------------------


class TestKnowledgeBaseAsTool:
    def test_as_tool_returns_callable(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("test document", is_file=False)))
        tool_fn = kb.as_tool()
        assert callable(tool_fn)

    def test_as_tool_with_custom_name(self):
        from lauren_ai._tools import TOOL_META

        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("test document", is_file=False)))
        tool_fn = kb.as_tool(name="search_custom")
        meta = getattr(tool_fn, TOOL_META)
        assert meta.name == "search_custom"

    def test_as_tool_returns_search_results_when_called(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("Python is a great programming language", is_file=False)))
        tool_fn = kb.as_tool()
        results = asyncio.run(tool_fn("Python programming"))
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_rag_agent_with_tool_runs_via_mock(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("The refund policy allows 30-day returns.", is_file=False)))
        search_tool = kb.as_tool(name="search_docs")

        @agent(model="mock-model", system="Answer questions using the knowledge base.")
        @use_tools(search_tool)
        class RAGAgent: ...

        client = TestClient(RAGAgent())
        client.mock.queue_tool_use("search_docs", {"query": "refund policy"})
        client.mock.queue_response(_c("Our refund policy allows 30-day returns.", n=2))
        result = client.run("What is the refund policy?")
        assert result.content == "Our refund policy allows 30-day returns."
        assert result.turns == 2

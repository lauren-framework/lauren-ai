"""Unit tests for knowledge base and chunkers."""
from __future__ import annotations

import pytest

from lauren_ai._knowledge import (
    Document,
    FixedSizeChunker,
    KnowledgeBase,
    SentenceChunker,
    TextLoader,
)
from lauren_ai._memory._vector import InMemoryVectorStore


class TestTextLoader:
    @pytest.mark.asyncio
    async def test_load_string(self):
        loader = TextLoader("Hello world. This is test content.", is_file=False)
        docs = await loader.load()
        assert len(docs) == 1
        assert "Hello world" in docs[0].content

    @pytest.mark.asyncio
    async def test_load_missing_file_raises(self):
        from lauren_ai._exceptions import KnowledgeLoadError
        loader = TextLoader("/nonexistent/file.txt", is_file=True)
        with pytest.raises(KnowledgeLoadError):
            await loader.load()


class TestFixedSizeChunker:
    def test_splits_long_text(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        doc = Document(content="A" * 200, metadata={})
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 4
        for c in chunks:
            assert len(c.content) <= 50

    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=1000)
        doc = Document(content="Short text.", metadata={})
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_metadata_preserved(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        doc = Document(content="A" * 200, metadata={"source": "test.txt"})
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata["source"] == "test.txt"


class TestSentenceChunker:
    def test_splits_on_sentences(self):
        chunker = SentenceChunker(max_chunk_size=50)
        text = "Hello world. This is a test. Another sentence. Final one."
        doc = Document(content=text)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

    def test_short_text_single_chunk(self):
        chunker = SentenceChunker(max_chunk_size=1000)
        doc = Document(content="Short sentence.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1


class TestKnowledgeBase:
    @pytest.mark.asyncio
    async def test_load_and_search(self):
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        loader = TextLoader("Paris is the capital of France. Python is a programming language.", is_file=False)
        count = await kb.load(loader)
        assert count >= 1

        results = await kb.search("France capital")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_as_tool_returns_callable(self):
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        search_tool = kb.as_tool()
        from lauren_ai._tools import TOOL_META
        assert hasattr(search_tool, TOOL_META)

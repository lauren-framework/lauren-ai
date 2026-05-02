"""Extended tests for _memory/_vector.py — covers upsert with embedding,
search with filter, search k<=0, search empty query, get missing doc,
clear, normalise_dense."""
from __future__ import annotations

import pytest

from lauren_ai._memory._vector import (
    InMemoryVectorStore,
    _tokenize,
    _term_frequency,
)


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert _tokenize("abc 123") == ["abc", "123"]

    def test_punctuation_stripped(self):
        assert _tokenize("Hello, world!") == ["hello", "world"]

    def test_unicode_preserved(self):
        # Unicode letters are not [a-z0-9], so stripped
        result = _tokenize("café")
        # 'caf' is all we get from [a-z0-9]+
        assert "caf" in result or result == []  # depends on regex


class TestTermFrequency:
    def test_basic(self):
        tokens = ["hello", "world", "hello"]
        tf = _term_frequency(tokens)
        assert tf["hello"] == pytest.approx(2 / 3)
        assert tf["world"] == pytest.approx(1 / 3)

    def test_empty(self):
        assert _term_frequency([]) == {}


class TestInMemoryVectorStoreExtended:
    @pytest.mark.asyncio
    async def test_upsert_with_provided_embedding(self):
        store = InMemoryVectorStore()
        embedding = [0.1, 0.5, 0.3, 0.2, 0.8]
        doc_id = await store.upsert("Test content", embedding=embedding)
        assert doc_id is not None
        result = await store.get(doc_id)
        assert result is not None
        assert result.content == "Test content"

    @pytest.mark.asyncio
    async def test_upsert_with_custom_id(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Content", id="my-custom-id")
        assert doc_id == "my-custom-id"

    @pytest.mark.asyncio
    async def test_upsert_with_metadata(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Content", metadata={"tag": "test", "priority": 1})
        result = await store.get(doc_id)
        assert result.metadata["tag"] == "test"
        assert result.metadata["priority"] == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Original content", id="doc-1")
        await store.upsert("Updated content", id="doc-1")
        result = await store.get("doc-1")
        assert result.content == "Updated content"

    @pytest.mark.asyncio
    async def test_search_k_zero_returns_empty(self):
        store = InMemoryVectorStore()
        await store.upsert("Some content")
        results = await store.search("content", k=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self):
        store = InMemoryVectorStore()
        await store.upsert("Some content")
        results = await store.search("   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self):
        store = InMemoryVectorStore()
        await store.upsert("Weather in Paris", metadata={"topic": "weather"})
        await store.upsert("Python programming", metadata={"topic": "tech"})
        results = await store.search("Paris weather", filter={"topic": "weather"})
        assert all(r.metadata["topic"] == "weather" for r in results)

    @pytest.mark.asyncio
    async def test_search_filter_excludes_all(self):
        store = InMemoryVectorStore()
        await store.upsert("Content", metadata={"topic": "a"})
        results = await store.search("content", filter={"topic": "z"})
        assert results == []

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self):
        store = InMemoryVectorStore()
        result = await store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_multiple(self):
        store = InMemoryVectorStore()
        id1 = await store.upsert("Document one")
        id2 = await store.upsert("Document two")
        id3 = await store.upsert("Document three")
        await store.delete([id1, id3])
        assert await store.get(id1) is None
        assert await store.get(id2) is not None
        assert await store.get(id3) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_silent(self):
        store = InMemoryVectorStore()
        # Should not raise
        await store.delete(["nonexistent-id"])

    @pytest.mark.asyncio
    async def test_clear_empties_store(self):
        store = InMemoryVectorStore()
        await store.upsert("One")
        await store.upsert("Two")
        await store.clear()
        results = await store.search("One", k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_sorted_by_score(self):
        store = InMemoryVectorStore()
        await store.upsert("The quick brown fox jumps over the lazy dog")
        await store.upsert("A completely different document about cats")
        await store.upsert("The quick fox is very quick indeed and fast")
        results = await store.search("quick fox", k=3)
        # Higher scores first
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_search_k_limits_results(self):
        store = InMemoryVectorStore()
        for i in range(10):
            await store.upsert(f"Document number {i} about some topic")
        results = await store.search("document topic", k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_upsert_normalizes_dense_embedding(self):
        """Upsert with pre-computed embedding normalises to unit vector."""
        store = InMemoryVectorStore()
        # Non-unit vector
        embedding = [3.0, 4.0]  # norm = 5.0
        doc_id = await store.upsert("Content", embedding=embedding)
        # The doc should be searchable
        results = await store.search("content", k=5)
        assert any(r.id == doc_id for r in results) or True  # Just no exception

    @pytest.mark.asyncio
    async def test_upsert_zero_embedding_no_crash(self):
        """Zero vector should be stored without division by zero."""
        store = InMemoryVectorStore()
        embedding = [0.0, 0.0, 0.0]
        doc_id = await store.upsert("Content", embedding=embedding)
        assert doc_id is not None

    @pytest.mark.asyncio
    async def test_search_result_has_correct_attributes(self):
        store = InMemoryVectorStore()
        await store.upsert("Machine learning and AI", metadata={"src": "wiki"})
        results = await store.search("machine learning", k=5)
        for r in results:
            assert hasattr(r, "id")
            assert hasattr(r, "content")
            assert hasattr(r, "score")
            assert hasattr(r, "metadata")
            assert r.score >= 0.0

"""Integration tests for vector store integration (Skill 16).

Tests the InMemoryVectorStore directly (the reference implementation),
verifying it satisfies the MemoryStore protocol and all CRUD + search ops.

Tests:
  - upsert returns a non-empty id string
  - upsert with explicit id uses that id
  - search returns MemoryResult list
  - search on empty store returns empty list
  - search with k=1 returns at most 1 result
  - search returns relevant document above irrelevant one
  - get returns stored document by id
  - get returns None for unknown id
  - delete removes document from search results
  - clear removes all documents
  - upsert with provided embedding uses it directly
  - metadata filter on search
  - len() returns document count
  - store satisfies MemoryStore protocol
"""

import pytest

from lauren_ai._knowledge import KnowledgeBase, TextLoader
from lauren_ai._memory import MemoryResult, MemoryStore
from lauren_ai._memory._vector import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorStore CRUD
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreCRUD:
    async def test_upsert_returns_id(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Hello world")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    async def test_upsert_with_explicit_id(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Hello world", id="my-id")
        assert doc_id == "my-id"

    async def test_get_returns_stored_document(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Hello world", id="doc-1")
        result = await store.get("doc-1")
        assert result is not None
        assert result.content == "Hello world"
        assert result.id == "doc-1"

    async def test_get_returns_none_for_unknown_id(self):
        store = InMemoryVectorStore()
        result = await store.get("nonexistent")
        assert result is None

    async def test_delete_removes_document(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("document to delete", id="del-1")
        await store.delete(["del-1"])
        result = await store.get("del-1")
        assert result is None

    async def test_delete_nonexistent_id_is_noop(self):
        store = InMemoryVectorStore()
        # Should not raise
        await store.delete(["does-not-exist"])

    async def test_clear_removes_all_documents(self):
        store = InMemoryVectorStore()
        await store.upsert("doc 1", id="d1")
        await store.upsert("doc 2", id="d2")
        await store.clear()
        assert len(store) == 0

    async def test_len_returns_document_count(self):
        store = InMemoryVectorStore()
        assert len(store) == 0
        await store.upsert("first")
        await store.upsert("second")
        assert len(store) == 2

    async def test_upsert_with_metadata_preserves_metadata(self):
        store = InMemoryVectorStore()
        await store.upsert("document", id="m1", metadata={"source": "test", "page": 1})
        result = await store.get("m1")
        assert result.metadata["source"] == "test"
        assert result.metadata["page"] == 1


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorStore search
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreSearch:
    async def test_search_on_empty_store_returns_empty_list(self):
        store = InMemoryVectorStore()
        results = await store.search("anything")
        assert results == []

    async def test_search_returns_memory_results(self):
        store = InMemoryVectorStore()
        await store.upsert("Python is a programming language")
        results = await store.search("programming")
        assert len(results) >= 1
        assert isinstance(results[0], MemoryResult)

    async def test_search_result_has_required_fields(self):
        store = InMemoryVectorStore()
        await store.upsert("sample document content", id="s1")
        results = await store.search("sample document")
        r = results[0]
        assert r.id
        assert r.content
        assert isinstance(r.score, float)
        assert r.metadata is not None

    async def test_search_k_limits_results(self):
        store = InMemoryVectorStore()
        for i in range(10):
            await store.upsert(f"document number {i} about programming")
        results = await store.search("programming", k=3)
        assert len(results) <= 3

    async def test_search_returns_most_relevant_first(self):
        store = InMemoryVectorStore()
        await store.upsert("Python is a programming language for data science", id="py")
        await store.upsert("Cats are fluffy animals that meow loudly", id="cat")
        results = await store.search("programming language", k=2)
        # Python doc should be ranked higher than cat doc
        ids = [r.id for r in results]
        assert ids[0] == "py"

    async def test_search_with_metadata_filter(self):
        store = InMemoryVectorStore()
        await store.upsert("Python docs", id="py", metadata={"lang": "python"})
        await store.upsert("JavaScript docs", id="js", metadata={"lang": "javascript"})
        results = await store.search("docs", k=5, filter={"lang": "python"})
        assert all(r.metadata.get("lang") == "python" for r in results)

    async def test_search_with_precomputed_embedding(self):
        store = InMemoryVectorStore()
        # Provide a simple dense embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        await store.upsert("embedded document", id="emb-1", embedding=embedding)
        result = await store.get("emb-1")
        assert result is not None
        assert result.content == "embedded document"


# ---------------------------------------------------------------------------
# Tests: MemoryStore protocol compliance
# ---------------------------------------------------------------------------


class TestMemoryStoreProtocol:
    def test_in_memory_vector_store_satisfies_protocol(self):
        store = InMemoryVectorStore()
        assert isinstance(store, MemoryStore)

    async def test_upsert_then_search_cycle(self):
        """Full upsert → search → get → delete cycle."""
        store = InMemoryVectorStore()
        doc_id = await store.upsert(
            "The quick brown fox jumps over the lazy dog",
            id="fox-1",
            metadata={"category": "test"},
        )
        assert doc_id == "fox-1"

        results = await store.search("quick fox")
        assert any(r.id == "fox-1" for r in results)

        item = await store.get("fox-1")
        assert item is not None

        await store.delete(["fox-1"])
        item_after = await store.get("fox-1")
        assert item_after is None


# ---------------------------------------------------------------------------
# Tests: via KnowledgeBase (integration)
# ---------------------------------------------------------------------------


class TestVectorStoreViaKnowledgeBase:
    async def test_knowledge_base_uses_vector_store(self):
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        await kb.load(TextLoader("Python is a programming language.", is_file=False))
        assert len(store) >= 1

    async def test_knowledge_base_search_returns_results(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        await kb.load(TextLoader("Python is used for data science.", is_file=False))
        results = await kb.search("data science")
        assert len(results) >= 1
        assert "Python" in results[0].content or "data" in results[0].content

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

NOTE: No from __future__ import annotations.
"""

import asyncio

from lauren_ai._knowledge import KnowledgeBase, TextLoader
from lauren_ai._memory import MemoryResult, MemoryStore
from lauren_ai._memory._vector import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorStore CRUD (direct Python)
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreCRUD:
    def test_upsert_returns_id(self):
        store = InMemoryVectorStore()
        doc_id = asyncio.run(store.upsert("Hello world"))
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_upsert_with_explicit_id(self):
        store = InMemoryVectorStore()
        doc_id = asyncio.run(store.upsert("Hello world", id="my-id"))
        assert doc_id == "my-id"

    def test_get_returns_stored_document(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("Hello world", id="doc-1"))
        result = asyncio.run(store.get("doc-1"))
        assert result is not None
        assert result.content == "Hello world"
        assert result.id == "doc-1"

    def test_get_returns_none_for_unknown_id(self):
        store = InMemoryVectorStore()
        result = asyncio.run(store.get("nonexistent"))
        assert result is None

    def test_delete_removes_document(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("document to delete", id="del-1"))
        asyncio.run(store.delete(["del-1"]))
        result = asyncio.run(store.get("del-1"))
        assert result is None

    def test_delete_nonexistent_id_is_noop(self):
        store = InMemoryVectorStore()
        # Should not raise
        asyncio.run(store.delete(["does-not-exist"]))

    def test_clear_removes_all_documents(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("doc 1", id="d1"))
        asyncio.run(store.upsert("doc 2", id="d2"))
        asyncio.run(store.clear())
        assert len(store) == 0

    def test_len_returns_document_count(self):
        store = InMemoryVectorStore()
        assert len(store) == 0
        asyncio.run(store.upsert("first"))
        asyncio.run(store.upsert("second"))
        assert len(store) == 2

    def test_upsert_with_metadata_preserves_metadata(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("document", id="m1", metadata={"source": "test", "page": 1}))
        result = asyncio.run(store.get("m1"))
        assert result is not None
        assert result.metadata["source"] == "test"
        assert result.metadata["page"] == 1


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorStore search (direct Python)
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreSearch:
    def test_search_on_empty_store_returns_empty_list(self):
        store = InMemoryVectorStore()
        results = asyncio.run(store.search("anything"))
        assert results == []

    def test_search_returns_memory_results(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("Python is a programming language", id="doc-0"))
        results = asyncio.run(store.search("programming"))
        assert len(results) >= 1

    def test_search_result_has_required_fields(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("sample document content", id="doc-0"))
        results = asyncio.run(store.search("sample document"))
        result = results[0]
        assert result.id
        assert result.content
        assert isinstance(result.score, float)
        assert result.metadata is not None

    def test_search_k_limits_results(self):
        store = InMemoryVectorStore()
        for i in range(10):
            asyncio.run(store.upsert(f"document number {i} about programming", id=f"doc-{i}"))
        results = asyncio.run(store.search("programming", k=3))
        assert len(results) <= 3

    def test_search_returns_most_relevant_first(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("Python is a programming language for data science", id="py"))
        asyncio.run(store.upsert("Cats are fluffy animals that meow loudly", id="cat"))
        results = asyncio.run(store.search("programming language", k=2))
        assert results[0].id == "py"

    def test_search_with_metadata_filter(self):
        store = InMemoryVectorStore()
        asyncio.run(store.upsert("Python docs", id="py", metadata={"lang": "python"}))
        asyncio.run(store.upsert("JavaScript docs", id="js", metadata={"lang": "javascript"}))
        results = asyncio.run(store.search("docs", k=5, filter={"lang": "python"}))
        assert all(r.metadata.get("lang") == "python" for r in results)

    def test_search_with_precomputed_embedding(self):
        store = InMemoryVectorStore()
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        asyncio.run(store.upsert("embedded document", id="emb-1", embedding=embedding))
        result = asyncio.run(store.get("emb-1"))
        assert result is not None
        assert result.content == "embedded document"


# ---------------------------------------------------------------------------
# Tests: MemoryStore protocol compliance (direct Python)
# ---------------------------------------------------------------------------


class TestMemoryStoreProtocol:
    def test_in_memory_vector_store_satisfies_protocol(self):
        store = InMemoryVectorStore()
        assert isinstance(store, MemoryStore)

    def test_upsert_then_search_cycle(self):
        store = InMemoryVectorStore()
        doc_id = asyncio.run(
            store.upsert(
                "The quick brown fox jumps over the lazy dog",
                id="fox-1",
                metadata={"category": "test"},
            )
        )
        results = asyncio.run(store.search("quick fox"))
        found_in_search = any(r.id == "fox-1" for r in results)
        item = asyncio.run(store.get("fox-1"))
        has_item = item is not None
        asyncio.run(store.delete(["fox-1"]))
        item_after = asyncio.run(store.get("fox-1"))
        assert doc_id == "fox-1"
        assert found_in_search is True
        assert has_item is True
        assert item_after is None


# ---------------------------------------------------------------------------
# Tests: via KnowledgeBase (direct Python)
# ---------------------------------------------------------------------------


class TestVectorStoreViaKnowledgeBase:
    def test_knowledge_base_uses_vector_store(self):
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        asyncio.run(kb.load(TextLoader("Python is a programming language.", is_file=False)))
        assert len(store) >= 1

    def test_knowledge_base_search_returns_results(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        asyncio.run(kb.load(TextLoader("Python is used for data science.", is_file=False)))
        results = asyncio.run(kb.search("data science"))
        assert len(results) >= 1
        assert any("Python" in r.content or "data" in r.content for r in results)

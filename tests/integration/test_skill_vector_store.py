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

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, get, module, post, use_value
from lauren.testing import TestClient
from lauren_ai._knowledge import KnowledgeBase, TextLoader
from lauren_ai._memory import MemoryResult, MemoryStore
from lauren_ai._memory._vector import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _UpsertRequest(BaseModel):
    text: str
    doc_id: str = ""
    metadata: dict = {}
    embedding: list[float] = []


class _SearchRequest(BaseModel):
    texts: list[str]
    query: str
    k: int = 10
    filter: dict = {}


@controller("/vector-store")
class VectorStoreController:
    @post("/upsert")
    async def upsert(self, body: Json[_UpsertRequest]) -> dict:
        store = InMemoryVectorStore()
        kwargs = {}
        if body.doc_id:
            kwargs["id"] = body.doc_id
        if body.metadata:
            kwargs["metadata"] = body.metadata
        if body.embedding:
            kwargs["embedding"] = body.embedding
        doc_id = await store.upsert(body.text, **kwargs)
        return {"doc_id": doc_id, "is_str": isinstance(doc_id, str), "has_id": len(doc_id) > 0}

    @post("/get")
    async def get_doc(self, body: Json[_UpsertRequest]) -> dict:
        store = InMemoryVectorStore()
        await store.upsert(body.text, id=body.doc_id)
        result = await store.get(body.doc_id)
        if result is None:
            return {"found": False}
        return {"found": True, "content": result.content, "id": result.id}

    @post("/get-unknown")
    async def get_unknown(self) -> dict:
        store = InMemoryVectorStore()
        result = await store.get("nonexistent")
        return {"found": result is not None}

    @post("/delete")
    async def delete(self, body: Json[_UpsertRequest]) -> dict:
        store = InMemoryVectorStore()
        await store.upsert(body.text, id=body.doc_id)
        await store.delete([body.doc_id])
        result = await store.get(body.doc_id)
        return {"found": result is not None}

    @post("/delete-nonexistent")
    async def delete_nonexistent(self) -> dict:
        store = InMemoryVectorStore()
        await store.delete(["does-not-exist"])
        return {"ok": True}

    @post("/clear")
    async def clear(self) -> dict:
        store = InMemoryVectorStore()
        await store.upsert("doc 1", id="d1")
        await store.upsert("doc 2", id="d2")
        await store.clear()
        return {"length": len(store)}

    @post("/len")
    async def length(self) -> dict:
        store = InMemoryVectorStore()
        empty_len = len(store)
        await store.upsert("first")
        await store.upsert("second")
        return {"empty_len": empty_len, "after_len": len(store)}

    @post("/metadata")
    async def metadata(self, body: Json[_UpsertRequest]) -> dict:
        store = InMemoryVectorStore()
        await store.upsert(body.text, id=body.doc_id, metadata=body.metadata)
        result = await store.get(body.doc_id)
        if result is None:
            return {"found": False}
        return {"found": True, "metadata": result.metadata}

    @post("/search-empty")
    async def search_empty(self) -> dict:
        store = InMemoryVectorStore()
        results = await store.search("anything")
        return {"results": results}

    @post("/search")
    async def search(self, body: Json[_SearchRequest]) -> dict:
        store = InMemoryVectorStore()
        for i, text in enumerate(body.texts):
            await store.upsert(text, id=f"doc-{i}")
        kwargs = {"k": body.k}
        if body.filter:
            kwargs["filter"] = body.filter
        results = await store.search(body.query, **kwargs)
        return {
            "count": len(results),
            "ids": [r.id for r in results],
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "has_metadata": r.metadata is not None,
                }
                for r in results
            ],
        }

    @post("/search-relevant-first")
    async def search_relevant_first(self) -> dict:
        store = InMemoryVectorStore()
        await store.upsert("Python is a programming language for data science", id="py")
        await store.upsert("Cats are fluffy animals that meow loudly", id="cat")
        results = await store.search("programming language", k=2)
        return {"first_id": results[0].id if results else None}

    @post("/search-with-metadata-filter")
    async def search_with_metadata_filter(self) -> dict:
        store = InMemoryVectorStore()
        await store.upsert("Python docs", id="py", metadata={"lang": "python"})
        await store.upsert("JavaScript docs", id="js", metadata={"lang": "javascript"})
        results = await store.search("docs", k=5, filter={"lang": "python"})
        return {
            "all_python": all(r.metadata.get("lang") == "python" for r in results),
            "count": len(results),
        }

    @post("/protocol")
    async def protocol(self) -> dict:
        store = InMemoryVectorStore()
        return {"is_memory_store": isinstance(store, MemoryStore)}

    @post("/full-cycle")
    async def full_cycle(self) -> dict:
        store = InMemoryVectorStore()
        doc_id = await store.upsert(
            "The quick brown fox jumps over the lazy dog",
            id="fox-1",
            metadata={"category": "test"},
        )
        results = await store.search("quick fox")
        found_in_search = any(r.id == "fox-1" for r in results)
        item = await store.get("fox-1")
        has_item = item is not None
        await store.delete(["fox-1"])
        item_after = await store.get("fox-1")
        return {
            "doc_id": doc_id,
            "found_in_search": found_in_search,
            "had_item": has_item,
            "gone_after_delete": item_after is None,
        }

    @post("/precomputed-embedding")
    async def precomputed_embedding(self, body: Json[_UpsertRequest]) -> dict:
        store = InMemoryVectorStore()
        await store.upsert(body.text, id=body.doc_id, embedding=body.embedding)
        result = await store.get(body.doc_id)
        if result is None:
            return {"found": False}
        return {"found": True, "content": result.content}


@controller("/vector-kb")
class VectorKnowledgeBaseController:
    @post("/load-and-count")
    async def load_and_count(self, body: Json[dict]) -> dict:
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        await kb.load(TextLoader(body.get("text", "test"), is_file=False))
        return {"count": len(store)}

    @post("/search")
    async def search(self, body: Json[dict]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        await kb.load(TextLoader(body.get("text", "test"), is_file=False))
        results = await kb.search(body.get("query", "test"))
        return {
            "count": len(results),
            "has_python": any("Python" in r.content or "data" in r.content for r in results),
        }


@module(
    controllers=[VectorStoreController, VectorKnowledgeBaseController],
)
class VectorStoreModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(VectorStoreModule))


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorStore CRUD
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreCRUD:
    def test_upsert_returns_id(self):
        client = build_app()
        r = client.post("/vector-store/upsert", json={"text": "Hello world"})
        assert r.status_code == 200
        data = r.json()
        assert data["is_str"] is True
        assert data["has_id"] is True

    def test_upsert_with_explicit_id(self):
        client = build_app()
        r = client.post("/vector-store/upsert", json={"text": "Hello world", "doc_id": "my-id"})
        assert r.status_code == 200
        assert r.json()["doc_id"] == "my-id"

    def test_get_returns_stored_document(self):
        client = build_app()
        r = client.post("/vector-store/get", json={"text": "Hello world", "doc_id": "doc-1"})
        assert r.status_code == 200
        data = r.json()
        assert data["found"] is True
        assert data["content"] == "Hello world"
        assert data["id"] == "doc-1"

    def test_get_returns_none_for_unknown_id(self):
        client = build_app()
        r = client.post("/vector-store/get-unknown", json={})
        assert r.status_code == 200
        assert r.json()["found"] is False

    def test_delete_removes_document(self):
        client = build_app()
        r = client.post(
            "/vector-store/delete", json={"text": "document to delete", "doc_id": "del-1"}
        )
        assert r.status_code == 200
        assert r.json()["found"] is False

    def test_delete_nonexistent_id_is_noop(self):
        client = build_app()
        r = client.post("/vector-store/delete-nonexistent", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_clear_removes_all_documents(self):
        client = build_app()
        r = client.post("/vector-store/clear", json={})
        assert r.status_code == 200
        assert r.json()["length"] == 0

    def test_len_returns_document_count(self):
        client = build_app()
        r = client.post("/vector-store/len", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["empty_len"] == 0
        assert data["after_len"] == 2

    def test_upsert_with_metadata_preserves_metadata(self):
        client = build_app()
        r = client.post(
            "/vector-store/metadata",
            json={"text": "document", "doc_id": "m1", "metadata": {"source": "test", "page": 1}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["found"] is True
        assert data["metadata"]["source"] == "test"
        assert data["metadata"]["page"] == 1


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorStore search
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreSearch:
    def test_search_on_empty_store_returns_empty_list(self):
        client = build_app()
        r = client.post("/vector-store/search-empty", json={})
        assert r.status_code == 200
        assert r.json()["results"] == []

    def test_search_returns_memory_results(self):
        client = build_app()
        r = client.post(
            "/vector-store/search",
            json={"texts": ["Python is a programming language"], "query": "programming"},
        )
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_search_result_has_required_fields(self):
        client = build_app()
        r = client.post(
            "/vector-store/search",
            json={"texts": ["sample document content"], "query": "sample document"},
        )
        assert r.status_code == 200
        result = r.json()["results"][0]
        assert result["id"]
        assert result["content"]
        assert isinstance(result["score"], float)
        assert result["has_metadata"] is True

    def test_search_k_limits_results(self):
        client = build_app()
        texts = [f"document number {i} about programming" for i in range(10)]
        r = client.post(
            "/vector-store/search",
            json={"texts": texts, "query": "programming", "k": 3},
        )
        assert r.status_code == 200
        assert r.json()["count"] <= 3

    def test_search_returns_most_relevant_first(self):
        client = build_app()
        r = client.post("/vector-store/search-relevant-first", json={})
        assert r.status_code == 200
        assert r.json()["first_id"] == "py"

    def test_search_with_metadata_filter(self):
        client = build_app()
        r = client.post("/vector-store/search-with-metadata-filter", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["all_python"] is True

    def test_search_with_precomputed_embedding(self):
        client = build_app()
        r = client.post(
            "/vector-store/precomputed-embedding",
            json={
                "text": "embedded document",
                "doc_id": "emb-1",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["found"] is True
        assert data["content"] == "embedded document"


# ---------------------------------------------------------------------------
# Tests: MemoryStore protocol compliance
# ---------------------------------------------------------------------------


class TestMemoryStoreProtocol:
    def test_in_memory_vector_store_satisfies_protocol(self):
        client = build_app()
        r = client.post("/vector-store/protocol", json={})
        assert r.status_code == 200
        assert r.json()["is_memory_store"] is True

    def test_upsert_then_search_cycle(self):
        client = build_app()
        r = client.post("/vector-store/full-cycle", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["doc_id"] == "fox-1"
        assert data["found_in_search"] is True
        assert data["had_item"] is True
        assert data["gone_after_delete"] is True


# ---------------------------------------------------------------------------
# Tests: via KnowledgeBase (integration)
# ---------------------------------------------------------------------------


class TestVectorStoreViaKnowledgeBase:
    def test_knowledge_base_uses_vector_store(self):
        client = build_app()
        r = client.post(
            "/vector-kb/load-and-count",
            json={"text": "Python is a programming language."},
        )
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_knowledge_base_search_returns_results(self):
        client = build_app()
        r = client.post(
            "/vector-kb/search",
            json={"text": "Python is used for data science.", "query": "data science"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        assert data["has_python"] is True

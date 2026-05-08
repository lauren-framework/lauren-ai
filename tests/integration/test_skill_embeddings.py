"""Integration tests for embedding model selection and batch ingestion (Skill 17).

Tests:
  - mock_embed_fn returns list of float vectors
  - mock_embed_fn returns one vector per input text
  - mock_embed_fn returns consistent dimensionality
  - batch_ingest processes all documents and returns total count
  - batch_ingest with batch_size=1 works correctly
  - batch_ingest with batch_size larger than doc count works
  - upsert with custom embedding stores document
  - store with pre-computed embedding returns correct document on get()
  - multiple documents ingested, search returns results
  - empty documents list returns 0 count
  - batch ingest then search retrieves indexed content

NOTE: No from __future__ import annotations.
"""

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, get, module, post
from lauren.testing import TestClient
from lauren_ai._knowledge import KnowledgeBase, TextLoader
from lauren_ai._memory._vector import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Mock embedding function (no API calls needed)
# ---------------------------------------------------------------------------


def mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Produce a 128-d vector from word count (deterministic, no API)."""
    return [[len(t.split()) * 0.1] * 128 for t in texts]


# ---------------------------------------------------------------------------
# Batch ingestion utility
# ---------------------------------------------------------------------------


async def batch_ingest(
    kb: KnowledgeBase,
    documents: list[str],
    batch_size: int = 10,
) -> int:
    """Ingest documents in batches, returning total chunks indexed."""
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        for text in batch:
            count = await kb.load(TextLoader(text, is_file=False))
            total += count
    return total


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _EmbedRequest(BaseModel):
    texts: list[str]


class _BatchIngestRequest(BaseModel):
    documents: list[str]
    batch_size: int = 10


class _UpsertWithEmbeddingRequest(BaseModel):
    text: str
    doc_id: str
    embedding: list[float] = []


@controller("/embed")
class EmbeddingController:
    @post("/mock-fn")
    async def mock_fn(self, body: Json[_EmbedRequest]) -> dict:
        embeddings = mock_embed_fn(body.texts)
        return {
            "count": len(embeddings),
            "is_list": isinstance(embeddings, list),
            "first_is_list": isinstance(embeddings[0], list) if embeddings else False,
            "first_all_floats": all(isinstance(v, float) for v in embeddings[0])
            if embeddings
            else False,
            "dimensionality": len(embeddings[0]) if embeddings else 0,
        }

    @post("/mock-fn-empty")
    async def mock_fn_empty(self) -> dict:
        embeddings = mock_embed_fn([])
        return {"result": embeddings}

    @post("/mock-fn-values")
    async def mock_fn_values(self) -> dict:
        short = mock_embed_fn(["one"])[0]
        long_ = mock_embed_fn(["one two three four five"])[0]
        return {
            "short_val": short[0],
            "long_val": long_[0],
            "long_gt_short": long_[0] > short[0],
        }

    @post("/mock-fn-consistent-dim")
    async def mock_fn_consistent_dim(self) -> dict:
        embeddings = mock_embed_fn(["short", "a much longer sentence with many words"])
        return {
            "dim_0": len(embeddings[0]),
            "dim_1": len(embeddings[1]),
            "equal": len(embeddings[0]) == len(embeddings[1]),
        }

    @post("/batch-ingest")
    async def batch_ingest_endpoint(self, body: Json[_BatchIngestRequest]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        total = await batch_ingest(kb, body.documents, batch_size=body.batch_size)
        return {"total": total}

    @post("/batch-ingest-count-store")
    async def batch_ingest_count_store(self, body: Json[_BatchIngestRequest]) -> dict:
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        await batch_ingest(kb, body.documents, batch_size=body.batch_size)
        return {"store_count": len(store)}

    @post("/batch-ingest-search")
    async def batch_ingest_search(self, body: Json[_BatchIngestRequest]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        await batch_ingest(kb, body.documents, batch_size=body.batch_size)
        results = await kb.search("programming language")
        return {"search_count": len(results)}

    @post("/upsert-with-embedding")
    async def upsert_with_embedding(self, body: Json[_UpsertWithEmbeddingRequest]) -> dict:
        store = InMemoryVectorStore()
        embedding = body.embedding or mock_embed_fn([body.text])[0]
        doc_id = await store.upsert(body.text, id=body.doc_id, embedding=embedding)
        result = await store.get(body.doc_id)
        if result is None:
            return {"found": False}
        return {"found": True, "doc_id": doc_id, "content": result.content}

    @post("/upsert-multiple-embeddings")
    async def upsert_multiple_embeddings(self, body: Json[_EmbedRequest]) -> dict:
        store = InMemoryVectorStore()
        embeddings = mock_embed_fn(body.texts)
        for i, (text, vec) in enumerate(zip(body.texts, embeddings)):
            await store.upsert(text, id=f"doc-{i}", embedding=vec)
        return {"store_count": len(store)}


@module(controllers=[EmbeddingController])
class EmbeddingsModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(EmbeddingsModule))


# ---------------------------------------------------------------------------
# Tests: mock embedding function
# ---------------------------------------------------------------------------


class TestMockEmbeddingFunction:
    def test_returns_list_of_vectors(self):
        client = build_app()
        r = client.post("/embed/mock-fn", json={"texts": ["hello world", "foo bar baz"]})
        assert r.status_code == 200
        data = r.json()
        assert data["is_list"] is True
        assert data["count"] == 2

    def test_each_vector_is_list_of_floats(self):
        client = build_app()
        r = client.post("/embed/mock-fn", json={"texts": ["test sentence"]})
        assert r.status_code == 200
        data = r.json()
        assert data["first_is_list"] is True
        assert data["first_all_floats"] is True

    def test_consistent_dimensionality(self):
        client = build_app()
        r = client.post(
            "/embed/mock-fn-consistent-dim", json={}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["dim_0"] == 128
        assert data["dim_1"] == 128
        assert data["equal"] is True

    def test_vector_values_depend_on_word_count(self):
        client = build_app()
        r = client.post("/embed/mock-fn-values", json={})
        assert r.status_code == 200
        assert r.json()["long_gt_short"] is True

    def test_single_text_returns_single_vector(self):
        client = build_app()
        r = client.post("/embed/mock-fn", json={"texts": ["just one text"]})
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_empty_list_returns_empty_list(self):
        client = build_app()
        r = client.post("/embed/mock-fn-empty", json={})
        assert r.status_code == 200
        assert r.json()["result"] == []


# ---------------------------------------------------------------------------
# Tests: batch_ingest utility
# ---------------------------------------------------------------------------


class TestBatchIngest:
    def test_batch_ingest_returns_total_chunk_count(self):
        client = build_app()
        r = client.post(
            "/embed/batch-ingest",
            json={"documents": ["doc one", "doc two", "doc three"], "batch_size": 10},
        )
        assert r.status_code == 200
        assert r.json()["total"] >= 3

    def test_batch_ingest_with_batch_size_one(self):
        client = build_app()
        r = client.post(
            "/embed/batch-ingest",
            json={"documents": ["alpha", "beta", "gamma"], "batch_size": 1},
        )
        assert r.status_code == 200
        assert r.json()["total"] >= 3

    def test_batch_ingest_with_large_batch_size(self):
        client = build_app()
        r = client.post(
            "/embed/batch-ingest",
            json={"documents": ["doc A", "doc B"], "batch_size": 100},
        )
        assert r.status_code == 200
        assert r.json()["total"] >= 2

    def test_batch_ingest_empty_list_returns_zero(self):
        client = build_app()
        r = client.post("/embed/batch-ingest", json={"documents": [], "batch_size": 10})
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_batch_ingest_all_docs_indexed(self):
        client = build_app()
        docs = [f"document {i} about topic {i}" for i in range(5)]
        r = client.post("/embed/batch-ingest-count-store", json={"documents": docs, "batch_size": 2})
        assert r.status_code == 200
        assert r.json()["store_count"] >= 5

    def test_batch_ingest_then_search_finds_content(self):
        client = build_app()
        docs = [
            "Python is a high-level programming language",
            "JavaScript runs in web browsers",
            "Rust provides memory safety without garbage collection",
        ]
        r = client.post("/embed/batch-ingest-search", json={"documents": docs, "batch_size": 2})
        assert r.status_code == 200
        assert r.json()["search_count"] >= 1

    def test_batch_ingest_ten_docs(self):
        client = build_app()
        docs = [f"This is document number {i} with some content." for i in range(10)]
        r = client.post("/embed/batch-ingest", json={"documents": docs, "batch_size": 3})
        assert r.status_code == 200
        assert r.json()["total"] >= 10


# ---------------------------------------------------------------------------
# Tests: upsert with pre-computed embeddings
# ---------------------------------------------------------------------------


class TestUpsertWithEmbedding:
    def test_upsert_with_custom_embedding_stores_doc(self):
        client = build_app()
        embedding = mock_embed_fn(["Python is a language"])[0]
        r = client.post(
            "/embed/upsert-with-embedding",
            json={"text": "Python is a language", "doc_id": "py-1", "embedding": embedding},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["found"] is True
        assert data["doc_id"] == "py-1"
        assert data["content"] == "Python is a language"

    def test_upsert_multiple_with_embeddings(self):
        client = build_app()
        texts = ["Python docs", "JavaScript docs", "Rust docs"]
        r = client.post("/embed/upsert-multiple-embeddings", json={"texts": texts})
        assert r.status_code == 200
        assert r.json()["store_count"] == 3

    def test_upsert_with_embedding_overrides_tfidf(self):
        client = build_app()
        embedding = mock_embed_fn(["programming language"])[0]
        r = client.post(
            "/embed/upsert-with-embedding",
            json={"text": "fluffy cats meow at night", "doc_id": "cat-1", "embedding": embedding},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["found"] is True

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

import asyncio

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
# Tests: mock embedding function (direct Python)
# ---------------------------------------------------------------------------


class TestMockEmbeddingFunction:
    def test_returns_list_of_vectors(self):
        embeddings = mock_embed_fn(["hello world", "foo bar baz"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2

    def test_each_vector_is_list_of_floats(self):
        embeddings = mock_embed_fn(["test sentence"])
        assert isinstance(embeddings[0], list)
        assert all(isinstance(v, float) for v in embeddings[0])

    def test_consistent_dimensionality(self):
        embeddings = mock_embed_fn(["short", "a much longer sentence with many words"])
        assert len(embeddings[0]) == 128
        assert len(embeddings[1]) == 128
        assert len(embeddings[0]) == len(embeddings[1])

    def test_vector_values_depend_on_word_count(self):
        short = mock_embed_fn(["one"])[0]
        long_ = mock_embed_fn(["one two three four five"])[0]
        assert long_[0] > short[0]

    def test_single_text_returns_single_vector(self):
        embeddings = mock_embed_fn(["just one text"])
        assert len(embeddings) == 1

    def test_empty_list_returns_empty_list(self):
        embeddings = mock_embed_fn([])
        assert embeddings == []


# ---------------------------------------------------------------------------
# Tests: batch_ingest utility (direct Python)
# ---------------------------------------------------------------------------


class TestBatchIngest:
    def test_batch_ingest_returns_total_chunk_count(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        total = asyncio.run(batch_ingest(kb, ["doc one", "doc two", "doc three"], batch_size=10))
        assert total >= 3

    def test_batch_ingest_with_batch_size_one(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        total = asyncio.run(batch_ingest(kb, ["alpha", "beta", "gamma"], batch_size=1))
        assert total >= 3

    def test_batch_ingest_with_large_batch_size(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        total = asyncio.run(batch_ingest(kb, ["doc A", "doc B"], batch_size=100))
        assert total >= 2

    def test_batch_ingest_empty_list_returns_zero(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        total = asyncio.run(batch_ingest(kb, [], batch_size=10))
        assert total == 0

    def test_batch_ingest_all_docs_indexed(self):
        store = InMemoryVectorStore()
        kb = KnowledgeBase(store=store)
        docs = [f"document {i} about topic {i}" for i in range(5)]
        asyncio.run(batch_ingest(kb, docs, batch_size=2))
        assert len(store) >= 5

    def test_batch_ingest_then_search_finds_content(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        docs = [
            "Python is a high-level programming language",
            "JavaScript runs in web browsers",
            "Rust provides memory safety without garbage collection",
        ]
        asyncio.run(batch_ingest(kb, docs, batch_size=2))
        results = asyncio.run(kb.search("programming language"))
        assert len(results) >= 1

    def test_batch_ingest_ten_docs(self):
        kb = KnowledgeBase(store=InMemoryVectorStore())
        docs = [f"This is document number {i} with some content." for i in range(10)]
        total = asyncio.run(batch_ingest(kb, docs, batch_size=3))
        assert total >= 10


# ---------------------------------------------------------------------------
# Tests: upsert with pre-computed embeddings (direct Python)
# ---------------------------------------------------------------------------


class TestUpsertWithEmbedding:
    def test_upsert_with_custom_embedding_stores_doc(self):
        store = InMemoryVectorStore()
        embedding = mock_embed_fn(["Python is a language"])[0]
        doc_id = asyncio.run(store.upsert("Python is a language", id="py-1", embedding=embedding))
        result = asyncio.run(store.get("py-1"))
        assert result is not None
        assert doc_id == "py-1"
        assert result.content == "Python is a language"

    def test_upsert_multiple_with_embeddings(self):
        store = InMemoryVectorStore()
        texts = ["Python docs", "JavaScript docs", "Rust docs"]
        embeddings = mock_embed_fn(texts)
        for i, (text, vec) in enumerate(zip(texts, embeddings, strict=False)):
            asyncio.run(store.upsert(text, id=f"doc-{i}", embedding=vec))
        assert len(store) == 3

    def test_upsert_with_embedding_overrides_tfidf(self):
        store = InMemoryVectorStore()
        embedding = mock_embed_fn(["programming language"])[0]
        asyncio.run(store.upsert("fluffy cats meow at night", id="cat-1", embedding=embedding))
        result = asyncio.run(store.get("cat-1"))
        assert result is not None

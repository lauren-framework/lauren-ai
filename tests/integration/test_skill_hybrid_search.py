"""Integration tests for hybrid search (dense + BM25) pattern (Skill 18).

Tests:
  - HybridSearch.index() adds documents
  - HybridSearch.search() returns list of dicts with id, text, score
  - search returns top_k results at most
  - BM25 scores exact keyword matches higher than unrelated docs
  - cosine similarity returns 1.0 for identical vectors
  - cosine similarity returns 0.0 for orthogonal vectors
  - alpha=1.0 uses only dense score
  - alpha=0.0 uses only sparse score
  - hybrid score is weighted combination of dense and sparse
  - most relevant document is returned first
  - empty corpus returns empty results
  - results are sorted by score descending

NOTE: No from __future__ import annotations.
"""

import math
from collections import Counter

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, module, post
from lauren.testing import TestClient


# ---------------------------------------------------------------------------
# HybridSearch implementation under test
# ---------------------------------------------------------------------------


class HybridSearch:
    def __init__(self):
        self._docs: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}

    def index(self, doc_id: str, text: str, embedding: list[float]) -> None:
        self._docs[doc_id] = text
        self._embeddings[doc_id] = embedding

    def _bm25_score(self, query: str, doc_text: str) -> float:
        query_terms = query.lower().split()
        doc_terms = doc_text.lower().split()
        doc_len = len(doc_terms)
        term_freq = Counter(doc_terms)
        avg_len = sum(len(d.split()) for d in self._docs.values()) / max(len(self._docs), 1)
        k1, b = 1.5, 0.75
        score = 0.0
        for term in query_terms:
            tf = term_freq.get(term, 0)
            idf = math.log(
                (len(self._docs) + 1)
                / (sum(1 for d in self._docs.values() if term in d.lower()) + 1)
                + 1
            )
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        return score

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 3,
        alpha: float = 0.5,
    ) -> list[dict]:
        results = []
        for doc_id, text in self._docs.items():
            dense_score = self._cosine_similarity(query_embedding, self._embeddings[doc_id])
            sparse_score = self._bm25_score(query, text)
            hybrid_score = alpha * dense_score + (1 - alpha) * min(sparse_score / 10.0, 1.0)
            results.append({"id": doc_id, "text": text, "score": hybrid_score})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Simple embedding helpers for tests
# ---------------------------------------------------------------------------


VOCAB = ["python", "javascript", "programming", "language", "web", "data", "science", "browser"]


def _make_embedding(words: list[str], vocab: list[str]) -> list[float]:
    """Create a simple bag-of-words embedding over a fixed vocabulary."""
    return [1.0 if w in words else 0.0 for w in vocab]


def _embed(text: str) -> list[float]:
    words = text.lower().split()
    return _make_embedding(words, VOCAB)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class _IndexRequest(BaseModel):
    doc_id: str
    text: str
    embedding: list[float]


class _SearchRequest(BaseModel):
    docs: list[dict]  # list of {doc_id, text, embedding}
    query: str
    query_embedding: list[float]
    top_k: int = 3
    alpha: float = 0.5


class _CosineRequest(BaseModel):
    a: list[float]
    b: list[float]


class _BM25Request(BaseModel):
    docs: list[dict]  # list of {doc_id, text, embedding}
    query: str
    doc_text: str


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


@controller("/hybrid")
class HybridSearchController:
    @post("/index-check")
    async def index_check(self, body: Json[_IndexRequest]) -> dict:
        hs = HybridSearch()
        hs.index(body.doc_id, body.text, body.embedding)
        return {"indexed": body.doc_id in hs._docs}

    @post("/index-multiple")
    async def index_multiple(self, body: Json[dict]) -> dict:
        hs = HybridSearch()
        for doc in body.get("docs", []):
            hs.index(doc["doc_id"], doc["text"], doc["embedding"])
        return {"count": len(hs._docs)}

    @post("/search-empty")
    async def search_empty(self, body: Json[dict]) -> dict:
        hs = HybridSearch()
        results = hs.search("anything", body.get("query_embedding", [0.0] * len(VOCAB)))
        return {"results": results}

    @post("/search")
    async def search(self, body: Json[_SearchRequest]) -> dict:
        hs = HybridSearch()
        for doc in body.docs:
            hs.index(doc["doc_id"], doc["text"], doc["embedding"])
        results = hs.search(body.query, body.query_embedding, top_k=body.top_k, alpha=body.alpha)
        return {
            "results": results,
            "count": len(results),
            "first_id": results[0]["id"] if results else None,
        }

    @post("/cosine")
    async def cosine(self, body: Json[_CosineRequest]) -> dict:
        hs = HybridSearch()
        score = hs._cosine_similarity(body.a, body.b)
        return {"score": score}

    @post("/bm25")
    async def bm25(self, body: Json[_BM25Request]) -> dict:
        hs = HybridSearch()
        for doc in body.docs:
            hs.index(doc["doc_id"], doc["text"], doc["embedding"])
        score = hs._bm25_score(body.query, body.doc_text)
        return {"score": score}

    @post("/bm25-compare")
    async def bm25_compare(self, body: Json[dict]) -> dict:
        docs = body.get("docs", [])
        query = body.get("query", "")
        match_text = body.get("match_text", "")
        nomatch_text = body.get("nomatch_text", "")
        hs = HybridSearch()
        for doc in docs:
            hs.index(doc["doc_id"], doc["text"], doc["embedding"])
        match_score = hs._bm25_score(query, match_text)
        nomatch_score = hs._bm25_score(query, nomatch_text)
        return {"match_score": match_score, "nomatch_score": nomatch_score, "match_gt": match_score > nomatch_score}

    @post("/search-nonnegative")
    async def search_nonnegative(self, body: Json[dict]) -> dict:
        hs = HybridSearch()
        hs.index("d1", "test document", _embed("test document"))
        results = hs.search("test", _embed("test"), alpha=0.5)
        return {"all_nonneg": all(r["score"] >= 0.0 for r in results)}


@module(controllers=[HybridSearchController])
class HybridSearchModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(HybridSearchModule))


# Helper to build doc dicts
def _doc(doc_id: str, text: str) -> dict:
    return {"doc_id": doc_id, "text": text, "embedding": _embed(text)}


# ---------------------------------------------------------------------------
# Tests: HybridSearch indexing
# ---------------------------------------------------------------------------


class TestHybridSearchIndex:
    def test_index_adds_document(self):
        client = build_app()
        r = client.post(
            "/hybrid/index-check",
            json={"doc_id": "d1", "text": "Python is a language", "embedding": _embed("Python is a language")},
        )
        assert r.status_code == 200
        assert r.json()["indexed"] is True

    def test_index_multiple_documents(self):
        client = build_app()
        r = client.post(
            "/hybrid/index-multiple",
            json={
                "docs": [
                    _doc("d1", "Python programming"),
                    _doc("d2", "JavaScript web"),
                ]
            },
        )
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_empty_corpus_search_returns_empty(self):
        client = build_app()
        r = client.post(
            "/hybrid/search-empty",
            json={"query_embedding": _embed("anything")},
        )
        assert r.status_code == 200
        assert r.json()["results"] == []


# ---------------------------------------------------------------------------
# Tests: cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        client = build_app()
        vec = [1.0, 0.0, 0.0]
        r = client.post("/hybrid/cosine", json={"a": vec, "b": vec})
        assert r.status_code == 200
        assert abs(r.json()["score"] - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        client = build_app()
        r = client.post("/hybrid/cosine", json={"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]})
        assert r.status_code == 200
        assert abs(r.json()["score"]) < 1e-6

    def test_zero_vector_returns_zero(self):
        client = build_app()
        r = client.post("/hybrid/cosine", json={"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]})
        assert r.status_code == 200
        assert r.json()["score"] == 0.0


# ---------------------------------------------------------------------------
# Tests: BM25 scoring
# ---------------------------------------------------------------------------


class TestBM25Scoring:
    def test_exact_keyword_match_has_positive_score(self):
        client = build_app()
        r = client.post(
            "/hybrid/bm25-compare",
            json={
                "docs": [
                    _doc("d1", "Python is a programming language"),
                    _doc("d2", "cats are fluffy animals"),
                ],
                "query": "python",
                "match_text": "Python is a programming language",
                "nomatch_text": "cats are fluffy animals",
            },
        )
        assert r.status_code == 200
        assert r.json()["match_gt"] is True

    def test_no_keyword_overlap_returns_zero_or_near(self):
        client = build_app()
        r = client.post(
            "/hybrid/bm25",
            json={
                "docs": [_doc("d1", "cats and dogs are pets")],
                "query": "programming",
                "doc_text": "cats and dogs are pets",
            },
        )
        assert r.status_code == 200
        assert r.json()["score"] == 0.0


# ---------------------------------------------------------------------------
# Tests: hybrid search results
# ---------------------------------------------------------------------------


class TestHybridSearchResults:
    def test_search_returns_list_of_dicts(self):
        client = build_app()
        r = client.post(
            "/hybrid/search",
            json={
                "docs": [_doc("d1", "Python programming")],
                "query": "python",
                "query_embedding": _embed("python"),
                "top_k": 3,
                "alpha": 0.5,
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        assert isinstance(results, list)
        assert all(isinstance(item, dict) for item in results)

    def test_search_result_has_required_keys(self):
        client = build_app()
        r = client.post(
            "/hybrid/search",
            json={
                "docs": [_doc("d1", "Python programming")],
                "query": "python",
                "query_embedding": _embed("python"),
                "top_k": 3,
                "alpha": 0.5,
            },
        )
        assert r.status_code == 200
        result = r.json()["results"][0]
        assert "id" in result
        assert "text" in result
        assert "score" in result

    def test_search_top_k_limits_results(self):
        client = build_app()
        docs = [_doc(f"d{i}", f"document {i} python programming") for i in range(5)]
        r = client.post(
            "/hybrid/search",
            json={
                "docs": docs,
                "query": "python programming",
                "query_embedding": _embed("python programming"),
                "top_k": 2,
                "alpha": 0.5,
            },
        )
        assert r.status_code == 200
        assert r.json()["count"] <= 2

    def test_search_results_sorted_descending_by_score(self):
        client = build_app()
        r = client.post(
            "/hybrid/search",
            json={
                "docs": [
                    _doc("py", "Python programming language"),
                    _doc("cat", "cats are fluffy animals that meow"),
                ],
                "query": "python programming",
                "query_embedding": _embed("Python programming"),
                "top_k": 2,
                "alpha": 0.5,
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_alpha_one_uses_only_dense_score(self):
        client = build_app()
        r = client.post(
            "/hybrid/search",
            json={
                "docs": [
                    _doc("py", "Python programming language"),
                    _doc("cat", "cats are fluffy animals"),
                ],
                "query": "python",
                "query_embedding": _embed("Python programming"),
                "top_k": 2,
                "alpha": 1.0,
            },
        )
        assert r.status_code == 200
        assert r.json()["first_id"] == "py"

    def test_alpha_zero_uses_only_sparse_score(self):
        client = build_app()
        zero_emb = [0.0] * len(VOCAB)
        r = client.post(
            "/hybrid/search",
            json={
                "docs": [
                    {"doc_id": "py", "text": "Python is a programming language", "embedding": zero_emb},
                    {"doc_id": "cat", "text": "cats are fluffy animals that meow", "embedding": zero_emb},
                ],
                "query": "python",
                "query_embedding": zero_emb,
                "top_k": 2,
                "alpha": 0.0,
            },
        )
        assert r.status_code == 200
        assert r.json()["first_id"] == "py"

    def test_hybrid_score_is_nonnegative(self):
        client = build_app()
        r = client.post("/hybrid/search-nonnegative", json={})
        assert r.status_code == 200
        assert r.json()["all_nonneg"] is True

    def test_most_relevant_doc_ranked_first(self):
        client = build_app()
        r = client.post(
            "/hybrid/search",
            json={
                "docs": [
                    _doc("programming", "Python is a high-level programming language"),
                    _doc("animals", "Cats are fluffy animals that meow"),
                    _doc("web", "JavaScript is used for web development"),
                ],
                "query": "python programming language",
                "query_embedding": _embed("Python programming"),
                "top_k": 3,
                "alpha": 0.5,
            },
        )
        assert r.status_code == 200
        assert r.json()["first_id"] == "programming"

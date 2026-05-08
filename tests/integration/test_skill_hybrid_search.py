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
"""

import math
from collections import Counter

import pytest


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


def _make_embedding(words: list[str], vocab: list[str]) -> list[float]:
    """Create a simple bag-of-words embedding over a fixed vocabulary."""
    return [1.0 if w in words else 0.0 for w in vocab]


VOCAB = ["python", "javascript", "programming", "language", "web", "data", "science", "browser"]


def _embed(text: str) -> list[float]:
    words = text.lower().split()
    return _make_embedding(words, VOCAB)


# ---------------------------------------------------------------------------
# Tests: HybridSearch indexing
# ---------------------------------------------------------------------------


class TestHybridSearchIndex:
    def test_index_adds_document(self):
        hs = HybridSearch()
        hs.index("d1", "Python is a language", _embed("Python is a language"))
        assert "d1" in hs._docs

    def test_index_multiple_documents(self):
        hs = HybridSearch()
        hs.index("d1", "Python programming", _embed("Python programming"))
        hs.index("d2", "JavaScript web", _embed("JavaScript web"))
        assert len(hs._docs) == 2

    def test_empty_corpus_search_returns_empty(self):
        hs = HybridSearch()
        results = hs.search("anything", _embed("anything"))
        assert results == []


# ---------------------------------------------------------------------------
# Tests: cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        hs = HybridSearch()
        vec = [1.0, 0.0, 0.0]
        assert abs(hs._cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        hs = HybridSearch()
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(hs._cosine_similarity(a, b)) < 1e-6

    def test_zero_vector_returns_zero(self):
        hs = HybridSearch()
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert hs._cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Tests: BM25 scoring
# ---------------------------------------------------------------------------


class TestBM25Scoring:
    def test_exact_keyword_match_has_positive_score(self):
        hs = HybridSearch()
        hs.index("d1", "Python is a programming language", _embed("Python programming language"))
        hs.index("d2", "cats are fluffy animals", _embed("cats fluffy animals"))
        score_match = hs._bm25_score("python", "Python is a programming language")
        score_nomatch = hs._bm25_score("python", "cats are fluffy animals")
        assert score_match > score_nomatch

    def test_no_keyword_overlap_returns_zero_or_near(self):
        hs = HybridSearch()
        hs.index("d1", "cats and dogs are pets", _embed("cats dogs pets"))
        score = hs._bm25_score("programming", "cats and dogs are pets")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: hybrid search results
# ---------------------------------------------------------------------------


class TestHybridSearchResults:
    def test_search_returns_list_of_dicts(self):
        hs = HybridSearch()
        hs.index("d1", "Python programming", _embed("Python programming"))
        results = hs.search("python", _embed("python"), top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_search_result_has_required_keys(self):
        hs = HybridSearch()
        hs.index("d1", "Python programming", _embed("Python programming"))
        results = hs.search("python", _embed("python"))
        assert "id" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]

    def test_search_top_k_limits_results(self):
        hs = HybridSearch()
        for i in range(5):
            hs.index(f"d{i}", f"document {i} python programming", _embed("python programming"))
        results = hs.search("python programming", _embed("python programming"), top_k=2)
        assert len(results) <= 2

    def test_search_results_sorted_descending_by_score(self):
        hs = HybridSearch()
        hs.index("py", "Python programming language", _embed("Python programming language"))
        hs.index("cat", "cats are fluffy animals that meow", _embed("cats fluffy animals"))
        results = hs.search("python programming", _embed("Python programming"), top_k=2, alpha=0.5)
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_alpha_one_uses_only_dense_score(self):
        hs = HybridSearch()
        py_emb = _embed("Python programming language")
        cat_emb = _embed("cats fluffy animals meow")
        hs.index("py", "Python programming language", py_emb)
        hs.index("cat", "cats are fluffy animals", cat_emb)
        query_emb = _embed("Python programming")
        results_dense = hs.search("python", query_emb, top_k=2, alpha=1.0)
        # With alpha=1.0, only dense (cosine) score matters
        assert results_dense[0]["id"] == "py"

    def test_alpha_zero_uses_only_sparse_score(self):
        hs = HybridSearch()
        py_emb = [0.0] * len(VOCAB)  # zero embedding — cosine = 0 for all
        cat_emb = [0.0] * len(VOCAB)
        hs.index("py", "Python is a programming language", py_emb)
        hs.index("cat", "cats are fluffy animals that meow", cat_emb)
        results = hs.search("python", [0.0] * len(VOCAB), top_k=2, alpha=0.0)
        # With alpha=0, only BM25 score matters — "python" matches "py" doc
        assert results[0]["id"] == "py"

    def test_hybrid_score_is_nonnegative(self):
        hs = HybridSearch()
        hs.index("d1", "test document", _embed("test document"))
        results = hs.search("test", _embed("test"), alpha=0.5)
        assert all(r["score"] >= 0.0 for r in results)

    def test_most_relevant_doc_ranked_first(self):
        hs = HybridSearch()
        hs.index(
            "programming",
            "Python is a high-level programming language",
            _embed("Python programming language"),
        )
        hs.index("animals", "Cats are fluffy animals that meow", _embed("cats fluffy animals"))
        hs.index("web", "JavaScript is used for web development", _embed("JavaScript web"))

        query_emb = _embed("Python programming")
        results = hs.search("python programming language", query_emb, top_k=3, alpha=0.5)
        assert results[0]["id"] == "programming"

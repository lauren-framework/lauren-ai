"""Integration tests for document re-ranking (Skill 19).

Tests:
  - DocumentReranker.rerank() returns sorted list with most relevant first
  - rerank with top_k limits results
  - rerank with top_k=None returns all results
  - simple_overlap score is higher for matching terms
  - simple_overlap score is 0 for no overlap
  - custom score_fn is used when provided
  - rerank preserves doc_id and text fields
  - rerank of single result returns same result
  - rerank of empty list returns empty list
  - results are sorted descending by re-rank score

NOTE: No from __future__ import annotations.
"""

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, module, post
from lauren.testing import TestClient


# ---------------------------------------------------------------------------
# SearchResult dataclass + DocumentReranker implementation under test
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float


class DocumentReranker:
    def __init__(self, score_fn: Callable[[str, str], float] | None = None):
        self._score_fn = score_fn or self._simple_overlap

    def _simple_overlap(self, query: str, doc: str) -> float:
        q_words = set(query.lower().split())
        d_words = set(doc.lower().split())
        return len(q_words & d_words) / max(len(q_words), 1)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        reranked = sorted(
            results,
            key=lambda r: self._score_fn(query, r.text),
            reverse=True,
        )
        return reranked[:top_k] if top_k is not None else reranked


def _make_results() -> list[SearchResult]:
    return [
        SearchResult("cat", "cats are fluffy animals that meow at night", 0.3),
        SearchResult("py", "Python is a high-level programming language for data science", 0.5),
        SearchResult("js", "JavaScript is used for web development in browsers", 0.4),
    ]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class _RerankRequest(BaseModel):
    query: str
    documents: list[dict]  # list of {doc_id, text, score}
    top_k: int | None = None
    score_fn: str = "overlap"  # "overlap" or "length" or "short_first"


class _OverlapRequest(BaseModel):
    query: str
    doc: str


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


def _build_results(docs: list[dict]) -> list[SearchResult]:
    return [SearchResult(d["doc_id"], d["text"], d["score"]) for d in docs]


@controller("/rerank")
class RerankController:
    @post("/rerank")
    async def rerank(self, body: Json[_RerankRequest]) -> dict:
        if body.score_fn == "length":

            def score_fn(query: str, doc: str) -> float:
                return float(len(doc))

            reranker = DocumentReranker(score_fn=score_fn)
        elif body.score_fn == "short_first":

            def score_fn(query: str, doc: str) -> float:
                return 1.0 / (len(doc) + 1)

            reranker = DocumentReranker(score_fn=score_fn)
        else:
            reranker = DocumentReranker()

        results = _build_results(body.documents)
        reranked = reranker.rerank(body.query, results, top_k=body.top_k)
        return {
            "count": len(reranked),
            "doc_ids": [r.doc_id for r in reranked],
            "scores": [reranker._score_fn(body.query, r.text) for r in reranked],
        }

    @post("/overlap")
    async def overlap(self, body: Json[_OverlapRequest]) -> dict:
        reranker = DocumentReranker()
        score = reranker._simple_overlap(body.query, body.doc)
        return {"score": score}

    @post("/rerank-empty")
    async def rerank_empty(self, body: Json[dict]) -> dict:
        reranker = DocumentReranker()
        result = reranker.rerank(body.get("query", "query"), [])
        return {"result": result}

    @post("/rerank-single")
    async def rerank_single(self, body: Json[dict]) -> dict:
        reranker = DocumentReranker()
        single = [SearchResult("d1", "Python is great", 0.9)]
        reranked = reranker.rerank(body.get("query", "Python"), single)
        return {"count": len(reranked), "first_id": reranked[0].doc_id if reranked else None}

    @post("/check-sorted")
    async def check_sorted(self, body: Json[dict]) -> dict:
        reranker = DocumentReranker()
        results = _make_results()
        query = body.get("query", "Python programming language data science")
        reranked = reranker.rerank(query, results)
        scores = [reranker._score_fn(query, r.text) for r in reranked]
        return {
            "sorted_desc": scores == sorted(scores, reverse=True),
            "doc_ids": [r.doc_id for r in reranked],
        }

    @post("/check-preserves-fields")
    async def check_preserves_fields(self, body: Json[dict]) -> dict:
        reranker = DocumentReranker()
        results = _make_results()
        reranked = reranker.rerank("python", results)
        doc_ids = {r.doc_id for r in reranked}
        return {
            "all_present": doc_ids == {"cat", "py", "js"},
            "count": len(reranked),
        }

    @post("/custom-score-called")
    async def custom_score_called(self) -> dict:
        call_log: list = []

        def custom_score(query: str, doc: str) -> float:
            call_log.append((query, doc))
            return float(len(doc))

        reranker = DocumentReranker(score_fn=custom_score)
        results = _make_results()
        reranker.rerank("anything", results)
        return {"call_count": len(call_log), "expected": len(results)}


@module(controllers=[RerankController])
class RerankingModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(RerankingModule))


_DEFAULT_DOCS = [
    {"doc_id": "cat", "text": "cats are fluffy animals that meow at night", "score": 0.3},
    {"doc_id": "py", "text": "Python is a high-level programming language for data science", "score": 0.5},
    {"doc_id": "js", "text": "JavaScript is used for web development in browsers", "score": 0.4},
]


# ---------------------------------------------------------------------------
# Tests: DocumentReranker
# ---------------------------------------------------------------------------


class TestDocumentReranker:
    def test_rerank_returns_list(self):
        client = build_app()
        r = client.post(
            "/rerank/rerank",
            json={"query": "python programming", "documents": _DEFAULT_DOCS},
        )
        assert r.status_code == 200
        assert isinstance(r.json()["doc_ids"], list)

    def test_rerank_most_relevant_first(self):
        client = build_app()
        r = client.post(
            "/rerank/rerank",
            json={"query": "python programming language", "documents": _DEFAULT_DOCS},
        )
        assert r.status_code == 200
        assert r.json()["doc_ids"][0] == "py"

    def test_rerank_with_top_k_limits_results(self):
        client = build_app()
        r = client.post(
            "/rerank/rerank",
            json={"query": "programming", "documents": _DEFAULT_DOCS, "top_k": 2},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_rerank_with_top_k_none_returns_all(self):
        client = build_app()
        r = client.post(
            "/rerank/rerank",
            json={"query": "programming", "documents": _DEFAULT_DOCS, "top_k": None},
        )
        assert r.status_code == 200
        assert r.json()["count"] == len(_DEFAULT_DOCS)

    def test_rerank_empty_list_returns_empty(self):
        client = build_app()
        r = client.post("/rerank/rerank-empty", json={"query": "query"})
        assert r.status_code == 200
        assert r.json()["result"] == []

    def test_rerank_single_result_returns_same(self):
        client = build_app()
        r = client.post("/rerank/rerank-single", json={"query": "Python"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["first_id"] == "d1"

    def test_rerank_preserves_doc_id_and_text(self):
        client = build_app()
        r = client.post("/rerank/check-preserves-fields", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["all_present"] is True
        assert data["count"] == 3

    def test_rerank_results_sorted_descending(self):
        client = build_app()
        r = client.post(
            "/rerank/check-sorted",
            json={"query": "Python programming language data science"},
        )
        assert r.status_code == 200
        assert r.json()["sorted_desc"] is True

    def test_custom_score_fn_is_used(self):
        client = build_app()
        r = client.post("/rerank/custom-score-called", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["call_count"] == data["expected"]

    def test_custom_score_fn_affects_ranking(self):
        client = build_app()
        docs = [
            {"doc_id": "long", "text": "A very long document with lots and lots of words in it", "score": 0.5},
            {"doc_id": "short", "text": "Brief.", "score": 0.3},
        ]
        r = client.post(
            "/rerank/rerank",
            json={"query": "anything", "documents": docs, "score_fn": "short_first"},
        )
        assert r.status_code == 200
        assert r.json()["doc_ids"][0] == "short"


# ---------------------------------------------------------------------------
# Tests: simple_overlap scoring
# ---------------------------------------------------------------------------


class TestSimpleOverlapScore:
    def test_exact_match_returns_one(self):
        client = build_app()
        r = client.post("/rerank/overlap", json={"query": "python", "doc": "python"})
        assert r.status_code == 200
        assert r.json()["score"] == 1.0

    def test_no_overlap_returns_zero(self):
        client = build_app()
        r = client.post(
            "/rerank/overlap", json={"query": "python", "doc": "cats are fluffy"}
        )
        assert r.status_code == 200
        assert r.json()["score"] == 0.0

    def test_partial_overlap_returns_fraction(self):
        client = build_app()
        r = client.post(
            "/rerank/overlap",
            json={"query": "python programming", "doc": "python is great"},
        )
        assert r.status_code == 200
        score = r.json()["score"]
        assert 0.0 < score < 1.0

    def test_case_insensitive_matching(self):
        client = build_app()
        r1 = client.post("/rerank/overlap", json={"query": "python", "doc": "Python is great"})
        r2 = client.post("/rerank/overlap", json={"query": "PYTHON", "doc": "Python is great"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["score"] > 0.0
        assert r2.json()["score"] > 0.0

    def test_all_query_words_match(self):
        client = build_app()
        r = client.post(
            "/rerank/overlap",
            json={"query": "python language", "doc": "Python is a language"},
        )
        assert r.status_code == 200
        assert r.json()["score"] == 1.0

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


_DEFAULT_DOCS = [
    SearchResult("cat", "cats are fluffy animals that meow at night", 0.3),
    SearchResult("py", "Python is a high-level programming language for data science", 0.5),
    SearchResult("js", "JavaScript is used for web development in browsers", 0.4),
]


# ---------------------------------------------------------------------------
# Tests: DocumentReranker (direct Python)
# ---------------------------------------------------------------------------


class TestDocumentReranker:
    def test_rerank_returns_list(self):
        reranker = DocumentReranker()
        reranked = reranker.rerank("python programming", _DEFAULT_DOCS)
        assert isinstance(reranked, list)

    def test_rerank_most_relevant_first(self):
        reranker = DocumentReranker()
        reranked = reranker.rerank("python programming language", _DEFAULT_DOCS)
        assert reranked[0].doc_id == "py"

    def test_rerank_with_top_k_limits_results(self):
        reranker = DocumentReranker()
        reranked = reranker.rerank("programming", _DEFAULT_DOCS, top_k=2)
        assert len(reranked) == 2

    def test_rerank_with_top_k_none_returns_all(self):
        reranker = DocumentReranker()
        reranked = reranker.rerank("programming", _DEFAULT_DOCS, top_k=None)
        assert len(reranked) == len(_DEFAULT_DOCS)

    def test_rerank_empty_list_returns_empty(self):
        reranker = DocumentReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_single_result_returns_same(self):
        reranker = DocumentReranker()
        single = [SearchResult("d1", "Python is great", 0.9)]
        reranked = reranker.rerank("Python", single)
        assert len(reranked) == 1
        assert reranked[0].doc_id == "d1"

    def test_rerank_preserves_doc_id_and_text(self):
        reranker = DocumentReranker()
        reranked = reranker.rerank("python", _make_results())
        doc_ids = {r.doc_id for r in reranked}
        assert doc_ids == {"cat", "py", "js"}
        assert len(reranked) == 3

    def test_rerank_results_sorted_descending(self):
        reranker = DocumentReranker()
        results = _make_results()
        query = "Python programming language data science"
        reranked = reranker.rerank(query, results)
        scores = [reranker._score_fn(query, r.text) for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_custom_score_fn_is_used(self):
        call_log: list = []

        def custom_score(query: str, doc: str) -> float:
            call_log.append((query, doc))
            return float(len(doc))

        reranker = DocumentReranker(score_fn=custom_score)
        results = _make_results()
        reranker.rerank("anything", results)
        assert len(call_log) == len(results)

    def test_custom_score_fn_affects_ranking(self):
        docs = [
            SearchResult("long", "A very long document with lots and lots of words in it", 0.5),
            SearchResult("short", "Brief.", 0.3),
        ]

        def short_first(query: str, doc: str) -> float:
            return 1.0 / (len(doc) + 1)

        reranker = DocumentReranker(score_fn=short_first)
        reranked = reranker.rerank("anything", docs)
        assert reranked[0].doc_id == "short"


# ---------------------------------------------------------------------------
# Tests: simple_overlap scoring (direct Python)
# ---------------------------------------------------------------------------


class TestSimpleOverlapScore:
    def test_exact_match_returns_one(self):
        reranker = DocumentReranker()
        score = reranker._simple_overlap("python", "python")
        assert score == 1.0

    def test_no_overlap_returns_zero(self):
        reranker = DocumentReranker()
        score = reranker._simple_overlap("python", "cats are fluffy")
        assert score == 0.0

    def test_partial_overlap_returns_fraction(self):
        reranker = DocumentReranker()
        score = reranker._simple_overlap("python programming", "python is great")
        assert 0.0 < score < 1.0

    def test_case_insensitive_matching(self):
        reranker = DocumentReranker()
        score1 = reranker._simple_overlap("python", "Python is great")
        score2 = reranker._simple_overlap("PYTHON", "Python is great")
        assert score1 > 0.0
        assert score2 > 0.0

    def test_all_query_words_match(self):
        reranker = DocumentReranker()
        score = reranker._simple_overlap("python language", "Python is a language")
        assert score == 1.0

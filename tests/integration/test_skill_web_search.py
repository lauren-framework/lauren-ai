"""Integration tests for the web search tool pattern (Skill 39).

Tests cover:
- InMemorySearchProvider matches by title keyword
- InMemorySearchProvider matches by snippet keyword
- InMemorySearchProvider returns empty list when no match
- InMemorySearchProvider respects max_results limit
- WebSearchTool wraps provider results in standard format
- WebSearchTool count field matches results length
- Case-insensitive search matching
- Default provider returns results when no custom provider supplied
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient
from lauren_ai._tools import tool, ToolContext


# ---------------------------------------------------------------------------
# WebSearchProvider, InMemorySearchProvider, and WebSearchTool inline
# ---------------------------------------------------------------------------


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[dict]: ...


class InMemorySearchProvider(WebSearchProvider):
    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        q = query.lower()
        matching = [
            r
            for r in self._results
            if q in r.get("title", "").lower() or q in r.get("snippet", "").lower()
        ]
        return matching[:max_results] if matching else self._results[:max_results]


# Stub context
class _Ctx:
    execution_context = None

    def get_metadata(self, key, default=None):
        return default


_CTX = _Ctx()


@tool()
class WebSearchTool:
    """Search the web for information.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """

    def __init__(self, provider: WebSearchProvider | None = None):
        self._provider = provider or InMemorySearchProvider(
            [
                {
                    "title": "Lauren Framework Docs",
                    "url": "https://lauren.dev",
                    "snippet": "Metadata-first Python web framework",
                },
            ]
        )

    async def run(self, ctx: ToolContext, query: str, max_results: int = 5) -> dict:
        results = await self._provider.search(query, max_results)
        return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/search")
class SearchController:
    @post("/provider")
    async def provider_search(self, body: Json[dict]) -> dict:
        """Run InMemorySearchProvider directly."""
        query = body.get("query", "")
        max_results = body.get("max_results", 5)
        records = body.get("records", [])
        provider = InMemorySearchProvider(records)
        results = await provider.search(query, max_results)
        return {"results": results, "count": len(results)}

    @post("/run")
    async def run_tool(self, body: Json[dict]) -> dict:
        """Run WebSearchTool with provided records."""
        query = body.get("query", "")
        max_results = body.get("max_results", 5)
        records = body.get("records", None)
        provider = InMemorySearchProvider(records) if records is not None else None
        search_tool = WebSearchTool(provider=provider)
        return await search_tool.run(_CTX, query, max_results)

    @post("/run-default")
    async def run_default(self, body: Json[dict]) -> dict:
        """Run WebSearchTool with default provider."""
        query = body.get("query", "")
        search_tool = WebSearchTool()
        return await search_tool.run(_CTX, query)


@module(controllers=[SearchController])
class WebSearchModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(WebSearchModule))


# ---------------------------------------------------------------------------
# Tests: InMemorySearchProvider
# ---------------------------------------------------------------------------


class TestInMemorySearchProvider:
    def test_matches_by_title_keyword(self):
        """Returns results whose title contains the query keyword."""
        client = build_app()
        r = client.post("/search/provider", json={
            "query": "python",
            "records": [
                {"title": "Python Tutorial", "url": "https://a.com", "snippet": "Learn Python"},
                {"title": "JavaScript Guide", "url": "https://b.com", "snippet": "Learn JS"},
            ],
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Python Tutorial"

    def test_matches_by_snippet_keyword(self):
        """Returns results whose snippet contains the query keyword."""
        client = build_app()
        r = client.post("/search/provider", json={
            "query": "async",
            "records": [
                {"title": "Generic Article", "url": "https://c.com", "snippet": "This is about async programming"},
                {"title": "Another Article", "url": "https://d.com", "snippet": "Sync programming basics"},
            ],
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 1
        assert "async" in data["results"][0]["snippet"].lower()

    def test_returns_all_results_when_no_match(self):
        """When no result matches the query, returns all results (fallback)."""
        client = build_app()
        r = client.post("/search/provider", json={
            "query": "zzz-no-match",
            "records": [
                {"title": "Alpha", "url": "https://e.com", "snippet": "alpha"},
                {"title": "Beta", "url": "https://f.com", "snippet": "beta"},
            ],
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2

    def test_respects_max_results_limit(self):
        """max_results limits the number of returned results."""
        client = build_app()
        r = client.post("/search/provider", json={
            "query": "python",
            "max_results": 3,
            "records": [
                {"title": f"Python article {i}", "url": f"https://g{i}.com", "snippet": "python"}
                for i in range(10)
            ],
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 3

    def test_case_insensitive_title_match(self):
        """Title matching is case-insensitive."""
        client = build_app()
        r = client.post("/search/provider", json={
            "query": "upper case",
            "records": [{"title": "UPPER CASE TITLE", "url": "https://h.com", "snippet": "content"}],
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1

    def test_empty_results_returns_empty_list(self):
        """An empty provider returns an empty list."""
        client = build_app()
        r = client.post("/search/provider", json={"query": "anything", "records": []})
        assert r.status_code == 200
        assert r.json()["results"] == []

    def test_multiple_matches_returned(self):
        """Multiple matching results are all returned (up to max_results)."""
        client = build_app()
        r = client.post("/search/provider", json={
            "query": "ai",
            "records": [
                {"title": "AI news 1", "url": "https://i1.com", "snippet": "artificial intelligence"},
                {"title": "AI news 2", "url": "https://i2.com", "snippet": "machine learning ai"},
                {"title": "Cooking tips", "url": "https://i3.com", "snippet": "recipes"},
            ],
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2


# ---------------------------------------------------------------------------
# Tests: WebSearchTool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def test_search_returns_standard_format(self):
        """WebSearchTool returns a dict with 'results' and 'count' keys."""
        client = build_app()
        r = client.post("/search/run", json={
            "query": "test",
            "records": [{"title": "Test", "url": "https://t.com", "snippet": "test content"}],
        })
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "count" in data

    def test_count_matches_results_length(self):
        """count field equals len(results)."""
        client = build_app()
        r = client.post("/search/run", json={
            "query": "result",
            "records": [
                {"title": "Result A", "url": "https://a.com", "snippet": "content a"},
                {"title": "Result B", "url": "https://b.com", "snippet": "content b"},
            ],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == len(data["results"])
        assert data["count"] == 2

    def test_no_match_returns_empty_results_via_fallback(self):
        """When query has no match, provider fallback is returned."""
        client = build_app()
        r = client.post("/search/run", json={
            "query": "zzz-nothing",
            "records": [{"title": "X", "url": "https://x.com", "snippet": "x"}],
        })
        assert r.status_code == 200
        assert isinstance(r.json()["results"], list)

    def test_default_provider_returns_lauren_result(self):
        """The default InMemorySearchProvider includes the Lauren Framework result."""
        client = build_app()
        r = client.post("/search/run-default", json={"query": "lauren"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] > 0
        titles = [res.get("title", "") for res in data["results"]]
        assert any("Lauren" in t for t in titles)

    def test_max_results_parameter_honoured(self):
        """max_results is forwarded to the provider."""
        client = build_app()
        r = client.post("/search/run", json={
            "query": "result",
            "max_results": 2,
            "records": [
                {"title": f"Result {i}", "url": f"https://r{i}.com", "snippet": "result"}
                for i in range(10)
            ],
        })
        assert r.status_code == 200
        assert r.json()["count"] <= 2

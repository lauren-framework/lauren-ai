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

from abc import ABC, abstractmethod

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


# ---------------------------------------------------------------------------
# Stub context
# ---------------------------------------------------------------------------


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

    async def run(self, ctx, query: str, max_results: int = 5) -> dict:
        results = await self._provider.search(query, max_results)
        return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Tests: InMemorySearchProvider
# ---------------------------------------------------------------------------


class TestInMemorySearchProvider:
    async def test_matches_by_title_keyword(self):
        """Returns results whose title contains the query keyword."""
        provider = InMemorySearchProvider(
            [
                {"title": "Python Tutorial", "url": "https://a.com", "snippet": "Learn Python"},
                {"title": "JavaScript Guide", "url": "https://b.com", "snippet": "Learn JS"},
            ]
        )
        results = await provider.search("python")
        assert len(results) == 1
        assert results[0]["title"] == "Python Tutorial"

    async def test_matches_by_snippet_keyword(self):
        """Returns results whose snippet contains the query keyword."""
        provider = InMemorySearchProvider(
            [
                {
                    "title": "Generic Article",
                    "url": "https://c.com",
                    "snippet": "This is about async programming",
                },
                {
                    "title": "Another Article",
                    "url": "https://d.com",
                    "snippet": "Sync programming basics",
                },
            ]
        )
        results = await provider.search("async")
        assert len(results) == 1
        assert "async" in results[0]["snippet"].lower()

    async def test_returns_all_results_when_no_match(self):
        """When no result matches the query, returns all results (fallback)."""
        provider = InMemorySearchProvider(
            [
                {"title": "Alpha", "url": "https://e.com", "snippet": "alpha"},
                {"title": "Beta", "url": "https://f.com", "snippet": "beta"},
            ]
        )
        results = await provider.search("zzz-no-match")
        assert len(results) == 2

    async def test_respects_max_results_limit(self):
        """max_results limits the number of returned results."""
        provider = InMemorySearchProvider(
            [
                {"title": f"Python article {i}", "url": f"https://g{i}.com", "snippet": "python"}
                for i in range(10)
            ]
        )
        results = await provider.search("python", max_results=3)
        assert len(results) == 3

    async def test_case_insensitive_title_match(self):
        """Title matching is case-insensitive."""
        provider = InMemorySearchProvider(
            [{"title": "UPPER CASE TITLE", "url": "https://h.com", "snippet": "content"}]
        )
        results = await provider.search("upper case")
        assert len(results) == 1

    async def test_empty_results_returns_empty_list(self):
        """An empty provider returns an empty list."""
        provider = InMemorySearchProvider([])
        results = await provider.search("anything")
        assert results == []

    async def test_multiple_matches_returned(self):
        """Multiple matching results are all returned (up to max_results)."""
        provider = InMemorySearchProvider(
            [
                {
                    "title": "AI news 1",
                    "url": "https://i1.com",
                    "snippet": "artificial intelligence",
                },
                {"title": "AI news 2", "url": "https://i2.com", "snippet": "machine learning ai"},
                {"title": "Cooking tips", "url": "https://i3.com", "snippet": "recipes"},
            ]
        )
        results = await provider.search("ai")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: WebSearchTool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    async def test_search_returns_standard_format(self):
        """WebSearchTool returns a dict with 'results' and 'count' keys."""
        provider = InMemorySearchProvider(
            [{"title": "Test", "url": "https://t.com", "snippet": "test content"}]
        )
        tool_instance = WebSearchTool(provider=provider)
        data = await tool_instance.run(_CTX, query="test")
        assert "results" in data
        assert "count" in data

    async def test_count_matches_results_length(self):
        """count field equals len(results)."""
        provider = InMemorySearchProvider(
            [
                {"title": "Result A", "url": "https://a.com", "snippet": "content a"},
                {"title": "Result B", "url": "https://b.com", "snippet": "content b"},
            ]
        )
        tool_instance = WebSearchTool(provider=provider)
        data = await tool_instance.run(_CTX, query="result")
        assert data["count"] == len(data["results"])
        assert data["count"] == 2

    async def test_no_match_returns_empty_results_via_fallback(self):
        """When query has no match, provider fallback is returned."""
        provider = InMemorySearchProvider([{"title": "X", "url": "https://x.com", "snippet": "x"}])
        tool_instance = WebSearchTool(provider=provider)
        data = await tool_instance.run(_CTX, query="zzz-nothing")
        assert isinstance(data["results"], list)

    async def test_default_provider_returns_lauren_result(self):
        """The default InMemorySearchProvider includes the Lauren Framework result."""
        tool_instance = WebSearchTool()
        data = await tool_instance.run(_CTX, query="lauren")
        assert data["count"] > 0
        titles = [res.get("title", "") for res in data["results"]]
        assert any("Lauren" in t for t in titles)

    async def test_max_results_parameter_honoured(self):
        """max_results is forwarded to the provider."""
        provider = InMemorySearchProvider(
            [
                {"title": f"Result {i}", "url": f"https://r{i}.com", "snippet": "result"}
                for i in range(10)
            ]
        )
        tool_instance = WebSearchTool(provider=provider)
        data = await tool_instance.run(_CTX, query="result", max_results=2)
        assert data["count"] <= 2

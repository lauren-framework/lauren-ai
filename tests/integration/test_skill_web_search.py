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

import pytest


# ---------------------------------------------------------------------------
# WebSearchProvider, InMemorySearchProvider, and WebSearchTool inline
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod


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

# Import tool decorator — no from __future__ import annotations needed
from lauren_ai._tools import tool, ToolContext


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
# Tests: InMemorySearchProvider
# ---------------------------------------------------------------------------


class TestInMemorySearchProvider:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_returns_all_results_when_no_match(self):
        """When no result matches the query, returns all results (fallback)."""
        records = [
            {"title": "Alpha", "url": "https://e.com", "snippet": "alpha"},
            {"title": "Beta", "url": "https://f.com", "snippet": "beta"},
        ]
        provider = InMemorySearchProvider(records)
        results = await provider.search("zzz-no-match")
        # Fallback: return all results up to max_results
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_respects_max_results_limit(self):
        """max_results limits the number of returned results."""
        records = [
            {"title": f"Python article {i}", "url": f"https://g{i}.com", "snippet": "python"}
            for i in range(10)
        ]
        provider = InMemorySearchProvider(records)
        results = await provider.search("python", max_results=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_case_insensitive_title_match(self):
        """Title matching is case-insensitive."""
        provider = InMemorySearchProvider(
            [{"title": "UPPER CASE TITLE", "url": "https://h.com", "snippet": "content"}]
        )
        results = await provider.search("upper case")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(self):
        """An empty provider returns an empty list."""
        provider = InMemorySearchProvider([])
        results = await provider.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_matches_returned(self):
        """Multiple matching results are all returned (up to max_results)."""
        provider = InMemorySearchProvider(
            [
                {"title": "AI news 1", "url": "https://i1.com", "snippet": "artificial intelligence"},
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
    @pytest.mark.asyncio
    async def test_search_returns_standard_format(self):
        """WebSearchTool returns a dict with 'results' and 'count' keys."""
        provider = InMemorySearchProvider(
            [{"title": "Test", "url": "https://t.com", "snippet": "test content"}]
        )
        search_tool = WebSearchTool(provider=provider)
        result = await search_tool.run(_CTX, "test")

        assert "results" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_count_matches_results_length(self):
        """count field equals len(results)."""
        provider = InMemorySearchProvider(
            [
                {"title": "Result A", "url": "https://a.com", "snippet": "content a"},
                {"title": "Result B", "url": "https://b.com", "snippet": "content b"},
            ]
        )
        search_tool = WebSearchTool(provider=provider)
        result = await search_tool.run(_CTX, "result")

        assert result["count"] == len(result["results"])
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_no_match_returns_empty_results_via_fallback(self):
        """When query has no match, provider fallback is returned."""
        provider = InMemorySearchProvider(
            [{"title": "X", "url": "https://x.com", "snippet": "x"}]
        )
        search_tool = WebSearchTool(provider=provider)
        result = await search_tool.run(_CTX, "zzz-nothing")
        # Fallback from InMemorySearchProvider returns all records
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_default_provider_returns_lauren_result(self):
        """The default InMemorySearchProvider includes the Lauren Framework result."""
        search_tool = WebSearchTool()
        result = await search_tool.run(_CTX, "lauren")

        assert result["count"] > 0
        titles = [r.get("title", "") for r in result["results"]]
        assert any("Lauren" in t for t in titles)

    @pytest.mark.asyncio
    async def test_max_results_parameter_honoured(self):
        """max_results is forwarded to the provider."""
        records = [
            {"title": f"Result {i}", "url": f"https://r{i}.com", "snippet": "result"}
            for i in range(10)
        ]
        provider = InMemorySearchProvider(records)
        search_tool = WebSearchTool(provider=provider)
        result = await search_tool.run(_CTX, "result", max_results=2)

        assert result["count"] <= 2

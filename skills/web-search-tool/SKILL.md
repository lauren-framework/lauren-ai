---
name: web-search-tool
description: Abstract web search tool with an InMemorySearchProvider for tests and patterns for Tavily and SerpAPI in production. Use when an agent needs to retrieve current information from the web.
---

> Use `codemap find "WebSearchTool"` after adding the pattern to your project.

# Web Search Tool Integration

Define a `WebSearchProvider` ABC with an `InMemorySearchProvider` for tests
and concrete provider adapters for production.

## Abstract provider

```python
from abc import ABC, abstractmethod

class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return a list of result dicts with title, url, snippet keys."""
        ...
```

## In-memory provider (tests)

```python
class InMemorySearchProvider(WebSearchProvider):
    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        q = query.lower()
        matching = [
            r for r in self._results
            if q in r.get("title", "").lower() or q in r.get("snippet", "").lower()
        ]
        return matching[:max_results] if matching else self._results[:max_results]
```

## Tool

```python
from lauren_ai._tools import tool, ToolContext

@tool()
class WebSearchTool:
    """Search the web for information.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """

    def __init__(self, provider: WebSearchProvider | None = None):
        self._provider = provider or InMemorySearchProvider([
            {
                "title": "Lauren Framework Docs",
                "url": "https://lauren.dev",
                "snippet": "Metadata-first Python web framework",
            },
        ])

    async def run(self, ctx: ToolContext, query: str, max_results: int = 5) -> dict:
        results = await self._provider.search(query, max_results)
        return {"results": results, "count": len(results)}
```

## Tavily provider

```python
import httpx

class TavilySearchProvider(WebSearchProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self._api_key, "query": query, "max_results": max_results},
            )
            data = resp.json()
            return [
                {"title": r["title"], "url": r["url"], "snippet": r.get("content", "")}
                for r in data.get("results", [])
            ]
```

## SerpAPI provider

```python
import httpx

class SerpAPISearchProvider(WebSearchProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={"q": query, "api_key": self._api_key, "num": max_results},
            )
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": r.get("snippet", ""),
                }
                for r in data.get("organic_results", [])[:max_results]
            ]
```

## Notes

- The `InMemorySearchProvider` filters by query words in `title` or `snippet`
  — sufficient for deterministic tests without network calls.
- Inject a real provider at startup via DI: `use_value(WebSearchTool, WebSearchTool(TavilySearchProvider(api_key)))`.
- Limit `max_results` to keep the tool output within the model's context window.

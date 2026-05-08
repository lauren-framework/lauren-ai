---
name: rest-api-tool
description: Invoke external REST APIs from inside an agent using httpx.AsyncClient. Supports GET/POST/PUT/DELETE with optional auth header propagation from the execution context. Use when an agent needs to call external services, microservices, or webhooks.
---

> Use `codemap find "RestAPITool"` after adding the pattern to your project.

# REST API Invocation Tool with Auth Propagation

An `httpx`-based `@tool()` class that supports all common HTTP methods and
propagates authorization headers from the calling request context.

## Pattern

```python
from lauren_ai._tools import tool, ToolContext
import httpx
import json

@tool()
class RestAPITool:
    """Invoke a REST API endpoint.

    Args:
        url: The full URL to call (or path when base_url is set).
        method: HTTP method (GET, POST, PUT, DELETE).
        body: Optional JSON body for POST/PUT as a JSON string.
        headers: Optional additional headers as a JSON string.
    """

    def __init__(self, base_url: str = "", auth_header: str | None = None):
        self._base_url = base_url
        self._auth_header = auth_header

    async def run(
        self,
        ctx: ToolContext,
        url: str,
        method: str = "GET",
        body: str = "",
        headers: str = "",
    ) -> dict:
        full_url = self._base_url + url if url.startswith("/") else url
        extra_headers = json.loads(headers) if headers else {}

        # Static auth header from tool configuration
        if self._auth_header:
            extra_headers["Authorization"] = self._auth_header

        # Dynamic auth propagation from the originating request
        if ctx.execution_context and ctx.execution_context.request:
            auth = ctx.execution_context.request.headers.get("authorization")
            if auth:
                extra_headers["Authorization"] = auth

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                method_fn = getattr(client, method.lower())
                response = await method_fn(
                    full_url,
                    json=json.loads(body) if body else None,
                    headers=extra_headers,
                )
                return {"status": response.status_code, "body": response.text[:2000]}
            except Exception as e:
                return {"error": str(e)}
```

## Testing with mocked httpx

Use `unittest.mock.AsyncMock` to avoid real network calls:

```python
from unittest.mock import AsyncMock, MagicMock, patch

async def test_rest_tool_get():
    tool_instance = RestAPITool(base_url="https://api.example.com")
    ctx = make_tool_context()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"message": "ok"}'

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tool_instance.run(ctx, "/users", method="GET")

    assert result["status"] == 200
    assert "ok" in result["body"]
```

## Auth propagation chain

```
HTTP request ─► Lauren controller
                    │ execution_context.request.headers["authorization"]
                    ▼
            AgentRunner.run(..., execution_context=ctx)
                    │
                    ▼
            ToolContext.execution_context
                    │
                    ▼
            RestAPITool.run → httpx call with Authorization header
```

## Notes

- `httpx.AsyncClient` is created per-call. For high-throughput use,
  inject a shared client via the constructor (DI-friendly).
- The `body` and `headers` parameters are JSON strings so they can be
  safely passed through the LLM tool-call JSON encoding.
- Cap `response.text[:2000]` to avoid flooding the model context with
  large responses — adjust the limit or add a summarisation step.

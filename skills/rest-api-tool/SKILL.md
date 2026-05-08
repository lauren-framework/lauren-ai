---
name: rest-api-tool
description: Invoke external REST APIs from inside an agent using httpx.AsyncClient. Supports GET/POST/PUT/DELETE with optional auth header propagation from the execution context. Use when an agent needs to call external services, microservices, or webhooks.
---

> Use `codemap find "RestAPITool"` after adding the pattern to your project.

# REST API Invocation Tool with Auth Propagation

An `httpx`-based `@tool()` class that supports all common HTTP methods and
propagates authorization headers from the calling request context.

## Critical rule — name override for acronym classes

`RestAPITool` would auto-generate `rest_a_p_i_tool`. Override explicitly:

```python
@tool(name="rest_api_tool")
class RestAPITool:
    ...
```

## Pattern

```python
from lauren_ai._tools import tool, ToolContext
import httpx
import json

@tool(name="rest_api_tool")
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

## AgentRunner test pattern

Use `unittest.mock.patch` to intercept `httpx.AsyncClient` within the tool
call's context. Pass the tool instance to `@use_tools` and use `_Capture` to
inspect results.

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch
from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolResult
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


def _c(text):
    return Completion(id="c1", model="mock", content=text, tool_calls=[],
                      stop_reason="end_turn", usage=TokenUsage(10, 5))


def _make_agent(tool_instance):
    @agent(model=None, system="REST API test agent")
    @use_tools(tool_instance)
    class RestAPITestAgent(_Capture):
        def __init__(self):
            _Capture.__init__(self)
    return RestAPITestAgent()


def test_get_request_is_made():
    tool_inst = RestAPITool()
    agent_inst = _make_agent(tool_inst)
    client = TestClient(agent_inst)

    mock_resp = MagicMock(status_code=200, text='{"ok": true}')
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        client.mock.queue_tool_use(
            "rest_api_tool",
            {"url": "https://api.example.com/data", "method": "GET"},
        )
        client.mock.queue_response(_c("Data retrieved."))
        client.run("Fetch data")

    data = json.loads(agent_inst.captured[0].content)
    assert data["status"] == 200
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
- `body` and `headers` are JSON strings so they can be passed through
  the LLM tool-call JSON encoding.
- Cap `response.text[:2000]` to avoid flooding the model context.
- Always override the tool name with `@tool(name="rest_api_tool")` since
  the auto-generated name for `RestAPITool` would be `rest_a_p_i_tool`.

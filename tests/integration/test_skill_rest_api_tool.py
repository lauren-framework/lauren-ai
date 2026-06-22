"""Integration tests for the REST API invocation tool pattern (Skill 38).

Tests cover:
- GET request is made to the correct URL
- POST request is made with the correct body
- Auth header is propagated from tool config
- Auth header from execution_context.request is propagated
- Non-2xx status codes are returned without error
- Network errors return an error dict
- base_url prefix is prepended to path URLs

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# RestAPITool implementation
# ---------------------------------------------------------------------------


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
        import httpx

        full_url = self._base_url + url if url.startswith("/") else url
        extra_headers = json.loads(headers) if headers else {}

        if self._auth_header:
            extra_headers["Authorization"] = self._auth_header

        # execution_context lives on agent_context, not ToolContext directly.
        _agent_ctx = ctx.agent_context if ctx is not None else None
        _exec_ctx = getattr(_agent_ctx, "execution_context", None) if _agent_ctx is not None else None
        if _exec_ctx is not None:
            req = _exec_ctx.request if hasattr(_exec_ctx, "request") else None
            if req and hasattr(req, "headers"):
                auth = req.headers.get("authorization")
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text, *, n=1, stop="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock",
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(10, 5),
    )


def _mock_response(status: int = 200, text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


def _make_mock_client(method: str, mock_resp=None, error: str | None = None):
    """Build a mocked httpx.AsyncClient context manager."""
    mock_method_fn = AsyncMock(side_effect=Exception(error)) if error else AsyncMock(return_value=mock_resp)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    setattr(mock_client, method.lower(), mock_method_fn)
    return mock_client, mock_method_fn


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------


def _make_agent(tool_instance):
    """Create a fresh test agent with the given RestAPITool instance."""

    @agent(model="mock-model", system="REST API test agent")
    @use_tools(tool_instance)
    class RestAPITestAgent(_Capture):
        def __init__(self):
            _Capture.__init__(self)

    return RestAPITestAgent()


# ---------------------------------------------------------------------------
# Tests: HTTP method dispatch
# ---------------------------------------------------------------------------


class TestRestAPIToolMethods:
    def test_get_request_is_made(self):
        """A GET request is dispatched to the provided URL."""
        tool_inst = RestAPITool()
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_resp = _mock_response(200, '{"ok": true}')
        mock_client, _ = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use(
                "rest_api_tool",
                {"url": "https://api.example.com/data", "method": "GET"},
            )
            client.mock.queue_response(_c("Data retrieved."))
            client.run("Fetch data")
        data = json.loads(agent_inst.captured[0].content)
        assert data["status"] == 200

    def test_post_request_sends_body(self):
        """A POST request is dispatched with the parsed JSON body."""
        tool_inst = RestAPITool()
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_resp = _mock_response(201, "created")
        mock_client, mock_fn = _make_mock_client("post", mock_resp)
        body_str = json.dumps({"name": "Alice"})
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use(
                "rest_api_tool",
                {"url": "https://api.example.com/users", "method": "POST", "body": body_str},
            )
            client.mock.queue_response(_c("Created."))
            client.run("Create user Alice")
        data = json.loads(agent_inst.captured[0].content)
        assert data["status"] == 201
        call_json = mock_fn.call_args.kwargs.get("json")
        assert call_json == {"name": "Alice"}

    def test_delete_request_is_dispatched(self):
        """A DELETE request is dispatched correctly."""
        tool_inst = RestAPITool()
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_resp = _mock_response(204, "")
        mock_client, _ = _make_mock_client("delete", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use(
                "rest_api_tool",
                {"url": "https://api.example.com/item/1", "method": "DELETE"},
            )
            client.mock.queue_response(_c("Deleted."))
            client.run("Delete item 1")
        data = json.loads(agent_inst.captured[0].content)
        assert data["status"] == 204


# ---------------------------------------------------------------------------
# Tests: URL construction
# ---------------------------------------------------------------------------


class TestRestAPIToolURL:
    def test_path_is_prepended_with_base_url(self):
        """When url starts with '/', base_url is prepended."""
        tool_inst = RestAPITool(base_url="https://api.example.com")
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use("rest_api_tool", {"url": "/users", "method": "GET"})
            client.mock.queue_response(_c("Users listed."))
            client.run("List users")
        called_url = mock_fn.call_args.args[0]
        assert called_url == "https://api.example.com/users"

    def test_absolute_url_ignores_base_url(self):
        """When url is absolute (no leading '/'), base_url is NOT prepended."""
        tool_inst = RestAPITool(base_url="https://base.example.com")
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use(
                "rest_api_tool",
                {"url": "https://other.example.com/path", "method": "GET"},
            )
            client.mock.queue_response(_c("Fetched."))
            client.run("Fetch other")
        called_url = mock_fn.call_args.args[0]
        assert called_url == "https://other.example.com/path"


# ---------------------------------------------------------------------------
# Tests: auth propagation
# ---------------------------------------------------------------------------


class TestRestAPIToolAuth:
    def test_static_auth_header_is_sent(self):
        """A static auth_header configured on the tool is sent with every request."""
        tool_inst = RestAPITool(auth_header="Bearer static-token")
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use(
                "rest_api_tool",
                {"url": "https://api.example.com/secure", "method": "GET"},
            )
            client.mock.queue_response(_c("Secured."))
            client.run("Access secure endpoint")
        sent_headers = mock_fn.call_args.kwargs.get("headers", {})
        assert sent_headers.get("Authorization") == "Bearer static-token"

    def test_auth_from_execution_context_overrides_static(self):
        """Auth from execution_context.request overrides the static auth_header.

        This test directly constructs a ToolContext (not via runner) because
        execution_context is a lauren-specific request-level object that the
        runner does not inject automatically in the test environment.
        """
        tool_inst = RestAPITool(auth_header="Bearer static-token")

        class _MockRequest:
            def __init__(self, auth):
                self.headers = {"authorization": auth}

        class _MockEC:
            def __init__(self, auth):
                self.request = _MockRequest(auth)

        import asyncio

        from unittest.mock import MagicMock  # noqa: PLC0415

        mock_agent_ctx = MagicMock()
        mock_agent_ctx.execution_context = _MockEC("Bearer request-token")
        ctx = ToolContext(
            agent_context=mock_agent_ctx,
            tool_use_id="tu1",
            turn=0,
        )
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            asyncio.run(tool_inst.run(ctx, url="https://api.example.com/secure", method="GET"))
        sent_headers = mock_fn.call_args.kwargs.get("headers", {})
        assert sent_headers.get("Authorization") == "Bearer request-token"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestRestAPIToolErrors:
    def test_network_error_returns_error_dict(self):
        """Network errors (e.g. connection refused) return an error dict."""
        tool_inst = RestAPITool()
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_client, _ = _make_mock_client("get", error="Connection refused")
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use(
                "rest_api_tool",
                {"url": "https://unreachable.example.com/", "method": "GET"},
            )
            client.mock.queue_response(_c("Network error."))
            client.run("Fetch unreachable")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data
        assert "Connection refused" in data["error"]

    def test_non_2xx_status_is_returned(self):
        """A 404 or 500 response is returned as-is (not raised as exception)."""
        tool_inst = RestAPITool()
        agent_inst = _make_agent(tool_inst)
        client = TestClient(agent_inst)
        mock_resp = _mock_response(404, "Not Found")
        mock_client, _ = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.mock.queue_tool_use(
                "rest_api_tool",
                {"url": "https://api.example.com/missing", "method": "GET"},
            )
            client.mock.queue_response(_c("Not found."))
            client.run("Get missing resource")
        data = json.loads(agent_inst.captured[0].content)
        assert data["status"] == 404
        assert "error" not in data

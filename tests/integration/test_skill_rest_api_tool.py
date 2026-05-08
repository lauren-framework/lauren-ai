"""Integration tests for the REST API invocation tool pattern (Skill 38).

Tests cover:
- GET request is made to the correct URL
- POST request is made with the correct body
- Auth header is propagated from tool config
- Auth header from execution_context.request is propagated
- Non-2xx status codes are returned without error
- Network errors return an error dict
- base_url prefix is prepended to path URLs
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from lauren_ai._tools import ToolContext


# ---------------------------------------------------------------------------
# RestAPITool implementation
# ---------------------------------------------------------------------------


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
        ctx,
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

        if ctx is not None and hasattr(ctx, "execution_context") and ctx.execution_context is not None:
            req = ctx.execution_context.request if hasattr(ctx.execution_context, "request") else None
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
# Helper stubs
# ---------------------------------------------------------------------------


class _MockRequest:
    def __init__(self, auth: str | None = None):
        self.headers = {}
        if auth:
            self.headers["authorization"] = auth


class _MockExecutionContext:
    def __init__(self, auth: str | None = None):
        self.request = _MockRequest(auth)


def _make_ctx(auth: str | None = None):
    ec = _MockExecutionContext(auth) if auth else None
    return ToolContext(
        agent_context=None,
        tool_use_id="tu1",
        turn=0,
        execution_context=ec,
    )


def _mock_response(status: int = 200, text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


def _make_mock_client(method: str, mock_resp=None, error: str | None = None):
    """Build a mocked httpx.AsyncClient context manager."""
    if error:
        mock_method_fn = AsyncMock(side_effect=Exception(error))
    else:
        mock_method_fn = AsyncMock(return_value=mock_resp)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    setattr(mock_client, method.lower(), mock_method_fn)
    return mock_client, mock_method_fn


# ---------------------------------------------------------------------------
# Tests: HTTP method dispatch
# ---------------------------------------------------------------------------


class TestRestAPIToolMethods:
    async def test_get_request_is_made(self):
        """A GET request is dispatched to the provided URL."""
        tool = RestAPITool()
        ctx = _make_ctx()
        mock_resp = _mock_response(200, '{"ok": true}')
        mock_client, _ = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(ctx, url="https://api.example.com/data", method="GET")
        assert result["status"] == 200

    async def test_post_request_sends_body(self):
        """A POST request is dispatched with the parsed JSON body."""
        tool = RestAPITool()
        ctx = _make_ctx()
        mock_resp = _mock_response(201, "created")
        mock_client, mock_fn = _make_mock_client("post", mock_resp)
        body = json.dumps({"name": "Alice"})
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(
                ctx, url="https://api.example.com/users", method="POST", body=body
            )
        assert result["status"] == 201
        call_json = mock_fn.call_args.kwargs.get("json")
        assert call_json == {"name": "Alice"}

    async def test_delete_request_is_dispatched(self):
        """A DELETE request is dispatched correctly."""
        tool = RestAPITool()
        ctx = _make_ctx()
        mock_resp = _mock_response(204, "")
        mock_client, _ = _make_mock_client("delete", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(
                ctx, url="https://api.example.com/item/1", method="DELETE"
            )
        assert result["status"] == 204


# ---------------------------------------------------------------------------
# Tests: URL construction
# ---------------------------------------------------------------------------


class TestRestAPIToolURL:
    async def test_path_is_prepended_with_base_url(self):
        """When url starts with '/', base_url is prepended."""
        tool = RestAPITool(base_url="https://api.example.com")
        ctx = _make_ctx()
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool.run(ctx, url="/users", method="GET")
        called_url = mock_fn.call_args.args[0]
        assert called_url == "https://api.example.com/users"

    async def test_absolute_url_ignores_base_url(self):
        """When url is absolute (no leading '/'), base_url is NOT prepended."""
        tool = RestAPITool(base_url="https://base.example.com")
        ctx = _make_ctx()
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool.run(ctx, url="https://other.example.com/path", method="GET")
        called_url = mock_fn.call_args.args[0]
        assert called_url == "https://other.example.com/path"


# ---------------------------------------------------------------------------
# Tests: auth propagation
# ---------------------------------------------------------------------------


class TestRestAPIToolAuth:
    async def test_static_auth_header_is_sent(self):
        """A static auth_header configured on the tool is sent with every request."""
        tool = RestAPITool(auth_header="Bearer static-token")
        ctx = _make_ctx()
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool.run(ctx, url="https://api.example.com/secure", method="GET")
        sent_headers = mock_fn.call_args.kwargs.get("headers", {})
        assert sent_headers.get("Authorization") == "Bearer static-token"

    async def test_auth_from_execution_context_overrides_static(self):
        """Auth from execution_context.request overrides the static auth_header."""
        tool = RestAPITool(auth_header="Bearer static-token")
        ctx = _make_ctx(auth="Bearer request-token")
        mock_resp = _mock_response(200, "ok")
        mock_client, mock_fn = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool.run(ctx, url="https://api.example.com/secure", method="GET")
        sent_headers = mock_fn.call_args.kwargs.get("headers", {})
        assert sent_headers.get("Authorization") == "Bearer request-token"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestRestAPIToolErrors:
    async def test_network_error_returns_error_dict(self):
        """Network errors (e.g. connection refused) return an error dict."""
        tool = RestAPITool()
        ctx = _make_ctx()
        mock_client, _ = _make_mock_client("get", error="Connection refused")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(
                ctx, url="https://unreachable.example.com/", method="GET"
            )
        assert "error" in result
        assert "Connection refused" in result["error"]

    async def test_non_2xx_status_is_returned(self):
        """A 404 or 500 response is returned as-is (not raised as exception)."""
        tool = RestAPITool()
        ctx = _make_ctx()
        mock_resp = _mock_response(404, "Not Found")
        mock_client, _ = _make_mock_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.run(
                ctx, url="https://api.example.com/missing", method="GET"
            )
        assert result["status"] == 404
        assert "error" not in result

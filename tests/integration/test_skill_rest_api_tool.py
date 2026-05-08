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

import pytest

from lauren_ai._tools import tool, ToolContext


# ---------------------------------------------------------------------------
# RestAPITool implementation (inline for test isolation)
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

        if ctx.execution_context and hasattr(ctx.execution_context, "request"):
            req = ctx.execution_context.request
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


def _make_stub_ctx(auth: str | None = None) -> ToolContext:
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


# ---------------------------------------------------------------------------
# Tests: HTTP method dispatch
# ---------------------------------------------------------------------------


class TestRestAPIToolMethods:
    @pytest.mark.asyncio
    async def test_get_request_is_made(self):
        """A GET request is dispatched to the provided URL."""
        tool_instance = RestAPITool()
        ctx = _make_stub_ctx()

        mock_get = AsyncMock(return_value=_mock_response(200, '{"ok": true}'))
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = mock_get

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool_instance.run(ctx, "https://api.example.com/data")

        assert result["status"] == 200
        mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_request_sends_body(self):
        """A POST request is dispatched with the parsed JSON body."""
        tool_instance = RestAPITool()
        ctx = _make_stub_ctx()

        mock_post = AsyncMock(return_value=_mock_response(201, "created"))
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = mock_post

        body = json.dumps({"name": "Alice"})

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool_instance.run(
                ctx, "https://api.example.com/users", method="POST", body=body
            )

        assert result["status"] == 201
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("json") == {"name": "Alice"}

    @pytest.mark.asyncio
    async def test_delete_request_is_dispatched(self):
        """A DELETE request is dispatched correctly."""
        tool_instance = RestAPITool()
        ctx = _make_stub_ctx()

        mock_delete = AsyncMock(return_value=_mock_response(204, ""))
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.delete = mock_delete

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool_instance.run(
                ctx, "https://api.example.com/item/1", method="DELETE"
            )

        assert result["status"] == 204
        mock_delete.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: URL construction
# ---------------------------------------------------------------------------


class TestRestAPIToolURL:
    @pytest.mark.asyncio
    async def test_path_is_prepended_with_base_url(self):
        """When url starts with '/', base_url is prepended."""
        tool_instance = RestAPITool(base_url="https://api.example.com")
        ctx = _make_stub_ctx()
        called_url = None

        async def mock_get(url, **kwargs):
            nonlocal called_url
            called_url = url
            return _mock_response()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool_instance.run(ctx, "/users")

        assert called_url == "https://api.example.com/users"

    @pytest.mark.asyncio
    async def test_absolute_url_ignores_base_url(self):
        """When url is absolute (no leading '/'), base_url is NOT prepended."""
        tool_instance = RestAPITool(base_url="https://base.example.com")
        ctx = _make_stub_ctx()
        called_url = None

        async def mock_get(url, **kwargs):
            nonlocal called_url
            called_url = url
            return _mock_response()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool_instance.run(ctx, "https://other.example.com/path")

        assert called_url == "https://other.example.com/path"


# ---------------------------------------------------------------------------
# Tests: auth propagation
# ---------------------------------------------------------------------------


class TestRestAPIToolAuth:
    @pytest.mark.asyncio
    async def test_static_auth_header_is_sent(self):
        """A static auth_header configured on the tool is sent with every request."""
        tool_instance = RestAPITool(auth_header="Bearer static-token")
        ctx = _make_stub_ctx()
        sent_headers = {}

        async def mock_get(url, **kwargs):
            sent_headers.update(kwargs.get("headers", {}))
            return _mock_response()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool_instance.run(ctx, "https://api.example.com/secure")

        assert sent_headers.get("Authorization") == "Bearer static-token"

    @pytest.mark.asyncio
    async def test_auth_from_execution_context_overrides_static(self):
        """Auth from execution_context.request overrides the static auth_header."""
        tool_instance = RestAPITool(auth_header="Bearer static-token")
        ctx = _make_stub_ctx(auth="Bearer request-token")
        sent_headers = {}

        async def mock_get(url, **kwargs):
            sent_headers.update(kwargs.get("headers", {}))
            return _mock_response()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await tool_instance.run(ctx, "https://api.example.com/secure")

        assert sent_headers.get("Authorization") == "Bearer request-token"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestRestAPIToolErrors:
    @pytest.mark.asyncio
    async def test_network_error_returns_error_dict(self):
        """Network errors (e.g. connection refused) return an error dict."""
        tool_instance = RestAPITool()
        ctx = _make_stub_ctx()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool_instance.run(ctx, "https://unreachable.example.com/")

        assert "error" in result
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_non_2xx_status_is_returned(self):
        """A 404 or 500 response is returned as-is (not raised as exception)."""
        tool_instance = RestAPITool()
        ctx = _make_stub_ctx()

        mock_get = AsyncMock(return_value=_mock_response(404, "Not Found"))
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = mock_get

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool_instance.run(ctx, "https://api.example.com/missing")

        assert result["status"] == 404
        assert "error" not in result

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

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient
from lauren_ai._tools import tool, ToolContext


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
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/api-tool")
class APIToolController:
    @post("/call")
    async def call(self, body: Json[dict]) -> dict:
        url = body.get("url", "")
        method = body.get("method", "GET")
        req_body = body.get("body", "")
        headers = body.get("headers", "")
        auth = body.get("auth", None)
        base_url = body.get("base_url", "")
        static_auth = body.get("static_auth", None)

        tool_instance = RestAPITool(base_url=base_url, auth_header=static_auth)
        ctx = _make_stub_ctx(auth)
        return await tool_instance.run(ctx, url=url, method=method, body=req_body, headers=headers)

    @post("/call-with-mock")
    async def call_with_mock(self, body: Json[dict]) -> dict:
        """Call the tool with a fully mocked httpx client."""
        url = body.get("url", "")
        method = body.get("method", "GET")
        req_body = body.get("body", "")
        auth = body.get("auth", None)
        base_url = body.get("base_url", "")
        static_auth = body.get("static_auth", None)
        mock_status = body.get("mock_status", 200)
        mock_text = body.get("mock_text", "ok")
        mock_error = body.get("mock_error", None)

        tool_instance = RestAPITool(base_url=base_url, auth_header=static_auth)
        ctx = _make_stub_ctx(auth)

        mock_resp = _mock_response(mock_status, mock_text)
        mock_method_fn = AsyncMock(return_value=mock_resp)
        if mock_error:
            mock_method_fn = AsyncMock(side_effect=Exception(mock_error))

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        setattr(mock_client, method.lower(), mock_method_fn)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool_instance.run(ctx, url=url, method=method, body=req_body)

        # Also capture what was called
        if not mock_error and mock_method_fn.called:
            call_kwargs = mock_method_fn.call_args
            called_url = call_kwargs.args[0] if call_kwargs.args else None
            result["_called_url"] = called_url
            result["_call_json"] = call_kwargs.kwargs.get("json")
            result["_call_headers"] = call_kwargs.kwargs.get("headers", {})
        return result


@module(controllers=[APIToolController])
class RestAPIModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(RestAPIModule))


# ---------------------------------------------------------------------------
# Tests: HTTP method dispatch
# ---------------------------------------------------------------------------


class TestRestAPIToolMethods:
    def test_get_request_is_made(self):
        """A GET request is dispatched to the provided URL."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://api.example.com/data",
            "method": "GET",
            "mock_status": 200,
            "mock_text": '{"ok": true}',
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == 200

    def test_post_request_sends_body(self):
        """A POST request is dispatched with the parsed JSON body."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://api.example.com/users",
            "method": "POST",
            "body": json.dumps({"name": "Alice"}),
            "mock_status": 201,
            "mock_text": "created",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == 201
        assert data["_call_json"] == {"name": "Alice"}

    def test_delete_request_is_dispatched(self):
        """A DELETE request is dispatched correctly."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://api.example.com/item/1",
            "method": "DELETE",
            "mock_status": 204,
            "mock_text": "",
        })
        assert r.status_code == 200
        assert r.json()["status"] == 204


# ---------------------------------------------------------------------------
# Tests: URL construction
# ---------------------------------------------------------------------------


class TestRestAPIToolURL:
    def test_path_is_prepended_with_base_url(self):
        """When url starts with '/', base_url is prepended."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "/users",
            "method": "GET",
            "base_url": "https://api.example.com",
            "mock_status": 200,
            "mock_text": "ok",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["_called_url"] == "https://api.example.com/users"

    def test_absolute_url_ignores_base_url(self):
        """When url is absolute (no leading '/'), base_url is NOT prepended."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://other.example.com/path",
            "method": "GET",
            "base_url": "https://base.example.com",
            "mock_status": 200,
            "mock_text": "ok",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["_called_url"] == "https://other.example.com/path"


# ---------------------------------------------------------------------------
# Tests: auth propagation
# ---------------------------------------------------------------------------


class TestRestAPIToolAuth:
    def test_static_auth_header_is_sent(self):
        """A static auth_header configured on the tool is sent with every request."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://api.example.com/secure",
            "method": "GET",
            "static_auth": "Bearer static-token",
            "mock_status": 200,
            "mock_text": "ok",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["_call_headers"].get("Authorization") == "Bearer static-token"

    def test_auth_from_execution_context_overrides_static(self):
        """Auth from execution_context.request overrides the static auth_header."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://api.example.com/secure",
            "method": "GET",
            "static_auth": "Bearer static-token",
            "auth": "Bearer request-token",
            "mock_status": 200,
            "mock_text": "ok",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["_call_headers"].get("Authorization") == "Bearer request-token"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestRestAPIToolErrors:
    def test_network_error_returns_error_dict(self):
        """Network errors (e.g. connection refused) return an error dict."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://unreachable.example.com/",
            "method": "GET",
            "mock_error": "Connection refused",
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "Connection refused" in data["error"]

    def test_non_2xx_status_is_returned(self):
        """A 404 or 500 response is returned as-is (not raised as exception)."""
        client = build_app()
        r = client.post("/api-tool/call-with-mock", json={
            "url": "https://api.example.com/missing",
            "method": "GET",
            "mock_status": 404,
            "mock_text": "Not Found",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == 404
        assert "error" not in data

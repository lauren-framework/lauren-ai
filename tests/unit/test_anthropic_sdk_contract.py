"""AnthropicTransport driven through the *real* ``anthropic`` SDK.

The tests that use fake clients cannot see SDK incompatibilities, because a fake
accepts any keyword argument.  Here the installed SDK, whichever line it is,
talks to a mock HTTP server, and the assertions are on what goes on the wire.
CI runs this module on each SDK line: ``nox -s tests_anthropic_sdk``.

* SDK 0.x declares ``temperature`` / ``top_p`` / ``top_k`` and sends them.
* SDK 1.x removed them (and moved to ``httpx2``) but still sends API version
  ``2023-06-01``, which accepts them through ``extra_body``.
* SDK 2.x sends API version ``2026-10-21``, which rejects them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lauren_ai import LLMConfig
from lauren_ai._exceptions import AuthTransportError, TransientTransportError, TransportError
from lauren_ai._transport import Message, RequestOptions, ToolSchema
from lauren_ai._transport._anthropic import AnthropicTransport

anthropic = pytest.importorskip("anthropic")

#: Major version of the installed SDK: 0, 1 or 2.
SDK_MAJOR = int(anthropic.__version__.split(".")[0])

if SDK_MAJOR >= 1:
    import httpx2 as http  # the SDK rejects an ``httpx`` client from 1.0 on
else:  # pragma: no cover - exercised on the 0.x CI line
    import httpx as http

_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


def _sse(events: list[tuple[str, dict[str, Any]]]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def _stream_events(*, tool: bool) -> list[tuple[str, dict[str, Any]]]:
    start = ("message_start", {"type": "message_start", "message": {**_MESSAGE, "content": [], "stop_reason": None}})
    if tool:
        return [
            start,
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 9},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    return [
        start,
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


class _Server:
    """A mock Anthropic API recording every request it receives."""

    def __init__(self) -> None:
        self.requests: list[http.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.status: int | None = None  # forced error status
        self.refuse_connections = False

    def handle(self, request: Any) -> Any:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else {}
        self.bodies.append(body)
        if self.refuse_connections:
            raise http.ConnectError("refused", request=request)
        if self.status is not None:
            return http.Response(self.status, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
        if request.url.path.endswith("/count_tokens"):
            return http.Response(200, json={"input_tokens": 42})
        if body.get("stream"):
            return http.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(_stream_events(tool=bool(body.get("tools")))),
            )
        return http.Response(
            200, json={**_MESSAGE, "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn"}
        )

    @property
    def last_body(self) -> dict[str, Any]:
        return self.bodies[-1]


@pytest.fixture
def server() -> _Server:
    return _Server()


def _transport(server: _Server, **config: Any) -> AnthropicTransport:
    client = http.AsyncClient(transport=http.MockTransport(server.handle))
    llm_config = LLMConfig.for_anthropic(
        model="claude-x", api_key="k", max_retries=0, client_options={"http_client": client}, **config
    )
    return AnthropicTransport(llm_config)


MESSAGES = [Message(role="user", content="hi")]
SAMPLING = ("temperature", "top_p", "top_k")


def _sampling_on_wire(body: dict[str, Any]) -> dict[str, Any]:
    return {key: body[key] for key in SAMPLING if key in body}


class TestRealSdkRequests:
    async def test_a_plain_request_succeeds(self, server: _Server) -> None:
        completion = await _transport(server).complete(MESSAGES, model="claude-x")
        assert completion.content == "hello"

    async def test_default_temperature_on_the_wire(self, server: _Server) -> None:
        await _transport(server).complete(MESSAGES, model="claude-x")
        # 0.x declares the parameter and keeps sending it; 1.x and later omit
        # the default, which is equivalent.
        expected = {"temperature": 1.0} if SDK_MAJOR == 0 else {}
        assert _sampling_on_wire(server.last_body) == expected

    async def test_non_default_sampling_on_the_wire(self, server: _Server) -> None:
        await _transport(server).complete(MESSAGES, model="claude-x", temperature=0.2, top_p=0.9)
        # 0.x: keyword arguments.  1.x: through extra_body.  2.x: the API
        # version rejects them, so they are not sent.
        expected = {} if SDK_MAJOR >= 2 else {"temperature": 0.2, "top_p": 0.9}
        assert _sampling_on_wire(server.last_body) == expected

    async def test_zero_temperature_on_the_wire(self, server: _Server) -> None:
        # Lauren's own summarisation call uses temperature=0.0; it must not
        # turn into an HTTP 400 on the newest SDK.
        completion = await _transport(server).complete(MESSAGES, model="claude-x", temperature=0.0)
        assert completion.content == "hello"
        expected = {} if SDK_MAJOR >= 2 else {"temperature": 0.0}
        assert _sampling_on_wire(server.last_body) == expected

    async def test_thinking_request_succeeds(self, server: _Server) -> None:
        await _transport(server).complete(MESSAGES, model="claude-x", thinking=True)
        assert server.last_body["thinking"]["type"] == "enabled"
        assert _sampling_on_wire(server.last_body) == {}

    @pytest.mark.skipif(SDK_MAJOR < 2, reason="API version 2026-10-21 is sent by SDK 2.x")
    async def test_sdk_2x_sends_the_new_api_version(self, server: _Server) -> None:
        await _transport(server).complete(MESSAGES, model="claude-x")
        assert server.requests[-1].headers["anthropic-version"] == "2026-10-21"

    async def test_pinning_the_old_api_version_keeps_sampling_on_the_wire(self, server: _Server) -> None:
        transport = _transport(server, default_headers={"anthropic-version": "2023-06-01"})
        await transport.complete(MESSAGES, model="claude-x", temperature=0.2)
        assert server.requests[-1].headers["anthropic-version"] == "2023-06-01"
        assert _sampling_on_wire(server.last_body) == {"temperature": 0.2}

    async def test_request_options_extra_body_survives_forwarding(self, server: _Server) -> None:
        await _transport(server).complete(
            MESSAGES,
            model="claude-x",
            temperature=0.2,
            request_options=RequestOptions(extra_body={"custom_flag": True}),
        )
        assert server.last_body["custom_flag"] is True


class TestRealSdkStreaming:
    async def _collect(self, transport: AnthropicTransport, **kwargs: Any) -> list[Any]:
        return [chunk async for chunk in await transport.complete(MESSAGES, model="claude-x", stream=True, **kwargs)]

    async def test_text_stream(self, server: _Server) -> None:
        chunks = await self._collect(_transport(server))
        assert "".join(getattr(chunk, "delta", "") or "" for chunk in chunks) == "hello"
        assert server.last_body["stream"] is True

    async def test_stream_sampling_on_the_wire(self, server: _Server) -> None:
        await self._collect(_transport(server), temperature=0.2)
        expected = {} if SDK_MAJOR >= 2 else {"temperature": 0.2}
        assert _sampling_on_wire(server.last_body) == expected

    async def test_tool_call_stream(self, server: _Server) -> None:
        tool = ToolSchema(
            name="get_weather",
            description="weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        chunks = await self._collect(_transport(server), tools=[tool])
        assert any(getattr(chunk, "tool_call_delta", None) or getattr(chunk, "tool_calls", None) for chunk in chunks)


class TestRealSdkOtherCalls:
    async def test_count_tokens(self, server: _Server) -> None:
        assert await _transport(server).count_tokens(MESSAGES, model="claude-x") == 42


class TestRealSdkErrorMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (429, TransientTransportError),
            (500, TransientTransportError),
            (529, TransientTransportError),
            (401, AuthTransportError),
            (403, AuthTransportError),
            (400, TransportError),
        ],
    )
    async def test_http_status_is_classified(self, server: _Server, status: int, expected: type[Exception]) -> None:
        server.status = status
        with pytest.raises(expected) as caught:
            await _transport(server).complete(MESSAGES, model="claude-x")
        assert type(caught.value) is expected

    async def test_connection_failure_is_transient(self, server: _Server) -> None:
        server.refuse_connections = True
        with pytest.raises(TransientTransportError):
            await _transport(server).complete(MESSAGES, model="claude-x")

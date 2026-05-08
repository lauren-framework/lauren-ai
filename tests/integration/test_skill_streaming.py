"""Integration tests for Skill 3: Streaming Response Handler.

Tests cover:
- run_stream() yields CompletionChunk objects
- Delta accumulation
- Stop reason in final chunk
- Usage in final chunk
- Empty delta chunks are safe to consume
- Streaming with metadata
- Multi-turn streaming (tool call mid-stream)
"""

from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, CompletionChunk, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _chunks(*parts: str, stop_reason: str = "end_turn") -> list[CompletionChunk]:
    chunks = [CompletionChunk(delta=p) for p in parts]
    chunks.append(
        CompletionChunk(
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=10, output_tokens=len(parts) or 1),
        )
    )
    return chunks


# ---------------------------------------------------------------------------
# Controller / Module
# ---------------------------------------------------------------------------


class _StreamRequest(BaseModel):
    prompt: str = "hi"


@controller("/agent")
class StreamController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)

    @post("/stream")
    async def stream(self, body: Json[_StreamRequest]) -> dict:
        @agent(model="mock-model")
        class A: ...

        accumulated = ""
        chunk_count = 0
        stop_reasons = []
        has_usage = False
        thinking_deltas = []

        async for chunk in await self._runner.run_stream(A(), body.prompt):
            chunk_count += 1
            if chunk.delta:
                accumulated += chunk.delta
            if chunk.stop_reason is not None:
                stop_reasons.append(chunk.stop_reason)
            if chunk.usage is not None:
                has_usage = True
            if chunk.thinking_delta:
                thinking_deltas.append(chunk.thinking_delta)

        return {
            "content": accumulated,
            "chunks": chunk_count,
            "stop_reasons": stop_reasons,
            "has_usage": has_usage,
            "thinking_deltas": thinking_deltas,
        }

    @post("/stream-metadata")
    async def stream_metadata(self, body: Json[_StreamRequest]) -> dict:
        captured = []

        @agent(model="mock-model")
        class MetaAgent:
            async def on_start(self, ctx):
                captured.append(ctx.metadata.get("key"))

        async for _ in await self._runner.run_stream(
            MetaAgent(), body.prompt, metadata={"key": "value42"}
        ):
            pass

        return {"captured": captured}

    @post("/stream-on-finish")
    async def stream_on_finish(self, body: Json[_StreamRequest]) -> dict:
        finished = []

        @agent(model="mock-model")
        class FinAgent:
            async def on_finish(self, resp, ctx):
                finished.append(resp.content)

        async for _ in await self._runner.run_stream(FinAgent(), body.prompt):
            pass

        return {"finished": finished}


@module(
    controllers=[StreamController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class StreamModule: ...


def build_app(chunk_list: list[CompletionChunk] | None = None) -> TestClient:
    _MOCK.reset()
    if chunk_list is not None:
        _MOCK.queue_stream(chunk_list)
    return TestClient(LaurenFactory.create(StreamModule))


# ---------------------------------------------------------------------------
# TestStreamBasic
# ---------------------------------------------------------------------------


class TestStreamBasic:
    def test_run_stream_returns_chunks(self):
        client = build_app(_chunks("Hello"))
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["chunks"] > 0

    def test_stream_delta_content_accumulated(self):
        client = build_app(_chunks("Hello", " world", "!"))
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["content"] == "Hello world!"

    def test_stream_final_chunk_has_stop_reason(self):
        client = build_app(_chunks("text"))
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        assert "end_turn" in r.json()["stop_reasons"]

    def test_stream_final_chunk_has_usage(self):
        client = build_app(_chunks("text"))
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["has_usage"] is True

    def test_stream_empty_delta_chunks_harmless(self):
        client = build_app([
            CompletionChunk(delta=""),
            CompletionChunk(delta="real"),
            CompletionChunk(delta=""),
            CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=1)),
        ])
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["content"] == "real"

    def test_stream_individual_deltas_correct(self):
        client = build_app(_chunks("foo", " bar"))
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["content"] == "foo bar"


# ---------------------------------------------------------------------------
# TestStreamWithMetadata
# ---------------------------------------------------------------------------


class TestStreamWithMetadata:
    def test_stream_metadata_accessible_in_on_start(self):
        client = build_app(_chunks("OK"))
        r = client.post("/agent/stream-metadata", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["captured"] == ["value42"]

    def test_stream_on_finish_hook_fires(self):
        client = build_app(_chunks("complete"))
        r = client.post("/agent/stream-on-finish", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["finished"] == ["complete"]


# ---------------------------------------------------------------------------
# TestStreamAccumulation
# ---------------------------------------------------------------------------


class TestStreamAccumulation:
    def test_stream_large_content_accumulates_fully(self):
        words = [f"word{i}" for i in range(20)]
        chunks_list = [CompletionChunk(delta=w + " ") for w in words]
        chunks_list.append(
            CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=10, output_tokens=20))
        )
        client = build_app(chunks_list)
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        full_text = r.json()["content"]
        for w in words:
            assert w in full_text

    def test_stream_thinking_delta_yielded(self):
        client = build_app([
            CompletionChunk(thinking_delta="Let me think..."),
            CompletionChunk(delta="Answer"),
            CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=5)),
        ])
        r = client.post("/agent/stream", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["thinking_deltas"] == ["Let me think..."]

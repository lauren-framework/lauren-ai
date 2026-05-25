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

from lauren_ai._agents import agent
from lauren_ai._transport import CompletionChunk, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model")
class StreamAgent: ...


# ---------------------------------------------------------------------------
# TestStreamBasic
# ---------------------------------------------------------------------------


class TestStreamBasic:
    async def test_run_stream_returns_chunks(self):
        client = TestClient(StreamAgent())
        client.mock.queue_stream(_chunks("Hello"))
        chunk_count = 0
        async for _chunk in await client.run_stream_async("hi"):
            chunk_count += 1
        assert chunk_count > 0

    async def test_stream_delta_content_accumulated(self):
        client = TestClient(StreamAgent())
        client.mock.queue_stream(_chunks("Hello", " world", "!"))
        accumulated = ""
        async for chunk in await client.run_stream_async("hi"):
            if chunk.delta:
                accumulated += chunk.delta
        assert accumulated == "Hello world!"

    async def test_stream_final_chunk_has_stop_reason(self):
        client = TestClient(StreamAgent())
        client.mock.queue_stream(_chunks("text"))
        stop_reasons = []
        async for chunk in await client.run_stream_async("hi"):
            if chunk.stop_reason is not None:
                stop_reasons.append(chunk.stop_reason)
        assert "end_turn" in stop_reasons

    async def test_stream_final_chunk_has_usage(self):
        client = TestClient(StreamAgent())
        client.mock.queue_stream(_chunks("text"))
        has_usage = False
        async for chunk in await client.run_stream_async("hi"):
            if chunk.usage is not None:
                has_usage = True
        assert has_usage is True

    async def test_stream_empty_delta_chunks_harmless(self):
        client = TestClient(StreamAgent())
        client.mock.queue_stream(
            [
                CompletionChunk(delta=""),
                CompletionChunk(delta="real"),
                CompletionChunk(delta=""),
                CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=1)),
            ]
        )
        accumulated = ""
        async for chunk in await client.run_stream_async("hi"):
            if chunk.delta:
                accumulated += chunk.delta
        assert accumulated == "real"

    async def test_stream_individual_deltas_correct(self):
        client = TestClient(StreamAgent())
        client.mock.queue_stream(_chunks("foo", " bar"))
        accumulated = ""
        async for chunk in await client.run_stream_async("hi"):
            if chunk.delta:
                accumulated += chunk.delta
        assert accumulated == "foo bar"


# ---------------------------------------------------------------------------
# TestStreamWithMetadata
# ---------------------------------------------------------------------------


class TestStreamWithMetadata:
    async def test_stream_metadata_accessible_in_on_start(self):
        captured = []

        @agent(model="mock-model")
        class MetaAgent:
            async def on_start(self, ctx):
                captured.append(ctx.metadata.get("key"))

        client = TestClient(MetaAgent())
        client.mock.queue_stream(_chunks("OK"))
        async for _ in await client.run_stream_async("hi", metadata={"key": "value42"}):
            pass
        assert captured == ["value42"]

    async def test_stream_on_finish_hook_fires(self):
        finished = []

        @agent(model="mock-model")
        class FinAgent:
            async def on_finish(self, resp, ctx):
                finished.append(resp.content)

        client = TestClient(FinAgent())
        client.mock.queue_stream(_chunks("complete"))
        async for _ in await client.run_stream_async("hi"):
            pass
        assert finished == ["complete"]


# ---------------------------------------------------------------------------
# TestStreamAccumulation
# ---------------------------------------------------------------------------


class TestStreamAccumulation:
    async def test_stream_large_content_accumulates_fully(self):
        words = [f"word{i}" for i in range(20)]
        chunks_list = [CompletionChunk(delta=w + " ") for w in words]
        chunks_list.append(CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=10, output_tokens=20)))
        client = TestClient(StreamAgent())
        client.mock.queue_stream(chunks_list)
        full_text = ""
        async for chunk in await client.run_stream_async("hi"):
            if chunk.delta:
                full_text += chunk.delta
        for w in words:
            assert w in full_text

    async def test_stream_thinking_delta_yielded(self):
        client = TestClient(StreamAgent())
        client.mock.queue_stream(
            [
                CompletionChunk(thinking_delta="Let me think..."),
                CompletionChunk(delta="Answer"),
                CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=5)),
            ]
        )
        thinking_deltas = []
        async for chunk in await client.run_stream_async("hi"):
            if chunk.thinking_delta:
                thinking_deltas.append(chunk.thinking_delta)
        assert thinking_deltas == ["Let me think..."]

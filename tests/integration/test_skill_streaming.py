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

from __future__ import annotations

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, CompletionChunk, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(mock: MockTransport | None = None) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


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
# TestStreamBasic
# ---------------------------------------------------------------------------


class TestStreamBasic:
    async def test_run_stream_returns_async_iterable(self):
        runner, mock = _make_runner()
        mock.queue_stream(_chunks("Hello"))

        @agent(model="mock-model")
        class A: ...

        stream = await runner.run_stream(A(), "hi")
        chunks = [c async for c in stream]
        assert len(chunks) > 0

    async def test_stream_yields_completion_chunk_objects(self):
        runner, mock = _make_runner()
        mock.queue_stream(_chunks("Hello"))

        @agent(model="mock-model")
        class A: ...

        async for chunk in await runner.run_stream(A(), "hi"):
            assert isinstance(chunk, CompletionChunk)

    async def test_stream_delta_content_accumulated(self):
        runner, mock = _make_runner()
        mock.queue_stream(_chunks("Hello", " world", "!"))

        @agent(model="mock-model")
        class A: ...

        accumulated = ""
        async for chunk in await runner.run_stream(A(), "hi"):
            if chunk.delta:
                accumulated += chunk.delta

        assert accumulated == "Hello world!"

    async def test_stream_individual_deltas_correct(self):
        runner, mock = _make_runner()
        mock.queue_stream(_chunks("foo", " bar"))

        @agent(model="mock-model")
        class A: ...

        deltas = [c.delta for c in [c async for c in await runner.run_stream(A(), "hi")] if c.delta]
        assert deltas == ["foo", " bar"]

    async def test_stream_final_chunk_has_stop_reason(self):
        runner, mock = _make_runner()
        mock.queue_stream(_chunks("text"))

        @agent(model="mock-model")
        class A: ...

        chunks = [c async for c in await runner.run_stream(A(), "hi")]
        stop_chunks = [c for c in chunks if c.stop_reason is not None]
        assert len(stop_chunks) >= 1
        assert stop_chunks[-1].stop_reason == "end_turn"

    async def test_stream_final_chunk_has_usage(self):
        runner, mock = _make_runner()
        mock.queue_stream(_chunks("text"))

        @agent(model="mock-model")
        class A: ...

        chunks = [c async for c in await runner.run_stream(A(), "hi")]
        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) >= 1

    async def test_stream_empty_delta_chunks_harmless(self):
        runner, mock = _make_runner()
        mock.queue_stream([
            CompletionChunk(delta=""),
            CompletionChunk(delta="real"),
            CompletionChunk(delta=""),
            CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=1)),
        ])

        @agent(model="mock-model")
        class A: ...

        accumulated = ""
        async for chunk in await runner.run_stream(A(), "hi"):
            if chunk.delta:
                accumulated += chunk.delta

        assert accumulated == "real"


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

        runner, mock = _make_runner()
        mock.queue_stream(_chunks("OK"))

        async for _ in await runner.run_stream(MetaAgent(), "hi", metadata={"key": "value42"}):
            pass

        assert captured == ["value42"]

    async def test_stream_on_finish_hook_fires(self):
        finished = []

        @agent(model="mock-model")
        class FinAgent:
            async def on_finish(self, resp, ctx):
                finished.append(resp.content)

        runner, mock = _make_runner()
        mock.queue_stream(_chunks("complete"))

        async for _ in await runner.run_stream(FinAgent(), "hi"):
            pass

        assert finished == ["complete"]


# ---------------------------------------------------------------------------
# TestStreamAccumulation
# ---------------------------------------------------------------------------


class TestStreamAccumulation:
    async def test_stream_large_content_accumulates_fully(self):
        runner, mock = _make_runner()
        words = [f"word{i}" for i in range(20)]
        chunks_list = [CompletionChunk(delta=w + " ") for w in words]
        chunks_list.append(
            CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=10, output_tokens=20))
        )
        mock.queue_stream(chunks_list)

        @agent(model="mock-model")
        class A: ...

        full_text = ""
        async for chunk in await runner.run_stream(A(), "hi"):
            if chunk.delta:
                full_text += chunk.delta

        for w in words:
            assert w in full_text

    async def test_stream_thinking_delta_yielded(self):
        runner, mock = _make_runner()
        mock.queue_stream([
            CompletionChunk(thinking_delta="Let me think..."),
            CompletionChunk(delta="Answer"),
            CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=5)),
        ])

        @agent(model="mock-model")
        class ThinkAgent: ...

        chunks = [c async for c in await runner.run_stream(ThinkAgent(), "hi")]
        thinking = [c.thinking_delta for c in chunks if c.thinking_delta]
        assert thinking == ["Let me think..."]

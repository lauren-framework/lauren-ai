"""Tests for lauren_ai._extractors — Agent, Completion, Embed, StreamCompletion."""

from __future__ import annotations

import pytest

from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage


def _compl(content: str = "ok") -> Completion:
    return Completion(
        id="c1",
        model="mock",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


def _make_transport_config(responses=None):
    cfg, mock = LLMConfig.for_testing()
    for r in responses or []:
        mock.queue_response(_compl(r))
    return mock, cfg


# ---------------------------------------------------------------------------
# Mock execution contexts
# ---------------------------------------------------------------------------


class _Ctx:
    """Minimal mock execution context."""

    def __init__(self, metadata=None, request=None):
        self._meta = metadata or {}
        self.request = request

    def get_metadata(self, key, default=None):
        return self._meta.get(key, default)


class _Request:
    def __init__(self, body=None, state_body=None, embed_input=None):
        self.body = body

        class _State:
            pass

        s = _State()
        if state_body is not None:
            s.body = state_body
        if embed_input is not None:
            s.embed_input = embed_input
        self.state = s


# ---------------------------------------------------------------------------
# Agent extractor
# ---------------------------------------------------------------------------


class TestAgentExtractor:
    @pytest.mark.asyncio
    async def test_extract_resolves_from_container(self):
        from lauren_ai._extractors import Agent

        sentinel = object()

        class FakeContainer:
            async def resolve(self, cls):
                return sentinel

        class FakeExtraction:
            inner_type = object

        extractor = Agent.__new__(Agent)
        extractor._runner = None
        extractor._container = FakeContainer()

        result = await extractor.extract(_Ctx(), FakeExtraction())
        assert result is sentinel

    def test_class_getitem_returns_annotated(self):
        from typing import Annotated, get_args, get_origin

        from lauren_ai._extractors import Agent

        class MyAgent:
            pass

        result = Agent[MyAgent]
        # Should be Annotated[MyAgent, Agent]
        assert get_origin(result) is Annotated
        args = get_args(result)
        assert args[0] is MyAgent
        assert args[1] is Agent


# ---------------------------------------------------------------------------
# Completion extractor
# ---------------------------------------------------------------------------


class TestCompletionExtractor:
    @pytest.mark.asyncio
    async def test_extract_with_explicit_prompt_metadata(self):
        """Priority 1: uses completion_prompt from metadata."""
        from lauren_ai._extractors import Completion as CompletionExtractor

        mock, cfg = _make_transport_config(["result content", "result content"])
        extractor = CompletionExtractor.__new__(CompletionExtractor)
        extractor._transport = mock
        extractor._config = cfg

        class _Extraction:
            inner_type = str

        ctx = _Ctx(metadata={"completion_prompt": "What is 2+2?"})
        result = await extractor.extract(ctx, _Extraction())
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_extract_prompt_from_field_name(self):
        """Priority 2: reads prompt from request state body field."""
        from lauren_ai._extractors import Completion as CompletionExtractor

        mock, cfg = _make_transport_config(["computed result", "computed result"])
        extractor = CompletionExtractor.__new__(CompletionExtractor)
        extractor._transport = mock
        extractor._config = cfg

        class _Extraction:
            inner_type = str

        req = _Request(state_body={"query": "tell me about AI"})
        ctx = _Ctx(
            metadata={"completion_prompt_field": "query"},
            request=req,
        )
        result = await extractor.extract(ctx, _Extraction())
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_extract_fallback_to_request_body(self):
        """Priority 3: falls back to repr(request.body)."""
        from lauren_ai._extractors import Completion as CompletionExtractor

        mock, cfg = _make_transport_config(["fallback result", "fallback result"])
        extractor = CompletionExtractor.__new__(CompletionExtractor)
        extractor._transport = mock
        extractor._config = cfg

        class _Extraction:
            inner_type = str

        req = _Request(body=b"raw body bytes")
        ctx = _Ctx(request=req)
        result = await extractor.extract(ctx, _Extraction())
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_extract_no_context_returns_empty_prompt(self):
        """No context at all → empty prompt → still completes."""
        from lauren_ai._extractors import Completion as CompletionExtractor

        mock, cfg = _make_transport_config(["empty", "empty"])
        extractor = CompletionExtractor.__new__(CompletionExtractor)
        extractor._transport = mock
        extractor._config = cfg

        class _Extraction:
            inner_type = str

        ctx = _Ctx()  # no metadata, no request
        result = await extractor.extract(ctx, _Extraction())
        assert isinstance(result, str)

    def test_resolve_prompt_explicit(self):
        from lauren_ai._extractors import Completion as CompletionExtractor

        ctx = _Ctx(metadata={"completion_prompt": "my prompt"})
        assert CompletionExtractor._resolve_prompt(ctx) == "my prompt"

    def test_resolve_prompt_field_from_body(self):
        from lauren_ai._extractors import Completion as CompletionExtractor

        req = _Request(state_body={"q": "question here"})
        ctx = _Ctx(metadata={"completion_prompt_field": "q"}, request=req)
        result = CompletionExtractor._resolve_prompt(ctx)
        assert result == "question here"

    def test_resolve_prompt_body_fallback(self):
        from lauren_ai._extractors import Completion as CompletionExtractor

        req = _Request(body=b"raw bytes")
        ctx = _Ctx(request=req)
        result = CompletionExtractor._resolve_prompt(ctx)
        assert "raw bytes" in result

    def test_resolve_prompt_empty(self):
        from lauren_ai._extractors import Completion as CompletionExtractor

        assert CompletionExtractor._resolve_prompt(_Ctx()) == ""

    def test_class_getitem(self):
        from typing import Annotated, get_origin

        from lauren_ai._extractors import Completion as CompletionExtractor

        result = CompletionExtractor[str]
        assert get_origin(result) is Annotated


# ---------------------------------------------------------------------------
# Embed extractor
# ---------------------------------------------------------------------------


class TestEmbedExtractor:
    @pytest.mark.asyncio
    async def test_extract_from_state_embed_input(self):
        from lauren_ai._extractors import Embed

        class FakeEmbedding:
            vector = [0.1, 0.2, 0.3]

        class FakeTransport:
            async def embed(self, texts, model=None, dimensions=None):
                return [FakeEmbedding()]

        cfg = LLMConfig(provider="anthropic", model="m", api_key="k")
        extractor = Embed.__new__(Embed)
        extractor._transport = FakeTransport()
        extractor._config = cfg

        req = _Request(embed_input="search query")
        ctx = _Ctx(request=req)

        class _Extraction:
            pass

        result = await extractor.extract(ctx, _Extraction())
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_extract_falls_back_to_body(self):
        from lauren_ai._extractors import Embed

        class FakeEmbedding:
            vector = [0.5]

        class FakeTransport:
            async def embed(self, texts, model=None, dimensions=None):
                return [FakeEmbedding()]

        cfg = LLMConfig(provider="anthropic", model="m", api_key="k")
        extractor = Embed.__new__(Embed)
        extractor._transport = FakeTransport()
        extractor._config = cfg

        req = _Request(body=b"hello world")
        ctx = _Ctx(request=req)

        class _Extraction:
            pass

        result = await extractor.extract(ctx, _Extraction())
        assert result == [0.5]

    @pytest.mark.asyncio
    async def test_extract_empty_embeddings_returns_empty(self):
        from lauren_ai._extractors import Embed

        class FakeTransport:
            async def embed(self, texts, model=None, dimensions=None):
                return []

        cfg = LLMConfig(provider="anthropic", model="m", api_key="k")
        extractor = Embed.__new__(Embed)
        extractor._transport = FakeTransport()
        extractor._config = cfg

        ctx = _Ctx()

        class _Extraction:
            pass

        result = await extractor.extract(ctx, _Extraction())
        assert result == []

    def test_resolve_embed_input_from_state(self):
        from lauren_ai._extractors import Embed

        req = _Request(embed_input="my query")
        ctx = _Ctx(request=req)
        assert Embed._resolve_embed_input(ctx) == "my query"

    def test_resolve_embed_input_from_bytes_body(self):
        from lauren_ai._extractors import Embed

        req = _Request(body=b"bytes content")
        ctx = _Ctx(request=req)
        result = Embed._resolve_embed_input(ctx)
        assert "bytes content" in result

    def test_resolve_embed_input_from_str_body(self):
        from lauren_ai._extractors import Embed

        req = _Request(body="string body")
        ctx = _Ctx(request=req)
        result = Embed._resolve_embed_input(ctx)
        assert "string body" in result

    def test_resolve_embed_input_empty(self):
        from lauren_ai._extractors import Embed

        assert Embed._resolve_embed_input(_Ctx()) == ""

    def test_class_getitem(self):
        from typing import Annotated, get_origin

        from lauren_ai._extractors import Embed

        result = Embed[list]
        assert get_origin(result) is Annotated


# ---------------------------------------------------------------------------
# StreamCompletion extractor
# ---------------------------------------------------------------------------


class TestStreamCompletionExtractor:
    @pytest.mark.asyncio
    async def test_extract_returns_stream(self):
        from lauren_ai._extractors import StreamCompletion
        from lauren_ai._transport import CompletionChunk

        async def fake_stream():
            yield CompletionChunk(delta="hello", stop_reason="end_turn", usage=None)

        class FakeTransport:
            async def complete(self, messages, **kwargs):
                return fake_stream()

        cfg = LLMConfig(provider="anthropic", model="m", api_key="k")
        extractor = StreamCompletion.__new__(StreamCompletion)
        extractor._transport = FakeTransport()
        extractor._config = cfg

        ctx = _Ctx(metadata={"completion_prompt": "Hi"})

        class _Extraction:
            pass

        stream = await extractor.extract(ctx, _Extraction())
        chunks = []
        async for chunk in stream:
            chunks.append(chunk.delta)
        assert chunks == ["hello"]

    def test_resolve_prompt_explicit(self):
        from lauren_ai._extractors import StreamCompletion

        ctx = _Ctx(metadata={"completion_prompt": "stream prompt"})
        assert StreamCompletion._resolve_prompt(ctx) == "stream prompt"

    def test_resolve_prompt_field_from_body(self):
        from lauren_ai._extractors import StreamCompletion

        req = _Request(state_body={"msg": "user input"})
        ctx = _Ctx(metadata={"completion_prompt_field": "msg"}, request=req)
        assert StreamCompletion._resolve_prompt(ctx) == "user input"

    def test_resolve_prompt_body_fallback(self):
        from lauren_ai._extractors import StreamCompletion

        req = _Request(body=b"raw")
        ctx = _Ctx(request=req)
        assert "raw" in StreamCompletion._resolve_prompt(ctx)

    def test_resolve_prompt_empty(self):
        from lauren_ai._extractors import StreamCompletion

        assert StreamCompletion._resolve_prompt(_Ctx()) == ""

    def test_class_getitem(self):
        from typing import Annotated, get_origin

        from lauren_ai._extractors import StreamCompletion

        result = StreamCompletion[str]
        assert get_origin(result) is Annotated

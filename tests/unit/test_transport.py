"""Unit tests for transport types and MockTransport."""
from __future__ import annotations

import pytest

from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    Embedding,
    Message,
    TokenUsage,
    ToolCall,
    ToolSchema,
)
from lauren_ai._transport._mock import MockTransport
from lauren_ai._exceptions import EmptyQueueError


class TestTokenUsage:
    def test_total_tokens(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_cost_usd_known_model(self):
        u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("claude-opus-4-6")
        # 15.0/1M input + 75.0/1M output = 90.0 USD
        assert cost == pytest.approx(90.0)

    def test_cost_usd_unknown_model(self):
        u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("unknown-model")
        # Default fallback: 1.0/1M input + 3.0/1M output = 4.0 USD
        assert cost == pytest.approx(4.0)

    def test_add(self):
        a = TokenUsage(input_tokens=10, output_tokens=5)
        b = TokenUsage(input_tokens=20, output_tokens=15)
        c = a + b
        assert c.input_tokens == 30
        assert c.output_tokens == 20


class TestMessage:
    def test_user_message(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_assistant_message(self):
        m = Message(role="assistant", content="Hi!")
        assert m.role == "assistant"

    def test_tool_result_content_block(self):
        from lauren_ai._transport import ContentBlock
        block = ContentBlock.tool_result_block(tool_use_id="tc1", content="result")
        assert block.type == "tool_result"
        assert block.tool_use_id == "tc1"
        assert block.content == "result"


class TestCompletion:
    def test_basic(self):
        c = Completion(
            id="c1",
            model="mock",
            content="Hello",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=5, output_tokens=10),
        )
        assert c.content == "Hello"
        assert c.stop_reason == "end_turn"
        assert c.usage.total_tokens == 15


class TestMockTransport:
    @pytest.mark.asyncio
    async def test_queue_and_dequeue(self):
        mock = MockTransport()
        completion = Completion(
            id="c1",
            model="mock",
            content="Test response",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        mock.queue_response(completion)
        result = await mock.complete(
            [Message(role="user", content="Hello")],
            model="mock",
        )
        assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_empty_queue_raises(self):
        mock = MockTransport()
        with pytest.raises(EmptyQueueError):
            await mock.complete(
                [Message(role="user", content="Hi")],
                model="mock",
            )

    @pytest.mark.asyncio
    async def test_calls_recorded(self):
        mock = MockTransport()
        mock.queue_response(
            Completion(
                id="c1", model="mock", content="Hi",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            )
        )
        await mock.complete([Message(role="user", content="Hello")], model="mock")
        assert len(mock.calls) == 1

    @pytest.mark.asyncio
    async def test_queue_tool_use(self):
        mock = MockTransport()
        mock.queue_tool_use("get_weather", {"city": "Paris"})
        result = await mock.complete(
            [Message(role="user", content="Weather?")],
            model="mock",
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].input == {"city": "Paris"}

    def test_reset(self):
        mock = MockTransport()
        mock.queue_response(
            Completion(
                id="c1", model="mock", content="x",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        )
        mock.reset()
        assert len(mock.calls) == 0

    @pytest.mark.asyncio
    async def test_embed(self):
        mock = MockTransport()
        embeddings = await mock.embed(["hello", "world"], model="mock-embed")
        assert len(embeddings) == 2
        # Default mock embeddings: zero vectors of dimension 1536
        assert len(embeddings[0].vector) == 1536

    @pytest.mark.asyncio
    async def test_count_tokens_heuristic(self):
        mock = MockTransport()
        tokens = await mock.count_tokens(
            [Message(role="user", content="Hello world")],
            model="mock",
        )
        # "Hello world" = 11 chars → ≈ 2 tokens (11 // 4 = 2)
        assert tokens >= 1

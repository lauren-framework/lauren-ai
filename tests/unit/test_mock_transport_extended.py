"""Extended tests for _transport/_mock.py — covers streaming, chunk aggregation,
embed queuing, and token counting."""

from __future__ import annotations

import pytest

from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    Embedding,
    Message,
    TokenUsage,
    ToolSchema,
    estimate_message_tokens,
)
from lauren_ai._transport._mock import (
    MockTransport,
    _aggregate_chunks,
    _completion_as_stream,
    _iter_chunks,
    _messages_token_count,
)

# ---------------------------------------------------------------------------
# Helper functions tests
# ---------------------------------------------------------------------------


class TestEstimateMessageTokens:
    """The shared dict-safe heuristic that every transport's count_tokens uses."""

    def test_basic_message(self):
        # 4-chars-per-token: a 400-char user message ≈ 100 tokens.
        assert estimate_message_tokens([{"role": "user", "content": "a" * 400}]) == 100

    def test_dict_and_dataclass_agree(self):
        from lauren_ai._transport import ContentBlock

        text = "hello there friend"
        as_dict = [{"role": "user", "content": text}]
        as_obj = [Message(role="user", content=[ContentBlock(type="text", text=text)])]
        assert estimate_message_tokens(as_dict) == estimate_message_tokens(as_obj)

    def test_counts_tool_use_input(self):
        # tool_use input must be counted (it is part of the billed request) even
        # though it is never truncated.
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"blob": "Z" * 4000}},
                ],
            }
        ]
        assert estimate_message_tokens(msgs) >= 1000  # ~4000 chars / 4


class TestMessagesTokenCount:
    def test_string_message(self):
        msgs = [Message(role="user", content="Hello")]
        count = _messages_token_count(msgs, system=None, tools=None)
        assert count >= 1

    def test_with_system(self):
        msgs = [Message(role="user", content="Hi")]
        count = _messages_token_count(msgs, system="You are helpful.", tools=None)
        assert count > _messages_token_count(msgs, system=None, tools=None)

    def test_with_tools(self):
        msgs = [Message(role="user", content="Hi")]
        tools = [
            ToolSchema(
                name="search",
                description="Search the web",
                input_schema={"type": "object", "properties": {}},
            )
        ]
        count = _messages_token_count(msgs, system=None, tools=tools)
        assert count > 0

    def test_with_content_blocks(self):
        from lauren_ai._transport import ContentBlock

        msgs = [Message(role="user", content=[ContentBlock(type="text", text="Hello there")])]
        count = _messages_token_count(msgs, system=None, tools=None)
        assert count >= 1


class TestIterChunks:
    @pytest.mark.asyncio
    async def test_iter_chunks_yields_all(self):
        chunks = [
            CompletionChunk(delta="Hello "),
            CompletionChunk(delta="world"),
        ]
        result = []
        async for chunk in _iter_chunks(chunks):
            result.append(chunk)
        assert len(result) == 2
        assert result[0].delta == "Hello "
        assert result[1].delta == "world"


class TestCompletionAsStream:
    @pytest.mark.asyncio
    async def test_streams_content_and_stop(self):
        completion = Completion(
            id="c1",
            model="mock",
            content="Hello world",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        chunks = []
        async for chunk in _completion_as_stream(completion):
            chunks.append(chunk)
        # Should yield text chunk and final stop chunk
        assert len(chunks) == 2
        assert chunks[0].delta == "Hello world"
        assert chunks[1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_empty_content_skips_text_chunk(self):
        completion = Completion(
            id="c1",
            model="mock",
            content="",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=5, output_tokens=0),
        )
        chunks = []
        async for chunk in _completion_as_stream(completion):
            chunks.append(chunk)
        # Only the stop chunk
        assert len(chunks) == 1
        assert chunks[0].stop_reason == "end_turn"


class TestAggregateChunks:
    def test_basic_text(self):
        chunks = [
            CompletionChunk(delta="Hello "),
            CompletionChunk(delta="world"),
            CompletionChunk(delta="", stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=2)),
        ]
        completion = _aggregate_chunks(chunks, model="test-model")
        assert completion.content == "Hello world"
        assert completion.stop_reason == "end_turn"
        assert completion.usage.input_tokens == 5
        assert completion.model == "test-model"

    def test_tool_call_aggregation(self):
        from lauren_ai._transport import ToolCallDelta

        chunks = [
            CompletionChunk(
                delta="",
                tool_call_delta=ToolCallDelta(
                    tool_use_id="tc1",
                    name="search",
                    input_delta='{"query": "',
                ),
            ),
            CompletionChunk(
                delta="",
                tool_call_delta=ToolCallDelta(
                    tool_use_id="tc1",
                    name=None,
                    input_delta='hello"}',
                ),
            ),
            CompletionChunk(delta="", stop_reason="tool_use"),
        ]
        completion = _aggregate_chunks(chunks, model="test-model")
        assert completion.stop_reason == "tool_use"
        assert len(completion.tool_calls) == 1
        tc = completion.tool_calls[0]
        assert tc.name == "search"
        assert tc.input == {"query": "hello"}

    def test_invalid_json_in_tool_call(self):
        from lauren_ai._transport import ToolCallDelta

        chunks = [
            CompletionChunk(
                delta="",
                tool_call_delta=ToolCallDelta(
                    tool_use_id="tc1",
                    name="tool",
                    input_delta="not valid json",
                ),
            ),
            CompletionChunk(delta="", stop_reason="tool_use"),
        ]
        completion = _aggregate_chunks(chunks, model="test-model")
        tc = completion.tool_calls[0]
        assert "_raw" in tc.input  # Fallback for invalid JSON

    def test_invalid_stop_reason_normalized(self):
        chunks = [
            CompletionChunk(delta="hi", stop_reason="unknown_reason"),
        ]
        completion = _aggregate_chunks(chunks, model="test")
        assert completion.stop_reason == "end_turn"

    def test_id_generated(self):
        chunks = [CompletionChunk(delta="hi")]
        completion = _aggregate_chunks(chunks, model="test")
        assert completion.id.startswith("mock_")


# ---------------------------------------------------------------------------
# MockTransport extended tests
# ---------------------------------------------------------------------------


class TestMockTransportExtended:
    @pytest.mark.asyncio
    async def test_streaming_with_completion_wraps(self):
        mock = MockTransport()
        completion = Completion(
            id="c1",
            model="mock",
            content="Hello!",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=5, output_tokens=3),
        )
        mock.queue_response(completion)
        stream = await mock.complete(
            [Message.user("Hi")],
            model="mock",
            stream=True,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_streaming_with_chunklist(self):
        mock = MockTransport()
        mock.queue_stream(
            [
                CompletionChunk(delta="Hello "),
                CompletionChunk(delta="world"),
                CompletionChunk(
                    delta="",
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=5, output_tokens=2),
                ),
            ]
        )
        stream = await mock.complete(
            [Message.user("Hi")],
            model="mock",
            stream=True,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello "

    @pytest.mark.asyncio
    async def test_aggregate_stream_to_completion(self):
        """Test that a queued stream is aggregated when stream=False."""
        mock = MockTransport()
        mock.queue_stream(
            [
                CompletionChunk(delta="Hello "),
                CompletionChunk(delta="world"),
                CompletionChunk(
                    delta="",
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=5, output_tokens=2),
                ),
            ]
        )
        result = await mock.complete(
            [Message.user("Hi")],
            model="mock",
            stream=False,
        )
        assert isinstance(result, Completion)
        assert result.content == "Hello world"

    @pytest.mark.asyncio
    async def test_queue_stream_empty_raises(self):
        mock = MockTransport()
        with pytest.raises(ValueError) as exc_info:
            mock.queue_stream([])
        assert "non-empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_queue_embed_empty_raises(self):
        mock = MockTransport()
        with pytest.raises(ValueError) as exc_info:
            mock.queue_embed([])
        assert "non-empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_with_queued_response(self):
        mock = MockTransport()
        mock.queue_embed(
            [
                Embedding(index=0, vector=[0.1, 0.2]),
                Embedding(index=1, vector=[0.3, 0.4]),
            ]
        )
        result = await mock.embed(["hello", "world"], model="test-embed")
        assert len(result) == 2
        assert result[0].vector == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_embed_default_zeros(self):
        mock = MockTransport()
        result = await mock.embed(["a", "b", "c"], model="test-embed")
        assert len(result) == 3
        for emb in result:
            assert len(emb.vector) == 1536
            assert all(v == 0.0 for v in emb.vector)

    @pytest.mark.asyncio
    async def test_embed_custom_dimensions(self):
        mock = MockTransport()
        result = await mock.embed(["test"], model="test-embed", dimensions=256)
        assert len(result[0].vector) == 256

    @pytest.mark.asyncio
    async def test_queue_tool_use_auto_id(self):
        mock = MockTransport()
        mock.queue_tool_use("get_weather", {"city": "Tokyo"})
        result = await mock.complete([Message.user("Weather?")], model="mock")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_use_id.startswith("toolu_")

    @pytest.mark.asyncio
    async def test_queue_tool_use_custom_id(self):
        mock = MockTransport()
        mock.queue_tool_use("get_weather", {"city": "Tokyo"}, tool_use_id="custom_id_123")
        result = await mock.complete([Message.user("Weather?")], model="mock")
        assert result.tool_calls[0].tool_use_id == "custom_id_123"

    @pytest.mark.asyncio
    async def test_queue_tool_use_with_content(self):
        mock = MockTransport()
        mock.queue_tool_use("search", {}, content="Let me search that for you.")
        result = await mock.complete([Message.user("Search?")], model="mock")
        assert result.content == "Let me search that for you."

    @pytest.mark.asyncio
    async def test_multiple_calls_recorded(self):
        mock = MockTransport()
        for i in range(3):
            mock.queue_response(
                Completion(
                    id=f"c{i}",
                    model="mock",
                    content=f"Response {i}",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                )
            )
        for _ in range(3):
            await mock.complete([Message.user("Hi")], model="mock")
        assert len(mock.calls) == 3

    def test_reset_clears_all(self):
        mock = MockTransport()
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="x",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        )
        mock.queue_embed([Embedding(index=0, vector=[1.0])])
        mock.reset()
        assert len(mock.calls) == 0
        assert len(mock._queue) == 0
        assert len(mock._embed_responses) == 0

    @pytest.mark.asyncio
    async def test_count_tokens_with_system_and_tools(self):
        mock = MockTransport()
        tools = [
            ToolSchema(
                name="search",
                description="Search the web",
                input_schema={"type": "object"},
            )
        ]
        count = await mock.count_tokens(
            [Message.user("Hello")],
            model="mock",
            system="You are helpful.",
            tools=tools,
        )
        assert count > 0

    @pytest.mark.asyncio
    async def test_call_records_all_params(self):
        mock = MockTransport()
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="OK",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=2),
            )
        )
        await mock.complete(
            [Message.user("Hello")],
            model="claude-haiku-4-5",
            system="Be helpful",
            max_tokens=512,
            temperature=0.5,
            stream=False,
        )
        call = mock.calls[0]
        assert call.model == "claude-haiku-4-5"
        assert call.system == "Be helpful"
        assert call.max_tokens == 512
        assert call.temperature == pytest.approx(0.5)
        assert call.stream is False

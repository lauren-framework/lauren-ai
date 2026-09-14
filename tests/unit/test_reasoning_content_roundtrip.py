"""Regression tests for OpenAI-compatible ``reasoning_content`` replay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace as dc_replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._config import LLMConfig
from lauren_ai._memory import ShortTermMemory, _shrink_message
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._transport import Completion, CompletionChunk, TokenUsage, ToolCall, ToolCallDelta
from lauren_ai._transport._mock import MockTransport
from lauren_ai._transport._openai import (
    OpenAITransport,
    _actionable_provider_message,
    _message_to_openai,
)


def _completion(**overrides: Any) -> Completion:
    values: dict[str, Any] = {
        "id": "c1",
        "model": "test-model",
        "content": "",
        "tool_calls": [],
        "stop_reason": "end_turn",
        "usage": TokenUsage(input_tokens=1, output_tokens=1),
    }
    values.update(overrides)
    return Completion(**values)


def _transport() -> OpenAITransport:
    config, _ = LLMConfig.for_testing()
    return OpenAITransport(dc_replace(config, provider="openai"))


def test_non_stream_response_captures_reasoning_content_from_mapping() -> None:
    result = _transport()._response_to_completion(
        {
            "id": "chatcmpl-1",
            "model": "gateway-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "reasoning_content": "exact\nreasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "goal_implemented", "arguments": '{"summary":"ok"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        },
        model="gateway-model",
    )

    assert result.reasoning_content == "exact\nreasoning"
    assert result.tool_calls[0].tool_use_id == "call-1"
    assert result.stop_reason == "tool_use"


def test_absent_and_empty_reasoning_content_are_distinct() -> None:
    absent = _transport()._response_to_completion(
        {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
        model="model",
    )
    empty = _transport()._response_to_completion(
        {"choices": [{"finish_reason": "stop", "message": {"content": "ok", "reasoning_content": ""}}]},
        model="model",
    )

    assert absent.reasoning_content is None
    assert empty.reasoning_content == ""


class _AsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _AsyncStream:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def __aiter__(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        for chunk in self._chunks:
            yield chunk


class _StreamingCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks

    async def create(self, **_kwargs: Any) -> _AsyncStream:
        return _AsyncStream(self.chunks)


def _openai_stream_chunk(*, reasoning: str | None = None, text: str | None = None, finish: str | None = None) -> Any:
    delta = SimpleNamespace(content=text, reasoning_content=reasoning, tool_calls=[])
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.mark.asyncio
async def test_stream_captures_reasoning_deltas_separately_from_text() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=_StreamingCompletions(
                [
                    _openai_stream_chunk(reasoning="first "),
                    _openai_stream_chunk(reasoning="second", text="visible"),
                    _openai_stream_chunk(finish="stop"),
                ]
            )
        )
    )

    chunks = [
        chunk
        async for chunk in _transport()._stream(
            client,
            {"model": "model", "messages": [], "stream": True},
            model="model",
        )
    ]

    assert [chunk.reasoning_content_delta for chunk in chunks if chunk.reasoning_content_delta is not None] == [
        "first ",
        "second",
    ]
    assert "visible" in "".join(chunk.delta for chunk in chunks)
    assert all(chunk.reasoning_content_delta not in ("visible",) for chunk in chunks)


@pytest.mark.asyncio
async def test_stream_does_not_duplicate_final_aggregate_reasoning_field() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=_StreamingCompletions(
                [
                    _openai_stream_chunk(reasoning="first "),
                    _openai_stream_chunk(reasoning="first second", finish="stop"),
                ]
            )
        )
    )

    chunks = [
        chunk
        async for chunk in _transport()._stream(
            client,
            {"model": "model", "messages": [], "stream": True},
            model="model",
        )
    ]

    assert (
        "".join(chunk.reasoning_content_delta or "" for chunk in chunks if chunk.reasoning_content_delta is not None)
        == "first second"
    )


def _agent_with_tool(tool_function: Any) -> Any:
    metadata = getattr(tool_function, TOOL_META)
    agent_decorator = cast(Any, agent(model="test-model"))
    tools_decorator = cast(Any, use_tools(cast(Callable[..., Any], tool_function)))

    @agent_decorator
    @tools_decorator
    class _Agent:
        pass

    cast(Any, _Agent).__lauren_ai_agent__.tools = {metadata.name: (tool_function, metadata)}
    return cast(Any, _Agent)


@pytest.mark.asyncio
async def test_runner_attaches_streamed_reasoning_to_tool_call_message() -> None:
    seen: list[str] = []

    @tool()
    async def goal_implemented(summary: str) -> str:
        """Record a completed goal."""
        seen.append(summary)
        return "recorded"

    Agent = _agent_with_tool(goal_implemented)
    tool_name = getattr(goal_implemented, TOOL_META).name
    transport = MockTransport()
    transport.queue_stream(
        [
            CompletionChunk(reasoning_content_delta="must preserve "),
            CompletionChunk(reasoning_content_delta="this"),
            CompletionChunk(tool_call_delta=ToolCallDelta(tool_use_id="call-1", name=tool_name, input_delta="")),
            CompletionChunk(
                tool_call_delta=ToolCallDelta(
                    tool_use_id="call-1",
                    name=None,
                    input_delta='{"summary":"done"}',
                )
            ),
            CompletionChunk(stop_reason="tool_use"),
        ]
    )
    transport.queue_stream([CompletionChunk(delta="finished"), CompletionChunk(stop_reason="end_turn")])

    memory = ShortTermMemory(max_tokens=200_000)
    runner = AgentRunnerBase(transport=transport)
    async for _ in await runner.run_stream(Agent(), "implement it", memory=memory):
        pass

    assert seen == ["done"]
    assistant = next(message for message in memory._messages if message.get("role") == "assistant")
    assert assistant["reasoning_content"] == "must preserve this"
    assert any(block.get("type") == "tool_use" for block in assistant["content"])


def test_openai_serializer_replays_reasoning_with_tool_call() -> None:
    memory = ShortTermMemory(max_tokens=100_000)
    memory.add_assistant(
        _completion(
            reasoning_content="do not drop me",
            tool_calls=[ToolCall(tool_use_id="call-1", name="goal_implemented", input={"summary": "ok"})],
            stop_reason="tool_use",
        )
    )

    payload = _message_to_openai(memory.messages()[0])[0]

    assert payload["reasoning_content"] == "do not drop me"
    assert payload["tool_calls"][0]["id"] == "call-1"


def test_snapshot_restore_and_compaction_preserve_reasoning_verbatim() -> None:
    memory = ShortTermMemory(max_tokens=100_000)
    memory.add_assistant(_completion(content="visible", reasoning_content="R\n" * 50))
    snapshot = memory.snapshot()

    restored = ShortTermMemory(max_tokens=100_000)
    restored.restore(snapshot)
    message = restored.messages()[0]
    shrunk = _shrink_message(message, target_chars=10)

    assert message["reasoning_content"] == "R\n" * 50
    assert shrunk["reasoning_content"] == "R\n" * 50


def test_messages_without_reasoning_keep_the_existing_wire_shape() -> None:
    assert _message_to_openai({"role": "assistant", "content": "ordinary"}) == [
        {"role": "assistant", "content": "ordinary"}
    ]


def test_malformed_restored_reasoning_content_is_rejected() -> None:
    memory = ShortTermMemory(max_tokens=100_000)

    with pytest.raises(ValueError, match="reasoning_content must be a string"):
        memory.add_assistant({"role": "assistant", "content": "old", "reasoning_content": 42})


def test_missing_reasoning_error_gets_a_non_retryable_recovery_hint() -> None:
    message = _actionable_provider_message(
        ValueError("The reasoning_content in the thinking mode must be passed back to the API")
    )

    assert "non-retryable" in message
    assert "start a new conversation" in message
    assert "reasoning_content" in message

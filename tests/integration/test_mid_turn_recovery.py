"""Fault-injection coverage for provider-step recovery (PRD-182)."""

from __future__ import annotations

import copy

import pytest

from lauren_ai import AgentConfig
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._exceptions import TransientTransportError
from lauren_ai._memory import ShortTermMemory
from lauren_ai._tools import _add_to_tool_map, tool
from lauren_ai._transport import Completion, CompletionChunk, TokenUsage
from lauren_ai._transport._mock import MockTransport

pytestmark = pytest.mark.integration


class _LateFailureStream:
    """Emit a partial response and then fail like a broken socket."""

    def __init__(self, *chunks: CompletionChunk) -> None:
        self._chunks = chunks

    def __aiter__(self):  # noqa: ANN204 - protocol-compatible async iterator
        return self._iterate()

    async def _iterate(self):  # noqa: ANN202 - test-only iterator
        for chunk in self._chunks:
            yield chunk
        raise TransientTransportError("late disconnect", status_code=503)


class _FaultTransport(MockTransport):
    """Mock transport that fails only the second provider step."""

    def __init__(self, *, fail_every_call_after_first: bool = False) -> None:
        super().__init__()
        self.calls_seen: list[list[object]] = []
        self._call_number = 0
        self._fail_every_call_after_first = fail_every_call_after_first

    async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201 - test double
        self.calls_seen.append(copy.deepcopy(messages))
        if self._call_number == 1 or (self._fail_every_call_after_first and self._call_number >= 1):
            self._call_number += 1
            return _LateFailureStream(CompletionChunk(delta="partial provider output"))
        self._call_number += 1
        return await super().complete(messages, **kwargs)


class _SignalRecorder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.signals: list[object] = []

    async def on_signal(self, signal: object) -> None:
        self.names.append(type(signal).__name__)
        self.signals.append(signal)


def _make_agent():
    @tool()
    async def record_note(value: str) -> str:
        return f"recorded:{value}"

    tools = {}
    _add_to_tool_map(tools, record_note)

    @use_tools(record_note)
    @agent(model="mock-model")
    class RecoveryAgent:
        pass

    RecoveryAgent.__lauren_ai_agent__.tools = tools
    return RecoveryAgent


def _completion(content: str) -> Completion:
    return Completion(
        id="final",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


@pytest.mark.asyncio
async def test_late_step_failure_retries_only_that_step_and_preserves_prior_messages() -> None:
    transport = _FaultTransport()
    transport.queue_tool_use("record_note", {"value": "step one"})
    transport.queue_response(_completion("completed after retry"))
    recorder = _SignalRecorder()
    memory = ShortTermMemory()
    runner = AgentRunnerBase(transport=transport)

    chunks = []
    async for chunk in await runner.run_stream(
        _make_agent(),
        "perform the work",
        memory=memory,
        config_override=AgentConfig(
            max_turns=4,
            transport_max_retries=1,
            transport_retry_base_delay_s=0,
        ),
        event_sinks=[recorder],
    ):
        chunks.append(chunk)

    assert len(transport.calls_seen) == 3
    # The failed second-step request and its retry are byte-equivalent and both
    # contain the already committed assistant/tool exchange.
    assert transport.calls_seen[1] == transport.calls_seen[2]
    assert len(transport.calls_seen[1]) == 3
    assert transport.calls_seen[1][0]["role"] == "user"
    assert transport.calls_seen[1][0]["content"] == "perform the work"
    assert transport.calls_seen[1][1]["role"] == "assistant"
    assert transport.calls_seen[1][2]["role"] == "user"
    assert transport.calls_seen[1][2]["content"][0]["content"] == "recorded:step one"

    messages = memory.messages()
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == "perform the work"
    assert messages[-1]["content"] == "completed after retry"
    assert not any(chunk.delta == "partial provider output" for chunk in chunks)
    assert recorder.names.count("AgentStepCommitted") == 2
    assert recorder.names.count("AgentStepRetryScheduled") == 1
    assert recorder.names.count("AgentStepInterrupted") == 1


@pytest.mark.asyncio
async def test_exhausted_step_failure_keeps_committed_context_and_quarantines_partial_text() -> None:
    transport = _FaultTransport(fail_every_call_after_first=True)
    transport.queue_tool_use("record_note", {"value": "step one"})
    recorder = _SignalRecorder()
    memory = ShortTermMemory()
    runner = AgentRunnerBase(transport=transport)

    with pytest.raises(TransientTransportError):
        async for _chunk in await runner.run_stream(
            _make_agent(),
            "perform the work",
            memory=memory,
            config_override=AgentConfig(
                max_turns=4,
                transport_max_retries=1,
                transport_retry_base_delay_s=0,
            ),
            event_sinks=[recorder],
        ):
            pass

    messages = memory.messages()
    assert len(messages) == 3
    assert messages[0]["content"] == "perform the work"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert all(message.get("content") != "partial provider output" for message in messages)
    assert recorder.names.count("AgentStepCommitted") == 1
    assert recorder.names.count("AgentStepInterrupted") == 2
    interrupted = [signal for signal in recorder.signals if type(signal).__name__ == "AgentStepInterrupted"]
    assert [vars(signal)["partial_text"] for signal in interrupted] == [
        "partial provider output",
        "partial provider output",
    ]


@pytest.mark.asyncio
async def test_follow_up_request_uses_the_same_memory_after_failed_turn() -> None:
    transport = _FaultTransport(fail_every_call_after_first=True)
    transport.queue_tool_use("record_note", {"value": "step one"})
    memory = ShortTermMemory()
    runner = AgentRunnerBase(transport=transport)

    with pytest.raises(TransientTransportError):
        async for _chunk in await runner.run_stream(
            _make_agent(),
            "perform the work",
            memory=memory,
            config_override=AgentConfig(
                max_turns=4,
                transport_max_retries=1,
                transport_retry_base_delay_s=0,
            ),
        ):
            pass

    # A later user message is a new logical turn, not a new conversation. The
    # failed turn's valid prefix must appear exactly once in the next request.
    transport._fail_every_call_after_first = False
    transport.queue_response(_completion("follow-up acknowledged"))
    async for _chunk in await runner.run_stream(
        _make_agent(),
        "what was completed?",
        memory=memory,
        config_override=AgentConfig(max_turns=1),
    ):
        pass

    follow_up = transport.calls_seen[-1]
    assert [message["role"] for message in follow_up] == [
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert follow_up[0]["content"] == "perform the work"
    assert follow_up[2]["content"][0]["content"] == "recorded:step one"
    assert follow_up[3]["content"] == "what was completed?"

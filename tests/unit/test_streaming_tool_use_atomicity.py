"""Tests for streaming tool-use atomicity (prd-streaming-tool-use-atomicity).

Covers three bugs fixed together:

1. Cancellation window between add_assistant and the inner try: — now closed by
   moving add_assistant inside the try/except BaseException block.
2. ensure_valid() called before each memory.messages() in _stream_loop to
   persist heals into _messages before further mutations.
3. _completion_as_stream now emits tool_call_delta chunks so mock transport
   + stream=True correctly exercises the tool execution path.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai._memory import ShortTermMemory
from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)
from lauren_ai._transport._mock import MockTransport, _completion_as_stream


# ── Fix 3: _completion_as_stream emits tool_call_delta ───────────────────────


class TestCompletionAsStreamToolCallDelta:
    @pytest.mark.asyncio
    async def test_emits_tool_call_delta_for_single_tool(self) -> None:
        """_completion_as_stream must emit a tool_call_delta for each tool call."""
        completion = Completion(
            id="msg_1",
            model="mock",
            content="",
            tool_calls=[ToolCall(tool_use_id="toolu_abc", name="my_tool", input={"k": "v"})],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        chunks = [chunk async for chunk in _completion_as_stream(completion)]

        tool_deltas = [c for c in chunks if c.tool_call_delta is not None]
        assert len(tool_deltas) == 1, f"Expected 1 tool_call_delta, got {len(tool_deltas)}"
        delta = tool_deltas[0].tool_call_delta
        assert delta.tool_use_id == "toolu_abc"
        assert delta.name == "my_tool"
        assert json.loads(delta.input_delta) == {"k": "v"}

    @pytest.mark.asyncio
    async def test_emits_tool_call_deltas_for_parallel_tools(self) -> None:
        """Multiple parallel tool calls each get their own delta chunk."""
        completion = Completion(
            id="msg_2",
            model="mock",
            content="",
            tool_calls=[ToolCall(tool_use_id=f"toolu_0{i}", name=f"tool_{i}", input={"n": i}) for i in range(1, 4)],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        chunks = [chunk async for chunk in _completion_as_stream(completion)]

        tool_deltas = [c for c in chunks if c.tool_call_delta is not None]
        assert len(tool_deltas) == 3

        ids = {c.tool_call_delta.tool_use_id for c in tool_deltas}
        assert ids == {"toolu_01", "toolu_02", "toolu_03"}

    @pytest.mark.asyncio
    async def test_no_tool_call_delta_when_no_tools(self) -> None:
        completion = Completion(
            id="msg_3",
            model="mock",
            content="Hello",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=5, output_tokens=10),
        )
        chunks = [chunk async for chunk in _completion_as_stream(completion)]
        tool_deltas = [c for c in chunks if c.tool_call_delta is not None]
        assert tool_deltas == []

    @pytest.mark.asyncio
    async def test_stop_reason_and_usage_always_in_final_chunk(self) -> None:
        completion = Completion(
            id="msg_4",
            model="mock",
            content="",
            tool_calls=[ToolCall(tool_use_id="tu_x", name="t", input={})],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=1, output_tokens=2),
        )
        chunks = [chunk async for chunk in _completion_as_stream(completion)]
        stop_chunks = [c for c in chunks if c.stop_reason is not None]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].stop_reason == "tool_use"
        assert stop_chunks[0].usage is not None


# ── Fix 3: queue_tool_use + stream=True now exercises tool execution ──────────


class TestMockTransportStreamToolUse:
    @pytest.mark.asyncio
    async def test_queue_tool_use_stream_emits_tool_call_delta(self) -> None:
        """queue_tool_use() + stream=True must produce tool_call_delta chunks."""
        mock = MockTransport()
        mock.queue_tool_use("calculator", {"expr": "2+2"}, tool_use_id="toolu_calc")

        stream = await mock.complete([], model="mock", stream=True)
        chunks = [chunk async for chunk in stream]

        tool_deltas = [c for c in chunks if c.tool_call_delta is not None]
        assert len(tool_deltas) == 1, "queue_tool_use + stream=True must now produce tool_call_delta chunks"
        delta = tool_deltas[0].tool_call_delta
        assert delta.tool_use_id == "toolu_calc"
        assert delta.name == "calculator"
        assert json.loads(delta.input_delta) == {"expr": "2+2"}

    @pytest.mark.asyncio
    async def test_memory_accumulates_tool_calls_from_stream(self) -> None:
        """_stream_loop builds accumulated_tool_calls from tool_call_delta chunks.

        Before Fix 3, queue_tool_use + stream=True always produced
        accumulated_tool_calls=[] because no tool_call_delta was emitted.
        """
        mock = MockTransport()
        mock.queue_tool_use("search", {"q": "docs"}, tool_use_id="toolu_search")

        memory = ShortTermMemory()
        memory.add_user("find the docs")

        stream = await mock.complete([], model="mock", stream=True)

        # Simulate what _stream_loop does when processing chunks
        partial_inputs: dict[str, str] = {}
        partial_names: dict[str, str] = {}
        stop_reason = None

        async for chunk in stream:
            if chunk.tool_call_delta is not None:
                tcd = chunk.tool_call_delta
                if tcd.name:
                    partial_names[tcd.tool_use_id] = tcd.name
                partial_inputs.setdefault(tcd.tool_use_id, "")
                partial_inputs[tcd.tool_use_id] += tcd.input_delta
            if chunk.stop_reason is not None:
                stop_reason = chunk.stop_reason

        assert stop_reason == "tool_use"
        assert "toolu_search" in partial_names
        assert partial_names["toolu_search"] == "search"
        assert json.loads(partial_inputs["toolu_search"]) == {"q": "docs"}


# ── Fix 1: add_assistant inside try/except BaseException ─────────────────────


class TestAddAssistantCancellationWindow:
    def test_ensure_valid_heals_after_add_assistant_no_tool_results(self) -> None:
        """Simulate: add_assistant committed, GeneratorExit before add_tool_results.

        After the fix, ensure_valid() is called in the except BaseException block
        which now covers add_assistant itself.  Memory should be consistent.
        """
        memory = ShortTermMemory()
        memory.add_user("please do something")

        # Simulate add_assistant with tool calls committed
        completion = Completion(
            id="c1",
            model="mock",
            content="",
            tool_calls=[
                ToolCall(tool_use_id="call_01", name="list_directory", input={"path": "."}),
                ToolCall(tool_use_id="call_02", name="search_files", input={"pattern": "*.md"}),
                ToolCall(tool_use_id="call_03", name="search_files", input={"pattern": "*.rst"}),
            ],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=50, output_tokens=30),
        )
        memory.add_assistant(completion)

        # State: orphaned tool_use — GeneratorExit arrived before tool execution
        assert len(memory._messages) == 2
        assert memory._messages[1]["role"] == "assistant"

        # ensure_valid() is what the except BaseException block calls (Fix 1)
        memory.ensure_valid()

        # Memory is now healed: one consolidated synthetic result
        tool_result_msgs = [
            m
            for m in memory._messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 1, "ensure_valid() must produce ONE consolidated synthetic result message"
        result_ids = {b["tool_use_id"] for b in tool_result_msgs[0]["content"] if b.get("type") == "tool_result"}
        assert result_ids == {"call_01", "call_02", "call_03"}

    def test_memory_consistent_after_heal_then_new_user_message(self) -> None:
        """After healing, add_user for the next turn leaves memory in valid state."""
        memory = ShortTermMemory()
        memory.add_user("original intent")

        completion = Completion(
            id="c1",
            model="m",
            content="",
            tool_calls=[ToolCall(tool_use_id="tu_1", name="t", input={})],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        memory.add_assistant(completion)

        # Fix 1: ensure_valid() in except BaseException
        memory.ensure_valid()

        # Fix 2: ensure_valid() called again at top of next loop iteration
        memory.ensure_valid()  # idempotent — must not double-heal

        memory.add_user("next turn message")

        msgs = memory.messages()

        # Check: no orphaned tool_use in the messages Anthropic would receive
        from lauren_ai._memory import _get_tool_call_ids, _get_tool_result_ids

        for i, msg in enumerate(msgs):
            if msg.get("role") == "assistant" and _get_tool_call_ids(msg):
                tool_ids = _get_tool_call_ids(msg)
                if i + 1 < len(msgs):
                    next_result_ids = _get_tool_result_ids(msgs[i + 1])
                    assert tool_ids <= next_result_ids, (
                        f"assistant at index {i} has tool_use IDs {tool_ids} "
                        f"not covered by next message's results {next_result_ids}"
                    )


# ── Fix 2: ensure_valid() before messages() in _stream_loop ──────────────────


class TestEnsureValidBeforeMessages:
    def test_ensure_valid_persists_heal_into_messages(self) -> None:
        """ensure_valid() mutates _messages; subsequent add_user builds on healed state."""
        memory = ShortTermMemory()
        memory.add_user("task")

        completion = Completion(
            id="c1",
            model="m",
            content="",
            tool_calls=[ToolCall(tool_use_id="tu_A", name="t", input={})],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        memory.add_assistant(completion)

        # Simulate Fix 2: ensure_valid() called before memory.messages() next loop
        memory.ensure_valid()

        # _messages now has the synthetic result in it
        assert any(
            m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in memory._messages
        ), "_messages must be healed in-place by ensure_valid()"

        # Now add the next user turn
        memory.add_user("follow-up")

        # memory.messages() should not produce a duplicate synthetic result
        msgs = memory.messages()
        synthetic_count = sum(
            1
            for m in msgs
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        assert synthetic_count == 1, (
            f"Expected exactly 1 synthetic result after ensure_valid + add_user, got {synthetic_count}"
        )

    def test_double_ensure_valid_is_idempotent(self) -> None:
        """Calling ensure_valid() twice must not double-inject synthetic results."""
        memory = ShortTermMemory()
        memory.add_user("t")

        completion = Completion(
            id="c1",
            model="m",
            content="",
            tool_calls=[ToolCall(tool_use_id="tu_1", name="t", input={})],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=5, output_tokens=3),
        )
        memory.add_assistant(completion)

        memory.ensure_valid()
        memory.ensure_valid()  # second call — must be idempotent

        result_msgs = [
            m
            for m in memory._messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(result_msgs) == 1, (
            f"Double ensure_valid must not duplicate synthetic results, got {len(result_msgs)}"
        )


# ── Regression: the exact scenario from the 400 error ────────────────────────


class TestAnthropicParallelToolUseRegression:
    @pytest.mark.asyncio
    async def test_seven_parallel_tools_from_stream_produce_valid_history(self) -> None:
        """Reproduce: agent makes 7 parallel file reads, stream is interrupted.

        The exact scenario from the production error:
          messages.1: tool_use ids found without tool_result blocks immediately after
          call_01_... call_02_... ... call_07_...
        """
        from lauren_ai._transport._anthropic import _message_to_anthropic

        memory = ShortTermMemory()
        memory.add_user("please concise the docs")

        # Agent responds with 7 parallel tool calls
        completion = Completion(
            id="c1",
            model="mock",
            content="",
            tool_calls=[
                ToolCall(tool_use_id=f"call_0{i}_xxx{i}", name="search_files", input={"pattern": f"**/*.{ext}"})
                for i, ext in enumerate(["md", "rst", "txt", "mdx", "html", "pdf", "doc"], start=1)
            ],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=100, output_tokens=80),
        )
        memory.add_assistant(completion)

        # Generator cancelled (GeneratorExit) — Fix 1 triggers ensure_valid()
        memory.ensure_valid()

        # Fix 2: ensure_valid() also called at top of next loop iteration
        memory.ensure_valid()

        # Next run_stream would add new user message
        memory.add_user("please concise the docs")  # re-sent

        # Convert to Anthropic format — must not raise and must have valid roles
        msgs_snapshot = memory.messages()
        converted = [_message_to_anthropic(m) for m in msgs_snapshot]

        for msg in converted:
            assert msg["role"] in ("user", "assistant"), f"Invalid Anthropic role: {msg['role']!r}"

        # Find the assistant message and check that the immediately following
        # message covers all its tool_use IDs
        for i, msg in enumerate(converted):
            if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                tool_use_ids = {b["id"] for b in msg["content"] if isinstance(b, dict) and b.get("type") == "tool_use"}
                if tool_use_ids:
                    assert i + 1 < len(converted), "Assistant must have a following message"
                    next_msg = converted[i + 1]
                    result_ids = {
                        b["tool_use_id"]
                        for b in (next_msg.get("content") or [])
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                    assert tool_use_ids <= result_ids, (
                        f"Anthropic would see orphaned tool_use IDs: {tool_use_ids - result_ids}"
                    )

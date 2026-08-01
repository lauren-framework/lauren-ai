"""Tests for the extended-thinking block round-trip (PRD-137 A+B+C+D).

When a response is in thinking mode and the assistant turn contains tool_use,
Anthropic requires the `thinking` blocks (with their signatures) to be passed
back — before the tool_use blocks — on the next request.  These tests cover the
full round-trip:

* **A** transport streaming captures `signature_delta` + `redacted_thinking`,
* **B** the runner reconstructs ordered `thinking_blocks` from the stream,
* **C** memory stores them first (thinking → text → tool_use),
* **D** the request serializer round-trips them verbatim.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from unittest.mock import MagicMock

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._config import LLMConfig
from lauren_ai._memory import ShortTermMemory
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    RedactedThinkingBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)
from lauren_ai._transport._anthropic import (
    AnthropicTransport,
    _content_block_to_anthropic,
    _message_to_anthropic,
)
from lauren_ai._transport._mock import MockTransport

# ── helpers ───────────────────────────────────────────────────────────────────


def _anthropic_transport() -> AnthropicTransport:
    config, _ = LLMConfig.for_testing()
    return AnthropicTransport(dc_replace(config, provider="anthropic"))


def _fake_stream(events: list[dict]):
    """Fake Anthropic SDK streaming context manager yielding mock events."""

    async def _aiter():
        for ev in events:
            chunk = MagicMock()
            chunk.type = ev["type"]
            if ev["type"] == "content_block_start":
                block = MagicMock()
                block.type = ev.get("block_type", "text")
                block.id = ev.get("block_id", None)
                block.name = ev.get("block_name", None)
                block.data = ev.get("data", "")
                chunk.content_block = block
            elif ev["type"] == "content_block_delta":
                delta = MagicMock()
                delta.type = ev.get("delta_type", "text_delta")
                delta.text = ev.get("text", "")
                delta.thinking = ev.get("thinking", "")
                delta.signature = ev.get("signature", "")
                delta.partial_json = ev.get("partial_json", "")
                chunk.delta = delta
            elif ev["type"] == "message_delta":
                delta = MagicMock()
                delta.stop_reason = ev.get("stop_reason", "end_turn")
                chunk.delta = delta
                chunk.usage = None
            yield chunk

    class _FakeStream:
        async def __aenter__(self):
            return _aiter()

        async def __aexit__(self, *_):
            pass

    return _FakeStream()


def _make_agent_with_tool(t):
    tool_meta = getattr(t, TOOL_META)

    @agent(model="test")
    @use_tools(t)
    class _Agent:
        pass

    _Agent.__lauren_ai_agent__.tools = {tool_meta.name: (t, tool_meta)}
    return _Agent


def _compl(**kw) -> Completion:
    base = dict(
        id="c1",
        model="m",
        content="",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    base.update(kw)
    return Completion(**base)


# ── A: streaming captures signature + redacted ─────────────────────────────────


class TestStreamCapture:
    @pytest.mark.asyncio
    async def test_captures_thinking_text_and_signature(self):
        transport = _anthropic_transport()
        events = [
            {"type": "content_block_start", "block_type": "thinking"},
            {"type": "content_block_delta", "delta_type": "thinking_delta", "thinking": "I should check the file"},
            {"type": "content_block_delta", "delta_type": "signature_delta", "signature": "SIGabc"},
            {"type": "content_block_stop"},
            {"type": "content_block_delta", "delta_type": "text_delta", "text": "Here is the answer"},
            {"type": "message_delta", "stop_reason": "end_turn"},
        ]
        client = MagicMock()
        client.messages.stream = MagicMock(return_value=_fake_stream(events))

        chunks = [c async for c in transport._stream(client, {"model": "test", "messages": []})]
        assert any(c.thinking_delta == "I should check the file" for c in chunks)
        assert any(c.thinking_signature == "SIGabc" for c in chunks)

    @pytest.mark.asyncio
    async def test_captures_redacted_thinking(self):
        transport = _anthropic_transport()
        events = [
            {"type": "content_block_start", "block_type": "redacted_thinking", "data": "OPAQUE=="},
            {"type": "content_block_stop"},
            {"type": "message_delta", "stop_reason": "end_turn"},
        ]
        client = MagicMock()
        client.messages.stream = MagicMock(return_value=_fake_stream(events))

        chunks = [c async for c in transport._stream(client, {"model": "test", "messages": []})]
        assert any(c.redacted_thinking_data == "OPAQUE==" for c in chunks)


# ── B + C: runner reconstructs + memory stores thinking first ──────────────────


class TestRunStreamStoresThinking:
    @pytest.mark.asyncio
    async def test_thinking_stored_before_tool_use(self):
        dispatched: list[str] = []

        @tool()
        async def greet(name: str) -> str:
            """Greet."""
            dispatched.append(name)
            return f"Hello {name}"

        Agent = _make_agent_with_tool(greet)
        tool_name = getattr(greet, TOOL_META).name

        mock = MockTransport()
        mock.queue_stream(
            [
                CompletionChunk(thinking_delta="I will greet them"),
                CompletionChunk(thinking_signature="SIG1"),
                CompletionChunk(delta="calling the tool"),
                CompletionChunk(tool_call_delta=ToolCallDelta(tool_use_id="t1", name=tool_name, input_delta="")),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="t1", name=None, input_delta='{"name": "Al"}')
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        mock.queue_stream([CompletionChunk(delta="Done."), CompletionChunk(stop_reason="end_turn")])

        mem = ShortTermMemory(max_tokens=200_000)
        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(Agent(), "greet Al", memory=mem):
            pass

        assert dispatched == ["Al"]
        asst = next(
            m
            for m in mem._messages
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        types = [b["type"] for b in asst["content"]]
        assert types[0] == "thinking", f"thinking must be first, got {types}"
        assert asst["content"][0]["thinking"] == "I will greet them"
        assert asst["content"][0]["signature"] == "SIG1"
        assert types.index("thinking") < types.index("tool_use")

    @pytest.mark.asyncio
    async def test_thinking_without_signature_still_preserved(self):
        # A gateway that omits signature_delta: the trailing thinking text must
        # still be preserved (finalised at end of stream with an empty signature).
        mock = MockTransport()
        mock.queue_stream(
            [
                CompletionChunk(thinking_delta="unsigned reasoning"),
                CompletionChunk(delta="answer"),
                CompletionChunk(stop_reason="end_turn"),
            ]
        )

        @agent(model="test")
        class Plain:
            pass

        Plain.__lauren_ai_agent__.tools = {}
        Plain.__lauren_ai_agent__.model = "test"
        mem = ShortTermMemory(max_tokens=200_000)
        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(Plain(), "hi", memory=mem):
            pass

        asst = mem._messages[-1]
        blocks = asst["content"]
        assert isinstance(blocks, list)
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "unsigned reasoning"


# ── C: add_assistant ordering (direct) ─────────────────────────────────────────


class TestAddAssistantOrdering:
    def test_thinking_first_then_text_then_tool_use(self):
        mem = ShortTermMemory(max_tokens=100_000)
        mem.add_assistant(
            _compl(
                content="text part",
                tool_calls=[ToolCall(tool_use_id="t1", name="f", input={"a": 1})],
                stop_reason="tool_use",
                thinking_blocks=[ThinkingBlock(thinking="reason", signature="S")],
            )
        )
        blocks = mem._messages[-1]["content"]
        assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use"]
        assert blocks[0]["signature"] == "S"

    def test_redacted_thinking_stored(self):
        mem = ShortTermMemory(max_tokens=100_000)
        mem.add_assistant(_compl(thinking_blocks=[RedactedThinkingBlock(data="D==")]))
        blocks = mem._messages[-1]["content"]
        assert blocks[0] == {"type": "redacted_thinking", "data": "D=="}

    def test_no_thinking_no_tools_stays_plain_string(self):
        mem = ShortTermMemory(max_tokens=100_000)
        mem.add_assistant(_compl(content="just text"))
        assert mem._messages[-1]["content"] == "just text"


# ── D: serialization round-trip ────────────────────────────────────────────────


class TestSerialization:
    def test_content_block_to_anthropic_thinking(self):
        out = _content_block_to_anthropic({"type": "thinking", "thinking": "t", "signature": "s"})
        assert out == {"type": "thinking", "thinking": "t", "signature": "s"}

    def test_content_block_to_anthropic_redacted(self):
        out = _content_block_to_anthropic({"type": "redacted_thinking", "data": "d"})
        assert out == {"type": "redacted_thinking", "data": "d"}

    def test_full_message_thinking_before_tool_use(self):
        # The regression: the assembled request must carry thinking before tool_use.
        mem = ShortTermMemory(max_tokens=100_000)
        mem.add_assistant(
            _compl(
                content="x",
                tool_calls=[ToolCall(tool_use_id="t1", name="f", input={})],
                stop_reason="tool_use",
                thinking_blocks=[ThinkingBlock(thinking="r", signature="S")],
            )
        )
        am = _message_to_anthropic(mem._messages[-1])
        types = [b["type"] for b in am["content"]]
        assert types[0] == "thinking"
        assert am["content"][0]["signature"] == "S"
        assert types.index("thinking") < types.index("tool_use")


# ── snapshot/restore (PRD-137 8) ───────────────────────────────────────────────


class TestPersistence:
    def test_thinking_survives_snapshot_restore(self):
        mem = ShortTermMemory(max_tokens=100_000)
        mem.add_assistant(
            _compl(
                tool_calls=[ToolCall(tool_use_id="t1", name="f", input={})],
                stop_reason="tool_use",
                thinking_blocks=[ThinkingBlock(thinking="r", signature="S")],
            )
        )
        snap = mem.snapshot()
        restored = ShortTermMemory(max_tokens=100_000)
        restored.restore(snap)
        blocks = restored._messages[-1]["content"]
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["signature"] == "S"


# ── E: thinking is never truncated by the context guard ────────────────────────


class TestThinkingImmutability:
    def test_thinking_not_truncated_while_tool_result_is(self):
        from lauren_ai._memory import _enforce_char_budget

        msgs = [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "X" * 5_000, "signature": "SIG"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Q" * 200_000}]},
        ]
        out = _enforce_char_budget(msgs, 8_000)
        thinking = next(b for b in out[1]["content"] if b["type"] == "thinking")
        assert thinking["thinking"] == "X" * 5_000, "thinking text must be untouched (signature integrity)"
        assert thinking["signature"] == "SIG"
        tool_result = out[2]["content"][0]
        assert len(str(tool_result["content"])) < 200_000, "the tool_result is what gets truncated"

    def test_redacted_thinking_not_truncated(self):
        from lauren_ai._memory import _shrink_message

        msg = {
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "D" * 50_000}],
        }
        out = _shrink_message(msg, 500)
        assert out["content"][0]["data"] == "D" * 50_000  # opaque payload untouched


# ── F: heal keeps thinking attached to its tool-using turn ─────────────────────


class TestHealKeepsThinking:
    def test_heal_does_not_strip_thinking(self):
        from lauren_ai._memory import _heal_dangling_tail

        snapshot = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reason", "signature": "S"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                ],
            },
            {"role": "user", "content": "moved on"},  # conversation progressed → heal inserts a result
        ]
        healed = _heal_dangling_tail(snapshot)
        asst = healed[1]["content"]
        assert asst[0]["type"] == "thinking" and asst[0]["signature"] == "S"
        assert any(b["type"] == "tool_use" for b in asst)
        # a synthetic tool_result was inserted right after the assistant turn
        inserted = healed[2]["content"]
        assert isinstance(inserted, list) and inserted[0]["type"] == "tool_result"
        assert inserted[0]["tool_use_id"] == "t1"

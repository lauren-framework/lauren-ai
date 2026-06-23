"""Tests for the hard pre-send context-budget guard (PRD-133 Layer C)."""

from __future__ import annotations

from lauren_ai import ToolResult
from lauren_ai._memory import (
    ShortTermMemory,
    _enforce_char_budget,
    _message_char_length,
    _shrink_message,
    _truncate_str,
)
from lauren_ai._transport import Completion, TokenUsage, ToolCall


def _total(messages: list) -> int:
    return sum(_message_char_length(m) for m in messages)


def _blocks(messages: list, btype: str) -> list:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            out.extend(b for b in content if isinstance(b, dict) and b.get("type") == btype)
    return out


class TestTruncateStr:
    def test_short_unchanged(self) -> None:
        assert _truncate_str("hello", 100) == "hello"

    def test_truncates_with_marker_to_target(self) -> None:
        out = _truncate_str("A" * 10_000, 1_000)
        assert len(out) <= 1_000
        assert "truncated" in out
        assert out.startswith("A") and out.endswith("A")  # head + tail kept


class TestShrinkMessage:
    def test_string_content(self) -> None:
        msg = {"role": "user", "content": "X" * 5_000}
        out = _shrink_message(msg, 500)
        assert _message_char_length(out) <= 500
        assert out["role"] == "user"

    def test_preserves_block_structure(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "Z" * 50_000},
            ],
        }
        out = _shrink_message(msg, 500)
        assert _message_char_length(out) <= 500
        block = out["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t1"  # id preserved
        assert "truncated" in block["content"]

    def test_does_not_truncate_tool_use_input(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Y" * 50_000},
                {"type": "tool_use", "tool_use_id": "t1", "name": "f", "input": {"a": 1}},
            ],
        }
        out = _shrink_message(msg, 500)
        tu = [b for b in out["content"] if b["type"] == "tool_use"][0]
        assert tu["input"] == {"a": 1}  # tool_use input untouched


class TestEnforceCharBudget:
    def test_within_budget_unchanged(self) -> None:
        msgs = [{"role": "user", "content": "small"}]
        assert _enforce_char_budget(msgs, 10_000) is msgs

    def test_over_budget_fits(self) -> None:
        msgs = [
            {"role": "user", "content": "question"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "Q" * 200_000}]},
        ]
        out = _enforce_char_budget(msgs, 4_000)
        assert _total(out) <= 4_000

    def test_zero_budget_noop(self) -> None:
        msgs = [{"role": "user", "content": "x" * 100}]
        assert _enforce_char_budget(msgs, 0) is msgs


class TestMessagesHardCap:
    """The end-to-end invariant: messages() never exceeds the budget."""

    def test_oversized_single_turn_is_capped(self) -> None:
        mem = ShortTermMemory(max_tokens=1_000)  # budget = 4_000 chars
        mem.add_user("please list everything")
        mem.add_assistant(
            Completion(
                id="c1",
                model="m",
                content="",
                tool_calls=[ToolCall(tool_use_id="t1", name="list_directory", input={"recursive": True})],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        )
        mem.add_tool_results([ToolResult.ok("X" * 500_000, tool_use_id="t1")])

        msgs = mem.messages()
        assert _total(msgs) <= 1_000 * 4, "request must fit the budget (no overflow)"
        # conversational user message survives (the floor is respected)
        assert any(m.get("role") == "user" and m.get("content") == "please list everything" for m in msgs)
        # tool_use/tool_result blocks both survive truncation (structure preserved)
        tus = _blocks(msgs, "tool_use")
        trs = _blocks(msgs, "tool_result")
        assert len(tus) == 1 and len(trs) == 1
        assert trs[0]["tool_use_id"] == "t1"  # the (truncated) tool_result keeps its id

    def test_normal_conversation_not_truncated(self) -> None:
        mem = ShortTermMemory(max_tokens=32_000)
        mem.add_user("hello")
        mem.add_assistant(
            Completion(
                id="c",
                model="m",
                content="hi there",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        )
        msgs = mem.messages()
        assert not any("truncated" in str(m.get("content", "")) for m in msgs)

"""Tests for Anthropic tool_use / tool_result conversation healing.

Covers the three-layer fix for the 400 error:
  "tool_use ids found without tool_result blocks immediately after"

Layer 1: Healer generates canonical (Anthropic-compatible) synthetic results.
Layer 2: _message_to_anthropic converts legacy role:"tool" messages defensively.
Layer 3: Streaming loop heals memory on cancellation between add_assistant and
         add_tool_result.
"""

from __future__ import annotations

import pytest

from lauren_ai._memory import (
    _INTERRUPTED_CONTENT,
    ShortTermMemory,
    _get_tool_result_ids,
    _heal_dangling_tail,
    _heal_dangling_tail_unconditional,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _assistant_with_tool_use(tool_use_id: str = "tu_1") -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll call the tool."},
            {"type": "tool_use", "id": tool_use_id, "name": "my_tool", "input": {}},
        ],
    }


def _tool_result(tool_use_id: str = "tu_1") -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "done"}],
    }


def _user(text: str = "hello") -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str = "hi") -> dict:
    return {"role": "assistant", "content": text}


# ── Layer 1: healer generates canonical format ────────────────────────────────


class TestHealerCanonicalFormat:
    def test_heal_dangling_tail_synthetic_uses_canonical_format(self) -> None:
        """Synthetic results must use role:'user' + tool_result block, not role:'tool'."""
        snapshot = [
            _user("do something"),
            _assistant_with_tool_use("tu_abc"),
            _user("next message"),  # has_moved_on = True
        ]
        healed = _heal_dangling_tail(snapshot)
        # The synthetic entry should be at index 2 (before the next user msg)
        synthetic = healed[2]
        assert synthetic["role"] == "user", "Synthetic result must have role:'user'"
        assert isinstance(synthetic["content"], list)
        assert synthetic["content"][0]["type"] == "tool_result"
        assert synthetic["content"][0]["tool_use_id"] == "tu_abc"
        assert "tool_call_id" not in synthetic
        assert synthetic.get("tool_call_id") is None

    def test_heal_dangling_tail_no_legacy_role_tool(self) -> None:
        """Healer must never generate role:'tool' (OpenAI-only format)."""
        snapshot = [_user(), _assistant_with_tool_use("tu_x"), _user("moved on")]
        healed = _heal_dangling_tail(snapshot)
        for msg in healed:
            assert msg.get("role") != "tool", f"Found legacy role:'tool' in {msg}"

    def test_heal_dangling_tail_unconditional_canonical_format(self) -> None:
        """_heal_dangling_tail_unconditional also uses canonical format."""
        snapshot = [_user(), _assistant_with_tool_use("tu_y")]
        healed = _heal_dangling_tail_unconditional(snapshot)
        synthetic = healed[-1]
        assert synthetic["role"] == "user"
        assert synthetic["content"][0]["type"] == "tool_result"
        assert synthetic["content"][0]["tool_use_id"] == "tu_y"

    def test_heal_dangling_tail_unconditional_no_legacy_role_tool(self) -> None:
        snapshot = [_user(), _assistant_with_tool_use("id1")]
        healed = _heal_dangling_tail_unconditional(snapshot)
        for msg in healed:
            assert msg.get("role") != "tool"

    def test_synthetic_content_is_interrupted_message(self) -> None:
        snapshot = [_user(), _assistant_with_tool_use("tu_z"), _user("next")]
        healed = _heal_dangling_tail(snapshot)
        synthetic = healed[2]
        assert synthetic["content"][0]["content"] == _INTERRUPTED_CONTENT

    def test_healed_result_detected_by_get_tool_result_ids(self) -> None:
        """The synthetic message must be recognised as a tool result."""
        snapshot = [_user(), _assistant_with_tool_use("tu_id"), _user("next")]
        healed = _heal_dangling_tail(snapshot)
        synthetic = healed[2]
        ids = _get_tool_result_ids(synthetic)
        assert "tu_id" in ids

    def test_no_healing_when_result_already_present(self) -> None:
        """If tool_result already exists, healer must not add duplicates."""
        snapshot = [
            _user(),
            _assistant_with_tool_use("tu_1"),
            _tool_result("tu_1"),
            _user("done"),
        ]
        healed = _heal_dangling_tail(snapshot)
        assert healed == snapshot

    def test_multiple_missing_ids_consolidated_into_one_message(self) -> None:
        """Multiple missing IDs must produce ONE message with all results.

        Anthropic requires all tool_results for one assistant turn to be in
        the *immediately following* user message.  Separate per-ID messages
        fail because the 'next message' only carries one result, leaving the
        rest orphaned — the exact error that triggered this fix.
        """
        snapshot = [
            _user(),
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "a", "name": "t1", "input": {}},
                    {"type": "tool_use", "id": "b", "name": "t2", "input": {}},
                ],
            },
            _user("next"),
        ]
        healed = _heal_dangling_tail(snapshot)
        synthetic_msgs = [
            m
            for m in healed
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        # Exactly ONE synthetic message must be injected (not two)
        assert len(synthetic_msgs) == 1, f"Expected 1 consolidated synthetic message, got {len(synthetic_msgs)}"
        # That one message must contain BOTH tool_result blocks
        healed_ids = {b["tool_use_id"] for b in synthetic_msgs[0]["content"] if b.get("type") == "tool_result"}
        assert healed_ids == {"a", "b"}

    def test_seven_parallel_tool_uses_consolidated(self) -> None:
        """Regression: 7 parallel tool_uses must produce ONE user message with 7 blocks."""
        tool_ids = [f"call_0{i}_xxx" for i in range(1, 8)]
        snapshot = [
            _user("explore the repo"),
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tid, "name": "read_file", "input": {}} for tid in tool_ids],
            },
        ]
        healed = _heal_dangling_tail_unconditional(snapshot)
        synthetic_msgs = [
            m
            for m in healed
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(synthetic_msgs) == 1, (
            f"7 parallel tool_uses must be healed into ONE user message, got {len(synthetic_msgs)} messages"
        )
        healed_ids = {b["tool_use_id"] for b in synthetic_msgs[0]["content"] if b.get("type") == "tool_result"}
        assert healed_ids == set(tool_ids), "All 7 IDs must be in the single message"


# ── Real execution: add_tool_results consolidation ────────────────────────────


class TestAddToolResultsConsolidation:
    """add_tool_results() must produce ONE message for parallel tool calls."""

    def _make_result(self, tool_use_id: str, content: str = "ok") -> object:
        from types import SimpleNamespace

        return SimpleNamespace(tool_use_id=tool_use_id, content=content, is_error=False)

    def test_single_result_adds_one_message(self) -> None:
        mem = ShortTermMemory()
        mem.add_tool_results([self._make_result("tu_1", "result")])
        assert len(mem._messages) == 1
        msg = mem._messages[0]
        assert msg["role"] == "user"
        assert msg["content"][0]["tool_use_id"] == "tu_1"

    def test_multiple_results_produce_one_consolidated_message(self) -> None:
        """3 parallel results must produce ONE user message with 3 blocks."""
        mem = ShortTermMemory()
        results = [self._make_result(f"call_0{i}") for i in range(1, 4)]
        mem.add_tool_results(results)
        assert len(mem._messages) == 1, f"Expected 1 consolidated message, got {len(mem._messages)}"
        msg = mem._messages[0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 3
        ids = {b["tool_use_id"] for b in msg["content"]}
        assert ids == {"call_01", "call_02", "call_03"}

    def test_empty_results_adds_nothing(self) -> None:
        mem = ShortTermMemory()
        mem.add_tool_results([])
        assert mem._messages == []

    def test_consolidated_results_pass_anthropic_conversion(self) -> None:
        """After consolidation, messages() produces valid Anthropic input."""
        from lauren_ai._transport._anthropic import _message_to_anthropic

        mem = ShortTermMemory()
        mem.add_user("find files")
        mem._messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"call_0{i}", "name": "search", "input": {}} for i in range(1, 4)
                ],
            }
        )
        results = [self._make_result(f"call_0{i}") for i in range(1, 4)]
        mem.add_tool_results(results)

        converted = [_message_to_anthropic(m) for m in mem.messages()]
        for msg in converted:
            assert msg["role"] in ("user", "assistant"), f"Invalid role: {msg['role']!r}"
        # The tool_result message must be at index 2 (right after assistant)
        assert converted[2]["role"] == "user"
        content = converted[2]["content"]
        assert len(content) == 3
        assert all(b["type"] == "tool_result" for b in content)

    def test_consecutive_parallel_turns_each_consolidated(self) -> None:
        """Two sequential rounds of parallel tool calls each produce one message."""
        mem = ShortTermMemory()
        mem.add_user("go")
        mem._messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "a1", "name": "t", "input": {}},
                    {"type": "tool_use", "id": "a2", "name": "t", "input": {}},
                ],
            }
        )
        mem.add_tool_results([self._make_result("a1"), self._make_result("a2")])
        mem._messages.append({"role": "assistant", "content": "got results"})
        mem._messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "b1", "name": "t", "input": {}},
                    {"type": "tool_use", "id": "b2", "name": "t", "input": {}},
                ],
            }
        )
        mem.add_tool_results([self._make_result("b1"), self._make_result("b2")])

        # Count consolidated result messages
        result_msgs = [
            m
            for m in mem._messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(result_msgs) == 2, "Each parallel turn should produce one result message"
        assert len(result_msgs[0]["content"]) == 2
        assert len(result_msgs[1]["content"]) == 2


# ── Layer 2: _message_to_anthropic defensive conversion ───────────────────────


class TestAnthropicMessageConversion:
    def test_legacy_role_tool_converted_to_user_tool_result(self) -> None:
        """role:'tool' message must be converted to role:'user' / tool_result."""
        from lauren_ai._transport._anthropic import _message_to_anthropic

        legacy = {"role": "tool", "tool_call_id": "tc_123", "content": "[interrupted]"}
        result = _message_to_anthropic(legacy)

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        block = result["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tc_123"
        assert block["content"] == "[interrupted]"

    def test_legacy_role_tool_empty_tool_call_id(self) -> None:
        """A legacy result without an ID is blocked before provider I/O."""
        from lauren_ai._exceptions import ToolConversationIntegrityError
        from lauren_ai._transport._anthropic import _message_to_anthropic

        legacy = {"role": "tool", "content": "result"}
        with pytest.raises(ToolConversationIntegrityError, match="without an ID"):
            _message_to_anthropic(legacy)

    def test_normal_user_message_unchanged(self) -> None:
        from lauren_ai._transport._anthropic import _message_to_anthropic

        msg = {"role": "user", "content": "hello"}
        assert _message_to_anthropic(msg) == {"role": "user", "content": "hello"}

    def test_normal_assistant_message_unchanged(self) -> None:
        from lauren_ai._transport._anthropic import _message_to_anthropic

        msg = {"role": "assistant", "content": "hi"}
        assert _message_to_anthropic(msg) == {"role": "assistant", "content": "hi"}

    def test_canonical_tool_result_block_passes_through(self) -> None:
        """New canonical format (role:'user' / tool_result) must pass through correctly."""
        from lauren_ai._transport._anthropic import _message_to_anthropic

        canonical = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "done"}],
        }
        result = _message_to_anthropic(canonical)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "tu_1"


# ── Layer 3: streaming loop heals on cancellation ────────────────────────────


class TestStreamingLoopHealing:
    def test_ensure_valid_heals_orphaned_tool_use(self) -> None:
        """ensure_valid() on a memory with orphaned tool_use produces valid history."""
        mem = ShortTermMemory()
        mem.add_user("please call the tool")

        # Simulate: assistant message with tool_use committed, then cancellation
        # (no tool_result added)
        mem._messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_orphan", "name": "t", "input": {}}],
            }
        )

        # ensure_valid() should inject a synthetic result
        mem.ensure_valid()

        messages = mem._messages
        # The synthetic result should appear immediately after the assistant msg
        synthetic = messages[2]
        assert synthetic["role"] == "user"
        assert isinstance(synthetic["content"], list)
        assert synthetic["content"][0]["type"] == "tool_result"
        assert synthetic["content"][0]["tool_use_id"] == "tu_orphan"

    def test_ensure_valid_canonical_format_on_orphaned_tool_use(self) -> None:
        """After ensure_valid(), no role:'tool' messages should exist."""
        mem = ShortTermMemory()
        mem.add_user("go")
        mem._messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": "f", "input": {}}],
            }
        )
        mem.ensure_valid()
        for msg in mem._messages:
            assert msg.get("role") != "tool", f"Found legacy role:'tool': {msg}"

    def test_memory_consistent_after_healed_orphan_allows_new_user_turn(self) -> None:
        """After healing, a new user message can be added without duplication."""
        mem = ShortTermMemory()
        mem.add_user("first")
        mem._messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_1", "name": "t", "input": {}}],
            }
        )
        mem.ensure_valid()
        mem.add_user("second")

        # No duplicate synthetic entries
        tool_results = [
            m
            for m in mem._messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m.get("content", []))
        ]
        assert len(tool_results) == 1, f"Expected 1 synthetic result, got {len(tool_results)}"

    def test_healed_history_passes_anthropic_message_conversion(self) -> None:
        """A healed history must convert to valid Anthropic messages without error."""
        from lauren_ai._transport._anthropic import _message_to_anthropic

        mem = ShortTermMemory()
        mem.add_user("go")
        mem._messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_1", "name": "t", "input": {}}],
            }
        )
        mem.ensure_valid()
        mem.add_user("continue")

        # Must not raise; all messages must have valid Anthropic roles
        converted = [_message_to_anthropic(m) for m in mem._messages]
        for msg in converted:
            assert msg["role"] in ("user", "assistant"), f"Invalid Anthropic role: {msg['role']!r} in {msg}"

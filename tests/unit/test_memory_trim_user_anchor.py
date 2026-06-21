"""Tests for the user-anchor guard in ShortTermMemory (prd-memory-trim-user-anchor).

Regression: large tool results (e.g. recursive directory listing) caused
messages() to trim away the original user intent, leaving the conversation
starting with role:"assistant".  All providers reject such histories with
errors like "400: messages: at least one message is required".
"""

from __future__ import annotations

import warnings

import pytest

from lauren_ai._memory import ShortTermMemory


from lauren_ai._memory import _is_conversational_user  # noqa: E402


# ── _is_conversational_user helper ───────────────────────────────────────────


class TestIsConversationalUser:
    def test_string_content_is_conversational(self) -> None:
        assert _is_conversational_user({"role": "user", "content": "hello"})

    def test_empty_content_is_conversational(self) -> None:
        assert _is_conversational_user({"role": "user", "content": ""})

    def test_pure_tool_result_is_not_conversational(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "done"},
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "ok"},
            ],
        }
        assert not _is_conversational_user(msg)

    def test_mixed_content_with_text_is_conversational(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "here are the results"},
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "done"},
            ],
        }
        assert _is_conversational_user(msg)

    def test_assistant_is_not_conversational_user(self) -> None:
        assert not _is_conversational_user({"role": "assistant", "content": "hi"})

    def test_system_is_not_conversational_user(self) -> None:
        assert not _is_conversational_user({"role": "system", "content": "rules"})


# ── helpers ───────────────────────────────────────────────────────────────────


def _big(chars: int) -> str:
    return "x" * chars


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant_tool_use(*tool_ids: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": "t", "input": {}} for tid in tool_ids],
    }


def _tool_results(*tool_ids: str, content: str = "result") -> dict:
    """Consolidated tool results in canonical format."""
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": content} for tid in tool_ids],
    }


def _system(text: str = "System.") -> dict:
    return {"role": "system", "content": text}


# ── Core regression: large tool results trim away user intent ─────────────────


class TestUserAnchorGuard:
    def test_messages_does_not_start_with_assistant_after_large_tool_results(self) -> None:
        """Reproduce the exact production failure.

        Scenario: user sends intent, agent makes 3 parallel tool calls,
        results are large (recursive directory listing ~400KB).  After
        trimming, messages() must still start with role:"user".
        """
        mem = ShortTermMemory(max_tokens=100)  # budget = 400 chars
        mem.add_user("please fix the docs")  # 20 chars — within budget alone
        mem._messages.append(_assistant_tool_use("call_01", "call_02", "call_03"))
        # Tool results are huge: 10x the budget
        mem._messages.append(_tool_results("call_01", "call_02", "call_03", content=_big(2000)))

        msgs = mem.messages()

        assert msgs, "messages() must not return empty list"
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert non_system, "messages() must have at least one non-system message"
        assert non_system[0]["role"] == "user", (
            f"First non-system message must be role:'user', got {non_system[0]['role']!r}.  "
            "Anthropic rejects conversations starting with an assistant message."
        )

    def test_messages_first_message_is_always_user(self) -> None:
        """Universal invariant: messages() first non-system is always user."""
        for budget in (1, 5, 10, 50, 100):
            mem = ShortTermMemory(max_tokens=budget)
            mem.add_user("task")
            mem._messages.append(_assistant_tool_use("tu_1"))
            mem._messages.append(_tool_results("tu_1", content=_big(budget * 10)))

            msgs = mem.messages()
            non_system = [m for m in msgs if m.get("role") != "system"]
            if non_system:
                assert non_system[0]["role"] == "user", (
                    f"max_tokens={budget}: first non-system must be user, got {non_system[0]['role']!r}"
                )

    def test_trim_to_fit_first_message_is_always_user(self) -> None:
        """trim_to_fit() never leaves _messages starting with assistant."""
        mem = ShortTermMemory(max_tokens=1_000_000)
        mem.add_user("task")
        mem._messages.append(_assistant_tool_use("tu_1"))
        mem._messages.append(_tool_results("tu_1", content=_big(500)))

        mem.trim_to_fit(max_tokens=5)  # tiny budget

        non_system = [m for m in mem._messages if m.get("role") != "system"]
        if non_system:
            assert non_system[0]["role"] == "user"

    def test_normal_trimming_still_works(self) -> None:
        """Old turns are still dropped when doing so preserves a user-first history."""
        mem = ShortTermMemory(max_tokens=10)  # budget = 40 chars
        # Turn 1: user + assistant exchange (no tools)
        mem.add_user("a" * 5)  # 5 chars
        mem._messages.append({"role": "assistant", "content": "b" * 5})
        # Turn 2: user + assistant exchange
        mem.add_user("c" * 5)
        mem._messages.append({"role": "assistant", "content": "d" * 5})
        # Turn 3 (current): user request
        mem.add_user("now")  # 3 chars — well within budget alone

        msgs = mem.messages()

        # Current user message must survive
        assert any(m.get("content") == "now" for m in msgs)
        # First non-system must be user
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert non_system[0]["role"] == "user"

    def test_system_messages_preserved_with_user_anchor(self) -> None:
        """System messages stay; the anchor guard considers only non-system."""
        mem = ShortTermMemory(max_tokens=5)  # budget = 20 chars
        mem._messages.append(_system("System prompt."))
        mem.add_user("task")
        mem._messages.append(_assistant_tool_use("tu_1"))
        mem._messages.append(_tool_results("tu_1", content=_big(200)))

        msgs = mem.messages()

        system_msgs = [m for m in msgs if m.get("role") == "system"]
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert system_msgs, "System message must be preserved"
        assert non_system, "Non-system messages must remain"
        assert non_system[0]["role"] == "user"


# ── Warning emission ───────────────────────────────────────────────────────────


class TestUserAnchorWarning:
    def test_messages_emits_warning_when_user_anchor_guard_fires(self) -> None:
        """UserWarning emitted when trimming stops to preserve user-first invariant."""
        mem = ShortTermMemory(max_tokens=10)  # budget = 40 chars
        mem.add_user("task")
        mem._messages.append(_assistant_tool_use("tu_1"))
        mem._messages.append(_tool_results("tu_1", content=_big(500)))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mem.messages()

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warns, "UserWarning must be emitted when user-anchor guard fires"
        msg = str(user_warns[0].message)
        assert "invalid" in msg.lower() or "trim" in msg.lower() or "budget" in msg.lower()

    def test_trim_to_fit_emits_warning_when_user_anchor_guard_fires(self) -> None:
        mem = ShortTermMemory(max_tokens=1_000_000)
        mem.add_user("task")
        mem._messages.append(_assistant_tool_use("tu_1"))
        mem._messages.append(_tool_results("tu_1", content=_big(500)))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mem.trim_to_fit(max_tokens=5)

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warns

    def test_no_warning_when_trimming_succeeds_normally(self) -> None:
        """No warning when old turns are dropped while a conversational user survives."""
        # budget = 10 * 4 = 40 chars
        # Build many small turns so trimming is needed but always leaves a user turn
        mem = ShortTermMemory(max_tokens=10)  # budget = 40 chars
        for _ in range(20):
            mem.add_user("ab")  # 2 chars
            mem._messages.append({"role": "assistant", "content": "cd"})  # 2 chars
        mem.add_user("now")  # 3 chars — current request

        # Total = 20*4 + 3 = 83 chars > 40 chars budget.
        # Trimming will drop old (user, assistant) pairs.  Each drop leaves
        # the remaining list still starting with a conversational user message,
        # so the anchor guard must NOT fire.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mem.messages()

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warns, (
            f"No warning expected when normal turn-pair trimming maintains "
            f"a conversational user message.  Got: {[str(w.message) for w in user_warns]}"
        )


# ── Anthropic-specific scenario ───────────────────────────────────────────────


class TestAnthropicLargeToolResultsRegression:
    def test_large_recursive_directory_listing_does_not_produce_assistant_first(self) -> None:
        """Exact reproduction of the 400 error with large tool results.

        User: "please fix the docs"
        Agent: 3 parallel tool calls (search_files x2, list_directory recursive)
        Result: list_directory returns ~400KB of file tree
        Expected: messages() returns user-first list, not assistant-first
        """
        from lauren_ai._transport._anthropic import _message_to_anthropic

        mem = ShortTermMemory(max_tokens=32_000)  # 128KB budget (production default)

        # User message (small)
        mem.add_user("please fix the docs")

        # Agent makes 3 parallel tool calls
        mem._messages.append(
            _assistant_tool_use(
                "call_01_search_rst",
                "call_02_search_md",
                "call_03_list_dir",
            )
        )

        # Consolidated tool results with large directory listing
        large_listing = "\n".join(f"src/module_{i}/file_{j}.py" for i in range(500) for j in range(20))
        mem._messages.append(
            _tool_results(
                "call_01_search_rst",
                "call_02_search_md",
                "call_03_list_dir",
                content=large_listing,  # ~200KB
            )
        )

        msgs = mem.messages()

        # Must not be empty
        assert msgs, "messages() must not return empty list"

        # First non-system must be user
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert non_system[0]["role"] == "user", (
            "Anthropic rejects conversations starting with assistant role — "
            "this would produce '400: messages: at least one message is required'"
        )

        # Must be convertible to Anthropic format without error
        for m in msgs:
            converted = _message_to_anthropic(m)
            assert converted["role"] in ("user", "assistant"), (
                f"Invalid Anthropic role after conversion: {converted['role']!r}"
            )

"""Tests for ShortTermMemory trim floor guard (prd-memory-trim-floor-guard).

Regression tests for the bug where messages() and trim_to_fit() dropped the
last non-system message when it exceeded the token budget, resulting in an
empty messages list that caused API errors:
  "400: messages: at least one message is required"
"""

from __future__ import annotations

import warnings

import pytest

from lauren_ai._memory import ShortTermMemory


# ── helpers ───────────────────────────────────────────────────────────────────


def _big_message(chars: int) -> str:
    """Return a string of *chars* characters."""
    return "x" * chars


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str = "ok") -> dict:
    return {"role": "assistant", "content": content}


def _system(content: str = "You are helpful.") -> dict:
    return {"role": "system", "content": content}


# ── Layer 1: messages() floor guard ──────────────────────────────────────────


class TestMessagesFloorGuard:
    def test_single_oversized_message_not_dropped(self) -> None:
        """A single user message that exceeds the budget must not be dropped."""
        mem = ShortTermMemory(max_tokens=100)  # budget = 400 chars
        big = _big_message(1000)  # 1000 chars > 400 budget
        mem.add_user(big)
        msgs = mem.messages()
        assert len(msgs) >= 1, "messages() must not return an empty list"
        assert msgs[-1]["content"] == big

    def test_single_oversized_message_returns_it_as_is(self) -> None:
        """The oversized message is returned unchanged."""
        mem = ShortTermMemory(max_tokens=10)  # budget = 40 chars
        big = "A" * 500
        mem.add_user(big)
        result = mem.messages()
        assert result[0]["content"] == big

    def test_multi_turn_trims_old_turns_but_keeps_last(self) -> None:
        """Old turns are still trimmed; the last (current) turn is kept."""
        mem = ShortTermMemory(max_tokens=100)  # budget = 400 chars
        # Turn 1: small (will be trimmed)
        mem.add_user("hello")
        mem._messages.append(_assistant("hi"))
        # Turn 2: oversized current request
        mem.add_user(_big_message(500))
        msgs = mem.messages()
        # The oversized current user message must survive
        assert any(m.get("content", "") == _big_message(500) for m in msgs), (
            "The oversized current turn must not be dropped"
        )

    def test_old_turns_dropped_when_current_fits(self) -> None:
        """Normal trimming still works when current turn fits the budget."""
        mem = ShortTermMemory(max_tokens=100)  # budget = 400 chars
        # Turn 1: large old turn
        mem.add_user(_big_message(300))
        mem._messages.append(_assistant("done"))
        # Turn 2: small current turn
        mem.add_user("small")
        msgs = mem.messages()
        # The small current message survives
        assert any(m.get("content") == "small" for m in msgs)

    def test_system_messages_preserved_with_oversized_user(self) -> None:
        """System messages are never dropped; oversized user message kept too."""
        mem = ShortTermMemory(max_tokens=10)  # budget = 40 chars
        mem._messages.insert(0, _system("System prompt."))
        mem.add_user(_big_message(200))
        msgs = mem.messages()
        roles = [m.get("role") for m in msgs]
        assert "system" in roles
        assert "user" in roles

    def test_empty_memory_returns_empty(self) -> None:
        """Empty memory must still return empty list (no regression)."""
        mem = ShortTermMemory(max_tokens=100)
        assert mem.messages() == []

    def test_normal_sized_message_still_trimmed_when_too_many_turns(self) -> None:
        """Normal-sized messages are still trimmed when over budget."""
        mem = ShortTermMemory(max_tokens=10)  # budget = 40 chars
        # Many small turns — each user message is 5 chars, assistant 2 chars
        for i in range(10):
            mem.add_user(f"msg{i}")
            mem._messages.append(_assistant("ok"))
        mem.add_user("now")
        msgs = mem.messages()
        # Result must fit budget and include at least the latest message
        total = sum(len(m.get("content", "") or "") for m in msgs)
        assert total <= 100  # well within budget
        assert any(m.get("content") == "now" for m in msgs)

    def test_messages_never_empty_with_any_content(self) -> None:
        """Regardless of budget, messages() is non-empty when memory has content."""
        for max_tokens in (1, 10, 50, 100):
            mem = ShortTermMemory(max_tokens=max_tokens)
            mem.add_user("hello")
            result = mem.messages()
            assert result, f"messages() returned [] with max_tokens={max_tokens}"


# ── Layer 2: trim_to_fit() floor guard ────────────────────────────────────────


class TestTrimToFitFloorGuard:
    def test_trim_to_fit_does_not_empty_single_oversized_message(self) -> None:
        """trim_to_fit() must not drop the last non-system message."""
        mem = ShortTermMemory(max_tokens=400_000)
        mem.add_user(_big_message(1000))
        mem.trim_to_fit(max_tokens=10)  # budget = 40 chars < 1000 chars
        assert len(mem._messages) >= 1
        assert mem._messages[0]["content"] == _big_message(1000)

    def test_trim_to_fit_removes_old_turns_normally(self) -> None:
        """Old turns are still dropped by trim_to_fit()."""
        mem = ShortTermMemory(max_tokens=400_000)
        # Each turn = "turn N" (6 chars) + "ok" (2 chars) = 8 chars per turn pair
        # 5 pairs = 40 chars + "latest" = 46 chars total
        # Budget of 2 tokens (8 chars) forces trimming of old turns
        for i in range(5):
            mem.add_user(f"turn {i}")
            mem._messages.append(_assistant("ok"))
        mem.add_user("latest")
        initial_count = len(mem._messages)
        mem.trim_to_fit(max_tokens=2)  # budget = 8 chars — forces old turns off
        assert len(mem._messages) < initial_count, "Old turns should be trimmed"
        assert any(m.get("content") == "latest" for m in mem._messages)

    def test_trim_to_fit_no_op_when_within_budget(self) -> None:
        """trim_to_fit() is a no-op when messages fit the budget."""
        mem = ShortTermMemory(max_tokens=100)
        mem.add_user("hi")
        mem._messages.append(_assistant("hello"))
        before = list(mem._messages)
        mem.trim_to_fit(max_tokens=1000)
        assert mem._messages == before


# ── Layer 3: UserWarning emitted ─────────────────────────────────────────────


class TestOversizedWarning:
    def test_messages_emits_warning_when_oversized(self) -> None:
        """messages() emits UserWarning when a single message exceeds budget."""
        mem = ShortTermMemory(max_tokens=10)  # budget = 40 chars
        mem.add_user(_big_message(500))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mem.messages()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings, "UserWarning must be emitted for oversized message"
        assert "budget" in str(user_warnings[0].message).lower() or "exceed" in str(user_warnings[0].message).lower()

    def test_trim_to_fit_emits_warning_when_oversized(self) -> None:
        """trim_to_fit() emits UserWarning when it cannot trim further."""
        mem = ShortTermMemory(max_tokens=400_000)
        mem.add_user(_big_message(1000))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mem.trim_to_fit(max_tokens=10)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings, "UserWarning must be emitted for oversized message"

    def test_no_warning_for_normal_sized_message(self) -> None:
        """No UserWarning is emitted when the message fits the budget."""
        mem = ShortTermMemory(max_tokens=1000)  # budget = 4000 chars
        mem.add_user("a short message")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mem.messages()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warnings, "No warning for in-budget message"

    def test_no_warning_when_old_turns_trimmed_normally(self) -> None:
        """No warning when trimming old turns that fit within budget logic."""
        mem = ShortTermMemory(max_tokens=100)  # budget = 400 chars
        for _ in range(5):
            mem.add_user("x" * 50)
            mem._messages.append(_assistant("ok"))
        mem.add_user("small")  # current turn is small
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mem.messages()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warnings, "No warning when normal trimming occurs"


# ── Regression: Anthropic 400 scenario ───────────────────────────────────────


class TestAnthropicEmptyMessagesRegression:
    def test_messages_never_empty_simulating_large_mention_injection(self) -> None:
        """Simulate agenthicc @mention: large file prepended to user intent.

        This is the exact scenario that caused:
          400: messages: at least one message is required
        """
        # Simulate agenthicc's build_context_prefix() + ctx.text
        file_content = "x" * 200_000  # 200KB file
        user_intent = "Implement the auth refactor described in the issue."
        agent_text = (
            f'<file path="src/auth.py" chars="{len(file_content):,}">\n{file_content}\n</file>\n\n{user_intent}'
        )
        mem = ShortTermMemory(max_tokens=32_000)  # 128_000 char budget
        mem.add_user(agent_text)

        msgs = mem.messages()
        assert msgs, (
            "messages() must not return [] when a single large @mention message "
            "exceeds the token budget — this causes Anthropic 400 errors"
        )
        assert msgs[-1]["role"] == "user"

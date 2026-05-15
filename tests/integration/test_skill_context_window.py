"""Integration tests for Skill 46: Agent Context Window Management & Trimming.

Tests cover:
- trim_messages keeps last max_turns turns (user+assistant pairs)
- trim_messages preserves system messages
- estimate_tokens returns non-zero for non-empty messages
- ShortTermMemory(max_tokens=N) is configurable
- ShortTermMemory.messages() returns trimmed snapshot
- ShortTermMemory trims oldest messages when over budget
- Manual trim via trim_to_fit mutates buffer

NOTE: from __future__ import annotations IS safe here (no @tool definitions).
"""

from __future__ import annotations

from lauren_ai import ShortTermMemory, agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Helper utilities from the skill
# ---------------------------------------------------------------------------


def trim_messages(messages: list[dict], max_turns: int = 10) -> list[dict]:
    """Keep system message + last max_turns turns (user+assistant pairs)."""
    system = [m for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) > max_turns * 2:
        turns = turns[-(max_turns * 2) :]
    return system + turns


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: 4 chars per token."""
    return sum(len(str(m.get("content", ""))) // 4 for m in messages)


def _build_messages(n_turns: int) -> list[dict]:
    """Build n_turns of user+assistant messages."""
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"User message {i}"})
        msgs.append({"role": "assistant", "content": f"Assistant reply {i}"})
    return msgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    runner = AgentRunner(transport=mock)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests: trim_messages helper (direct)
# ---------------------------------------------------------------------------


class TestTrimMessages:
    def test_trim_keeps_last_n_turns(self):
        msgs = _build_messages(20)  # 40 messages total
        trimmed = trim_messages(msgs, max_turns=5)
        non_system = [m for m in trimmed if m.get("role") != "system"]
        assert len(non_system) == 10  # 5 turns * 2

    def test_trim_below_limit_unchanged(self):
        msgs = _build_messages(3)  # 6 messages, under limit of 10
        trimmed = trim_messages(msgs, max_turns=10)
        assert len(trimmed) == 6

    def test_trim_preserves_system_message(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        msgs += _build_messages(15)
        trimmed = trim_messages(msgs, max_turns=5)
        system_msgs = [m for m in trimmed if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful."

    def test_trim_exact_limit_unchanged(self):
        msgs = _build_messages(10)  # exactly 10 turns
        trimmed = trim_messages(msgs, max_turns=10)
        assert len(trimmed) == 20

    def test_trim_default_max_turns(self):
        msgs = _build_messages(15)
        trimmed = trim_messages(msgs)  # default max_turns=10
        non_system = [m for m in trimmed if m.get("role") != "system"]
        assert len(non_system) == 20


# ---------------------------------------------------------------------------
# Tests: estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_estimate_non_zero_for_non_empty(self):
        msgs = [{"role": "user", "content": "Hello, how are you today?"}]
        assert estimate_tokens(msgs) > 0

    def test_estimate_zero_for_empty(self):
        msgs = []
        assert estimate_tokens(msgs) == 0

    def test_estimate_scales_with_content_length(self):
        short_msgs = [{"role": "user", "content": "Hi"}]
        long_msgs = [{"role": "user", "content": "Hi " * 100}]
        assert estimate_tokens(long_msgs) > estimate_tokens(short_msgs)

    def test_estimate_multiple_messages(self):
        msgs = _build_messages(5)
        total = estimate_tokens(msgs)
        assert total > 0


# ---------------------------------------------------------------------------
# Tests: ShortTermMemory sliding window
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    def test_short_term_memory_configurable(self):
        mem = ShortTermMemory(max_tokens=4000)
        assert mem._max_tokens == 4000

    def test_empty_memory_returns_empty_list(self):
        mem = ShortTermMemory(max_tokens=4000)
        assert mem.messages() == []

    def test_add_user_message(self):
        mem = ShortTermMemory(max_tokens=4000)
        mem.add_user("Hello")
        msgs = mem.messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_token_estimate_grows_with_content(self):
        mem = ShortTermMemory(max_tokens=4000)
        mem.add_user("Hello world, this is a test message of some length.")
        assert mem.token_estimate > 0

    def test_messages_trimmed_when_over_budget(self):
        # 10 tokens budget = 40 chars; add many long messages
        mem = ShortTermMemory(max_tokens=10)
        for _i in range(20):
            mem.add_user("A" * 50)  # 50 chars = ~12 tokens each
        trimmed = mem.messages()
        # Should have fewer messages than were added
        assert len(trimmed) < 20

    def test_messages_not_mutate_internal_buffer(self):
        mem = ShortTermMemory(max_tokens=4000)
        mem.add_user("test")
        snapshot1 = mem.messages()
        snapshot1.clear()
        snapshot2 = mem.messages()
        assert len(snapshot2) == 1  # internal buffer not affected

    def test_clear_empties_buffer(self):
        mem = ShortTermMemory(max_tokens=4000)
        mem.add_user("Hello")
        mem.clear()
        assert len(mem.messages()) == 0

    def test_snapshot_is_deep_copy(self):
        mem = ShortTermMemory(max_tokens=4000)
        mem.add_user("Hello")
        snap = mem.snapshot()
        snap["messages"][0]["content"] = "mutated"
        assert mem.messages()[0]["content"] == "Hello"

    def test_restore_loads_messages(self):
        mem = ShortTermMemory(max_tokens=4000)
        mem.add_user("original")
        snap = mem.snapshot()
        mem.clear()
        mem.restore(snap)
        assert len(mem.messages()) == 1

    def test_trim_to_fit_mutates_buffer(self):
        mem = ShortTermMemory(max_tokens=100_000)
        for _ in range(10):
            mem.add_user("X" * 100)  # 100 chars = 25 tokens each → 250 total
        before = len(mem._messages)
        mem.trim_to_fit(max_tokens=50)  # 50 tokens = 200 chars budget
        after = len(mem._messages)
        assert after < before


# ---------------------------------------------------------------------------
# Tests: agent with ShortTermMemory via TestClient
# ---------------------------------------------------------------------------


class TestAgentWithShortTermMemory:
    async def test_agent_runs_with_memory_config(self):
        @agent(
            model="mock-model",
            system="You are helpful.",
            memory=ShortTermMemory(max_tokens=4000),
            conversation_store=InMemoryConversationStore(),
        )
        class MemAgent: ...

        client = TestClient(MemAgent())
        client.mock.queue_response(_completion("Hello!"))
        resp = await client.run_async("Hi there")
        assert resp.content == "Hello!"

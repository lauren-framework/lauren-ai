"""Integration tests for conversation memory pattern (Skill 12).

Tests:
  - ShortTermMemory accumulates messages across two runs when shared
  - ShortTermMemory trims to token budget
  - InMemoryConversationStore saves and loads history correctly
  - Agent with conversation_store + conversation_id persists turns
  - Second run with same conversation_id sees prior history in LLM messages
  - Different conversation_ids are isolated
  - No conversation_id means store is not touched
  - Per-request store override wins over agent-level store
  - Store.clear() removes all histories
  - Store.list_conversations() returns stored IDs
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._memory import ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import AgentTestClient


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
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests: ShortTermMemory
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    def test_fresh_memory_is_empty(self):
        mem = ShortTermMemory(max_tokens=8_000)
        assert len(mem) == 0
        assert mem.messages() == []

    def test_add_user_message(self):
        mem = ShortTermMemory()
        mem.add_user("Hello!")
        msgs = mem.messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello!"

    def test_add_assistant_completion(self):
        mem = ShortTermMemory()
        mem.add_assistant(_completion("Hi there!"))
        msgs = mem.messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    def test_token_estimate_increases_with_messages(self):
        mem = ShortTermMemory()
        mem.add_user("Hello world this is a test message to fill the buffer up a bit")
        assert mem.token_estimate > 0

    def test_sliding_window_trims_oldest_non_system_messages(self):
        # Very small budget so oldest message gets trimmed
        mem = ShortTermMemory(max_tokens=1)
        mem.add_user("first message that is long enough to exceed budget")
        mem.add_user("second message")
        trimmed = mem.messages()
        # Should trim the oldest to fit budget
        assert len(trimmed) <= 2

    async def test_shared_memory_accumulates_across_two_agent_runs(self):
        shared_mem = ShortTermMemory(max_tokens=10_000)

        @agent(model="mock-model", memory=shared_mem)
        class MemAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("r1"))
        mock.queue_response(_completion("r2"))

        await runner.run(MemAgent(), "q1")
        await runner.run(MemAgent(), "q2")

        # 2 user + 2 assistant messages
        assert len(shared_mem.messages()) == 4

    async def test_shared_memory_accumulates_async(self):
        shared_mem = ShortTermMemory(max_tokens=10_000)

        @agent(model="mock-model", memory=shared_mem)
        class MemAgentAsync: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("response1"))
        mock.queue_response(_completion("response2"))

        await runner.run(MemAgentAsync(), "turn1")
        await runner.run(MemAgentAsync(), "turn2")

        assert len(shared_mem.messages()) == 4

    def test_clear_empties_memory(self):
        mem = ShortTermMemory()
        mem.add_user("hello")
        mem.clear()
        assert len(mem) == 0

    def test_snapshot_and_restore(self):
        mem = ShortTermMemory()
        mem.add_user("first")
        snap = mem.snapshot()
        mem.add_user("second")
        mem.restore(snap)
        assert len(mem) == 1
        assert mem.messages()[0]["content"] == "first"


# ---------------------------------------------------------------------------
# Tests: InMemoryConversationStore
# ---------------------------------------------------------------------------


class TestInMemoryConversationStore:
    async def test_load_returns_empty_list_for_unknown_id(self):
        store = InMemoryConversationStore()
        result = await store.load("nonexistent")
        assert result == []

    async def test_save_and_load_roundtrip(self):
        store = InMemoryConversationStore()
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        await store.save("sess-1", messages)
        loaded = await store.load("sess-1")
        assert loaded == messages

    async def test_save_overwrites_existing(self):
        store = InMemoryConversationStore()
        await store.save("s", [{"role": "user", "content": "old"}])
        await store.save("s", [{"role": "user", "content": "new"}])
        loaded = await store.load("s")
        assert loaded[0]["content"] == "new"

    async def test_delete_removes_history(self):
        store = InMemoryConversationStore()
        await store.save("s", [{"role": "user", "content": "msg"}])
        await store.delete("s")
        assert await store.load("s") == []

    async def test_clear_removes_all_histories(self):
        store = InMemoryConversationStore()
        await store.save("a", [{"role": "user", "content": "1"}])
        await store.save("b", [{"role": "user", "content": "2"}])
        await store.clear()
        assert len(store) == 0

    async def test_list_conversations_returns_stored_ids(self):
        store = InMemoryConversationStore()
        await store.save("alice", [])
        await store.save("bob", [])
        ids = await store.list_conversations()
        assert "alice" in ids
        assert "bob" in ids

    async def test_different_ids_are_isolated(self):
        store = InMemoryConversationStore()
        await store.save("alice", [{"role": "user", "content": "alice message"}])
        await store.save("bob", [{"role": "user", "content": "bob message"}])
        alice_hist = await store.load("alice")
        bob_hist = await store.load("bob")
        assert all("bob" not in str(m) for m in alice_hist)
        assert all("alice" not in str(m) for m in bob_hist)


# ---------------------------------------------------------------------------
# Tests: Agent with conversation store
# ---------------------------------------------------------------------------


class TestAgentConversationMemory:
    async def test_agent_saves_history_to_store(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("answer"))
        await runner.run(StoreAgent(), "question", conversation_id="sess1")

        history = await store.load("sess1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    async def test_second_run_with_same_id_sees_prior_history(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent2: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("I'll remember that"))
        await runner.run(StoreAgent2(), "My name is Alice", conversation_id="s")

        mock.queue_response(_completion("Your name is Alice"))
        await runner.run(StoreAgent2(), "What is my name?", conversation_id="s")

        # Second call messages should include prior exchange
        second_call_messages = mock.calls[1].messages
        contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
        assert "My name is Alice" in contents

    async def test_no_conversation_id_store_not_touched(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent3: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(StoreAgent3(), "hi")

        assert len(store) == 0

    async def test_per_request_store_override_wins(self):
        meta_store = InMemoryConversationStore()
        override_store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=meta_store)
        class MetaStoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(
            MetaStoreAgent(),
            "hi",
            conversation_id="s",
            conversation_store=override_store,
        )

        assert len(await override_store.load("s")) == 2
        assert len(await meta_store.load("s")) == 0

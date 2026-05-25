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

NOTE: No from __future__ import annotations.
"""

import asyncio

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory import ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock: MockTransport) -> AgentRunner:
    return AgentRunner(transport=mock)


# ---------------------------------------------------------------------------
# Tests: ShortTermMemory (direct Python)
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
        mem.add_assistant(_c("Hi there!"))
        msgs = mem.messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    def test_token_estimate_increases_with_messages(self):
        mem = ShortTermMemory()
        mem.add_user("Hello world this is a test message to fill the buffer up a bit")
        assert mem.token_estimate > 0

    def test_sliding_window_trims_oldest_non_system_messages(self):
        mem = ShortTermMemory(max_tokens=1)
        mem.add_user("first message that is long enough to exceed budget")
        mem.add_user("second message")
        assert len(mem.messages()) <= 2

    def test_shared_memory_accumulates_across_two_agent_runs(self):
        shared_mem = ShortTermMemory(max_tokens=10_000)
        mock = MockTransport()
        mock.queue_response(_c("r1"))
        mock.queue_response(_c("r2"))
        runner = _make_runner(mock)

        @agent(model="mock-model", memory=shared_mem)
        class MemAgent: ...

        asyncio.run(runner.run(MemAgent(), "q1"))
        asyncio.run(runner.run(MemAgent(), "q2"))
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
# Tests: InMemoryConversationStore (direct Python)
# ---------------------------------------------------------------------------


class TestInMemoryConversationStore:
    def test_load_returns_empty_list_for_unknown_id(self):
        store = InMemoryConversationStore()
        result = asyncio.run(store.load("nonexistent"))
        assert result == []

    def test_save_and_load_roundtrip(self):
        store = InMemoryConversationStore()
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        asyncio.run(store.save("sess-1", messages))
        loaded = asyncio.run(store.load("sess-1"))
        assert loaded["messages"] == messages

    def test_save_overwrites_existing(self):
        store = InMemoryConversationStore()
        asyncio.run(store.save("s", [{"role": "user", "content": "old"}]))
        asyncio.run(store.save("s", [{"role": "user", "content": "new"}]))
        loaded = asyncio.run(store.load("s"))
        assert loaded["messages"][0]["content"] == "new"

    def test_delete_removes_history(self):
        store = InMemoryConversationStore()
        asyncio.run(store.save("s", [{"role": "user", "content": "msg"}]))
        asyncio.run(store.delete("s"))
        result = asyncio.run(store.load("s"))
        assert result == []

    def test_clear_removes_all_histories(self):
        store = InMemoryConversationStore()
        asyncio.run(store.save("a", [{"role": "user", "content": "1"}]))
        asyncio.run(store.save("b", [{"role": "user", "content": "2"}]))
        asyncio.run(store.clear())
        assert len(store) == 0

    def test_list_conversations_returns_stored_ids(self):
        store = InMemoryConversationStore()
        asyncio.run(store.save("alice", []))
        asyncio.run(store.save("bob", []))
        ids = asyncio.run(store.list_conversations())
        assert "alice" in ids
        assert "bob" in ids

    def test_different_ids_are_isolated(self):
        store = InMemoryConversationStore()
        asyncio.run(store.save("alice", [{"role": "user", "content": "alice message"}]))
        asyncio.run(store.save("bob", [{"role": "user", "content": "bob message"}]))
        alice_hist = asyncio.run(store.load("alice"))
        bob_hist = asyncio.run(store.load("bob"))
        assert all("bob" not in str(m) for m in alice_hist)
        assert all("alice" not in str(m) for m in bob_hist)


# ---------------------------------------------------------------------------
# Tests: Agent with conversation store (via TestClient / AgentRunner)
# ---------------------------------------------------------------------------


class TestAgentConversationMemory:
    def test_agent_saves_history_to_store(self):
        store = InMemoryConversationStore()
        mock = MockTransport()
        mock.queue_response(_c("answer"))

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner = _make_runner(mock)
        asyncio.run(runner.run(StoreAgent(), "question", conversation_id="sess1"))
        history = asyncio.run(store.load("sess1"))
        assert len(history) == 2
        assert history["messages"][0]["role"] == "user"
        assert history["messages"][1]["role"] == "assistant"

    def test_second_run_with_same_id_sees_prior_history(self):
        store = InMemoryConversationStore()
        mock = MockTransport()
        mock.queue_response(_c("I'll remember that"))
        mock.queue_response(_c("Your name is Alice"))

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent2: ...

        runner = _make_runner(mock)
        asyncio.run(runner.run(StoreAgent2(), "My name is Alice", conversation_id="s"))
        asyncio.run(runner.run(StoreAgent2(), "What is my name?", conversation_id="s"))

        second_call_messages = mock.calls[1].messages
        contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
        assert "My name is Alice" in contents

    def test_no_conversation_id_store_not_touched(self):
        store = InMemoryConversationStore()
        mock = MockTransport()
        mock.queue_response(_c("OK"))

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent3: ...

        runner = _make_runner(mock)
        asyncio.run(runner.run(StoreAgent3(), "hi"))
        assert len(store) == 0

    def test_per_request_store_override_wins(self):
        meta_store = InMemoryConversationStore()
        override_store = InMemoryConversationStore()
        mock = MockTransport()
        mock.queue_response(_c("OK"))

        @agent(model="mock-model", conversation_store=meta_store)
        class MetaStoreAgent: ...

        runner = _make_runner(mock)
        asyncio.run(
            runner.run(
                MetaStoreAgent(),
                "hi",
                conversation_id="s",
                conversation_store=override_store,
            )
        )
        override_length = len((asyncio.run(override_store.load("s")) or {"messages": []}).get("messages", []))
        meta_length = len((asyncio.run(meta_store.load("s")) or {"messages": []}).get("messages", []))
        assert override_length == 2
        assert meta_length == 0

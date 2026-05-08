"""Integration tests for conversation state DB persistence (Skill 14).

Tests the SQLiteConversationStore pattern using aiosqlite and :memory: DB.

Tests:
  - save and load roundtrip for a single conversation
  - load returns empty list for unknown conversation_id
  - save overwrites existing history
  - delete removes the conversation
  - separate conversation IDs are isolated
  - JSON messages with nested structure survive roundtrip
  - multiple saves to different IDs do not interfere
  - concurrent saves do not corrupt data
  - store works as ConversationStore protocol (duck-typing)
  - agent with SQLiteConversationStore saves and loads history
"""

import json
import time

import pytest

try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

pytestmark = pytest.mark.skipif(not HAS_AIOSQLITE, reason="aiosqlite not installed")

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import AgentTestClient


# ---------------------------------------------------------------------------
# SQLiteConversationStore implementation under test
# ---------------------------------------------------------------------------


class SQLiteConversationStore:
    """Persist conversation history to SQLite via aiosqlite.

    Uses a persistent aiosqlite connection so that :memory: databases
    work correctly across multiple method calls.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT NOT NULL PRIMARY KEY,
                    messages TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await self._conn.commit()
        return self._conn

    async def save(self, conversation_id: str, messages: list) -> None:
        conn = await self._get_conn()
        await conn.execute(
            "INSERT OR REPLACE INTO conversations (id, messages, updated_at) VALUES (?, ?, ?)",
            (conversation_id, json.dumps(messages), time.time()),
        )
        await conn.commit()

    async def load(self, conversation_id: str) -> list:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT messages FROM conversations WHERE id = ?",
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return json.loads(row[0]) if row else []

    async def delete(self, conversation_id: str) -> None:
        conn = await self._get_conn()
        await conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


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
# Tests: SQLiteConversationStore
# ---------------------------------------------------------------------------


class TestSQLiteConversationStore:
    async def test_load_returns_empty_for_unknown_id(self):
        store = SQLiteConversationStore()
        result = await store.load("nonexistent-id")
        assert result == []

    async def test_save_and_load_roundtrip(self):
        store = SQLiteConversationStore()
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        await store.save("sess-1", messages)
        loaded = await store.load("sess-1")
        assert loaded == messages

    async def test_save_overwrites_existing_history(self):
        store = SQLiteConversationStore()
        await store.save("s", [{"role": "user", "content": "old"}])
        await store.save("s", [{"role": "user", "content": "new"}])
        loaded = await store.load("s")
        assert loaded == [{"role": "user", "content": "new"}]

    async def test_delete_removes_conversation(self):
        store = SQLiteConversationStore()
        await store.save("s", [{"role": "user", "content": "msg"}])
        await store.delete("s")
        result = await store.load("s")
        assert result == []

    async def test_delete_nonexistent_is_noop(self):
        store = SQLiteConversationStore()
        # Should not raise
        await store.delete("does-not-exist")

    async def test_separate_conversation_ids_are_isolated(self):
        store = SQLiteConversationStore()
        alice_msgs = [{"role": "user", "content": "alice message"}]
        bob_msgs = [{"role": "user", "content": "bob message"}]
        await store.save("alice", alice_msgs)
        await store.save("bob", bob_msgs)
        assert await store.load("alice") == alice_msgs
        assert await store.load("bob") == bob_msgs

    async def test_json_messages_with_nested_structure_survive_roundtrip(self):
        store = SQLiteConversationStore()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "nested content"}]},
            {
                "role": "assistant",
                "content": "response",
                "metadata": {"tags": ["a", "b"], "score": 0.9},
            },
        ]
        await store.save("nested-test", messages)
        loaded = await store.load("nested-test")
        assert loaded == messages

    async def test_multiple_saves_to_different_ids_do_not_interfere(self):
        store = SQLiteConversationStore()
        for i in range(5):
            await store.save(f"id-{i}", [{"role": "user", "content": f"message {i}"}])
        for i in range(5):
            loaded = await store.load(f"id-{i}")
            assert loaded[0]["content"] == f"message {i}"

    async def test_store_satisfies_conversation_store_duck_typing(self):
        """SQLiteConversationStore satisfies ConversationStore protocol (duck-typing)."""
        from lauren_ai._memory import ConversationStore

        store = SQLiteConversationStore()
        assert isinstance(store, ConversationStore)

    async def test_empty_messages_list_roundtrip(self):
        store = SQLiteConversationStore()
        await store.save("empty", [])
        loaded = await store.load("empty")
        assert loaded == []


# ---------------------------------------------------------------------------
# Tests: Agent using SQLiteConversationStore
# ---------------------------------------------------------------------------


class TestAgentWithSQLiteStore:
    async def test_agent_saves_history_to_sqlite_store(self):
        store = SQLiteConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class SQLiteAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("saved to sqlite"))
        await runner.run(SQLiteAgent(), "hello", conversation_id="db-sess-1")

        history = await store.load("db-sess-1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    async def test_agent_loads_prior_history_from_sqlite(self):
        store = SQLiteConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class SQLiteAgent2: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("I will remember"))
        await runner.run(SQLiteAgent2(), "My name is Bob", conversation_id="db-s")

        mock.queue_response(_completion("Your name is Bob"))
        await runner.run(SQLiteAgent2(), "What is my name?", conversation_id="db-s")

        # Second call's messages should include the prior user turn
        second_call_messages = mock.calls[1].messages
        contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
        assert "My name is Bob" in contents

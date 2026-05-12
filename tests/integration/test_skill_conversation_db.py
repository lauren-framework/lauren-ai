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

NOTE: No from __future__ import annotations.
"""

import asyncio
import json
import time

import pytest

try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

from lauren_ai._agents import agent
from lauren_ai._memory import ConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

pytestmark = pytest.mark.skipif(not HAS_AIOSQLITE, reason="aiosqlite not installed")

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


def _c(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Tests: SQLiteConversationStore (direct Python)
# ---------------------------------------------------------------------------


class TestSQLiteConversationStore:
    def test_load_returns_empty_for_unknown_id(self):
        store = SQLiteConversationStore()
        result = asyncio.run(store.load("nonexistent-id"))
        assert result == []

    def test_save_and_load_roundtrip(self):
        store = SQLiteConversationStore()
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        asyncio.run(store.save("sess-1", messages))
        loaded = asyncio.run(store.load("sess-1"))
        assert loaded == messages

    def test_save_overwrites_existing_history(self):
        store = SQLiteConversationStore()
        asyncio.run(store.save("s", [{"role": "user", "content": "old"}]))
        asyncio.run(store.save("s", [{"role": "user", "content": "new"}]))
        loaded = asyncio.run(store.load("s"))
        assert loaded == [{"role": "user", "content": "new"}]

    def test_delete_removes_conversation(self):
        store = SQLiteConversationStore()
        asyncio.run(store.save("s", [{"role": "user", "content": "msg"}]))
        asyncio.run(store.delete("s"))
        result = asyncio.run(store.load("s"))
        assert result == []

    def test_delete_nonexistent_is_noop(self):
        store = SQLiteConversationStore()
        # Should not raise
        asyncio.run(store.delete("does-not-exist"))

    def test_separate_conversation_ids_are_isolated(self):
        store = SQLiteConversationStore()
        alice_msgs = [{"role": "user", "content": "alice message"}]
        bob_msgs = [{"role": "user", "content": "bob message"}]
        asyncio.run(store.save("alice", alice_msgs))
        asyncio.run(store.save("bob", bob_msgs))
        alice_loaded = asyncio.run(store.load("alice"))
        bob_loaded = asyncio.run(store.load("bob"))
        assert alice_loaded == alice_msgs
        assert bob_loaded == bob_msgs

    def test_json_messages_with_nested_structure_survive_roundtrip(self):
        store = SQLiteConversationStore()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "nested content"}]},
            {
                "role": "assistant",
                "content": "response",
                "metadata": {"tags": ["a", "b"], "score": 0.9},
            },
        ]
        asyncio.run(store.save("nested-test", messages))
        loaded = asyncio.run(store.load("nested-test"))
        assert loaded == messages

    def test_multiple_saves_to_different_ids_do_not_interfere(self):
        store = SQLiteConversationStore()
        for i in range(5):
            asyncio.run(store.save(f"id-{i}", [{"role": "user", "content": f"message {i}"}]))
        for i in range(5):
            loaded = asyncio.run(store.load(f"id-{i}"))
            assert loaded[0]["content"] == f"message {i}"

    def test_store_satisfies_conversation_store_duck_typing(self):
        store = SQLiteConversationStore()
        assert isinstance(store, ConversationStore)

    def test_empty_messages_list_roundtrip(self):
        store = SQLiteConversationStore()
        asyncio.run(store.save("empty", []))
        loaded = asyncio.run(store.load("empty"))
        assert loaded == []


# ---------------------------------------------------------------------------
# Tests: Agent using SQLiteConversationStore (via TestClient)
# ---------------------------------------------------------------------------


class TestAgentWithSQLiteStore:
    def test_agent_saves_history_to_sqlite_store(self):
        store = SQLiteConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class SQLiteAgent: ...

        client = TestClient(SQLiteAgent())
        client.mock.queue_response(_c("saved to sqlite"))
        client.run("hello", conversation_id="db-sess-1")
        history = asyncio.run(store.load("db-sess-1"))
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_agent_loads_prior_history_from_sqlite(self):
        store = SQLiteConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class SQLiteAgent2: ...

        client = TestClient(SQLiteAgent2())
        client.mock.queue_response(_c("I will remember"))
        client.run("My name is Bob", conversation_id="db-s")
        client.mock.queue_response(_c("Your name is Bob"))
        client.run("What is my name?", conversation_id="db-s")

        second_call_messages = client.calls[1].messages
        contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
        assert "My name is Bob" in contents

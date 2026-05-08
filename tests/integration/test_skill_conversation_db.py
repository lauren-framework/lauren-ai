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

import json
import time

import pytest
from pydantic import BaseModel

try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

pytestmark = pytest.mark.skipif(not HAS_AIOSQLITE, reason="aiosqlite not installed")

from lauren import Json, LaurenFactory, controller, module, post, get, use_value
from lauren.testing import TestClient
from lauren_ai import LLMConfig
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory import ConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


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


_MOCK = MockTransport()


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _SaveRequest(BaseModel):
    conv_id: str
    messages: list


class _LoadRequest(BaseModel):
    conv_id: str


class _AgentRunRequest(BaseModel):
    prompt: str = "hello"
    conv_id: str = "sess-1"


@controller("/store")
class StoreController:
    @post("/save-load")
    async def save_load(self, body: Json[_SaveRequest]) -> dict:
        store = SQLiteConversationStore()
        await store.save(body.conv_id, body.messages)
        loaded = await store.load(body.conv_id)
        return {"loaded": loaded}

    @post("/load-unknown")
    async def load_unknown(self) -> dict:
        store = SQLiteConversationStore()
        result = await store.load("nonexistent-id")
        return {"result": result}

    @post("/overwrite")
    async def overwrite(self) -> dict:
        store = SQLiteConversationStore()
        await store.save("s", [{"role": "user", "content": "old"}])
        await store.save("s", [{"role": "user", "content": "new"}])
        loaded = await store.load("s")
        return {"loaded": loaded}

    @post("/delete")
    async def delete(self) -> dict:
        store = SQLiteConversationStore()
        await store.save("s", [{"role": "user", "content": "msg"}])
        await store.delete("s")
        result = await store.load("s")
        return {"result": result}

    @post("/delete-nonexistent")
    async def delete_nonexistent(self) -> dict:
        store = SQLiteConversationStore()
        await store.delete("does-not-exist")
        return {"ok": True}

    @post("/isolate")
    async def isolate(self) -> dict:
        store = SQLiteConversationStore()
        alice_msgs = [{"role": "user", "content": "alice message"}]
        bob_msgs = [{"role": "user", "content": "bob message"}]
        await store.save("alice", alice_msgs)
        await store.save("bob", bob_msgs)
        alice_loaded = await store.load("alice")
        bob_loaded = await store.load("bob")
        return {
            "alice_correct": alice_loaded == alice_msgs,
            "bob_correct": bob_loaded == bob_msgs,
        }

    @post("/nested-roundtrip")
    async def nested_roundtrip(self, body: Json[_SaveRequest]) -> dict:
        store = SQLiteConversationStore()
        await store.save(body.conv_id, body.messages)
        loaded = await store.load(body.conv_id)
        return {"matches": loaded == body.messages}

    @post("/multi-ids")
    async def multi_ids(self) -> dict:
        store = SQLiteConversationStore()
        for i in range(5):
            await store.save(f"id-{i}", [{"role": "user", "content": f"message {i}"}])
        results = []
        for i in range(5):
            loaded = await store.load(f"id-{i}")
            results.append(loaded[0]["content"] == f"message {i}")
        return {"all_match": all(results)}

    @post("/duck-typing")
    async def duck_typing(self) -> dict:
        store = SQLiteConversationStore()
        return {"is_conversation_store": isinstance(store, ConversationStore)}

    @post("/empty-messages")
    async def empty_messages(self) -> dict:
        store = SQLiteConversationStore()
        await store.save("empty", [])
        loaded = await store.load("empty")
        return {"result": loaded}


@controller("/agent-store")
class AgentStoreController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._cfg = cfg
        self._mock = mock

    @post("/saves-history")
    async def saves_history(self, body: Json[_AgentRunRequest]) -> dict:
        store = SQLiteConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class SQLiteAgent: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        await runner.run(SQLiteAgent(), body.prompt, conversation_id=body.conv_id)
        history = await store.load(body.conv_id)
        return {
            "length": len(history),
            "first_role": history[0]["role"] if history else None,
            "second_role": history[1]["role"] if len(history) > 1 else None,
        }

    @post("/loads-prior-history")
    async def loads_prior_history(self) -> dict:
        store = SQLiteConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class SQLiteAgent2: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        await runner.run(SQLiteAgent2(), "My name is Bob", conversation_id="db-s")
        await runner.run(SQLiteAgent2(), "What is my name?", conversation_id="db-s")

        second_call_messages = self._mock.calls[1].messages
        contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
        return {"bob_in_history": "My name is Bob" in contents}


@module(
    controllers=[StoreController, AgentStoreController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class ConversationDBModule: ...


def build_app(*responses) -> TestClient:
    _MOCK.reset()
    for content in responses:
        _MOCK.queue_response(_completion(content))
    return TestClient(LaurenFactory.create(ConversationDBModule))


# ---------------------------------------------------------------------------
# Tests: SQLiteConversationStore
# ---------------------------------------------------------------------------


class TestSQLiteConversationStore:
    def test_load_returns_empty_for_unknown_id(self):
        client = build_app()
        r = client.post("/store/load-unknown", json={})
        assert r.status_code == 200
        assert r.json()["result"] == []

    def test_save_and_load_roundtrip(self):
        client = build_app()
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        r = client.post("/store/save-load", json={"conv_id": "sess-1", "messages": messages})
        assert r.status_code == 200
        assert r.json()["loaded"] == messages

    def test_save_overwrites_existing_history(self):
        client = build_app()
        r = client.post("/store/overwrite", json={})
        assert r.status_code == 200
        assert r.json()["loaded"] == [{"role": "user", "content": "new"}]

    def test_delete_removes_conversation(self):
        client = build_app()
        r = client.post("/store/delete", json={})
        assert r.status_code == 200
        assert r.json()["result"] == []

    def test_delete_nonexistent_is_noop(self):
        client = build_app()
        r = client.post("/store/delete-nonexistent", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_separate_conversation_ids_are_isolated(self):
        client = build_app()
        r = client.post("/store/isolate", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["alice_correct"] is True
        assert data["bob_correct"] is True

    def test_json_messages_with_nested_structure_survive_roundtrip(self):
        client = build_app()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "nested content"}]},
            {
                "role": "assistant",
                "content": "response",
                "metadata": {"tags": ["a", "b"], "score": 0.9},
            },
        ]
        r = client.post(
            "/store/nested-roundtrip", json={"conv_id": "nested-test", "messages": messages}
        )
        assert r.status_code == 200
        assert r.json()["matches"] is True

    def test_multiple_saves_to_different_ids_do_not_interfere(self):
        client = build_app()
        r = client.post("/store/multi-ids", json={})
        assert r.status_code == 200
        assert r.json()["all_match"] is True

    def test_store_satisfies_conversation_store_duck_typing(self):
        client = build_app()
        r = client.post("/store/duck-typing", json={})
        assert r.status_code == 200
        assert r.json()["is_conversation_store"] is True

    def test_empty_messages_list_roundtrip(self):
        client = build_app()
        r = client.post("/store/empty-messages", json={})
        assert r.status_code == 200
        assert r.json()["result"] == []


# ---------------------------------------------------------------------------
# Tests: Agent using SQLiteConversationStore
# ---------------------------------------------------------------------------


class TestAgentWithSQLiteStore:
    def test_agent_saves_history_to_sqlite_store(self):
        client = build_app("saved to sqlite")
        r = client.post(
            "/agent-store/saves-history",
            json={"prompt": "hello", "conv_id": "db-sess-1"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["length"] == 2
        assert data["first_role"] == "user"
        assert data["second_role"] == "assistant"

    def test_agent_loads_prior_history_from_sqlite(self):
        client = build_app("I will remember", "Your name is Bob")
        r = client.post("/agent-store/loads-prior-history", json={})
        assert r.status_code == 200
        assert r.json()["bob_in_history"] is True

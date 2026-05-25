"""Integration tests for the first-party SQLite persistent storage backends."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import lauren_ai.testing as ai_testing
from lauren_ai import (
    MemoryFact,
    SQLiteConversationStore,
    SQLiteStoreBackend,
    SQLiteStoreConfig,
    SQLiteUserMemoryStore,
    SQLiteVectorStore,
    agent,
)
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._knowledge import KnowledgeBase
from lauren_ai._middleware import conversation_middleware
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


def _completion(content: str, *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _db_path(tmp_path: Path, name: str) -> str:
    return str(tmp_path / f"{name}.sqlite3")


class TestSQLiteConversationStore:
    @pytest.mark.asyncio
    async def test_roundtrip_normalizes_legacy_message_lists(self, tmp_path: Path) -> None:
        store = await SQLiteConversationStore.connect(_db_path(tmp_path, "conversation-roundtrip"))
        try:
            messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
            await store.save("sess-1", messages)
            loaded = await store.load("sess-1")
            assert loaded["messages"] == messages
            assert loaded["summary"] is None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_persists_across_reopen(self, tmp_path: Path) -> None:
        db_path = _db_path(tmp_path, "conversation-reopen")
        store = await SQLiteConversationStore.connect(db_path)
        await store.save("sess-2", {"messages": [{"role": "user", "content": "persist me"}], "summary": "summary"})
        await store.close()

        reopened = await SQLiteConversationStore.connect(db_path)
        try:
            loaded = await reopened.load("sess-2")
            assert loaded["messages"][0]["content"] == "persist me"
            assert loaded["summary"] == "summary"
        finally:
            await reopened.close()

    @pytest.mark.asyncio
    async def test_concurrent_saves_keep_conversation_ids_isolated(self, tmp_path: Path) -> None:
        store = await SQLiteConversationStore.connect(_db_path(tmp_path, "conversation-concurrency"))
        try:
            async with asyncio.TaskGroup() as task_group:
                for index in range(5):
                    task_group.create_task(
                        store.save(
                            f"sess-{index}",
                            [{"role": "user", "content": f"message-{index}"}],
                        )
                    )
            conversations = await store.list_conversations()
            assert conversations == [f"sess-{index}" for index in range(5)]
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_runner_loads_prior_history_from_durable_store(self, tmp_path: Path) -> None:
        store = await SQLiteConversationStore.connect(_db_path(tmp_path, "runner-history"))

        @agent(model="mock-model")
        class PersistentAgent:
            pass

        transport = MockTransport()
        runner = AgentRunner(transport=transport)
        try:
            transport.queue_response(_completion("first"))
            await runner.run(PersistentAgent(), "Hello", conversation_id="conv-1", conversation_store=store)

            transport.queue_response(_completion("second", n=2))
            await runner.run(PersistentAgent(), "What did I say?", conversation_id="conv-1", conversation_store=store)

            contents = [item["content"] for item in transport.calls[1].messages if isinstance(item.get("content"), str)]
            assert "Hello" in contents
            assert "first" in contents
            assert "What did I say?" in contents
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_conversation_middleware_roundtrip_works_with_sqlite_store(self, tmp_path: Path) -> None:
        store = await SQLiteConversationStore.connect(_db_path(tmp_path, "middleware-history"))
        middleware_cls = conversation_middleware(store, header="x-conversation-id", cookie=None, auto_create=False)
        middleware = middleware_cls(conv_store=store)

        class FakeRequest:
            def __init__(self) -> None:
                self.headers = {"x-conversation-id": "conv-mw"}
                self.cookies = {}
                self.state = SimpleNamespace()

        request = FakeRequest()
        try:
            await store.save("conv-mw", [{"role": "user", "content": "before"}])

            async def call_next(req: FakeRequest) -> str:
                req.state.updated_conversation = {
                    "messages": [{"role": "assistant", "content": "after"}],
                    "summary": None,
                }
                return "ok"

            response = await middleware.dispatch(request, call_next)
            saved = await store.load("conv-mw")

            assert response == "ok"
            assert request.state.conversation_history["messages"][0]["content"] == "before"
            assert saved["messages"][0]["content"] == "after"
        finally:
            await store.close()


class TestSQLiteUserMemoryStore:
    @pytest.mark.asyncio
    async def test_crud_and_search(self, tmp_path: Path) -> None:
        store = await SQLiteUserMemoryStore.connect(_db_path(tmp_path, "user-memory-crud"))
        fact = MemoryFact(
            memory_id="m1",
            user_id="alice",
            content="Alice prefers morning meetings",
            topics=["preferences", "schedule"],
        )
        try:
            await store.add(fact)
            loaded = await store.get("alice", "m1")
            matches = await store.search("alice", "morning")
            listed = await store.list("alice", topic="preferences")

            assert loaded is not None
            assert loaded.content == fact.content
            assert [item.memory_id for item in matches] == ["m1"]
            assert [item.memory_id for item in listed] == ["m1"]

            await store.update("m1", content="Alice prefers afternoon meetings", confidence=0.6)
            updated = await store.get("alice", "m1")
            assert updated is not None
            assert updated.content == "Alice prefers afternoon meetings"
            assert updated.confidence == 0.6

            await store.delete("m1")
            assert await store.get("alice", "m1") is None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_persists_across_reopen_and_clear_is_user_scoped(self, tmp_path: Path) -> None:
        db_path = _db_path(tmp_path, "user-memory-reopen")
        store = await SQLiteUserMemoryStore.connect(db_path)
        try:
            await store.add(MemoryFact(memory_id="a1", user_id="alice", content="Alice likes Python"))
            await store.add(MemoryFact(memory_id="b1", user_id="bob", content="Bob likes Rust"))
        finally:
            await store.close()

        reopened = await SQLiteUserMemoryStore.connect(db_path)
        try:
            assert (await reopened.get("alice", "a1")) is not None
            assert (await reopened.get("bob", "b1")) is not None
            await reopened.clear("alice")
            assert await reopened.get("alice", "a1") is None
            assert (await reopened.get("bob", "b1")) is not None
        finally:
            await reopened.close()


class TestSQLiteVectorStore:
    @pytest.mark.asyncio
    async def test_search_filter_and_get(self, tmp_path: Path) -> None:
        store = await SQLiteVectorStore.connect(_db_path(tmp_path, "vector-search"))
        try:
            await store.upsert("Refund policy for annual plans", id="d1", metadata={"topic": "billing"})
            await store.upsert("Vacation policy for employees", id="d2", metadata={"topic": "hr"})

            results = await store.search("refund annual", k=2)
            filtered = await store.search("policy", k=5, filter={"topic": "billing"})
            loaded = await store.get("d1")

            assert results[0].id == "d1"
            assert [item.id for item in filtered] == ["d1"]
            assert loaded is not None
            assert loaded.metadata["topic"] == "billing"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_persists_across_reopen(self, tmp_path: Path) -> None:
        db_path = _db_path(tmp_path, "vector-reopen")
        store = await SQLiteVectorStore.connect(db_path)
        try:
            await store.upsert("Escalation path for urgent incidents", id="doc-1", metadata={"team": "ops"})
        finally:
            await store.close()

        reopened = await SQLiteVectorStore.connect(db_path)
        try:
            results = await reopened.search("urgent incidents", k=3)
            assert [item.id for item in results] == ["doc-1"]
        finally:
            await reopened.close()

    @pytest.mark.asyncio
    async def test_knowledge_base_survives_reopen(self, tmp_path: Path) -> None:
        db_path = _db_path(tmp_path, "knowledge-base")
        store = await SQLiteVectorStore.connect(db_path)
        kb = KnowledgeBase(store=store)

        class FakeLoader:
            async def load(self) -> list[SimpleNamespace]:
                return [
                    SimpleNamespace(
                        content="Shipping policy allows expedited delivery.",
                        metadata={"source": "faq"},
                        id="doc-shipping",
                    )
                ]

        try:

            class FakeChunker:
                def chunk(self, document: SimpleNamespace) -> list[SimpleNamespace]:
                    return [document]

            kb = KnowledgeBase(store=store, chunker=FakeChunker())
            await kb.load(FakeLoader())
        finally:
            await store.close()

        reopened = await SQLiteVectorStore.connect(db_path)
        try:
            reopened_kb = KnowledgeBase(store=reopened)
            results = await reopened_kb.search("expedited shipping", top_k=3, filter_metadata={"source": "faq"})
            assert [item.id for item in results] == ["doc-shipping"]
        finally:
            await reopened.close()


class TestSharedSQLiteBackend:
    @pytest.mark.asyncio
    async def test_shared_backend_supports_all_store_types(self, tmp_path: Path) -> None:
        backend = await SQLiteStoreBackend.connect(
            SQLiteStoreConfig(database_path=_db_path(tmp_path, "shared-backend"), table_prefix="lauren_")
        )
        conversation_store = SQLiteConversationStore.from_backend(backend)
        user_memory_store = SQLiteUserMemoryStore.from_backend(backend)
        vector_store = SQLiteVectorStore.from_backend(backend)

        try:
            await conversation_store.save("conv-shared", [{"role": "user", "content": "hi"}])
            await user_memory_store.add(MemoryFact(memory_id="m-shared", user_id="u1", content="User likes docs"))
            await vector_store.upsert("Documentation for deployment", id="doc-shared", metadata={"kind": "docs"})

            loaded_conversation = await conversation_store.load("conv-shared")
            loaded_fact = await user_memory_store.get("u1", "m-shared")
            search_results = await vector_store.search("deployment docs", k=2)

            assert loaded_conversation["messages"][0]["content"] == "hi"
            assert loaded_fact is not None
            assert loaded_fact.content == "User likes docs"
            assert [item.id for item in search_results] == ["doc-shared"]
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_testclient_works_with_sqlite_conversation_store(self, tmp_path: Path) -> None:
        store = await SQLiteConversationStore.connect(_db_path(tmp_path, "test-client"))

        @agent(model="mock-model", conversation_store=store)
        class DurableAgent:
            pass

        client = ai_testing.TestClient(DurableAgent())
        try:
            client.mock.queue_response(_completion("persisted"))
            result = await client.run_async("hello", conversation_id="client-conv")
            history = await store.load("client-conv")

            assert result.content == "persisted"
            assert history["messages"][0]["content"] == "hello"
            assert history["messages"][1]["content"] == "persisted"
        finally:
            await store.close()

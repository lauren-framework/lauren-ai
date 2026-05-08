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

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, get, module, post, use_value
from lauren.testing import TestClient
from lauren_ai import LLMConfig
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory import ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Module-level mock and shared state
# ---------------------------------------------------------------------------

_MOCK = MockTransport()
_SHARED_MEM = ShortTermMemory(max_tokens=10_000)
_STORE = InMemoryConversationStore()


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


class _RunRequest(BaseModel):
    prompt: str = "hi"
    conversation_id: str = ""


@controller("/memory")
class ShortTermMemoryController:
    @get("/fresh")
    async def fresh(self) -> dict:
        mem = ShortTermMemory(max_tokens=8_000)
        return {"length": len(mem), "messages": mem.messages()}

    @get("/add-user")
    async def add_user(self) -> dict:
        mem = ShortTermMemory()
        mem.add_user("Hello!")
        msgs = mem.messages()
        return {"length": len(msgs), "role": msgs[0]["role"], "content": msgs[0]["content"]}

    @get("/add-assistant")
    async def add_assistant(self) -> dict:
        mem = ShortTermMemory()
        c = _completion("Hi there!")
        mem.add_assistant(c)
        msgs = mem.messages()
        return {"length": len(msgs), "role": msgs[0]["role"]}

    @get("/token-estimate")
    async def token_estimate(self) -> dict:
        mem = ShortTermMemory()
        mem.add_user("Hello world this is a test message to fill the buffer up a bit")
        return {"token_estimate": mem.token_estimate}

    @get("/trim")
    async def trim(self) -> dict:
        mem = ShortTermMemory(max_tokens=1)
        mem.add_user("first message that is long enough to exceed budget")
        mem.add_user("second message")
        return {"length": len(mem.messages())}

    @get("/clear")
    async def clear(self) -> dict:
        mem = ShortTermMemory()
        mem.add_user("hello")
        mem.clear()
        return {"length": len(mem)}

    @get("/snapshot")
    async def snapshot_restore(self) -> dict:
        mem = ShortTermMemory()
        mem.add_user("first")
        snap = mem.snapshot()
        mem.add_user("second")
        mem.restore(snap)
        return {"length": len(mem), "content": mem.messages()[0]["content"]}

    @post("/shared-run")
    async def shared_run(self, body: Json[_RunRequest], mock: MockTransport) -> dict:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools={}, config=cfg)

        @agent(model="mock-model", memory=_SHARED_MEM)
        class MemAgent: ...

        await runner.run(MemAgent(), body.prompt)
        return {"shared_mem_length": len(_SHARED_MEM.messages())}


@controller("/store")
class StoreController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._cfg = cfg
        self._mock = mock

    @get("/load-empty")
    async def load_empty(self) -> dict:
        store = InMemoryConversationStore()
        result = await store.load("nonexistent")
        return {"result": result}

    @post("/save-load")
    async def save_load(self, body: Json[dict]) -> dict:
        store = InMemoryConversationStore()
        messages = body.get("messages", [])
        conv_id = body.get("conv_id", "s1")
        await store.save(conv_id, messages)
        loaded = await store.load(conv_id)
        return {"loaded": loaded}

    @post("/overwrite")
    async def overwrite(self) -> dict:
        store = InMemoryConversationStore()
        await store.save("s", [{"role": "user", "content": "old"}])
        await store.save("s", [{"role": "user", "content": "new"}])
        loaded = await store.load("s")
        return {"content": loaded[0]["content"]}

    @post("/delete")
    async def delete(self) -> dict:
        store = InMemoryConversationStore()
        await store.save("s", [{"role": "user", "content": "msg"}])
        await store.delete("s")
        result = await store.load("s")
        return {"result": result}

    @post("/clear-all")
    async def clear_all(self) -> dict:
        store = InMemoryConversationStore()
        await store.save("a", [{"role": "user", "content": "1"}])
        await store.save("b", [{"role": "user", "content": "2"}])
        await store.clear()
        return {"length": len(store)}

    @get("/list")
    async def list_convs(self) -> dict:
        store = InMemoryConversationStore()
        await store.save("alice", [])
        await store.save("bob", [])
        ids = await store.list_conversations()
        return {"alice": "alice" in ids, "bob": "bob" in ids}

    @post("/isolate")
    async def isolate(self) -> dict:
        store = InMemoryConversationStore()
        await store.save("alice", [{"role": "user", "content": "alice message"}])
        await store.save("bob", [{"role": "user", "content": "bob message"}])
        alice_hist = await store.load("alice")
        bob_hist = await store.load("bob")
        return {
            "alice_clean": all("bob" not in str(m) for m in alice_hist),
            "bob_clean": all("alice" not in str(m) for m in bob_hist),
        }

    @post("/agent-saves")
    async def agent_saves(self, body: Json[_RunRequest]) -> dict:
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        conv_id = body.conversation_id or "sess1"
        await runner.run(StoreAgent(), body.prompt, conversation_id=conv_id)
        history = await store.load(conv_id)
        return {
            "length": len(history),
            "first_role": history[0]["role"] if history else None,
            "second_role": history[1]["role"] if len(history) > 1 else None,
        }

    @post("/agent-history")
    async def agent_history(self) -> dict:
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent2: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        await runner.run(StoreAgent2(), "My name is Alice", conversation_id="s")
        await runner.run(StoreAgent2(), "What is my name?", conversation_id="s")

        second_call_messages = self._mock.calls[1].messages
        contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
        return {"alice_in_history": "My name is Alice" in contents}

    @post("/agent-no-id")
    async def agent_no_id(self) -> dict:
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent3: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        await runner.run(StoreAgent3(), "hi")
        return {"store_length": len(store)}

    @post("/agent-override-store")
    async def agent_override_store(self) -> dict:
        meta_store = InMemoryConversationStore()
        override_store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=meta_store)
        class MetaStoreAgent: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        await runner.run(
            MetaStoreAgent(),
            "hi",
            conversation_id="s",
            conversation_store=override_store,
        )
        return {
            "override_length": len(await override_store.load("s")),
            "meta_length": len(await meta_store.load("s")),
        }


@module(
    controllers=[ShortTermMemoryController, StoreController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class ConversationMemoryModule: ...


def build_app(*responses) -> TestClient:
    _MOCK.reset()
    _SHARED_MEM.clear()
    for content in responses:
        _MOCK.queue_response(_completion(content))
    return TestClient(LaurenFactory.create(ConversationMemoryModule))


# ---------------------------------------------------------------------------
# Tests: ShortTermMemory
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    def test_fresh_memory_is_empty(self):
        client = build_app()
        r = client.get("/memory/fresh")
        assert r.status_code == 200
        data = r.json()
        assert data["length"] == 0
        assert data["messages"] == []

    def test_add_user_message(self):
        client = build_app()
        r = client.get("/memory/add-user")
        assert r.status_code == 200
        data = r.json()
        assert data["length"] == 1
        assert data["role"] == "user"
        assert data["content"] == "Hello!"

    def test_add_assistant_completion(self):
        client = build_app()
        r = client.get("/memory/add-assistant")
        assert r.status_code == 200
        data = r.json()
        assert data["length"] == 1
        assert data["role"] == "assistant"

    def test_token_estimate_increases_with_messages(self):
        client = build_app()
        r = client.get("/memory/token-estimate")
        assert r.status_code == 200
        assert r.json()["token_estimate"] > 0

    def test_sliding_window_trims_oldest_non_system_messages(self):
        client = build_app()
        r = client.get("/memory/trim")
        assert r.status_code == 200
        assert r.json()["length"] <= 2

    def test_shared_memory_accumulates_across_two_agent_runs(self):
        client = build_app("r1", "r2")
        r1 = client.post("/memory/shared-run", json={"prompt": "q1"})
        r2 = client.post("/memory/shared-run", json={"prompt": "q2"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["shared_mem_length"] == 4

    def test_clear_empties_memory(self):
        client = build_app()
        r = client.get("/memory/clear")
        assert r.status_code == 200
        assert r.json()["length"] == 0

    def test_snapshot_and_restore(self):
        client = build_app()
        r = client.get("/memory/snapshot")
        assert r.status_code == 200
        data = r.json()
        assert data["length"] == 1
        assert data["content"] == "first"


# ---------------------------------------------------------------------------
# Tests: InMemoryConversationStore
# ---------------------------------------------------------------------------


class TestInMemoryConversationStore:
    def test_load_returns_empty_list_for_unknown_id(self):
        client = build_app()
        r = client.get("/store/load-empty")
        assert r.status_code == 200
        assert r.json()["result"] == []

    def test_save_and_load_roundtrip(self):
        client = build_app()
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        r = client.post("/store/save-load", json={"messages": messages, "conv_id": "sess-1"})
        assert r.status_code == 200
        assert r.json()["loaded"] == messages

    def test_save_overwrites_existing(self):
        client = build_app()
        r = client.post("/store/overwrite", json={})
        assert r.status_code == 200
        assert r.json()["content"] == "new"

    def test_delete_removes_history(self):
        client = build_app()
        r = client.post("/store/delete", json={})
        assert r.status_code == 200
        assert r.json()["result"] == []

    def test_clear_removes_all_histories(self):
        client = build_app()
        r = client.post("/store/clear-all", json={})
        assert r.status_code == 200
        assert r.json()["length"] == 0

    def test_list_conversations_returns_stored_ids(self):
        client = build_app()
        r = client.get("/store/list")
        assert r.status_code == 200
        data = r.json()
        assert data["alice"] is True
        assert data["bob"] is True

    def test_different_ids_are_isolated(self):
        client = build_app()
        r = client.post("/store/isolate", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["alice_clean"] is True
        assert data["bob_clean"] is True


# ---------------------------------------------------------------------------
# Tests: Agent with conversation store
# ---------------------------------------------------------------------------


class TestAgentConversationMemory:
    def test_agent_saves_history_to_store(self):
        client = build_app("answer")
        r = client.post("/store/agent-saves", json={"prompt": "question", "conversation_id": "sess1"})
        assert r.status_code == 200
        data = r.json()
        assert data["length"] == 2
        assert data["first_role"] == "user"
        assert data["second_role"] == "assistant"

    def test_second_run_with_same_id_sees_prior_history(self):
        client = build_app("I'll remember that", "Your name is Alice")
        r = client.post("/store/agent-history", json={})
        assert r.status_code == 200
        assert r.json()["alice_in_history"] is True

    def test_no_conversation_id_store_not_touched(self):
        client = build_app("OK")
        r = client.post("/store/agent-no-id", json={})
        assert r.status_code == 200
        assert r.json()["store_length"] == 0

    def test_per_request_store_override_wins(self):
        client = build_app("OK")
        r = client.post("/store/agent-override-store", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["override_length"] == 2
        assert data["meta_length"] == 0

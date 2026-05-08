"""Integration tests for long-term vector memory with @remember (Skill 13).

Tests:
  - @remember attaches REMEMBER_META with correct configuration
  - RememberMeta.inject / extract / top_k values match decorator args
  - Agent decorated with @remember can run with MockTransport
  - InMemoryUserMemoryStore add/search roundtrip
  - MemoryFact reinforce increases confidence
  - MemoryFact decay reduces confidence
  - Store.clear() removes facts for a given user_id
  - Store.list() returns facts for user_id
  - Store.search() returns matching facts by content
  - Multiple users' facts are isolated

NOTE: No from __future__ import annotations.
"""

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, get, module, post, use_value
from lauren.testing import TestClient
from lauren_ai import LLMConfig
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory import REMEMBER_META, RememberMeta, remember
from lauren_ai._memory._in_memory_user import InMemoryUserMemoryStore
from lauren_ai._memory._user import MemoryFact
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Module-level shared state
# ---------------------------------------------------------------------------

_MOCK = MockTransport()
_STORE = InMemoryUserMemoryStore()


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_fact(user_id: str, content: str, memory_id: str = "f1") -> MemoryFact:
    return MemoryFact(memory_id=memory_id, user_id=user_id, content=content)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _AgentRequest(BaseModel):
    prompt: str = "hello"
    user_id: str = "user-1"


@controller("/remember-meta")
class RememberMetaController:
    @get("/inject-true")
    async def inject_true(self) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, inject=True)
        class InjectAgent: ...

        meta = getattr(InjectAgent, REMEMBER_META)
        return {"inject": meta.inject, "is_remember_meta": isinstance(meta, RememberMeta)}

    @get("/extract-true")
    async def extract_true(self) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, extract=True)
        class ExtractAgent: ...

        meta = getattr(ExtractAgent, REMEMBER_META)
        return {"extract": meta.extract}

    @get("/extract-false")
    async def extract_false(self) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, extract=False)
        class NoExtractAgent: ...

        meta = getattr(NoExtractAgent, REMEMBER_META)
        return {"extract": meta.extract}

    @get("/top-k")
    async def top_k(self) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, top_k=10)
        class TopKAgent: ...

        meta = getattr(TopKAgent, REMEMBER_META)
        return {"top_k": meta.top_k}

    @get("/inject-false")
    async def inject_false(self) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, inject=False)
        class NoInjectAgent: ...

        meta = getattr(NoInjectAgent, REMEMBER_META)
        return {"inject": meta.inject}

    @get("/attaches-meta")
    async def attaches_meta(self) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None, system="personalized assistant")
        @remember(store=store, extract=True, inject=True, top_k=5)
        class PA: ...

        meta = getattr(PA, REMEMBER_META)
        return {"has_meta": isinstance(meta, RememberMeta)}


@controller("/remember-agent")
class RememberAgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._cfg = cfg
        self._mock = mock

    @post("/run")
    async def run(self, body: Json[_AgentRequest]) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None, system="You are a personalized assistant.")
        @remember(store=store, extract=False, inject=False)
        class PersonalizedAgent: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        resp = await runner.run(PersonalizedAgent(), body.prompt)
        return {"content": resp.content}

    @post("/run-inject-only")
    async def run_inject_only(self, body: Json[_AgentRequest]) -> dict:
        store = InMemoryUserMemoryStore()

        @agent(model=None, system="Personalized assistant.")
        @remember(store=store, extract=False, inject=True, top_k=3)
        class InjectOnlyAgent: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        resp = await runner.run(InjectOnlyAgent(), body.prompt)
        return {"content": resp.content, "has_content": resp.content is not None}


@controller("/user-memory")
class UserMemoryController:
    def __init__(self, store: InMemoryUserMemoryStore) -> None:
        self._store = store

    @post("/add")
    async def add(self, body: Json[dict]) -> dict:
        fact = MemoryFact(
            memory_id=body["memory_id"],
            user_id=body["user_id"],
            content=body["content"],
        )
        await self._store.add(fact)
        return {"added": True}

    @get("/get/{user_id}/{memory_id}")
    async def get_fact(self, user_id: str, memory_id: str) -> dict:
        result = await self._store.get(user_id, memory_id)
        if result is None:
            return {"found": False, "content": None}
        return {"found": True, "content": result.content}

    @get("/search/{user_id}")
    async def search(self, user_id: str, q: str) -> dict:
        results = await self._store.search(user_id, q)
        return {"count": len(results), "contents": [f.content for f in results]}

    @post("/clear/{user_id}")
    async def clear(self, user_id: str) -> dict:
        await self._store.clear(user_id)
        facts = await self._store.list(user_id)
        return {"remaining": len(facts)}

    @get("/list/{user_id}")
    async def list_facts(self, user_id: str) -> dict:
        facts = await self._store.list(user_id)
        return {"count": len(facts)}

    @get("/count")
    async def count(self) -> dict:
        return {"count": len(self._store._facts) if hasattr(self._store, "_facts") else -1}


@controller("/memory-fact")
class MemoryFactController:
    @post("/reinforce")
    async def reinforce(self, body: Json[dict]) -> dict:
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        fact.confidence = body.get("initial_confidence", 0.5)
        before = fact.confidence
        fact.reinforce()
        return {"before": before, "after": fact.confidence, "increased": fact.confidence > before}

    @post("/reinforce-capped")
    async def reinforce_capped(self) -> dict:
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        fact.confidence = 1.0
        fact.reinforce()
        return {"confidence": fact.confidence, "capped": fact.confidence <= 1.0}

    @post("/decay")
    async def decay(self, body: Json[dict]) -> dict:
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        fact.confidence = 1.0
        factor = body.get("factor", 0.5)
        fact.decay(factor=factor)
        return {"confidence": fact.confidence}

    @get("/default-confidence")
    async def default_confidence(self) -> dict:
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        return {"confidence": fact.confidence}


@module(
    controllers=[
        RememberMetaController,
        RememberAgentController,
        UserMemoryController,
        MemoryFactController,
    ],
    providers=[
        use_value(provide=MockTransport, value=_MOCK),
        use_value(provide=InMemoryUserMemoryStore, value=_STORE),
    ],
)
class VectorMemoryModule: ...


def build_app(*responses) -> TestClient:
    _MOCK.reset()
    # Reset the shared store by replacing its internal state directly
    if hasattr(_STORE, "_facts"):
        _STORE._facts.clear()
    elif hasattr(_STORE, "_data"):
        _STORE._data.clear()
    for content in responses:
        _MOCK.queue_response(_completion(content))
    return TestClient(LaurenFactory.create(VectorMemoryModule))


# ---------------------------------------------------------------------------
# Tests: @remember decorator metadata
# ---------------------------------------------------------------------------


class TestRememberDecoratorMetadata:
    def test_remember_attaches_metadata(self):
        client = build_app()
        r = client.get("/remember-meta/attaches-meta")
        assert r.status_code == 200
        assert r.json()["has_meta"] is True

    def test_remember_inject_true(self):
        client = build_app()
        r = client.get("/remember-meta/inject-true")
        assert r.status_code == 200
        data = r.json()
        assert data["inject"] is True
        assert data["is_remember_meta"] is True

    def test_remember_extract_true(self):
        client = build_app()
        r = client.get("/remember-meta/extract-true")
        assert r.status_code == 200
        assert r.json()["extract"] is True

    def test_remember_extract_false(self):
        client = build_app()
        r = client.get("/remember-meta/extract-false")
        assert r.status_code == 200
        assert r.json()["extract"] is False

    def test_remember_top_k_stored(self):
        client = build_app()
        r = client.get("/remember-meta/top-k")
        assert r.status_code == 200
        assert r.json()["top_k"] == 10

    def test_remember_inject_false(self):
        client = build_app()
        r = client.get("/remember-meta/inject-false")
        assert r.status_code == 200
        assert r.json()["inject"] is False


# ---------------------------------------------------------------------------
# Tests: Agent with @remember runs
# ---------------------------------------------------------------------------


class TestRememberAgentRuns:
    def test_agent_with_remember_runs_successfully(self):
        client = build_app("Hello! I remember you.")
        r = client.post("/remember-agent/run", json={"prompt": "Hello"})
        assert r.status_code == 200
        assert r.json()["content"] == "Hello! I remember you."

    def test_agent_with_remember_inject_only_runs_successfully(self):
        client = build_app("Based on your preferences...")
        r = client.post("/remember-agent/run-inject-only", json={"prompt": "What should I use?"})
        assert r.status_code == 200
        assert r.json()["has_content"] is True


# ---------------------------------------------------------------------------
# Tests: InMemoryUserMemoryStore
# ---------------------------------------------------------------------------


class TestInMemoryUserMemoryStore:
    def test_add_and_get_fact(self):
        client = build_app()
        client.post(
            "/user-memory/add",
            json={"memory_id": "m1", "user_id": "user-1", "content": "User prefers dark mode"},
        )
        r = client.get("/user-memory/get/user-1/m1")
        assert r.status_code == 200
        data = r.json()
        assert data["found"] is True
        assert data["content"] == "User prefers dark mode"

    def test_get_returns_none_for_missing_fact(self):
        client = build_app()
        r = client.get("/user-memory/get/user-1/nonexistent")
        assert r.status_code == 200
        assert r.json()["found"] is False

    def test_search_finds_matching_fact(self):
        client = build_app()
        client.post(
            "/user-memory/add",
            json={"memory_id": "m1", "user_id": "user-1", "content": "User likes Python programming"},
        )
        r = client.get("/user-memory/search/user-1?q=python")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        assert "User likes Python programming" in data["contents"]

    def test_search_returns_empty_for_no_match(self):
        client = build_app()
        client.post(
            "/user-memory/add",
            json={"memory_id": "m1", "user_id": "user-1", "content": "User likes Python"},
        )
        r = client.get("/user-memory/search/user-1?q=javascript")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_different_users_facts_are_isolated(self):
        client = build_app()
        client.post(
            "/user-memory/add",
            json={"memory_id": "a1", "user_id": "alice", "content": "Alice prefers morning meetings"},
        )
        client.post(
            "/user-memory/add",
            json={"memory_id": "b1", "user_id": "bob", "content": "Bob works remotely"},
        )
        r_alice = client.get("/user-memory/search/alice?q=morning")
        r_bob = client.get("/user-memory/search/bob?q=remote")
        assert r_alice.status_code == 200
        assert r_bob.status_code == 200
        assert r_alice.json()["count"] >= 1
        assert r_bob.json()["count"] >= 1

    def test_clear_removes_user_facts(self):
        client = build_app()
        client.post(
            "/user-memory/add",
            json={"memory_id": "m1", "user_id": "user-1", "content": "fact 1"},
        )
        client.post(
            "/user-memory/add",
            json={"memory_id": "m2", "user_id": "user-1", "content": "fact 2"},
        )
        r = client.post("/user-memory/clear/user-1", json={})
        assert r.status_code == 200
        assert r.json()["remaining"] == 0

    def test_list_returns_user_facts(self):
        client = build_app()
        client.post(
            "/user-memory/add",
            json={"memory_id": "m1", "user_id": "user-1", "content": "fact A"},
        )
        client.post(
            "/user-memory/add",
            json={"memory_id": "m2", "user_id": "user-1", "content": "fact B"},
        )
        r = client.get("/user-memory/list/user-1")
        assert r.status_code == 200
        assert r.json()["count"] == 2


# ---------------------------------------------------------------------------
# Tests: MemoryFact reinforce/decay
# ---------------------------------------------------------------------------


class TestMemoryFact:
    def test_reinforce_increases_confidence(self):
        client = build_app()
        r = client.post("/memory-fact/reinforce", json={"initial_confidence": 0.5})
        assert r.status_code == 200
        assert r.json()["increased"] is True

    def test_reinforce_caps_at_1_0(self):
        client = build_app()
        r = client.post("/memory-fact/reinforce-capped", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["capped"] is True
        assert data["confidence"] <= 1.0

    def test_decay_reduces_confidence(self):
        client = build_app()
        r = client.post("/memory-fact/decay", json={"factor": 0.5})
        assert r.status_code == 200
        assert r.json()["confidence"] == 0.5

    def test_default_confidence_is_1_0(self):
        client = build_app()
        r = client.get("/memory-fact/default-confidence")
        assert r.status_code == 200
        assert r.json()["confidence"] == 1.0

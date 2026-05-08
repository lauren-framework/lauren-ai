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
"""

import pytest

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._memory import REMEMBER_META, RememberMeta, remember
from lauren_ai._memory._in_memory_user import InMemoryUserMemoryStore
from lauren_ai._memory._user import MemoryFact
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


def _make_fact(user_id: str, content: str, memory_id: str = "f1") -> MemoryFact:
    return MemoryFact(memory_id=memory_id, user_id=user_id, content=content)


# ---------------------------------------------------------------------------
# Tests: @remember decorator metadata
# ---------------------------------------------------------------------------


class TestRememberDecoratorMetadata:
    def test_remember_attaches_metadata(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None, system="personalized assistant")
        @remember(store=store, extract=True, inject=True, top_k=5)
        class PA: ...

        meta = getattr(PA, REMEMBER_META)
        assert isinstance(meta, RememberMeta)

    def test_remember_inject_true(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, inject=True)
        class InjectAgent: ...

        meta = getattr(InjectAgent, REMEMBER_META)
        assert meta.inject is True

    def test_remember_extract_true(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, extract=True)
        class ExtractAgent: ...

        meta = getattr(ExtractAgent, REMEMBER_META)
        assert meta.extract is True

    def test_remember_extract_false(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, extract=False)
        class NoExtractAgent: ...

        meta = getattr(NoExtractAgent, REMEMBER_META)
        assert meta.extract is False

    def test_remember_top_k_stored(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, top_k=10)
        class TopKAgent: ...

        meta = getattr(TopKAgent, REMEMBER_META)
        assert meta.top_k == 10

    def test_remember_inject_false(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None)
        @remember(store=store, inject=False)
        class NoInjectAgent: ...

        meta = getattr(NoInjectAgent, REMEMBER_META)
        assert meta.inject is False


# ---------------------------------------------------------------------------
# Tests: Agent with @remember runs with MockTransport
# ---------------------------------------------------------------------------


class TestRememberAgentRuns:
    async def test_agent_with_remember_runs_successfully(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None, system="You are a personalized assistant.")
        @remember(store=store, extract=False, inject=False)
        class PersonalizedAgent: ...

        mock = MockTransport()
        mock.queue_response(_completion("Hello! I remember you."))
        client = AgentTestClient(PersonalizedAgent(), mock)
        resp = await client.run_async("Hello")
        assert resp.content == "Hello! I remember you."

    async def test_agent_with_remember_inject_only_runs_successfully(self):
        store = InMemoryUserMemoryStore()

        @agent(model=None, system="Personalized assistant.")
        @remember(store=store, extract=False, inject=True, top_k=3)
        class InjectOnlyAgent: ...

        mock = MockTransport()
        mock.queue_response(_completion("Based on your preferences..."))
        client = AgentTestClient(InjectOnlyAgent(), mock)
        resp = await client.run_async("What should I use?")
        assert resp.content is not None


# ---------------------------------------------------------------------------
# Tests: InMemoryUserMemoryStore
# ---------------------------------------------------------------------------


class TestInMemoryUserMemoryStore:
    async def test_add_and_get_fact(self):
        store = InMemoryUserMemoryStore()
        fact = _make_fact("user-1", "User prefers dark mode", memory_id="m1")
        await store.add(fact)
        retrieved = await store.get("user-1", "m1")
        assert retrieved is not None
        assert retrieved.content == "User prefers dark mode"

    async def test_get_returns_none_for_missing_fact(self):
        store = InMemoryUserMemoryStore()
        result = await store.get("user-1", "nonexistent")
        assert result is None

    async def test_search_finds_matching_fact(self):
        store = InMemoryUserMemoryStore()
        fact = _make_fact("user-1", "User likes Python programming", memory_id="m1")
        await store.add(fact)
        results = await store.search("user-1", "python")
        assert len(results) >= 1
        assert results[0].content == "User likes Python programming"

    async def test_search_returns_empty_for_no_match(self):
        store = InMemoryUserMemoryStore()
        fact = _make_fact("user-1", "User likes Python", memory_id="m1")
        await store.add(fact)
        results = await store.search("user-1", "javascript")
        assert len(results) == 0

    async def test_different_users_facts_are_isolated(self):
        store = InMemoryUserMemoryStore()
        await store.add(_make_fact("alice", "Alice prefers morning meetings", memory_id="a1"))
        await store.add(_make_fact("bob", "Bob works remotely", memory_id="b1"))
        alice_facts = await store.search("alice", "morning")
        bob_facts = await store.search("bob", "remote")
        assert all(f.user_id == "alice" for f in alice_facts)
        assert all(f.user_id == "bob" for f in bob_facts)

    async def test_clear_removes_user_facts(self):
        store = InMemoryUserMemoryStore()
        await store.add(_make_fact("user-1", "fact 1", memory_id="m1"))
        await store.add(_make_fact("user-1", "fact 2", memory_id="m2"))
        await store.clear("user-1")
        facts = await store.list("user-1")
        assert facts == []

    async def test_list_returns_user_facts(self):
        store = InMemoryUserMemoryStore()
        await store.add(_make_fact("user-1", "fact A", memory_id="m1"))
        await store.add(_make_fact("user-1", "fact B", memory_id="m2"))
        facts = await store.list("user-1")
        assert len(facts) == 2


# ---------------------------------------------------------------------------
# Tests: MemoryFact reinforce/decay
# ---------------------------------------------------------------------------


class TestMemoryFact:
    def test_reinforce_increases_confidence(self):
        fact = _make_fact("u", "some fact")
        fact.confidence = 0.5
        fact.reinforce()
        assert fact.confidence > 0.5

    def test_reinforce_caps_at_1_0(self):
        fact = _make_fact("u", "some fact")
        fact.confidence = 1.0
        fact.reinforce()
        assert fact.confidence <= 1.0

    def test_decay_reduces_confidence(self):
        fact = _make_fact("u", "some fact")
        fact.confidence = 1.0
        fact.decay(factor=0.5)
        assert fact.confidence == 0.5

    def test_default_confidence_is_1_0(self):
        fact = _make_fact("u", "some fact")
        assert fact.confidence == 1.0

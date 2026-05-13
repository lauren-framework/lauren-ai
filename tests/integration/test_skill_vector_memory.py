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

import asyncio

from lauren_ai._agents import agent
from lauren_ai._memory import REMEMBER_META, RememberMeta, remember
from lauren_ai._memory._in_memory_user import InMemoryUserMemoryStore
from lauren_ai._memory._user import MemoryFact
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

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
# Tests: @remember decorator metadata
# ---------------------------------------------------------------------------


class TestRememberDecoratorMetadata:
    def test_remember_attaches_metadata(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model", system="personalized assistant")
        @remember(store=store, extract=True, inject=True, top_k=5)
        class PA: ...

        meta = getattr(PA, REMEMBER_META)
        assert isinstance(meta, RememberMeta)

    def test_remember_inject_true(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model")
        @remember(store=store, inject=True)
        class InjectAgent: ...

        meta = getattr(InjectAgent, REMEMBER_META)
        assert meta.inject is True
        assert isinstance(meta, RememberMeta)

    def test_remember_extract_true(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model")
        @remember(store=store, extract=True)
        class ExtractAgent: ...

        meta = getattr(ExtractAgent, REMEMBER_META)
        assert meta.extract is True

    def test_remember_extract_false(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model")
        @remember(store=store, extract=False)
        class NoExtractAgent: ...

        meta = getattr(NoExtractAgent, REMEMBER_META)
        assert meta.extract is False

    def test_remember_top_k_stored(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model")
        @remember(store=store, top_k=10)
        class TopKAgent: ...

        meta = getattr(TopKAgent, REMEMBER_META)
        assert meta.top_k == 10

    def test_remember_inject_false(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model")
        @remember(store=store, inject=False)
        class NoInjectAgent: ...

        meta = getattr(NoInjectAgent, REMEMBER_META)
        assert meta.inject is False


# ---------------------------------------------------------------------------
# Tests: Agent with @remember runs (via TestClient)
# ---------------------------------------------------------------------------


class TestRememberAgentRuns:
    def test_agent_with_remember_runs_successfully(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model", system="You are a personalized assistant.")
        @remember(store=store, extract=False, inject=False)
        class PersonalizedAgent: ...

        client = TestClient(PersonalizedAgent())
        client.mock.queue_response(_c("Hello! I remember you."))
        result = client.run("Hello")
        assert result.content == "Hello! I remember you."

    def test_agent_with_remember_inject_only_runs_successfully(self):
        store = InMemoryUserMemoryStore()

        @agent(model="mock-model", system="Personalized assistant.")
        @remember(store=store, extract=False, inject=True, top_k=3)
        class InjectOnlyAgent: ...

        client = TestClient(InjectOnlyAgent())
        client.mock.queue_response(_c("Based on your preferences..."))
        result = client.run("What should I use?")
        assert result.content is not None


# ---------------------------------------------------------------------------
# Tests: InMemoryUserMemoryStore (direct Python)
# ---------------------------------------------------------------------------


class TestInMemoryUserMemoryStore:
    def test_add_and_get_fact(self):
        store = InMemoryUserMemoryStore()
        fact = MemoryFact(memory_id="m1", user_id="user-1", content="User prefers dark mode")
        asyncio.run(store.add(fact))
        result = asyncio.run(store.get("user-1", "m1"))
        assert result is not None
        assert result.content == "User prefers dark mode"

    def test_get_returns_none_for_missing_fact(self):
        store = InMemoryUserMemoryStore()
        result = asyncio.run(store.get("user-1", "nonexistent"))
        assert result is None

    def test_search_finds_matching_fact(self):
        store = InMemoryUserMemoryStore()
        fact = MemoryFact(memory_id="m1", user_id="user-1", content="User likes Python programming")
        asyncio.run(store.add(fact))
        results = asyncio.run(store.search("user-1", "python"))
        assert len(results) >= 1
        assert "User likes Python programming" in [f.content for f in results]

    def test_search_returns_empty_for_no_match(self):
        store = InMemoryUserMemoryStore()
        fact = MemoryFact(memory_id="m1", user_id="user-1", content="User likes Python")
        asyncio.run(store.add(fact))
        results = asyncio.run(store.search("user-1", "javascript"))
        assert len(results) == 0

    def test_different_users_facts_are_isolated(self):
        store = InMemoryUserMemoryStore()
        asyncio.run(
            store.add(
                MemoryFact(
                    memory_id="a1", user_id="alice", content="Alice prefers morning meetings"
                )
            )
        )
        asyncio.run(
            store.add(MemoryFact(memory_id="b1", user_id="bob", content="Bob works remotely"))
        )
        alice_results = asyncio.run(store.search("alice", "morning"))
        bob_results = asyncio.run(store.search("bob", "remote"))
        assert len(alice_results) >= 1
        assert len(bob_results) >= 1

    def test_clear_removes_user_facts(self):
        store = InMemoryUserMemoryStore()
        asyncio.run(store.add(MemoryFact(memory_id="m1", user_id="user-1", content="fact 1")))
        asyncio.run(store.add(MemoryFact(memory_id="m2", user_id="user-1", content="fact 2")))
        asyncio.run(store.clear("user-1"))
        remaining = asyncio.run(store.list("user-1"))
        assert len(remaining) == 0

    def test_list_returns_user_facts(self):
        store = InMemoryUserMemoryStore()
        asyncio.run(store.add(MemoryFact(memory_id="m1", user_id="user-1", content="fact A")))
        asyncio.run(store.add(MemoryFact(memory_id="m2", user_id="user-1", content="fact B")))
        facts = asyncio.run(store.list("user-1"))
        assert len(facts) == 2


# ---------------------------------------------------------------------------
# Tests: MemoryFact reinforce/decay (direct Python)
# ---------------------------------------------------------------------------


class TestMemoryFact:
    def test_reinforce_increases_confidence(self):
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        fact.confidence = 0.5
        before = fact.confidence
        fact.reinforce()
        assert fact.confidence > before

    def test_reinforce_caps_at_1_0(self):
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        fact.confidence = 1.0
        fact.reinforce()
        assert fact.confidence <= 1.0

    def test_decay_reduces_confidence(self):
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        fact.confidence = 1.0
        fact.decay(factor=0.5)
        assert fact.confidence == 0.5

    def test_default_confidence_is_1_0(self):
        fact = MemoryFact(memory_id="f1", user_id="u", content="some fact")
        assert fact.confidence == 1.0

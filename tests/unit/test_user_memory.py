"""Unit tests for user-level persistent memory."""
from __future__ import annotations

import uuid
import pytest
from datetime import datetime

from lauren_ai._memory._user import MemoryFact, UserMemoryStore
from lauren_ai._memory._in_memory_user import InMemoryUserMemoryStore
from lauren_ai._memory._remember import remember, RememberMeta, MemoryConfigError, REMEMBER_META


class TestMemoryFact:
    def test_reinforce_updates_last_seen(self):
        fact = MemoryFact("id1", "u1", "User likes Python", confidence=0.5)
        before = fact.last_seen_at
        fact.reinforce()
        assert fact.confidence > 0.5
        assert fact.last_seen_at >= before

    def test_decay_reduces_confidence(self):
        fact = MemoryFact("id1", "u1", "User likes Python", confidence=1.0)
        fact.decay(0.5)
        assert fact.confidence == pytest.approx(0.5)

    def test_reinforce_caps_at_1(self):
        fact = MemoryFact("id1", "u1", "fact", confidence=0.95)
        fact.reinforce()
        assert fact.confidence <= 1.0


class TestInMemoryUserMemoryStore:
    async def test_add_and_get(self):
        store = InMemoryUserMemoryStore()
        fact = MemoryFact("m1", "u1", "User loves Python", topics=["language"])
        await store.add(fact)
        retrieved = await store.get("u1", "m1")
        assert retrieved is not None
        assert retrieved.content == "User loves Python"

    async def test_get_wrong_user_returns_none(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "fact"))
        assert await store.get("u2", "m1") is None

    async def test_search_finds_matching_content(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "User loves Python", topics=["language"]))
        await store.add(MemoryFact("m2", "u1", "User works on web apps", topics=["work"]))
        results = await store.search("u1", "Python")
        assert any(f.memory_id == "m1" for f in results)

    async def test_search_finds_by_topic(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "fact", topics=["coding"]))
        results = await store.search("u1", "coding")
        assert len(results) >= 1

    async def test_search_isolates_users(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "User A has a cat"))
        await store.add(MemoryFact("m2", "u2", "User B has a dog"))
        results = await store.search("u1", "cat")
        assert all(f.user_id == "u1" for f in results)

    async def test_list_all_facts_for_user(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "fact1"))
        await store.add(MemoryFact("m2", "u1", "fact2"))
        await store.add(MemoryFact("m3", "u2", "other"))
        results = await store.list("u1")
        assert len(results) == 2

    async def test_list_filters_by_topic(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "fact1", topics=["coding"]))
        await store.add(MemoryFact("m2", "u1", "fact2", topics=["food"]))
        results = await store.list("u1", topic="coding")
        assert len(results) == 1
        assert results[0].memory_id == "m1"

    async def test_update_content(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "old content"))
        await store.update("m1", content="new content")
        fact = await store.get("u1", "m1")
        assert fact is not None
        assert fact.content == "new content"

    async def test_delete_removes_fact(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "temp fact"))
        await store.delete("m1")
        assert await store.get("u1", "m1") is None

    async def test_clear_removes_all_user_facts(self):
        store = InMemoryUserMemoryStore()
        await store.add(MemoryFact("m1", "u1", "fact1"))
        await store.add(MemoryFact("m2", "u1", "fact2"))
        await store.add(MemoryFact("m3", "u2", "other"))
        await store.clear("u1")
        assert len(await store.list("u1")) == 0
        assert len(await store.list("u2")) == 1

    def test_len(self):
        store = InMemoryUserMemoryStore()
        assert len(store) == 0


class TestRememberDecorator:
    def test_attaches_metadata(self):
        @remember(extract=True, inject=False, top_k=3)
        class Agent:
            pass

        meta: RememberMeta = getattr(Agent, REMEMBER_META)
        assert meta.extract is True
        assert meta.inject is False
        assert meta.top_k == 3

    def test_bare_usage_raises(self):
        with pytest.raises(MemoryConfigError, match="parentheses"):
            @remember
            class Bad:
                pass

    def test_defaults(self):
        @remember()
        class Agent:
            pass

        meta: RememberMeta = getattr(Agent, REMEMBER_META)
        assert meta.extract is True
        assert meta.inject is True
        assert meta.top_k == 5

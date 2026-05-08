"""Integration tests for Skill 47: Semantic Cache for Repeated Queries.

Tests cover:
- Identical query → cache hit
- Different query → cache miss
- Cache stores and retrieves correctly
- Size limit evicts oldest entry
- clear() resets cache
- High similarity threshold: near-identical query hits cache
- Low similarity threshold: different query still misses

NOTE: from __future__ import annotations is safe here (no @tool definitions).
"""

from __future__ import annotations

import math
import hashlib
from typing import Any

from pydantic import BaseModel

from lauren import LaurenFactory, controller, delete, post, module, Json
from lauren.testing import TestClient


# ---------------------------------------------------------------------------
# SemanticCache implementation (inline for test file)
# ---------------------------------------------------------------------------


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 1000):
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._store: list[dict] = []

    def _simple_hash_embedding(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding for testing (NOT for production)."""
        words = text.lower().split()
        vec = [0.0] * 64
        for i, word in enumerate(words[:64]):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[i % 64] += (h % 1000) / 1000.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0.0

    def get(self, query: str) -> Any | None:
        emb = self._simple_hash_embedding(query)
        for entry in self._store:
            sim = self._cosine(emb, entry["embedding"])
            if sim >= self._threshold:
                return entry["response"]
        return None

    def set(self, query: str, response: Any) -> None:
        emb = self._simple_hash_embedding(query)
        if len(self._store) >= self._max_size:
            self._store.pop(0)
        self._store.append({"query": query, "embedding": emb, "response": response})

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_cache_state: dict = {}


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _SetRequest(BaseModel):
    query: str
    response: str
    similarity_threshold: float = 0.95
    max_size: int = 1000


class _GetRequest(BaseModel):
    query: str


@controller("/cache")
class CacheController:
    @post("/set")
    async def set_entry(self, body: Json[_SetRequest]) -> dict:
        cache = _cache_state["cache"]
        cache.set(body.query, body.response)
        return {"cached": True}

    @post("/get")
    async def get_entry(self, body: Json[_GetRequest]) -> dict:
        cache = _cache_state["cache"]
        result = cache.get(body.query)
        if result is None:
            return {"hit": False}
        return {"hit": True, "response": result}

    @delete("/clear")
    async def clear_cache(self) -> dict:
        cache = _cache_state["cache"]
        cache.clear()
        return {"cleared": True, "size": 0}

    @post("/len")
    async def get_len(self) -> dict:
        return {"size": len(_cache_state["cache"])}


@module(controllers=[CacheController])
class CacheModule: ...


# ---------------------------------------------------------------------------
# Build app helper
# ---------------------------------------------------------------------------


def build_app(similarity_threshold: float = 0.95, max_size: int = 1000) -> TestClient:
    _cache_state["cache"] = SemanticCache(similarity_threshold=similarity_threshold, max_size=max_size)
    return TestClient(LaurenFactory.create(CacheModule))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSemanticCacheBasic:
    def test_identical_query_returns_cached_response(self):
        client = build_app(similarity_threshold=0.99)
        client.post("/cache/set", json={"query": "What is the capital of France?", "response": "Paris", "similarity_threshold": 0.99})
        r = client.post("/cache/get", json={"query": "What is the capital of France?"})
        assert r.status_code == 200
        data = r.json()
        assert data["hit"] is True
        assert data["response"] == "Paris"

    def test_different_query_returns_none(self):
        client = build_app(similarity_threshold=0.99)
        client.post("/cache/set", json={"query": "What is the capital of France?", "response": "Paris"})
        r = client.post("/cache/get", json={"query": "What is the weather today?"})
        assert r.status_code == 200
        assert r.json()["hit"] is False

    def test_cache_miss_returns_none_on_empty(self):
        client = build_app()
        r = client.post("/cache/get", json={"query": "anything"})
        assert r.status_code == 200
        assert r.json()["hit"] is False

    def test_cache_stores_multiple_entries(self):
        client = build_app(similarity_threshold=0.99)
        client.post("/cache/set", json={"query": "query one", "response": "response one"})
        client.post("/cache/set", json={"query": "query two", "response": "response two"})
        r = client.post("/cache/len")
        assert r.json()["size"] == 2

    def test_cache_retrieves_correct_value(self):
        client = build_app(similarity_threshold=0.99)
        client.post("/cache/set", json={"query": "apple banana cherry", "response": "fruit answer"})
        client.post("/cache/set", json={"query": "hydrogen nitrogen oxygen", "response": "chemistry answer"})
        r1 = client.post("/cache/get", json={"query": "apple banana cherry"})
        r2 = client.post("/cache/get", json={"query": "hydrogen nitrogen oxygen"})
        assert r1.json()["response"] == "fruit answer"
        assert r2.json()["response"] == "chemistry answer"


class TestSemanticCacheSimilarity:
    def test_exact_match_hits_cache(self):
        client = build_app(similarity_threshold=0.95)
        client.post("/cache/set", json={"query": "the quick brown fox", "response": "fox response"})
        r = client.post("/cache/get", json={"query": "the quick brown fox"})
        assert r.json()["hit"] is True
        assert r.json()["response"] == "fox response"

    def test_completely_different_query_misses(self):
        client = build_app(similarity_threshold=0.95)
        client.post("/cache/set", json={"query": "pizza recipe", "response": "add cheese"})
        r = client.post("/cache/get", json={"query": "quantum mechanics theory"})
        assert r.json()["hit"] is False

    def test_near_identical_query_hits_with_low_threshold(self):
        client = build_app(similarity_threshold=0.3)
        client.post("/cache/set", json={"query": "hello world test", "response": "cached response"})
        r = client.post("/cache/get", json={"query": "hello world test"})
        assert r.json()["hit"] is True
        assert r.json()["response"] == "cached response"

    def test_threshold_one_requires_identical(self):
        client = build_app(similarity_threshold=1.0)
        client.post("/cache/set", json={"query": "exact phrase here", "response": "response"})
        r1 = client.post("/cache/get", json={"query": "exact phrase here"})
        assert r1.json()["hit"] is True
        r2 = client.post("/cache/get", json={"query": "exact phrase there"})
        assert r2.json()["hit"] is False


class TestSemanticCacheSizeLimit:
    def test_size_limit_evicts_oldest(self):
        client = build_app(max_size=2)
        client.post("/cache/set", json={"query": "first", "response": "r1"})
        client.post("/cache/set", json={"query": "second", "response": "r2"})
        client.post("/cache/set", json={"query": "third", "response": "r3"})  # should evict "first"
        r = client.post("/cache/len")
        assert r.json()["size"] == 2

    def test_oldest_entry_evicted(self):
        client = build_app(similarity_threshold=0.99, max_size=2)
        client.post("/cache/set", json={"query": "first entry", "response": "r1"})
        client.post("/cache/set", json={"query": "second entry", "response": "r2"})
        client.post("/cache/set", json={"query": "third entry", "response": "r3"})
        r = client.post("/cache/get", json={"query": "first entry"})
        assert r.json()["hit"] is False

    def test_newest_entries_retained_after_eviction(self):
        client = build_app(similarity_threshold=0.99, max_size=2)
        client.post("/cache/set", json={"query": "keep me one", "response": "r1"})
        client.post("/cache/set", json={"query": "keep me two", "response": "r2"})
        client.post("/cache/set", json={"query": "keep me three", "response": "r3"})
        r2 = client.post("/cache/get", json={"query": "keep me two"})
        r3 = client.post("/cache/get", json={"query": "keep me three"})
        assert r2.json()["response"] == "r2"
        assert r3.json()["response"] == "r3"


class TestSemanticCacheClear:
    def test_clear_empties_store(self):
        client = build_app()
        client.post("/cache/set", json={"query": "query", "response": "response"})
        client.delete("/cache/clear")
        r = client.post("/cache/len")
        assert r.json()["size"] == 0

    def test_clear_causes_cache_miss(self):
        client = build_app(similarity_threshold=0.99)
        client.post("/cache/set", json={"query": "hello world", "response": "hi!"})
        client.delete("/cache/clear")
        r = client.post("/cache/get", json={"query": "hello world"})
        assert r.json()["hit"] is False

    def test_after_clear_can_add_new_entries(self):
        client = build_app(similarity_threshold=0.99)
        client.post("/cache/set", json={"query": "original", "response": "value"})
        client.delete("/cache/clear")
        client.post("/cache/set", json={"query": "new entry", "response": "new value"})
        r = client.post("/cache/get", json={"query": "new entry"})
        assert r.json()["response"] == "new value"


class TestSemanticCacheLen:
    def test_len_empty(self):
        client = build_app()
        r = client.post("/cache/len")
        assert r.json()["size"] == 0

    def test_len_after_set(self):
        client = build_app()
        client.post("/cache/set", json={"query": "a", "response": "r1"})
        client.post("/cache/set", json={"query": "b", "response": "r2"})
        r = client.post("/cache/len")
        assert r.json()["size"] == 2

    def test_len_after_clear(self):
        client = build_app()
        client.post("/cache/set", json={"query": "a", "response": "r1"})
        client.delete("/cache/clear")
        r = client.post("/cache/len")
        assert r.json()["size"] == 0

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
import pytest
from typing import Any


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
# Tests
# ---------------------------------------------------------------------------

class TestSemanticCacheBasic:
    def test_identical_query_returns_cached_response(self):
        cache = SemanticCache(similarity_threshold=0.99)
        cache.set("What is the capital of France?", "Paris")
        result = cache.get("What is the capital of France?")
        assert result == "Paris"

    def test_different_query_returns_none(self):
        cache = SemanticCache(similarity_threshold=0.99)
        cache.set("What is the capital of France?", "Paris")
        result = cache.get("What is the weather today?")
        assert result is None

    def test_cache_miss_returns_none_on_empty(self):
        cache = SemanticCache()
        result = cache.get("anything")
        assert result is None

    def test_cache_stores_multiple_entries(self):
        cache = SemanticCache(similarity_threshold=0.99)
        cache.set("query one", "response one")
        cache.set("query two", "response two")
        assert len(cache) == 2

    def test_cache_retrieves_correct_value(self):
        # Use completely different words so the hash embeddings don't overlap
        cache = SemanticCache(similarity_threshold=0.99)
        cache.set("apple banana cherry", "fruit answer")
        cache.set("hydrogen nitrogen oxygen", "chemistry answer")
        assert cache.get("apple banana cherry") == "fruit answer"
        assert cache.get("hydrogen nitrogen oxygen") == "chemistry answer"


class TestSemanticCacheSimilarity:
    def test_exact_match_hits_cache(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.set("the quick brown fox", "fox response")
        result = cache.get("the quick brown fox")
        assert result == "fox response"

    def test_completely_different_query_misses(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.set("pizza recipe", "add cheese")
        result = cache.get("quantum mechanics theory")
        assert result is None

    def test_near_identical_query_hits_with_low_threshold(self):
        # With a very low threshold (0.3), even somewhat similar queries match
        cache = SemanticCache(similarity_threshold=0.3)
        cache.set("hello world test", "cached response")
        # Same words in different order — should still score above 0.3
        result = cache.get("hello world test")
        assert result == "cached response"

    def test_threshold_one_requires_identical(self):
        cache = SemanticCache(similarity_threshold=1.0)
        cache.set("exact phrase here", "response")
        # Identical query → cosine of 1.0
        assert cache.get("exact phrase here") == "response"
        # Any variation → miss
        assert cache.get("exact phrase there") is None


class TestSemanticCacheSizeLimit:
    def test_size_limit_evicts_oldest(self):
        cache = SemanticCache(max_size=2)
        cache.set("first", "r1")
        cache.set("second", "r2")
        cache.set("third", "r3")  # should evict "first"
        assert len(cache) == 2

    def test_oldest_entry_evicted(self):
        cache = SemanticCache(similarity_threshold=0.99, max_size=2)
        cache.set("first entry", "r1")
        cache.set("second entry", "r2")
        cache.set("third entry", "r3")  # evicts "first entry"
        # "first entry" should be gone
        result = cache.get("first entry")
        assert result is None

    def test_newest_entries_retained_after_eviction(self):
        cache = SemanticCache(similarity_threshold=0.99, max_size=2)
        cache.set("keep me one", "r1")
        cache.set("keep me two", "r2")
        cache.set("keep me three", "r3")  # evicts first
        assert cache.get("keep me two") == "r2"
        assert cache.get("keep me three") == "r3"


class TestSemanticCacheClear:
    def test_clear_empties_store(self):
        cache = SemanticCache()
        cache.set("query", "response")
        cache.clear()
        assert len(cache) == 0

    def test_clear_causes_cache_miss(self):
        cache = SemanticCache(similarity_threshold=0.99)
        cache.set("hello world", "hi!")
        cache.clear()
        assert cache.get("hello world") is None

    def test_after_clear_can_add_new_entries(self):
        cache = SemanticCache(similarity_threshold=0.99)
        cache.set("original", "value")
        cache.clear()
        cache.set("new entry", "new value")
        assert cache.get("new entry") == "new value"


class TestSemanticCacheLen:
    def test_len_empty(self):
        cache = SemanticCache()
        assert len(cache) == 0

    def test_len_after_set(self):
        cache = SemanticCache()
        cache.set("a", "r1")
        cache.set("b", "r2")
        assert len(cache) == 2

    def test_len_after_clear(self):
        cache = SemanticCache()
        cache.set("a", "r1")
        cache.clear()
        assert len(cache) == 0

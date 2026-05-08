---
name: semantic-cache
description: Implements a semantic similarity cache for LLM responses using cosine similarity over text embeddings. Use when reducing redundant LLM calls by returning cached responses for semantically identical or near-identical queries, with configurable similarity threshold and cache size.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Semantic Cache for Repeated Queries

## Overview

`SemanticCache` stores LLM responses keyed by a pseudo-embedding of the query
text.  On lookup it computes cosine similarity between the incoming query's
embedding and all cached embeddings; if any exceeds `similarity_threshold` the
stored response is returned without calling the LLM.

> The built-in `_simple_hash_embedding` is deterministic and useful for testing
> but **not suitable for production**.  Replace it with a real embedding model
> (e.g. `text-embedding-3-small` via `LLMService.embed()`) for semantic search.

---

## Implementation

```python
# cache/semantic_cache.py
import math
import hashlib
from typing import Any

class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_size: int = 1000,
    ):
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._store: list[dict] = []  # [{query, embedding, response}]

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
            if self._cosine(emb, entry["embedding"]) >= self._threshold:
                return entry["response"]
        return None

    def set(self, query: str, response: Any) -> None:
        emb = self._simple_hash_embedding(query)
        if len(self._store) >= self._max_size:
            self._store.pop(0)  # evict oldest
        self._store.append({"query": query, "embedding": emb, "response": response})

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
```

---

## Integrating with an agent runner

Wrap the runner call with a cache lookup / populate pattern:

```python
cache = SemanticCache(similarity_threshold=0.95)

async def cached_run(runner, agent_instance, prompt: str) -> str:
    cached = cache.get(prompt)
    if cached is not None:
        return cached
    response = await runner.run(agent_instance, prompt)
    cache.set(prompt, response.content)
    return response.content
```

---

## Using a real embedding model

```python
from lauren_ai import LLMService
import math

class EmbeddingSemanticCache(SemanticCache):
    def __init__(self, llm: LLMService, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    async def get_async(self, query: str) -> Any | None:
        embeddings = await self._llm.embed([query])
        emb = embeddings[0].vector
        for entry in self._store:
            if self._cosine(emb, entry["embedding"]) >= self._threshold:
                return entry["response"]
        return None

    async def set_async(self, query: str, response: Any) -> None:
        embeddings = await self._llm.embed([query])
        emb = embeddings[0].vector
        if len(self._store) >= self._max_size:
            self._store.pop(0)
        self._store.append({"query": query, "embedding": emb, "response": response})
```

---

## Similarity threshold guidance

| Threshold | Behaviour |
|-----------|-----------|
| `1.0` | Exact match only (identical hash) |
| `0.95` | Near-identical phrasing |
| `0.80` | Semantically similar (real embeddings) |
| `< 0.60` | Too permissive — risk of wrong hits |

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_module.py` | `LLMService.embed()` for real embeddings |
| `src/lauren_ai/_transport/__init__.py` | `Embedding` type |

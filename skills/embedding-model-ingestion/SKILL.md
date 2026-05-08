---
name: embedding-model-ingestion
description: Handles embedding model selection and batch document ingestion into vector stores. Use when processing large document corpora efficiently, swapping embedding models (OpenAI, Cohere, local), batching API calls to avoid rate limits, or building ingestion pipelines.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Embedding Model Selection & Batch Ingestion

## Overview

The built-in `InMemoryVectorStore` uses TF-IDF (no external embeddings needed).
For production use cases, you can provide pre-computed embeddings via the
`embedding=` parameter on `upsert()`, or supply a batch embedding function to
your ingestion pipeline.

---

## Batch ingestion pattern

```python
from lauren_ai._knowledge import KnowledgeBase, TextLoader, FixedSizeChunker
from lauren_ai._memory._vector import InMemoryVectorStore

async def batch_ingest(
    kb: KnowledgeBase,
    documents: list[str],
    batch_size: int = 10,
) -> int:
    """Ingest documents in batches, returning total chunks indexed."""
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        for text in batch:
            count = await kb.load(TextLoader(text, is_file=False))
            total += count
    return total
```

---

## Providing a custom embedding function

Replace the default TF-IDF with dense embeddings from any provider by passing
`embedding=` to `store.upsert()`:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed(texts: list[str]) -> list[list[float]]:
    resp = await client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]

store = InMemoryVectorStore()
embeddings = await embed(["Python is a language", "JavaScript runs in browsers"])
for text, vec in zip(texts, embeddings):
    await store.upsert(text, embedding=vec)
```

---

## Cohere embed API

```python
import cohere

co = cohere.AsyncClient()

async def cohere_embed(texts: list[str]) -> list[list[float]]:
    resp = await co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
    return resp.embeddings
```

---

## Mock embedding for tests

Use a simple deterministic embedding function in tests to avoid network calls:

```python
def mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Produce a 128-d vector from word count (deterministic, no API)."""
    return [[len(t.split()) * 0.1] * 128 for t in texts]
```

---

## Chunking before embedding

Always chunk large documents before embedding to stay within model context windows:

```python
chunker = FixedSizeChunker(chunk_size=512, overlap=64)
kb = KnowledgeBase(store=InMemoryVectorStore(), chunker=chunker)
await kb.load(TextLoader("docs/large_manual.txt"))
```

---

## Batch size tuning

| Provider | Recommended batch_size | Notes |
|----------|----------------------|-------|
| OpenAI `text-embedding-3-small` | 100 | Max 2048 input texts per request |
| Cohere `embed-english-v3.0` | 96 | Max 96 texts per request |
| Local models | 32–64 | Depends on GPU memory |

---

## Reference

- `lauren_ai._knowledge`: `KnowledgeBase`, `TextLoader`, `FixedSizeChunker`
- `lauren_ai._memory._vector`: `InMemoryVectorStore`
- Skills: `rag-pipeline`, `vector-store-integration`

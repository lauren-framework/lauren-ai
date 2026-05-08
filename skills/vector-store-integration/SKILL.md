---
name: vector-store-integration
description: Integrates external vector stores (Pinecone, Weaviate, pgvector) with lauren-ai using the MemoryStore protocol. Use when scaling beyond InMemoryVectorStore for production deployments, connecting to managed vector databases, or implementing custom vector store backends.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Vector Store Integration

## MemoryStore protocol

All vector store integrations implement the `MemoryStore` protocol:

```python
from lauren_ai._memory import MemoryStore, MemoryResult

class MemoryStore(Protocol):
    async def upsert(self, content: str, *, id=None, metadata=None, embedding=None) -> str: ...
    async def search(self, query: str, *, k: int = 5, filter=None) -> list[MemoryResult]: ...
    async def get(self, id: str) -> MemoryResult | None: ...
    async def delete(self, ids: list[str]) -> None: ...
    async def clear(self) -> None: ...
```

Swap backends by replacing the `store=` argument to `KnowledgeBase`.

---

## InMemoryVectorStore (development & testing)

The built-in TF-IDF implementation requires no dependencies:

```python
from lauren_ai._memory._vector import InMemoryVectorStore

store = InMemoryVectorStore()
doc_id = await store.upsert("Python is a programming language", metadata={"source": "docs"})
results = await store.search("programming", k=3)
```

---

## Pinecone integration

```python
from pinecone import Pinecone
from lauren_ai._memory import MemoryResult

class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str, embed_fn):
        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._embed = embed_fn

    async def upsert(self, content: str, *, id=None, metadata=None, embedding=None) -> str:
        import uuid
        doc_id = id or str(uuid.uuid4())
        vec = embedding or self._embed([content])[0]
        self._index.upsert(vectors=[(doc_id, vec, {"content": content, **(metadata or {})})])
        return doc_id

    async def search(self, query: str, *, k: int = 5, filter=None) -> list[MemoryResult]:
        query_vec = self._embed([query])[0]
        resp = self._index.query(vector=query_vec, top_k=k, include_metadata=True, filter=filter)
        return [
            MemoryResult(
                id=m.id,
                content=m.metadata.get("content", ""),
                score=m.score,
                metadata=m.metadata,
            )
            for m in resp.matches
        ]

    async def get(self, id: str) -> MemoryResult | None:
        resp = self._index.fetch(ids=[id])
        if id not in resp.vectors:
            return None
        v = resp.vectors[id]
        return MemoryResult(id=id, content=v.metadata.get("content", ""), score=1.0, metadata=v.metadata)

    async def delete(self, ids: list[str]) -> None:
        self._index.delete(ids=ids)

    async def clear(self) -> None:
        self._index.delete(delete_all=True)
```

---

## pgvector integration (asyncpg)

```python
import asyncpg
import json
from lauren_ai._memory import MemoryResult

class PgVectorStore:
    def __init__(self, dsn: str, embed_fn, table: str = "embeddings"):
        self._dsn = dsn
        self._embed = embed_fn
        self._table = table
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
            await self._pool.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB
                )
            """)
        return self._pool

    async def upsert(self, content: str, *, id=None, metadata=None, embedding=None) -> str:
        import uuid
        doc_id = id or str(uuid.uuid4())
        vec = embedding or self._embed([content])[0]
        pool = await self._get_pool()
        await pool.execute(
            f"INSERT INTO {self._table} (id, content, embedding, metadata) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (id) DO UPDATE "
            "SET content=$2, embedding=$3, metadata=$4",
            doc_id, content, vec, json.dumps(metadata or {}),
        )
        return doc_id

    async def search(self, query: str, *, k: int = 5, filter=None) -> list[MemoryResult]:
        query_vec = self._embed([query])[0]
        pool = await self._get_pool()
        rows = await pool.fetch(
            f"SELECT id, content, metadata, 1 - (embedding <=> $1) AS score "
            f"FROM {self._table} ORDER BY score DESC LIMIT $2",
            query_vec, k,
        )
        return [
            MemoryResult(id=r["id"], content=r["content"], score=r["score"], metadata=json.loads(r["metadata"]))
            for r in rows
        ]

    async def get(self, id: str) -> MemoryResult | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(f"SELECT * FROM {self._table} WHERE id=$1", id)
        if not row:
            return None
        return MemoryResult(id=id, content=row["content"], score=1.0, metadata=json.loads(row["metadata"]))

    async def delete(self, ids: list[str]) -> None:
        pool = await self._get_pool()
        await pool.execute(f"DELETE FROM {self._table} WHERE id = ANY($1)", ids)

    async def clear(self) -> None:
        pool = await self._get_pool()
        await pool.execute(f"TRUNCATE {self._table}")
```

---

## Wiring into KnowledgeBase

```python
kb = KnowledgeBase(store=PineconeVectorStore(...))
# All KnowledgeBase methods work identically regardless of the store backend
await kb.load(TextLoader("docs/manual.txt"))
results = await kb.search("installation steps", top_k=5)
```

---

## Reference

- `lauren_ai._memory`: `MemoryStore`, `MemoryResult`
- `lauren_ai._memory._vector`: `InMemoryVectorStore`
- `lauren_ai._knowledge`: `KnowledgeBase`
- Skills: `rag-pipeline`, `embedding-model-ingestion`, `hybrid-search`

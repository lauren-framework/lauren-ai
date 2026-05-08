---
name: conversation-state-db
description: Persists conversation history to a database (SQLite/Postgres) via a custom ConversationStore implementation. Use when conversation history must survive process restarts, implementing a production database-backed store, or replacing InMemoryConversationStore with durable storage.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Conversation State Serialization to Database

## Overview

`InMemoryConversationStore` is convenient for development but loses all history
on process restart.  Replace it with a database-backed implementation that
satisfies the `ConversationStore` protocol.

---

## SQLite implementation (aiosqlite)

```python
import asyncio
import json
import time

import aiosqlite

from lauren_ai._memory import ConversationStore


class SQLiteConversationStore:
    """Persist conversation history to SQLite via aiosqlite."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._initialized = False

    async def _ensure_init(self) -> None:
        if not self._initialized:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT NOT NULL,
                        messages TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                await db.commit()
            self._initialized = True

    async def save(self, conversation_id: str, messages: list) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO conversations (id, messages, updated_at) VALUES (?, ?, ?)",
                (conversation_id, json.dumps(messages), time.time()),
            )
            await db.commit()

    async def load(self, conversation_id: str) -> list:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT messages FROM conversations WHERE id = ?",
                (conversation_id,),
            ) as cursor:
                row = await cursor.fetchone()
                return json.loads(row[0]) if row else []

    async def delete(self, conversation_id: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await db.commit()
```

---

## Wiring with an agent

```python
store = SQLiteConversationStore(db_path="/tmp/conversations.db")

@agent(model="claude-opus-4-6", conversation_store=store)
class SupportAgent: ...

resp = await runner.run(SupportAgent(), "Hello", conversation_id="user-42")
```

---

## Postgres implementation (asyncpg)

For production multi-worker deployments, replace SQLite with Postgres.  The
interface is identical — only the driver changes:

```python
import asyncpg
import json

class PostgresConversationStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None

    async def _pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
        return self._pool

    async def save(self, conversation_id: str, messages: list) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (id, messages, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id) DO UPDATE
                SET messages = $2, updated_at = NOW()
                """,
                conversation_id,
                json.dumps(messages),
            )
    # ... load/delete follow the same pattern
```

---

## Schema migrations

Always create the schema before first use.  The pattern above uses
`CREATE TABLE IF NOT EXISTS` as a simple idempotent migration.  For production,
use a dedicated migration tool (Alembic, Flyway).

---

## Testing

Test the store directly without an agent — just call `save` and `load`:

```python
store = SQLiteConversationStore(db_path=":memory:")
messages = [{"role": "user", "content": "hello"}]
await store.save("s1", messages)
loaded = await store.load("s1")
assert loaded == messages
```

---

## Reference

- `lauren_ai._memory`: `ConversationStore` (protocol)
- `lauren_ai._memory._stores`: `InMemoryConversationStore` (reference implementation)
- Skills: `conversation-memory`, `managing-memory`

"""Persistent SQLite-backed implementations of the public memory protocols."""

from __future__ import annotations

__all__ = [
    "SQLiteConversationStore",
    "SQLiteStoreBackend",
    "SQLiteStoreConfig",
    "SQLiteUserMemoryStore",
    "SQLiteVectorStore",
]

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from lauren_ai._exceptions import StorageError

from . import MemoryResult
from ._user import MemoryFact
from ._vector import InMemoryVectorStore

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalise_snapshot(snapshot: Any) -> dict[str, Any]:
    if isinstance(snapshot, list):
        return {"messages": snapshot, "summary": None}
    if isinstance(snapshot, dict):
        messages = snapshot.get("messages", [])
        summary = snapshot.get("summary")
        return {"messages": messages, "summary": summary}
    raise StorageError(
        "Conversation snapshots must be a list of messages or a snapshot dict.",
        backend="sqlite",
    )


def _serialize_fact(fact: MemoryFact) -> tuple[str, str, str, str, float, str, str, str | None]:
    return (
        fact.memory_id,
        fact.user_id,
        fact.content,
        json.dumps(fact.topics),
        fact.confidence,
        fact.created_at.astimezone(UTC).isoformat(),
        fact.last_seen_at.astimezone(UTC).isoformat(),
        fact.source_conversation_id,
    )


def _row_to_fact(row: Any) -> MemoryFact:
    return MemoryFact(
        memory_id=str(row["memory_id"]),
        user_id=str(row["user_id"]),
        content=str(row["content"]),
        topics=cast(list[str], json.loads(str(row["topics_json"]))),
        confidence=float(row["confidence"]),
        created_at=_parse_datetime(str(row["created_at"])),
        last_seen_at=_parse_datetime(str(row["last_seen_at"])),
        source_conversation_id=cast(str | None, row["source_conversation_id"]),
    )


@dataclass(frozen=True, slots=True)
class SQLiteStoreConfig:
    """Configuration shared by the durable SQLite-backed stores."""

    database_path: str
    timeout_seconds: float = 30.0
    table_prefix: str = ""
    uri: bool = False
    pragmas: tuple[str, ...] = (
        "journal_mode=WAL",
        "synchronous=NORMAL",
        "foreign_keys=ON",
    )


class SQLiteStoreBackend:
    """Shared connection and schema manager for SQLite-backed stores."""

    def __init__(self, config: SQLiteStoreConfig | str = SQLiteStoreConfig(":memory:")) -> None:
        if isinstance(config, str):
            config = SQLiteStoreConfig(database_path=config)
        self._config = config
        self._connection: sqlite3.Connection | None = None
        self._connect_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._schemas_ready: set[str] = set()
        self._closed = False

    @classmethod
    async def connect(cls, config: SQLiteStoreConfig | str) -> SQLiteStoreBackend:
        """Create a backend and eagerly open the connection."""
        backend = cls(config)
        await backend._get_connection()
        return backend

    @property
    def config(self) -> SQLiteStoreConfig:
        """Return the immutable backend config."""
        return self._config

    async def close(self) -> None:
        """Close the shared SQLite connection."""
        async with self._connect_lock:
            conn = self._connection
            self._connection = None
            self._schemas_ready.clear()
            self._closed = True
            if conn is None:
                return
            async with self._operation_lock:
                await asyncio.to_thread(conn.close)

    async def ensure_schema(self, schema_id: str, statements: Sequence[str]) -> None:
        """Run schema initialization once per backend instance."""
        async with self._connect_lock:
            conn = await self._get_connection_unlocked()
            if schema_id in self._schemas_ready:
                return
            async with self._operation_lock:
                await asyncio.to_thread(self._ensure_schema_sync, conn, tuple(statements))
                self._schemas_ready.add(schema_id)

    async def fetch_one(self, query: str, parameters: Sequence[Any] = ()) -> Any | None:
        """Fetch a single row."""
        conn = await self._get_connection()
        async with self._operation_lock:
            return await asyncio.to_thread(self._fetch_one_sync, conn, query, tuple(parameters))

    async def fetch_all(self, query: str, parameters: Sequence[Any] = ()) -> list[Any]:
        """Fetch all matching rows."""
        conn = await self._get_connection()
        async with self._operation_lock:
            return await asyncio.to_thread(self._fetch_all_sync, conn, query, tuple(parameters))

    async def execute(self, query: str, parameters: Sequence[Any] = ()) -> None:
        """Execute a write query and commit it."""
        conn = await self._get_connection()
        async with self._operation_lock:
            await asyncio.to_thread(self._execute_sync, conn, query, tuple(parameters))

    async def _get_connection(self) -> sqlite3.Connection:
        async with self._connect_lock:
            return await self._get_connection_unlocked()

    async def _get_connection_unlocked(self) -> sqlite3.Connection:
        if self._closed and self._connection is None:
            self._closed = False
        if self._connection is not None:
            return self._connection

        try:
            conn = await asyncio.to_thread(
                sqlite3.connect,
                self._config.database_path,
                timeout=self._config.timeout_seconds,
                uri=self._config.uri,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            for pragma in self._config.pragmas:
                await asyncio.to_thread(conn.execute, f"PRAGMA {pragma}")
            await asyncio.to_thread(conn.commit)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(
                "Failed to open the SQLite storage backend.",
                backend="sqlite",
                location=self._config.database_path,
                cause=exc,
            ) from exc
        self._connection = conn
        return conn

    @staticmethod
    def _ensure_schema_sync(conn: sqlite3.Connection, statements: Sequence[str]) -> None:
        for statement in statements:
            conn.execute(statement)
        conn.commit()

    @staticmethod
    def _execute_sync(conn: sqlite3.Connection, query: str, parameters: Sequence[Any]) -> None:
        conn.execute(query, parameters)
        conn.commit()

    @staticmethod
    def _fetch_one_sync(conn: sqlite3.Connection, query: str, parameters: Sequence[Any]) -> Any | None:
        cursor = conn.execute(query, parameters)
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    @staticmethod
    def _fetch_all_sync(conn: sqlite3.Connection, query: str, parameters: Sequence[Any]) -> list[Any]:
        cursor = conn.execute(query, parameters)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()


class SQLiteConversationStore:
    """Durable conversation history store backed by SQLite."""

    def __init__(
        self,
        database_path: str = ":memory:",
        *,
        config: SQLiteStoreConfig | None = None,
        backend: SQLiteStoreBackend | None = None,
    ) -> None:
        self._backend = backend or SQLiteStoreBackend(config or SQLiteStoreConfig(database_path=database_path))
        self._owns_backend = backend is None

    @classmethod
    async def connect(cls, config: SQLiteStoreConfig | str) -> SQLiteConversationStore:
        """Create a store and eagerly initialize its backend."""
        backend = await SQLiteStoreBackend.connect(config)
        store = cls(backend=backend)
        await store._ensure_schema()
        return store

    @classmethod
    def from_backend(cls, backend: SQLiteStoreBackend) -> SQLiteConversationStore:
        """Create a store that shares an existing backend."""
        return cls(backend=backend)

    async def load(self, conversation_id: str) -> Any:
        await self._ensure_schema()
        row = await self._backend.fetch_one(
            f"SELECT snapshot_json FROM {self._table_name()} WHERE conversation_id = ?",
            (conversation_id,),
        )
        if row is None:
            return []
        try:
            snapshot = json.loads(str(row["snapshot_json"]))
        except (TypeError, ValueError) as exc:
            raise StorageError(
                "Failed to deserialize the stored conversation snapshot.",
                backend="sqlite",
                location=self._backend.config.database_path,
                cause=exc,
            ) from exc
        return _normalise_snapshot(snapshot)

    async def save(self, conversation_id: str, snapshot: Any) -> None:
        await self._ensure_schema()
        normalised = _normalise_snapshot(snapshot)
        try:
            payload = json.dumps(normalised, default=str)
        except (TypeError, ValueError) as exc:
            raise StorageError(
                "Failed to serialize the conversation snapshot for SQLite persistence.",
                backend="sqlite",
                location=self._backend.config.database_path,
                cause=exc,
            ) from exc
        await self._backend.execute(
            f"""
            INSERT INTO {self._table_name()} (conversation_id, snapshot_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_at = excluded.updated_at
            """,
            (conversation_id, payload, _utc_now_iso()),
        )

    async def delete(self, conversation_id: str) -> None:
        await self._ensure_schema()
        await self._backend.execute(
            f"DELETE FROM {self._table_name()} WHERE conversation_id = ?",
            (conversation_id,),
        )

    async def list_conversations(self) -> list[str]:
        await self._ensure_schema()
        rows = await self._backend.fetch_all(
            f"SELECT conversation_id FROM {self._table_name()} ORDER BY conversation_id ASC"
        )
        return [str(row["conversation_id"]) for row in rows]

    async def clear(self) -> None:
        await self._ensure_schema()
        await self._backend.execute(f"DELETE FROM {self._table_name()}")

    async def close(self) -> None:
        """Close the owned backend connection, if any."""
        if self._owns_backend:
            await self._backend.close()

    async def _ensure_schema(self) -> None:
        await self._backend.ensure_schema(
            schema_id=f"{self._backend.config.table_prefix}:conversation-store",
            statements=(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name()} (
                    conversation_id TEXT PRIMARY KEY,
                    snapshot_json   TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                )
                """,
            ),
        )

    def _table_name(self) -> str:
        return f"{self._backend.config.table_prefix}conversations"


class SQLiteUserMemoryStore:
    """Durable user fact store backed by SQLite."""

    def __init__(
        self,
        database_path: str = ":memory:",
        *,
        config: SQLiteStoreConfig | None = None,
        backend: SQLiteStoreBackend | None = None,
    ) -> None:
        self._backend = backend or SQLiteStoreBackend(config or SQLiteStoreConfig(database_path=database_path))
        self._owns_backend = backend is None

    @classmethod
    async def connect(cls, config: SQLiteStoreConfig | str) -> SQLiteUserMemoryStore:
        backend = await SQLiteStoreBackend.connect(config)
        store = cls(backend=backend)
        await store._ensure_schema()
        return store

    @classmethod
    def from_backend(cls, backend: SQLiteStoreBackend) -> SQLiteUserMemoryStore:
        return cls(backend=backend)

    async def add(self, fact: MemoryFact) -> None:
        await self._ensure_schema()
        await self._backend.execute(
            f"""
            INSERT INTO {self._table_name()} (
                memory_id,
                user_id,
                content,
                topics_json,
                confidence,
                created_at,
                last_seen_at,
                source_conversation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id = excluded.user_id,
                content = excluded.content,
                topics_json = excluded.topics_json,
                confidence = excluded.confidence,
                created_at = excluded.created_at,
                last_seen_at = excluded.last_seen_at,
                source_conversation_id = excluded.source_conversation_id
            """,
            _serialize_fact(fact),
        )

    async def get(self, user_id: str, memory_id: str) -> MemoryFact | None:
        await self._ensure_schema()
        row = await self._backend.fetch_one(
            f"SELECT * FROM {self._table_name()} WHERE user_id = ? AND memory_id = ?",
            (user_id, memory_id),
        )
        if row is None:
            return None
        return _row_to_fact(row)

    async def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryFact]:
        await self._ensure_schema()
        rows = await self._backend.fetch_all(
            f"SELECT * FROM {self._table_name()} WHERE user_id = ?",
            (user_id,),
        )
        query_lower = query.lower()
        matches = []
        for row in rows:
            fact = _row_to_fact(row)
            if query_lower in fact.content.lower() or any(query_lower in topic.lower() for topic in fact.topics):
                matches.append(fact)
        matches.sort(key=lambda fact: fact.confidence, reverse=True)
        return matches[:top_k]

    async def list(self, user_id: str, topic: str | None = None) -> list[MemoryFact]:
        await self._ensure_schema()
        rows = await self._backend.fetch_all(
            f"SELECT * FROM {self._table_name()} WHERE user_id = ? ORDER BY last_seen_at DESC",
            (user_id,),
        )
        facts = [_row_to_fact(row) for row in rows]
        if topic is not None:
            topic_lower = topic.lower()
            facts = [fact for fact in facts if any(topic_lower in item.lower() for item in fact.topics)]
        return facts

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
    ) -> None:
        await self._ensure_schema()
        row = await self._backend.fetch_one(
            f"SELECT * FROM {self._table_name()} WHERE memory_id = ?",
            (memory_id,),
        )
        if row is None:
            return
        fact = _row_to_fact(row)
        if content is not None:
            fact.content = content
        if confidence is not None:
            fact.confidence = confidence
        await self.add(fact)

    async def delete(self, memory_id: str) -> None:
        await self._ensure_schema()
        await self._backend.execute(
            f"DELETE FROM {self._table_name()} WHERE memory_id = ?",
            (memory_id,),
        )

    async def clear(self, user_id: str) -> None:
        await self._ensure_schema()
        await self._backend.execute(
            f"DELETE FROM {self._table_name()} WHERE user_id = ?",
            (user_id,),
        )

    async def close(self) -> None:
        if self._owns_backend:
            await self._backend.close()

    async def _ensure_schema(self) -> None:
        await self._backend.ensure_schema(
            schema_id=f"{self._backend.config.table_prefix}:user-memory-store",
            statements=(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name()} (
                    memory_id              TEXT PRIMARY KEY,
                    user_id                TEXT NOT NULL,
                    content                TEXT NOT NULL,
                    topics_json            TEXT NOT NULL,
                    confidence             REAL NOT NULL,
                    created_at             TEXT NOT NULL,
                    last_seen_at           TEXT NOT NULL,
                    source_conversation_id TEXT
                )
                """,
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name()}_user_id_idx
                ON {self._table_name()} (user_id)
                """,
            ),
        )

    def _table_name(self) -> str:
        return f"{self._backend.config.table_prefix}user_memory_facts"


class SQLiteVectorStore:
    """Durable document store backed by SQLite with TF-IDF search compatibility."""

    def __init__(
        self,
        database_path: str = ":memory:",
        *,
        config: SQLiteStoreConfig | None = None,
        backend: SQLiteStoreBackend | None = None,
    ) -> None:
        self._backend = backend or SQLiteStoreBackend(config or SQLiteStoreConfig(database_path=database_path))
        self._owns_backend = backend is None
        self._index_lock = asyncio.Lock()
        self._search_index: InMemoryVectorStore | None = None
        self._index_dirty = True

    @classmethod
    async def connect(cls, config: SQLiteStoreConfig | str) -> SQLiteVectorStore:
        backend = await SQLiteStoreBackend.connect(config)
        store = cls(backend=backend)
        await store._ensure_schema()
        return store

    @classmethod
    def from_backend(cls, backend: SQLiteStoreBackend) -> SQLiteVectorStore:
        return cls(backend=backend)

    async def upsert(
        self,
        content: str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        await self._ensure_schema()
        document_id = id or str(uuid.uuid4())
        try:
            metadata_json = json.dumps(metadata or {})
            embedding_json = json.dumps(embedding) if embedding is not None else None
        except (TypeError, ValueError) as exc:
            raise StorageError(
                "Failed to serialize the memory document for SQLite persistence.",
                backend="sqlite",
                location=self._backend.config.database_path,
                cause=exc,
            ) from exc
        await self._backend.execute(
            f"""
            INSERT INTO {self._table_name()} (
                document_id,
                content,
                metadata_json,
                embedding_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                content = excluded.content,
                metadata_json = excluded.metadata_json,
                embedding_json = excluded.embedding_json,
                created_at = excluded.created_at
            """,
            (document_id, content, metadata_json, embedding_json, _utc_now_iso()),
        )
        self._index_dirty = True
        return document_id

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        await self._ensure_schema()
        index = await self._get_search_index()
        return await index.search(query, k=k, filter=filter)

    async def get(self, id: str) -> MemoryResult | None:
        await self._ensure_schema()
        row = await self._backend.fetch_one(
            f"SELECT content, metadata_json FROM {self._table_name()} WHERE document_id = ?",
            (id,),
        )
        if row is None:
            return None
        try:
            metadata = cast(dict[str, Any], json.loads(str(row["metadata_json"])))
        except (TypeError, ValueError) as exc:
            raise StorageError(
                "Failed to deserialize the stored memory document metadata.",
                backend="sqlite",
                location=self._backend.config.database_path,
                cause=exc,
            ) from exc
        return MemoryResult(id=id, content=str(row["content"]), score=1.0, metadata=metadata)

    async def delete(self, ids: list[str]) -> None:
        await self._ensure_schema()
        for document_id in ids:
            await self._backend.execute(
                f"DELETE FROM {self._table_name()} WHERE document_id = ?",
                (document_id,),
            )
        self._index_dirty = True

    async def clear(self) -> None:
        await self._ensure_schema()
        await self._backend.execute(f"DELETE FROM {self._table_name()}")
        self._index_dirty = True

    async def close(self) -> None:
        if self._owns_backend:
            await self._backend.close()

    async def _get_search_index(self) -> InMemoryVectorStore:
        async with self._index_lock:
            if self._search_index is not None and not self._index_dirty:
                return self._search_index
            rows = await self._backend.fetch_all(
                f"SELECT document_id, content, metadata_json FROM {self._table_name()}"
            )
            index = InMemoryVectorStore()
            for row in rows:
                metadata = cast(dict[str, Any], json.loads(str(row["metadata_json"])))
                await index.upsert(
                    str(row["content"]),
                    id=str(row["document_id"]),
                    metadata=metadata,
                )
            self._search_index = index
            self._index_dirty = False
            return index

    async def _ensure_schema(self) -> None:
        await self._backend.ensure_schema(
            schema_id=f"{self._backend.config.table_prefix}:vector-store",
            statements=(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name()} (
                    document_id    TEXT PRIMARY KEY,
                    content        TEXT NOT NULL,
                    metadata_json  TEXT NOT NULL,
                    embedding_json TEXT,
                    created_at     TEXT NOT NULL
                )
                """,
            ),
        )

    def _table_name(self) -> str:
        return f"{self._backend.config.table_prefix}memory_documents"

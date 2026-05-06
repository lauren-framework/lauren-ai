from __future__ import annotations

"""User-level persistent memory that spans conversations."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@dataclass
class MemoryFact:
    """A single persisted fact about a user.

    Facts are stored with a confidence score [0.0–1.0] that
    can decay over time if not reinforced.
    """

    memory_id: str
    user_id: str
    content: str
    topics: list[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_conversation_id: str | None = None

    def reinforce(self) -> None:
        """Update last_seen_at and boost confidence towards 1.0."""
        self.last_seen_at = datetime.now(UTC)
        self.confidence = min(1.0, self.confidence + 0.1)

    def decay(self, factor: float = 0.8) -> None:
        """Reduce confidence by factor."""
        self.confidence *= factor


@runtime_checkable
class UserMemoryStore(Protocol):
    """Protocol for user-level persistent memory stores."""

    async def add(self, fact: MemoryFact) -> None: ...
    async def get(self, user_id: str, memory_id: str) -> MemoryFact | None: ...
    async def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryFact]: ...
    async def list(self, user_id: str, topic: str | None = None) -> list[MemoryFact]: ...
    async def update(
        self, memory_id: str, *, content: str | None = None, confidence: float | None = None
    ) -> None: ...
    async def delete(self, memory_id: str) -> None: ...
    async def clear(self, user_id: str) -> None: ...

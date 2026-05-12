"""In-memory implementation of UserMemoryStore for tests."""

from __future__ import annotations

from lauren_ai._memory._user import MemoryFact


class InMemoryUserMemoryStore:
    """In-process UserMemoryStore for testing and development.

    Uses simple substring matching for search (no vector similarity).
    """

    def __init__(self) -> None:
        self._facts: dict[str, MemoryFact] = {}  # memory_id -> fact

    async def add(self, fact: MemoryFact) -> None:
        self._facts[fact.memory_id] = fact

    async def get(self, user_id: str, memory_id: str) -> MemoryFact | None:
        fact = self._facts.get(memory_id)
        if fact and fact.user_id == user_id:
            return fact
        return None

    async def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryFact]:
        query_lower = query.lower()
        results = [
            f
            for f in self._facts.values()
            if f.user_id == user_id
            and (
                query_lower in f.content.lower() or any(query_lower in t.lower() for t in f.topics)
            )
        ]
        # Sort by confidence descending
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results[:top_k]

    async def list(self, user_id: str, topic: str | None = None) -> list[MemoryFact]:
        results = [f for f in self._facts.values() if f.user_id == user_id]
        if topic:
            results = [f for f in results if topic.lower() in [t.lower() for t in f.topics]]
        results.sort(key=lambda f: f.last_seen_at, reverse=True)
        return results

    async def update(
        self, memory_id: str, *, content: str | None = None, confidence: float | None = None
    ) -> None:
        fact = self._facts.get(memory_id)
        if fact is None:
            return
        if content is not None:
            fact.content = content
        if confidence is not None:
            fact.confidence = confidence

    async def delete(self, memory_id: str) -> None:
        self._facts.pop(memory_id, None)

    async def clear(self, user_id: str) -> None:
        to_delete = [mid for mid, f in self._facts.items() if f.user_id == user_id]
        for mid in to_delete:
            del self._facts[mid]

    def __len__(self) -> int:
        return len(self._facts)

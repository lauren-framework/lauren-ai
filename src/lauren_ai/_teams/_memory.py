from __future__ import annotations

"""In-call shared memory for team workers."""

from typing import Any


class TeamMemory:
    """Shared context store for a single TeamRunner.run() call.

    Workers can read prior worker outputs via ``get_all()`` and write
    new findings via ``set(key, value)``.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    async def set(self, key: str, value: Any) -> None:
        """Store a value under *key*.

        :param key: The storage key.
        :param value: The value to store.
        """
        self._store[key] = value

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by *key*, returning *default* when absent.

        :param key: The storage key.
        :param default: Fallback value when the key is not present.
        :return: The stored value or *default*.
        """
        return self._store.get(key, default)

    async def get_all(self) -> dict[str, Any]:
        """Return a snapshot of all stored key-value pairs.

        :return: A shallow copy of the internal store.
        """
        return dict(self._store)

    async def clear(self) -> None:
        """Remove all stored entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

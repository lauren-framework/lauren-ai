"""In-memory conversation history store.

``InMemoryConversationStore`` implements the ``ConversationStore`` protocol
using a plain Python dict.  Suitable for development, testing, and
single-process deployments.  Conversation histories are lost on process
restart.
"""

from __future__ import annotations

__all__ = [
    "InMemoryConversationStore",
]

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryConversationStore:
    """In-memory store for full conversation histories.

    Implements the ``ConversationStore`` protocol.  Each conversation is keyed
    by an arbitrary string identifier (typically a user ID or session UUID).
    Deep copies are used on both ``load`` and ``save`` so that the caller
    cannot inadvertently mutate stored data.

    :Example:

    .. code-block:: python

        store = InMemoryConversationStore()
        await store.save("session-abc", messages)
        loaded = await store.load("session-abc")
    """

    def __init__(self) -> None:
        # conversation_id → list[Message]
        self._histories: dict[str, list[Any]] = {}

    # ------------------------------------------------------------------
    # ConversationStore protocol
    # ------------------------------------------------------------------

    async def load(self, conversation_id: str) -> Any:
        """Load the conversation snapshot for *conversation_id*.

        Returns an empty list when the conversation does not exist (backward
        compat — callers that check ``if prior:`` still work on empty lists).

        :param conversation_id: Unique conversation identifier.
        :type conversation_id: str
        :return: A deep copy of the stored snapshot.  When the snapshot was
            created by ``ShortTermMemory.snapshot()`` this is a
            ``{"messages": [...], "summary": ...}`` dict; for legacy plain
            lists the raw list is returned.
        :rtype: dict[str, Any] | list[Any]
        """
        history = self._histories.get(conversation_id)
        if history is None:
            return []
        return copy.deepcopy(history)

    async def save(self, conversation_id: str, snapshot: Any) -> None:
        """Persist the conversation snapshot for *conversation_id*.

        Overwrites any existing entry for that identifier.  A deep copy is
        stored to prevent the caller from mutating the stored data.

        Accepts both the new dict snapshot format
        (``{"messages": [...], "summary": ...}``) produced by
        ``ShortTermMemory.snapshot()`` and the legacy plain ``list[Message]``
        format so that code written against the old API continues to work.
        Plain lists are automatically normalised to the dict format so that
        ``load()`` always returns a consistent shape.

        :param conversation_id: Unique conversation identifier.
        :type conversation_id: str
        :param snapshot: Snapshot dict or message list to persist.
        :type snapshot: dict[str, Any] | list[Any]
        """
        if not conversation_id:
            logger.warning(
                "lauren_ai.InMemoryConversationStore: save called with empty conversation_id — storing under empty key"
            )
        # Normalise legacy plain-list format to dict so load() is consistent.
        if isinstance(snapshot, list):
            snapshot = {"messages": snapshot, "summary": None}
        self._histories[conversation_id] = copy.deepcopy(snapshot)

    async def delete(self, conversation_id: str) -> None:
        """Delete the history for *conversation_id*.

        Silently does nothing when the conversation does not exist.

        :param conversation_id: Unique conversation identifier.
        :type conversation_id: str
        """
        self._histories.pop(conversation_id, None)

    # ------------------------------------------------------------------
    # Bonus methods
    # ------------------------------------------------------------------

    async def list_conversations(self) -> list[str]:
        """Return a sorted list of all stored conversation identifiers.

        :return: Sorted list of conversation IDs.
        :rtype: list[str]
        """
        return sorted(self._histories.keys())

    async def clear(self) -> None:
        """Remove all stored conversation histories.

        :return: None
        :rtype: None
        """
        self._histories.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of stored conversations.

        :return: Conversation count.
        :rtype: int
        """
        return len(self._histories)

    def __contains__(self, conversation_id: str) -> bool:
        """Support ``in`` operator to test whether a conversation exists.

        :param conversation_id: Conversation identifier to check.
        :type conversation_id: str
        :return: ``True`` when the conversation is stored.
        :rtype: bool
        """
        return conversation_id in self._histories

    def __repr__(self) -> str:  # pragma: no cover
        return f"InMemoryConversationStore(conversations={len(self._histories)})"

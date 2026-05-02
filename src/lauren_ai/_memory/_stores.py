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

    async def load(self, conversation_id: str) -> list[Any]:
        """Load the message history for *conversation_id*.

        Returns an empty list when the conversation does not exist.

        :param conversation_id: Unique conversation identifier.
        :type conversation_id: str
        :return: A deep copy of the stored message list (empty list when not
            found).
        :rtype: list[Message]
        """
        history = self._histories.get(conversation_id)
        if history is None:
            return []
        return copy.deepcopy(history)

    async def save(self, conversation_id: str, messages: list[Any]) -> None:
        """Persist the message history for *conversation_id*.

        Overwrites any existing history for that identifier.  A deep copy of
        *messages* is stored to prevent the caller from mutating the stored
        data.

        :param conversation_id: Unique conversation identifier.
        :type conversation_id: str
        :param messages: Ordered list of ``Message`` objects to persist.
        :type messages: list[Message]
        """
        if not conversation_id:
            logger.warning(
                "lauren_ai.InMemoryConversationStore: save called with empty "
                "conversation_id — storing under empty key"
            )
        self._histories[conversation_id] = copy.deepcopy(messages)

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

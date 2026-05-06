"""Memory system for lauren-ai.

Provides:

- ``ShortTermMemory`` — sliding-window conversation buffer with token-budget
  trimming.
- ``MemoryStore`` — protocol for long-term semantic (vector) memory.
- ``ConversationStore`` — protocol for persisting full conversation histories.
- ``MemoryResult`` — result dataclass returned by ``MemoryStore.search()``.
- ``MemoryFact`` — a single persisted fact about a user.
- ``UserMemoryStore`` — protocol for user-level persistent memory stores.
- ``InMemoryUserMemoryStore`` — in-process implementation for testing.
- ``remember`` — decorator for opting an ``@agent()`` class into user memory.
- ``RememberMeta`` — metadata attached by ``@remember()``.
- ``MemoryConfigError`` — raised when memory configuration is invalid.
"""

from __future__ import annotations

__all__ = [
    "ConversationStore",
    "MemoryResult",
    "MemoryStore",
    "ShortTermMemory",
    "MemoryFact",
    "UserMemoryStore",
    "InMemoryUserMemoryStore",
    "remember",
    "RememberMeta",
    "MemoryConfigError",
    "REMEMBER_META",
]

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


@dataclass
class MemoryResult:
    """A single result returned by ``MemoryStore.search()``.

    :param id: The unique identifier of the stored document.
    :type id: str
    :param content: The original text content.
    :type content: str
    :param score: Similarity score in the range ``[0.0, 1.0]``; higher is more
        similar.
    :type score: float
    :param metadata: Arbitrary key/value metadata attached at upsert time.
    :type metadata: dict[str, Any]
    """

    id: str
    content: str
    score: float
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# MemoryStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for long-term semantic memory (vector-backed or key-value).

    All implementations must be safe for concurrent async access.
    """

    async def upsert(
        self,
        content: str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Insert or update a document in the store.

        :param content: The text content to store.
        :type content: str
        :param id: Optional stable identifier.  A UUID is generated when
            ``None``.
        :type id: str | None
        :param metadata: Optional key/value metadata.
        :type metadata: dict[str, Any] | None
        :param embedding: Pre-computed embedding vector.  The store computes
            its own embedding when ``None``.
        :type embedding: list[float] | None
        :return: The document's identifier.
        :rtype: str
        """
        ...

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        """Semantic search over stored documents.

        :param query: Natural-language query string.
        :type query: str
        :param k: Maximum number of results to return.
        :type k: int
        :param filter: Optional metadata filter dict (exact-match semantics).
        :type filter: dict[str, Any] | None
        :return: Up to *k* results ordered by descending similarity score.
        :rtype: list[MemoryResult]
        """
        ...

    async def get(self, id: str) -> MemoryResult | None:
        """Retrieve a single document by its identifier.

        :param id: Document identifier.
        :type id: str
        :return: The document, or ``None`` if not found.
        :rtype: MemoryResult | None
        """
        ...

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by their identifiers.

        :param ids: List of document identifiers to remove.  Non-existent IDs
            are silently ignored.
        :type ids: list[str]
        """
        ...

    async def clear(self) -> None:
        """Remove all documents from the store.

        :return: None
        :rtype: None
        """
        ...


# ---------------------------------------------------------------------------
# ConversationStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConversationStore(Protocol):
    """Protocol for persisting and retrieving full conversation histories.

    Keyed by an arbitrary string ``conversation_id`` (typically a session or
    user identifier).
    """

    async def load(self, conversation_id: str) -> list[Any]:
        """Load the message history for *conversation_id*.

        :param conversation_id: Unique conversation / session identifier.
        :type conversation_id: str
        :return: Ordered list of ``Message`` objects (empty list when not
            found).
        :rtype: list[Message]
        """
        ...

    async def save(self, conversation_id: str, messages: list[Any]) -> None:
        """Persist the message history for *conversation_id*.

        Overwrites any existing history for that ID.

        :param conversation_id: Unique conversation / session identifier.
        :type conversation_id: str
        :param messages: Ordered list of ``Message`` objects to persist.
        :type messages: list[Message]
        """
        ...

    async def delete(self, conversation_id: str) -> None:
        """Delete the history for *conversation_id*.

        :param conversation_id: Unique conversation / session identifier.
        :type conversation_id: str
        """
        ...


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------


def _estimate_content_length(content: Any) -> int:
    """Heuristic character count for a message content value.

    Handles ``str``, ``list[ContentBlock]``, and anything JSON-serialisable.

    :param content: Message content (str, list, or other).
    :return: Approximate character count.
    :rtype: int
    """
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += len(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str):
                    total += len(text)
                else:
                    try:
                        total += len(json.dumps(block, default=str))
                    except (TypeError, ValueError):
                        total += len(str(block))
            else:
                try:
                    total += len(json.dumps(block, default=str))
                except (TypeError, ValueError):
                    total += len(str(block))
        return total
    try:
        return len(json.dumps(content, default=str))
    except (TypeError, ValueError):
        return len(str(content))


def _message_char_length(message: Any) -> int:
    """Return the character length of a ``Message`` object.

    Accepts both dataclass-style objects with a ``.content`` attribute and
    plain dicts with a ``"content"`` key.

    :param message: A message object or dict.
    :return: Character count of its content.
    :rtype: int
    """
    if isinstance(message, dict):
        return _estimate_content_length(message.get("content", ""))
    content = getattr(message, "content", "")
    return _estimate_content_length(content)


def _get_role(message: Any) -> str:
    """Return the role string of a message object.

    :param message: Message object or dict.
    :return: Role string (e.g. ``"user"``, ``"assistant"``).
    :rtype: str
    """
    if isinstance(message, dict):
        return message.get("role", "")
    return getattr(message, "role", "")


class ShortTermMemory:
    """Sliding-window conversation buffer for a single agent run.

    Stores the ordered message history and automatically trims to fit within a
    token budget when requested.  Uses the heuristic ``chars / 4 ≈ tokens``
    when no token-counting transport is available.

    :param max_tokens: Maximum number of tokens to retain in the window.
        Defaults to 40 000.
    :type max_tokens: int

    :Example:

    .. code-block:: python

        memory = ShortTermMemory(max_tokens=8000)
        memory.add_user("Hello, how are you?")
        memory.add_assistant(completion)
        msgs = memory.messages()  # trimmed to budget
    """

    _CHARS_PER_TOKEN: int = 4

    def __init__(self, max_tokens: int = 40_000) -> None:
        self._max_tokens = max_tokens
        self._messages: list[Any] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_user(self, content: str | list[Any]) -> None:
        """Append a user message to the buffer.

        :param content: Plain text string or list of content blocks.
        :type content: str | list[Any]
        """
        # Build a simple dict-based Message compatible with the transport layer
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, completion: Any) -> None:
        """Append an assistant completion to the buffer.

        Accepts a ``Completion`` dataclass (with ``.content`` and
        ``.tool_calls`` attributes) or a plain dict.

        :param completion: A ``Completion`` object or ``{"role": "assistant",
            "content": "..."}`` dict.
        :type completion: Any
        """
        if isinstance(completion, dict):
            self._messages.append(completion)
            return

        # Dataclass / object form
        content = getattr(completion, "content", "")
        tool_calls = getattr(completion, "tool_calls", [])

        if tool_calls:
            # Build a content list with text + tool_use blocks
            blocks: list[Any] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                tc_id = getattr(tc, "tool_use_id", getattr(tc, "id", ""))
                tc_name = getattr(tc, "name", "")
                tc_input = getattr(tc, "input", {})
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc_id,
                        "name": tc_name,
                        "input": tc_input,
                    }
                )
            self._messages.append({"role": "assistant", "content": blocks})
        else:
            self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, result: Any) -> None:
        """Append a tool result message to the buffer.

        Accepts a ``ToolResult`` dataclass or a plain dict.

        :param result: A ``ToolResult`` object or dict.
        :type result: Any
        """
        if isinstance(result, dict):
            self._messages.append(result)
            return

        tool_use_id = getattr(result, "tool_use_id", "")
        content = getattr(result, "content", "")
        is_error = getattr(result, "is_error", False)

        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                        "is_error": is_error,
                    }
                ],
            }
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def messages(self) -> list[Any]:
        """Return the current message list, trimmed to fit the token window.

        The trim is applied in-place on a copy; the internal buffer is NOT
        modified.  Call ``trim_to_fit()`` explicitly to mutate the buffer.

        :return: Ordered list of messages within the token budget.
        :rtype: list[Message]
        """
        # Return a shallow-copy snapshot (does not mutate internal state)
        snapshot = list(self._messages)
        budget_chars = self._max_tokens * self._CHARS_PER_TOKEN
        total_chars = sum(_message_char_length(m) for m in snapshot)

        while snapshot and total_chars > budget_chars:
            # Drop the oldest non-system message
            for idx, msg in enumerate(snapshot):
                if _get_role(msg) != "system":
                    removed = snapshot.pop(idx)
                    total_chars -= _message_char_length(removed)
                    break
            else:
                # All messages are system messages — nothing more to drop
                break

        return snapshot

    def trim_to_fit(self, max_tokens: int) -> None:
        """Drop oldest non-system messages until the token estimate fits.

        Unlike ``messages()`` this *mutates* the internal buffer.

        :param max_tokens: Target token budget.
        :type max_tokens: int
        """
        budget_chars = max_tokens * self._CHARS_PER_TOKEN
        while self._messages and self.token_estimate * self._CHARS_PER_TOKEN > budget_chars:
            for idx, msg in enumerate(self._messages):
                if _get_role(msg) != "system":
                    self._messages.pop(idx)
                    break
            else:
                break

    @property
    def token_estimate(self) -> int:
        """Heuristic token count: total character length of all messages divided by 4.

        :return: Estimated token count.
        :rtype: int
        """
        total_chars = sum(_message_char_length(m) for m in self._messages)
        return max(0, total_chars // self._CHARS_PER_TOKEN)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all messages from the buffer.

        :return: None
        :rtype: None
        """
        self._messages.clear()

    def snapshot(self) -> list[Any]:
        """Return a deep copy of the current message list.

        The returned list is independent of the internal buffer; mutations to
        it do not affect the memory.

        :return: Immutable snapshot of the conversation history.
        :rtype: list[Message]
        """
        return copy.deepcopy(self._messages)

    def restore(self, messages: list[Any]) -> None:
        """Restore the message buffer from a snapshot.

        :param messages: Ordered list of ``Message`` objects (typically
            produced by ``snapshot()``).
        :type messages: list[Message]
        """
        self._messages = list(messages)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of messages currently in the buffer.

        :return: Message count.
        :rtype: int
        """
        return len(self._messages)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ShortTermMemory(messages={len(self._messages)}, "
            f"token_estimate={self.token_estimate}, "
            f"max_tokens={self._max_tokens})"
        )


# ---------------------------------------------------------------------------
# User-level persistent memory (Section 36)
# ---------------------------------------------------------------------------

from lauren_ai._memory._in_memory_user import InMemoryUserMemoryStore  # noqa: E402
from lauren_ai._memory._remember import (  # noqa: E402
    REMEMBER_META,
    MemoryConfigError,
    RememberMeta,
    remember,
)
from lauren_ai._memory._user import MemoryFact, UserMemoryStore  # noqa: E402

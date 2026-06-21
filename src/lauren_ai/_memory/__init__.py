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
    "SQLiteConversationStore",
    "SQLiteStoreBackend",
    "SQLiteStoreConfig",
    "SQLiteUserMemoryStore",
    "SQLiteVectorStore",
    "remember",
    "RememberMeta",
    "MemoryConfigError",
    "REMEMBER_META",
]

import copy
import json
import warnings
from dataclasses import dataclass
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


def _has_tool_calls(message: Any) -> bool:
    """Return True if *message* is an assistant turn that contains tool calls.

    Handles both OpenAI format (``{"tool_calls": [...]}`` key) and Anthropic
    format (``content`` list containing ``{"type": "tool_use"}`` blocks).
    """
    if isinstance(message, dict):
        if message.get("role") != "assistant":
            return False
        if message.get("tool_calls"):
            return True
        content = message.get("content", "")
        if isinstance(content, list):
            return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
        return False
    if getattr(message, "role", "") != "assistant":
        return False
    if getattr(message, "tool_calls", None):
        return True
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return False


def _is_tool_result(message: Any) -> bool:
    """Return True if *message* is a tool result.

    Handles both OpenAI format (``role="tool"``) and Anthropic format
    (``role="user"`` with ``content`` containing ``{"type": "tool_result"}``
    blocks).
    """
    if isinstance(message, dict):
        if message.get("role") == "tool":
            return True
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, list):
                return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        return False
    if getattr(message, "role", "") == "tool":
        return True
    if getattr(message, "role", "") == "user":
        content = getattr(message, "content", "")
        if isinstance(content, list):
            return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def _is_conversational_user(message: Any) -> bool:
    """Return True when *message* is a user turn that can open a conversation.

    A ``role:"user"`` message whose content consists **entirely** of
    ``tool_result`` blocks cannot be the first message in a conversation —
    it is a tool-result response to an assistant turn, not a standalone user
    request.  All other user messages (string content, mixed content, or
    content-free) qualify as conversational anchors.

    :param message: Message dict or dataclass.
    :return: ``True`` when the message is a conversational user turn.
    :rtype: bool
    """
    if _get_role(message) != "user":
        return False
    content: Any = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    # String content (or no content) — regular user message.
    if not isinstance(content, list):
        return True
    # Mixed content list — check whether any block is NOT a tool_result.
    return any(not (isinstance(b, dict) and b.get("type") == "tool_result") for b in content)


def _drop_oldest_turn(msgs: list) -> tuple[list, int]:
    """Drop the oldest complete turn from *msgs*, preserving system messages.

    A "complete turn" is an atomic unit that must be removed together:
    - A plain user or assistant message (no tool calls) → remove just that one
    - An assistant message with tool calls **plus** all immediately following
      tool-result messages → remove the whole block

    Dropping a tool-call message without its results (or vice-versa) produces
    an orphaned ``tool`` message that causes a 400 from the API.

    :param msgs: Current message list (not mutated).
    :return: ``(new_list, chars_removed)`` tuple.
    """
    for start, msg in enumerate(msgs):
        if _get_role(msg) == "system":
            continue
        # Found the oldest non-system message.  Extend the slice to include all
        # immediately following tool results so the whole transaction is atomic.
        end = start + 1
        if _has_tool_calls(msg):
            while end < len(msgs) and _is_tool_result(msgs[end]):
                end += 1
        removed = msgs[start:end]
        chars_removed = sum(_message_char_length(m) for m in removed)
        return msgs[:start] + msgs[end:], chars_removed
    return msgs, 0


def _get_tool_call_ids(message: Any) -> set[str]:
    """Return the set of tool_call_ids that *message* is requesting results for.

    Returns an empty set for messages that are not assistant turns with tool
    calls.  Handles both OpenAI format (``tool_calls`` list) and Anthropic
    format (``content`` list of ``tool_use`` blocks).
    """
    ids: set[str] = set()
    if isinstance(message, dict):
        for tc in message.get("tool_calls") or []:
            if isinstance(tc, dict):
                tc_id = tc.get("id") or tc.get("tool_use_id", "")
                if tc_id:
                    ids.add(tc_id)
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tc_id = block.get("id", "")
                    if tc_id:
                        ids.add(tc_id)
    else:
        for tc in getattr(message, "tool_calls", None) or []:
            tc_id = getattr(tc, "id", None) or getattr(tc, "tool_use_id", "")
            if tc_id:
                ids.add(tc_id)
    return ids


def _get_tool_result_ids(message: Any) -> set[str]:
    """Return the set of tool_call_ids answered by *message*.

    Returns an empty set for non-tool-result messages.  Handles both OpenAI
    format (``role="tool"`` with ``tool_call_id``) and Anthropic format
    (``role="user"`` with ``tool_result`` content blocks).
    """
    ids: set[str] = set()
    if isinstance(message, dict):
        if message.get("role") == "tool":
            tc_id = message.get("tool_call_id", "")
            if tc_id:
                ids.add(tc_id)
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tc_id = block.get("tool_use_id", "")
                    if tc_id:
                        ids.add(tc_id)
    else:
        tc_id = getattr(message, "tool_call_id", "") or getattr(message, "tool_use_id", "")
        if tc_id:
            ids.add(tc_id)
    return ids


_INTERRUPTED_CONTENT = (
    "[Tool execution was interrupted before a result was received. "
    "The tool may or may not have completed. "
    "Decide whether to retry or proceed without this result.]"
)


def _heal_dangling_tail(snapshot: list) -> list:
    """Inject synthetic error results for any unanswered tool_calls.

    Scans forward to find the LAST assistant message that contains tool calls.
    If any of its ``tool_call_id``\\s lack a corresponding tool-result message
    anywhere later in *snapshot*, synthetic ``role="tool"`` messages are
    inserted immediately after the last existing tool result for that turn.

    This heals the "insufficient tool messages following tool_calls message"
    400 error that OpenAI returns when an agent turn is interrupted (by
    CancelledError, max_turns, or a network failure) after the assistant
    message is stored but before all tool results arrive.

    :param snapshot: Message list (not mutated — a new list is returned).
    :return: Healed message list.
    """
    # Find the index of the last assistant message that has tool calls.
    last_assistant_idx = -1
    for i, msg in enumerate(snapshot):
        if _has_tool_calls(msg):
            last_assistant_idx = i

    if last_assistant_idx == -1:
        return snapshot

    expected = _get_tool_call_ids(snapshot[last_assistant_idx])
    if not expected:
        return snapshot

    # Collect which IDs are answered anywhere after the assistant message.
    answered: set[str] = set()
    last_result_idx = last_assistant_idx
    for j in range(last_assistant_idx + 1, len(snapshot)):
        result_ids = _get_tool_result_ids(snapshot[j])
        if result_ids:
            answered.update(result_ids)
            last_result_idx = j

    missing = expected - answered
    if not missing:
        return snapshot

    # Only inject synthetic results if the conversation has moved past this
    # tool-calling exchange — i.e. there is at least one non-tool-result
    # message after the last tool result we found.  Without this guard,
    # messages() would mutate a freshly-added assistant turn that is still
    # mid-flight waiting for its tool results.
    has_moved_on = any(not _get_tool_result_ids(snapshot[j]) for j in range(last_result_idx + 1, len(snapshot)))
    if not has_moved_on:
        return snapshot

    # Insert synthetic error results immediately after the last existing tool
    # result for this turn (or directly after the assistant message if none).
    # Produce ONE consolidated canonical synthetic result message containing
    # ALL missing tool_result blocks.  Anthropic requires every tool_use in an
    # assistant turn to have a corresponding tool_result in the *immediately
    # following* user message — it does NOT accept separate per-tool messages.
    # Using a single user message with multiple content blocks satisfies both
    # Anthropic (native) and OpenAI (the transport converts each block to a
    # separate role:"tool" message, which OpenAI accepts).
    synthetic = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "content": _INTERRUPTED_CONTENT,
                }
                for tc_id in sorted(missing)  # sorted for determinism
            ],
        }
    ]
    insert_at = last_result_idx + 1
    return snapshot[:insert_at] + synthetic + snapshot[insert_at:]


def _heal_dangling_tail_unconditional(snapshot: list) -> list:
    """Like ``_heal_dangling_tail`` but without the ``has_moved_on`` guard.

    Injects synthetic error results for *any* unanswered tool_calls in the
    last assistant-with-tool-calls message, even if the conversation has not
    yet progressed past that turn.  Use this for pre-request healing only —
    it would incorrectly mutate mid-flight turns if called during a stream.
    """
    last_assistant_idx = -1
    for i, msg in enumerate(snapshot):
        if _has_tool_calls(msg):
            last_assistant_idx = i

    if last_assistant_idx == -1:
        return snapshot

    expected = _get_tool_call_ids(snapshot[last_assistant_idx])
    if not expected:
        return snapshot

    answered: set[str] = set()
    last_result_idx = last_assistant_idx
    for j in range(last_assistant_idx + 1, len(snapshot)):
        result_ids = _get_tool_result_ids(snapshot[j])
        if result_ids:
            answered.update(result_ids)
            last_result_idx = j

    missing = expected - answered
    if not missing:
        return snapshot

    synthetic = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "content": _INTERRUPTED_CONTENT,
                }
                for tc_id in sorted(missing)  # all missing IDs in ONE message
            ],
        }
    ]
    insert_at = last_result_idx + 1
    return snapshot[:insert_at] + synthetic + snapshot[insert_at:]


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
        self._summary: str | None = None

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

    def add_tool_results(self, results: list[Any]) -> None:
        """Append multiple tool results as a **single** consolidated message.

        Anthropic requires all ``tool_result`` blocks for a given assistant
        turn to appear in the *same* immediately-following user message.
        Calling ``add_tool_result`` in a loop creates N separate messages,
        which causes Anthropic to report that only the first ID is answered
        and the rest are missing (400 error).

        This method consolidates all results into one ``role:"user"`` message
        with multiple content blocks, satisfying Anthropic's constraint while
        remaining compatible with OpenAI (the transport converts each block to
        a separate ``role:"tool"`` message as needed).

        For a single result, delegates to ``add_tool_result`` unchanged.

        :param results: List of ``ToolResult`` objects or dicts to add.
        :type results: list
        """
        if not results:
            return
        if len(results) == 1:
            self.add_tool_result(results[0])
            return
        # Build consolidated content block list
        blocks: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, dict):
                blocks.append(result)
                continue
            tool_use_id = getattr(result, "tool_use_id", "")
            content = getattr(result, "content", "")
            is_error = getattr(result, "is_error", False)
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
            }
            blocks.append(block)
        self._messages.append({"role": "user", "content": blocks})

    # ------------------------------------------------------------------
    # Summary / compression helpers
    # ------------------------------------------------------------------

    @property
    def summary(self) -> str | None:
        """The compressed summary of older conversation turns, or ``None``
        when no summarisation has been performed yet.

        :return: Summary text, or ``None``.
        :rtype: str | None
        """
        return self._summary

    def set_summary(self, text: str) -> None:
        """Store *text* as the conversation summary.

        Called by the runner after a summarisation LLM call completes.
        The summary is persisted via ``snapshot()`` / ``restore()`` so
        resumed sessions carry it forward.

        :param text: Compressed summary of older conversation turns.
        :type text: str
        """
        self._summary = text

    def messages_to_summarize(self, keep_recent: int = 6) -> list[Any]:
        """Return the slice of messages that should be compressed.

        Returns the oldest ``(total - keep_recent)`` non-system messages.
        System messages are excluded because they are already managed
        separately (they are never dropped by ``messages()`` either).

        :param keep_recent: Number of most-recent non-system messages to
            preserve verbatim.  Defaults to 6 (≈ 3 user/assistant pairs).
        :type keep_recent: int
        :return: List of messages to feed to the summarisation LLM call.
        :rtype: list[Any]
        """
        non_system = [m for m in self._messages if _get_role(m) != "system"]
        to_compress = non_system[:-keep_recent] if len(non_system) > keep_recent else []
        return list(to_compress)

    def trim_to_recent(self, keep_recent: int = 6) -> None:
        """Drop all but the most-recent *keep_recent* non-system messages.

        Called by the runner after the summarisation call so the buffer
        only holds recent turns while the older context lives in
        ``self._summary``.

        :param keep_recent: Number of most-recent non-system messages to
            keep.  Defaults to 6.
        :type keep_recent: int
        """
        system_msgs = [m for m in self._messages if _get_role(m) == "system"]
        non_system = [m for m in self._messages if _get_role(m) != "system"]
        kept = non_system[-keep_recent:] if non_system else []
        self._messages = system_msgs + kept

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
            # Simulate dropping the oldest turn to see if anything non-system
            # would remain.  Two invariants are enforced:
            #
            # 1. Never produce an empty non-system message list — all providers
            #    return 400 "at least one message is required".
            # 2. Never produce a list whose first non-system message is not
            #    role:"user" — all providers require user-first conversations.
            #    This occurs when large tool results force the original user
            #    intent to be trimmed away, leaving an assistant turn first.
            candidate, removed_chars = _drop_oldest_turn(snapshot)
            if not removed_chars:
                break  # only system messages remain — nothing more to drop
            # Guard: never trim past the last *conversational* user message.
            # A tool_result user message cannot open a conversation — all
            # providers require the first non-system message to be a real user
            # turn, not a tool response.  Without this guard, large tool results
            # (e.g. recursive directory listings) can force trimming to remove
            # the original user intent, leaving only assistant+tool_result
            # which providers reject ("messages: at least one message required").
            if not any(_is_conversational_user(m) for m in candidate):
                warnings.warn(
                    f"ShortTermMemory: cannot trim history to fit the token "
                    f"budget ({budget_chars:,} chars equivalent) without "
                    f"removing the last conversational user message (current "
                    f"size: {total_chars:,} chars).  Sending as-is; consider "
                    "truncating large tool results upstream.",
                    UserWarning,
                    stacklevel=3,
                )
                break
            snapshot = candidate
            total_chars -= removed_chars

        # Heal any incomplete tool-call sequence at the tail (e.g. an agent
        # turn interrupted before all parallel tool results arrived).
        return _heal_dangling_tail(snapshot)

    def trim_to_fit(self, max_tokens: int) -> None:
        """Drop oldest non-system messages until the token estimate fits.

        Unlike ``messages()`` this *mutates* the internal buffer.

        :param max_tokens: Target token budget.
        :type max_tokens: int
        """
        budget_chars = max_tokens * self._CHARS_PER_TOKEN
        while self._messages:
            total_chars = sum(_message_char_length(m) for m in self._messages)
            if total_chars <= budget_chars:
                break
            candidate, removed_chars = _drop_oldest_turn(self._messages)
            if not removed_chars:
                break  # only system messages remain
            if not any(_is_conversational_user(m) for m in candidate):
                warnings.warn(
                    f"ShortTermMemory.trim_to_fit: cannot trim to "
                    f"{budget_chars:,} chars without removing the last "
                    f"conversational user message (current size: "
                    f"{total_chars:,} chars).  Keeping as-is.",
                    UserWarning,
                    stacklevel=2,
                )
                break
            self._messages = candidate
        # Heal dangling tail after mutating the buffer.
        self._messages = _heal_dangling_tail(self._messages)

    def ensure_valid(self) -> None:
        """Heal dangling tool_calls in-place before making an API request.

        Unlike ``messages()`` (which has a ``has_moved_on`` guard to avoid
        healing mid-flight tool calls) this method heals *unconditionally*.
        It should be called once immediately before each ``run_stream()`` /
        ``run()`` invocation to handle cases where a previous agent turn was
        interrupted while a tool was suspended — e.g. when the user cancels
        a plan-approval overlay after the LLM has already called the approval
        tool for a second time.

        The method is idempotent — calling it multiple times is safe.
        """
        self._messages = _heal_dangling_tail_unconditional(self._messages)

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

    def snapshot(self) -> Any:
        """Return a deep copy of the current memory state.

        The returned object includes both the message list and the
        conversation summary (if any).  It is independent of the internal
        buffer — mutations do not affect the memory.

        The format is a ``dict`` with ``"messages"`` and ``"summary"`` keys
        so that resumed sessions carry the summary forward.  Old snapshots
        that are plain ``list`` objects are still accepted by ``restore()``
        for backward compatibility.

        :return: Snapshot dict ``{"messages": [...], "summary": str | None}``.
        :rtype: dict[str, Any]
        """
        return {
            "messages": copy.deepcopy(self._messages),
            "summary": self._summary,
        }

    def restore(self, data: Any) -> None:
        """Restore the memory buffer from a snapshot.

        Accepts both the new ``dict`` snapshot format (``{"messages": [...],
        "summary": ...}``) and the legacy plain ``list`` format produced by
        older versions of ``snapshot()``.

        :param data: Snapshot produced by ``snapshot()``, or a plain list of
            message objects for backward compatibility.
        :type data: dict[str, Any] | list[Any]
        """
        if isinstance(data, list):
            # Legacy format — plain list with no summary
            self._messages = list(data)
            self._summary = None
        else:
            self._messages = list(data.get("messages", []))
            self._summary = data.get("summary")

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
        summary_tag = f", summary={len(self._summary)}ch" if self._summary else ""
        return (
            f"ShortTermMemory(messages={len(self._messages)}, "
            f"token_estimate={self.token_estimate}, "
            f"max_tokens={self._max_tokens}{summary_tag})"
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
from lauren_ai._memory._sqlite import (  # noqa: E402
    SQLiteConversationStore,
    SQLiteStoreBackend,
    SQLiteStoreConfig,
    SQLiteUserMemoryStore,
    SQLiteVectorStore,
)
from lauren_ai._memory._user import MemoryFact, UserMemoryStore  # noqa: E402

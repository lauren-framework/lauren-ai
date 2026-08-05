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
    "ToolCallRecord",
    "ToolExchange",
    "ToolHistoryIssue",
    "ToolHistoryReport",
    "ToolResultRecord",
    "extract_tool_call_ids",
    "extract_tool_result_ids",
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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from lauren_ai._exceptions import ToolConversationIntegrityError
from lauren_ai._memory._tool_integrity import (
    ToolCallRecord,
    ToolExchange,
    ToolHistoryIssue,
    ToolHistoryReport,
    ToolResultRecord,
    extract_tool_call_ids,
    extract_tool_result_ids,
    is_tool_call_message,
    is_tool_result_message,
)

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


def _thinking_block_to_dict(block: Any) -> dict[str, Any]:
    """Serialise a thinking / redacted-thinking block to its dict form (PRD-137).

    Accepts the ``ThinkingBlock`` / ``RedactedThinkingBlock`` dataclasses or an
    already-dict block (idempotent — so re-stored assistant turns are stable).
    The dict round-trips through snapshot/restore + the request serializer with
    the signature intact, which Anthropic requires when passing thinking back.
    """
    if isinstance(block, dict):
        return block
    if getattr(block, "type", "") == "redacted_thinking":
        return {"type": "redacted_thinking", "data": getattr(block, "data", "")}
    return {
        "type": "thinking",
        "thinking": getattr(block, "thinking", ""),
        "signature": getattr(block, "signature", ""),
    }


# ---------------------------------------------------------------------------
# Hard context-budget enforcement (PRD-133 Layer C)
# ---------------------------------------------------------------------------
#: Minimum characters kept when a single block is truncated to fit budget.
_TRUNCATION_MIN_KEEP: int = 400


def _truncate_str(text: str, target: int) -> str:
    """Shorten *text* to ~*target* chars, keeping a head + tail with a marker."""
    if target <= 0:
        return ""
    if len(text) <= target:
        return text
    marker = f"\n…[truncated {len(text) - target} chars to fit context budget]…\n"
    keep = max(_TRUNCATION_MIN_KEEP, target - len(marker))
    head = (keep * 2) // 3
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _block_text_field(block: Any) -> tuple[str | None, int]:
    """Return ``(field_name, length)`` of a block's truncatable text, or (None, 0).

    Only the ``text`` / ``content`` *string* fields are truncatable; ``tool_use``
    inputs and ids are left intact so the call stays valid.  ``thinking`` /
    ``redacted_thinking`` blocks are **never** truncatable — their signature is
    verified against the exact thinking text, so any edit invalidates it and the
    provider rejects the request (PRD-137 E).
    """
    if isinstance(block, dict):
        if block.get("type") in ("thinking", "redacted_thinking"):
            return None, 0
        if isinstance(block.get("text"), str):
            return "text", len(block["text"])
        if isinstance(block.get("content"), str):
            return "content", len(block["content"])
    return None, 0


def _shrink_message(message: Any, target_chars: int) -> Any:
    """Return a copy of *message* whose content is truncated to ~*target_chars*.

    Block structure is preserved (no block is removed) so ``tool_use`` /
    ``tool_result`` pairing stays valid — only text/content strings shrink.
    Only ``dict`` messages (the runtime ShortTermMemory format) are shrunk.
    """
    if not isinstance(message, dict):
        return message
    content = message.get("content", "")
    if isinstance(content, str):
        if len(content) <= target_chars:
            return message
        return {**message, "content": _truncate_str(content, target_chars)}
    if isinstance(content, list):
        cur = _estimate_content_length(content)
        if cur <= target_chars:
            return message
        excess = cur - target_chars
        new_blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        order = sorted(
            range(len(new_blocks)),
            key=lambda i: _block_text_field(new_blocks[i])[1],
            reverse=True,
        )
        for i in order:
            if excess <= 0:
                break
            field, blen = _block_text_field(new_blocks[i])
            if field is None or blen <= _TRUNCATION_MIN_KEEP:
                continue
            cut = min(blen - _TRUNCATION_MIN_KEEP, excess)
            new_blocks[i][field] = _truncate_str(new_blocks[i][field], blen - cut)
            excess -= cut
        return {**message, "content": new_blocks}
    return message


def _enforce_char_budget(messages: list[Any], budget_chars: int) -> list[Any]:
    """Hard-cap *messages* to *budget_chars* by truncating block contents.

    The sliding-window trim in :meth:`ShortTermMemory.messages` can leave the
    list over budget when a single un-droppable turn is itself too large (e.g. a
    recursive directory listing or a huge file read).  This shrinks block
    contents — never removing blocks — until the total fits, so the request is
    guaranteed to stay within budget instead of overflowing the model context
    window.
    """
    if budget_chars <= 0:
        return messages
    total = sum(_message_char_length(m) for m in messages)
    if total <= budget_chars:
        return messages
    result = list(messages)
    excess = total - budget_chars
    order = sorted(
        range(len(result)),
        key=lambda i: _message_char_length(result[i]),
        reverse=True,
    )
    for i in order:
        if excess <= 0:
            break
        cur = _message_char_length(result[i])
        if cur <= _TRUNCATION_MIN_KEEP:
            continue
        result[i] = _shrink_message(result[i], max(_TRUNCATION_MIN_KEEP, cur - excess))
        excess -= cur - _message_char_length(result[i])
    return result


def _tool_history_error(report: ToolHistoryReport) -> ToolConversationIntegrityError:
    """Convert a validation report into a safe public exception."""
    issue = report.first_issue
    if issue is None:
        raise ValueError("Cannot construct a tool-history error from a valid report")
    return ToolConversationIntegrityError(
        (
            "Tool conversation invariant violated: "
            f"code={issue.code} expected={issue.expected_count} observed={issue.observed_count}"
        ),
        code=issue.code,
        expected_count=issue.expected_count,
        observed_count=issue.observed_count,
        assistant_index=issue.assistant_index,
        repairable=issue.repairable,
    )


def _validate_tool_history(messages: list[Any]) -> ToolHistoryReport:
    """Validate assistant tool batches and their immediately following results."""
    issues: list[ToolHistoryIssue] = []
    consumed_result_indexes: set[int] = set()
    for index, message in enumerate(messages):
        expected = extract_tool_call_ids(message)
        if not is_tool_call_message(message):
            continue

        if any(not value for value in expected):
            issues.append(
                ToolHistoryIssue(
                    "empty_call_id",
                    "assistant tool-call batch contains an empty call ID",
                    index,
                    expected,
                    (),
                    False,
                )
            )
            continue
        if len(set(expected)) != len(expected):
            issues.append(
                ToolHistoryIssue(
                    "duplicate_call_id",
                    "assistant tool-call batch contains duplicate call IDs",
                    index,
                    expected,
                    (),
                    False,
                )
            )
            continue

        next_index = index + 1
        if next_index >= len(messages) or not is_tool_result_message(messages[next_index]):
            later_result = any(is_tool_result_message(item) for item in messages[next_index:])
            issues.append(
                ToolHistoryIssue(
                    "non_adjacent_results" if later_result else "missing_results",
                    "tool results do not immediately follow the assistant batch",
                    index,
                    expected,
                    (),
                    not later_result,
                )
            )
            continue

        observed: list[str] = []
        result_index = next_index
        while result_index < len(messages) and is_tool_result_message(messages[result_index]):
            consumed_result_indexes.add(result_index)
            observed.extend(extract_tool_result_ids(messages[result_index]))
            result_index += 1

        if any(not value for value in observed):
            issues.append(
                ToolHistoryIssue(
                    "empty_result_id",
                    "tool result exchange contains an empty call ID",
                    index,
                    expected,
                    tuple(observed),
                    False,
                )
            )
            continue
        if len(set(observed)) != len(observed):
            issues.append(
                ToolHistoryIssue(
                    "duplicate_result_id",
                    "tool result exchange contains duplicate call IDs",
                    index,
                    expected,
                    tuple(observed),
                    False,
                )
            )
            continue
        unknown = set(observed) - set(expected)
        if unknown:
            issues.append(
                ToolHistoryIssue(
                    "unknown_result_id",
                    "tool result exchange contains an ID not requested by the assistant",
                    index,
                    expected,
                    tuple(observed),
                    False,
                )
            )
            continue
        missing = set(expected) - set(observed)
        if missing:
            issues.append(
                ToolHistoryIssue(
                    "missing_results",
                    "tool result exchange does not answer every assistant call",
                    index,
                    expected,
                    tuple(observed),
                    True,
                )
            )
    for index, message in enumerate(messages):
        if is_tool_result_message(message) and index not in consumed_result_indexes:
            issues.append(
                ToolHistoryIssue(
                    "orphan_result",
                    "tool result is not attached to an assistant tool-call batch",
                    None,
                    (),
                    extract_tool_result_ids(message),
                    False,
                )
            )
    return ToolHistoryReport(tuple(issues))


def _synthetic_result(call_id: str) -> Any:
    """Create a provider-neutral error result without importing at module load."""
    from lauren_ai._tools import ToolResult  # noqa: PLC0415

    return ToolResult.error(
        "[Tool execution was interrupted before a result was received.]",
        tool_use_id=call_id,
        status="synthetic",
    )


_TOOL_OUTCOME_STATUSES = frozenset({"executed", "error", "rejected", "cancelled", "timed_out", "synthetic"})


def _outcome_status(result: Any, *, synthetic: bool) -> str:
    """Return a bounded transaction outcome classification."""
    if synthetic:
        return "synthetic"
    if isinstance(result, dict):
        candidate = result.get("status", "")
        is_error = bool(result.get("is_error", False))
    else:
        candidate = getattr(result, "status", "")
        is_error = bool(getattr(result, "is_error", False))
    status = str(candidate) if candidate in _TOOL_OUTCOME_STATUSES else ""
    return status or ("error" if is_error else "executed")


def _repair_tool_history(messages: list[Any]) -> tuple[list[Any], ToolHistoryReport]:
    """Repair only deterministic missing-result tails and validate again."""
    report = _validate_tool_history(messages)
    if report.ok:
        return list(messages), report
    if not report.repairable:
        raise _tool_history_error(report)

    repaired = copy.deepcopy(messages)
    for issue in sorted(report.issues, key=lambda item: item.assistant_index or -1, reverse=True):
        if issue.assistant_index is None:
            continue
        assistant_index = issue.assistant_index
        expected = issue.expected_ids
        observed: list[str] = []
        result_index = assistant_index + 1
        while result_index < len(repaired) and is_tool_result_message(repaired[result_index]):
            observed.extend(extract_tool_result_ids(repaired[result_index]))
            result_index += 1
        missing = [call_id for call_id in expected if call_id not in set(observed)]
        if not missing:
            continue
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": "[Tool execution was interrupted before a result was received.]",
                "is_error": True,
            }
            for call_id in missing
        ]
        first_result_index = assistant_index + 1
        if first_result_index < len(repaired) and is_tool_result_message(repaired[first_result_index]):
            first = repaired[first_result_index]
            if isinstance(first, dict) and first.get("role") == "user" and isinstance(first.get("content"), list):
                first["content"].extend(blocks)
            else:
                repaired[result_index:result_index] = [{"role": "user", "content": blocks}]
        else:
            repaired[result_index:result_index] = [{"role": "user", "content": blocks}]

    final = _validate_tool_history(repaired)
    if not final.ok:
        raise _tool_history_error(final)
    return repaired, final


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
        self._active_exchange: ToolExchange | None = None

    def validate_tool_history(self) -> ToolHistoryReport:
        """Validate every assistant tool exchange in the current history.

        The report contains only machine-readable codes, counts, and indexes;
        it never includes prompts, tool arguments, or tool output.  Callers
        that are about to contact a provider must reject a non-``ok`` report.
        """
        return _validate_tool_history(self._messages)

    @property
    def active_tool_exchange(self) -> ToolExchange | None:
        """Return the most recently started tool exchange, if any."""
        return self._active_exchange

    def repair_tool_history(self) -> ToolHistoryReport:
        """Repair deterministic missing-result exchanges in place.

        Missing results are filled with explicit error results.  Empty,
        duplicate, unknown, or non-adjacent IDs are not guessed at and raise
        :class:`ToolConversationIntegrityError` instead.
        """
        before = self.validate_tool_history()
        repaired, report = _repair_tool_history(self._messages)
        self._messages = repaired
        exchange = self._active_exchange
        if exchange is not None and exchange.state == "started" and not before.ok:
            self._abort_repaired_exchange(exchange)
        return report

    def _abort_repaired_exchange(self, exchange: ToolExchange) -> None:
        """Record repaired outcomes and close an in-flight exchange once."""
        observed = {tool_use_id for message in self._messages for tool_use_id in extract_tool_result_ids(message)}
        outcomes = tuple(
            ToolResultRecord(
                tool_use_id=call_id,
                status="synthetic" if call_id not in observed else "executed",
                synthetic=call_id not in observed,
            )
            for call_id in exchange.call_ids
        )
        aborted = exchange.with_outcomes(outcomes).aborted()
        self._active_exchange = aborted
        for outcome in outcomes:
            self.on_tool_exchange_result_recorded(aborted, outcome)
        self.on_tool_exchange_aborted(aborted, repaired=True)

    def begin_tool_exchange(self, tool_calls: Iterable[Any], run_id: str | None = None) -> ToolExchange:
        """Start an exchange for the most recently appended assistant batch."""
        if self._active_exchange is not None and self._active_exchange.state == "started":
            raise ToolConversationIntegrityError(
                "A tool exchange is already in progress",
                code="exchange_already_started",
                expected_count=len(self._active_exchange.call_ids),
                observed_count=0,
            )
        exchange = ToolExchange.from_tool_calls(tool_calls, run_id=run_id)
        if not self._messages or extract_tool_call_ids(self._messages[-1]) != exchange.call_ids:
            raise ToolConversationIntegrityError(
                "Tool exchange does not match the most recent assistant batch",
                code="exchange_batch_mismatch",
                expected_count=len(exchange.call_ids),
                observed_count=len(extract_tool_call_ids(self._messages[-1])) if self._messages else 0,
            )
        self._active_exchange = exchange
        self.on_tool_exchange_started(exchange)
        return exchange

    def commit_tool_exchange(
        self,
        exchange: ToolExchange,
        results: list[Any],
        *,
        on_unresolved: str = "synthesize_error_results",
    ) -> ToolExchange:
        """Commit one complete result batch for *exchange*.

        Results are reordered to the assistant's call order.  A missing result
        is synthesized when ``on_unresolved`` requests the default recovery;
        unknown, duplicate, or empty IDs always fail locally.
        """
        if exchange.state == "committed":
            return exchange
        if exchange.state == "aborted":
            raise ToolConversationIntegrityError(
                "Cannot commit an aborted tool exchange",
                code="exchange_already_aborted",
                expected_count=len(exchange.call_ids),
                observed_count=0,
            )
        current = self._active_exchange
        if current is not None and current.exchange_id != exchange.exchange_id:
            raise ToolConversationIntegrityError(
                "Tool exchange does not own the active transaction",
                code="exchange_owner_mismatch",
                expected_count=len(current.call_ids),
                observed_count=len(exchange.call_ids),
            )
        if current is not None and current.exchange_id == exchange.exchange_id:
            if current.state == "committed":
                return current
            if current.state == "aborted":
                raise ToolConversationIntegrityError(
                    "Cannot commit an aborted tool exchange",
                    code="exchange_already_aborted",
                    expected_count=len(exchange.call_ids),
                    observed_count=0,
                )
        expected = exchange.call_ids
        actual: list[str] = []
        for result in results:
            if isinstance(result, dict):
                actual.extend(extract_tool_result_ids(result))
            else:
                actual.append(str(getattr(result, "tool_use_id", "") or ""))
        if any(not value for value in actual):
            raise ToolConversationIntegrityError(
                "Tool exchange contains an empty result ID",
                code="empty_result_id",
                expected_count=len(expected),
                observed_count=len(actual),
            )
        if len(set(actual)) != len(actual):
            raise ToolConversationIntegrityError(
                "Tool exchange contains duplicate result IDs",
                code="duplicate_result_id",
                expected_count=len(expected),
                observed_count=len(actual),
            )
        if set(actual) - set(expected):
            raise ToolConversationIntegrityError(
                "Tool exchange contains an unknown result ID",
                code="unknown_result_id",
                expected_count=len(expected),
                observed_count=len(actual),
            )
        missing = [call_id for call_id in expected if call_id not in set(actual)]
        if missing and on_unresolved != "synthesize_error_results":
            raise ToolConversationIntegrityError(
                "Tool exchange does not contain a result for every call",
                code="missing_results",
                expected_count=len(expected),
                observed_count=len(actual),
                repairable=True,
            )

        by_id = {
            str(getattr(result, "tool_use_id", "") or ""): result for result in results if not isinstance(result, dict)
        }
        for result in results:
            if isinstance(result, dict):
                result_ids = extract_tool_result_ids(result)
                if len(result_ids) == 1:
                    by_id[result_ids[0]] = result
        ordered = [by_id[call_id] for call_id in expected if call_id in by_id]
        ordered.extend(_synthetic_result(call_id) for call_id in missing)
        self.add_tool_results(ordered)
        report = self.validate_tool_history()
        if not report.ok:
            raise _tool_history_error(report)
        synthetic_ids = set(missing)
        outcomes = tuple(
            ToolResultRecord(
                tool_use_id=call_id,
                status=_outcome_status(by_id.get(call_id), synthetic=call_id in synthetic_ids),
                synthetic=call_id in synthetic_ids,
            )
            for call_id in expected
        )
        committed = exchange.with_outcomes(outcomes).committed()
        self._active_exchange = committed
        for outcome in outcomes:
            self.on_tool_exchange_result_recorded(committed, outcome)
        self.on_tool_exchange_committed(committed)
        return committed

    def abort_tool_exchange(self, exchange: ToolExchange, *, repaired: bool = False) -> ToolExchange:
        """Mark an exchange aborted after unresolved calls were repaired."""
        current = self._active_exchange
        if current is not None and current.exchange_id == exchange.exchange_id and current.state == "aborted":
            return current
        if current is not None and current.exchange_id == exchange.exchange_id and current.state == "committed":
            return current
        if exchange.state == "committed":
            return exchange
        if exchange.state == "aborted":
            return exchange
        aborted = (
            current.aborted()
            if current is not None and current.exchange_id == exchange.exchange_id
            else exchange.aborted()
        )
        self._active_exchange = aborted
        self.on_tool_exchange_aborted(aborted, repaired=repaired)
        return aborted

    def on_tool_exchange_started(self, exchange: ToolExchange) -> None:
        """Lifecycle hook for durable memory implementations."""

    def on_tool_exchange_committed(self, exchange: ToolExchange) -> None:
        """Lifecycle hook for durable memory implementations."""

    def on_tool_exchange_result_recorded(self, exchange: ToolExchange, outcome: ToolResultRecord) -> None:
        """Lifecycle hook after one result is associated with an exchange."""

    def on_tool_exchange_aborted(self, exchange: ToolExchange, *, repaired: bool) -> None:
        """Lifecycle hook for durable memory implementations."""

    @property
    def max_tokens(self) -> int:
        """The live-window token budget that :meth:`messages` trims to.

        This is the budget the proactive compaction ladder defends: when the
        full buffer's exact token count approaches it, older turns are
        LLM-summarised rather than lossily truncated.

        :return: The configured token budget.
        :rtype: int
        """
        return self._max_tokens

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
        incoming_ids = extract_tool_call_ids(completion)
        if any(not value for value in incoming_ids):
            raise ToolConversationIntegrityError(
                "Assistant tool-call batch contains an empty call ID",
                code="empty_call_id",
                expected_count=len(incoming_ids),
                observed_count=0,
            )
        if len(set(incoming_ids)) != len(incoming_ids):
            raise ToolConversationIntegrityError(
                "Assistant tool-call batch contains duplicate call IDs",
                code="duplicate_call_id",
                expected_count=len(incoming_ids),
                observed_count=0,
            )
        if isinstance(completion, dict):
            self._messages.append(completion)
            return

        # Dataclass / object form
        content = getattr(completion, "content", "")
        tool_calls = getattr(completion, "tool_calls", [])
        thinking_blocks = getattr(completion, "thinking_blocks", []) or []

        if tool_calls or thinking_blocks:
            # Build a content list.  Anthropic extended thinking requires the
            # thinking / redacted_thinking blocks to appear FIRST — before text
            # and tool_use — and to be sent back verbatim with their signatures
            # (PRD-137 C).  Order: thinking → text → tool_use.
            blocks: list[Any] = [_thinking_block_to_dict(tb) for tb in thinking_blocks]
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
        expected = extract_tool_call_ids(self._messages[-1]) if self._messages else ()
        result_id = (
            extract_tool_result_ids(result)[0]
            if isinstance(result, dict) and extract_tool_result_ids(result)
            else str(getattr(result, "tool_use_id", "") or "")
        )
        if expected and (len(expected) != 1 or result_id != expected[0]):
            raise ToolConversationIntegrityError(
                "A single result cannot commit a multi-call assistant exchange",
                code="partial_result_batch",
                expected_count=len(expected),
                observed_count=1 if result_id else 0,
            )
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

        expected = extract_tool_call_ids(self._messages[-1]) if self._messages else ()
        if expected:
            actual: list[str] = []
            for result in results:
                if isinstance(result, dict):
                    actual.extend(extract_tool_result_ids(result))
                else:
                    actual.append(str(getattr(result, "tool_use_id", "") or ""))
            if any(not value for value in actual):
                raise ToolConversationIntegrityError(
                    "Tool result exchange contains an empty result ID",
                    code="empty_result_id",
                    expected_count=len(expected),
                    observed_count=len(actual),
                )
            if len(actual) != len(set(actual)):
                raise ToolConversationIntegrityError(
                    "Tool result exchange contains duplicate result IDs",
                    code="duplicate_result_id",
                    expected_count=len(expected),
                    observed_count=len(actual),
                )
            if set(actual) != set(expected):
                raise ToolConversationIntegrityError(
                    "Tool result exchange does not answer exactly the assistant batch",
                    code="result_batch_mismatch",
                    expected_count=len(expected),
                    observed_count=len(actual),
                )

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

    def _safe_keep_recent(self, keep_recent: int) -> int:
        """Adjust *keep_recent* so the recent window opens on a valid anchor.

        The summary boundary must never split an assistant ``tool_use`` from its
        following ``tool_result``: if the first *kept* message were a
        ``tool_result`` whose ``tool_use`` got summarised away, the provider
        would reject the request (orphan tool_result / non-user first message).
        This grows the kept count until the first kept non-system message is a
        conversational anchor (text user message or assistant turn), capping at
        "keep everything" (which simply means nothing is summarised this round).

        :param keep_recent: The requested number of recent messages to keep.
        :return: An adjusted count whose recent window starts on a safe anchor.
        :rtype: int
        """
        non_system = [m for m in self._messages if _get_role(m) != "system"]
        n = len(non_system)
        k = max(0, min(keep_recent, n))
        # non_system[n - k] is the first kept message when keeping the last k.
        while 0 < k < n and _is_tool_result(non_system[n - k]):
            k += 1
        return k

    def messages_to_summarize(self, keep_recent: int = 6) -> list[Any]:
        """Return the slice of messages that should be compressed.

        Returns the oldest non-system messages outside the (boundary-safe)
        recent window.  System messages are excluded because they are already
        managed separately (they are never dropped by ``messages()`` either).

        :param keep_recent: Number of most-recent non-system messages to
            preserve verbatim.  Defaults to 6 (≈ 3 user/assistant pairs).  The
            boundary is snapped so a ``tool_use``/``tool_result`` pair is never
            split (see :meth:`_safe_keep_recent`).
        :type keep_recent: int
        :return: List of messages to feed to the summarisation LLM call.
        :rtype: list[Any]
        """
        non_system = [m for m in self._messages if _get_role(m) != "system"]
        k = self._safe_keep_recent(keep_recent)
        to_compress = non_system[:-k] if k and len(non_system) > k else (non_system if k == 0 else [])
        return list(to_compress)

    def trim_to_recent(self, keep_recent: int = 6) -> None:
        """Drop all but the most-recent *keep_recent* non-system messages.

        Called by the runner after the summarisation call so the buffer
        only holds recent turns while the older context lives in
        ``self._summary``.  The boundary is snapped so a ``tool_use``/
        ``tool_result`` pair is never split (see :meth:`_safe_keep_recent`).

        :param keep_recent: Number of most-recent non-system messages to
            keep.  Defaults to 6.
        :type keep_recent: int
        """
        system_msgs = [m for m in self._messages if _get_role(m) == "system"]
        non_system = [m for m in self._messages if _get_role(m) != "system"]
        k = self._safe_keep_recent(keep_recent)
        kept = non_system[-k:] if k else []
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
                break
            snapshot = candidate
            total_chars -= removed_chars

        # Heal any incomplete tool-call sequence at the tail (e.g. an agent
        # turn interrupted before all parallel tool results arrived).
        snapshot = _heal_dangling_tail(snapshot)
        # PRD-133 Layer C: hard budget guarantee.  The sliding-window trim above
        # never drops past the last conversational user message, so a single
        # oversized turn (huge tool result) can still exceed the window.  Truncate
        # block contents so the returned list is *always* within budget — the
        # request can never overflow the model context window.
        return _enforce_char_budget(snapshot, budget_chars)

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
        before = self.validate_tool_history()
        self._messages, _ = _repair_tool_history(self._messages)
        if self._active_exchange is not None and self._active_exchange.state == "started" and not before.ok:
            self._abort_repaired_exchange(self._active_exchange)

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
        self._active_exchange = None

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
        snapshot: dict[str, Any] = {
            "messages": copy.deepcopy(self._messages),
            "summary": self._summary,
        }
        if self._active_exchange is not None:
            snapshot["tool_exchange"] = copy.deepcopy(self._active_exchange)
        return snapshot

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
            self._active_exchange = None
        else:
            self._messages = list(data.get("messages", []))
            self._summary = data.get("summary")
            self._active_exchange = data.get("tool_exchange")

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

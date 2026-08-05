"""Provider-neutral tool-call conversation integrity primitives."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ToolExchange",
    "ToolCallRecord",
    "ToolResultRecord",
    "ToolHistoryIssue",
    "ToolHistoryReport",
    "extract_tool_call_ids",
    "extract_tool_result_ids",
    "is_tool_call_message",
    "is_tool_result_message",
    "redact_tool_ids",
]


@dataclass(frozen=True, slots=True)
class ToolHistoryIssue:
    """One safe, provider-neutral conversation invariant violation."""

    code: str
    message: str
    assistant_index: int | None
    expected_ids: tuple[str, ...]
    observed_ids: tuple[str, ...]
    repairable: bool

    @property
    def expected_count(self) -> int:
        return len(self.expected_ids)

    @property
    def observed_count(self) -> int:
        return len(self.observed_ids)


@dataclass(frozen=True, slots=True)
class ToolHistoryReport:
    """Validation result for a canonical conversation history."""

    issues: tuple[ToolHistoryIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def repairable(self) -> bool:
        return bool(self.issues) and all(issue.repairable for issue in self.issues)

    @property
    def first_issue(self) -> ToolHistoryIssue | None:
        return self.issues[0] if self.issues else None


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """The bounded provider-neutral identity of one requested tool call."""

    tool_use_id: str
    name: str
    input: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultRecord:
    """The diagnostic outcome of one call in a :class:`ToolExchange`."""

    tool_use_id: str
    status: str
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class ToolExchange:
    """Lifecycle identity for one assistant tool-call batch."""

    exchange_id: str
    call_ids: tuple[str, ...]
    run_id: str | None = None
    state: str = "started"
    calls: tuple[ToolCallRecord, ...] = ()
    outcomes: tuple[ToolResultRecord, ...] = ()

    @classmethod
    def from_tool_calls(cls, tool_calls: Iterable[Any], run_id: str | None = None) -> ToolExchange:
        """Create an exchange and reject empty or duplicate call IDs."""
        records: list[ToolCallRecord] = []
        for call in tool_calls:
            if isinstance(call, dict):
                call_id = str(call.get("tool_use_id") or call.get("id") or "")
                name = str(call.get("name") or "")
                raw_input = call.get("input")
            else:
                call_id = str(getattr(call, "tool_use_id", None) or getattr(call, "id", "") or "")
                name = str(getattr(call, "name", "") or "")
                raw_input = getattr(call, "input", None)
            input_value = dict(raw_input) if isinstance(raw_input, Mapping) else {}
            records.append(ToolCallRecord(call_id, name, input_value))
        ids = tuple(record.tool_use_id for record in records)
        if not ids or any(not call_id for call_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Tool exchange requires non-empty unique tool-use IDs")
        return cls(exchange_id=uuid.uuid4().hex, call_ids=ids, run_id=run_id, calls=tuple(records))

    def committed(self) -> ToolExchange:
        return ToolExchange(
            self.exchange_id,
            self.call_ids,
            self.run_id,
            "committed",
            self.calls,
            self.outcomes,
        )

    def aborted(self) -> ToolExchange:
        return ToolExchange(
            self.exchange_id,
            self.call_ids,
            self.run_id,
            "aborted",
            self.calls,
            self.outcomes,
        )

    def with_outcomes(self, outcomes: Iterable[ToolResultRecord]) -> ToolExchange:
        """Return a copy carrying ordered execution outcomes."""
        return ToolExchange(
            self.exchange_id,
            self.call_ids,
            self.run_id,
            self.state,
            self.calls,
            tuple(outcomes),
        )


def _content(message: Any) -> Any:
    return message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")


def _role(message: Any) -> str:
    if isinstance(message, dict):
        return message.get("role", "")
    role = getattr(message, "role", None)
    if role:
        return role
    return "assistant" if hasattr(message, "tool_calls") else ""


def extract_tool_call_ids(message: Any) -> tuple[str, ...]:
    """Return ordered call IDs, retaining empty and duplicate values."""
    if _role(message) != "assistant":
        return ()
    ids: list[str] = []
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
        for call in calls:
            if isinstance(call, dict):
                ids.append(str(call.get("id") or call.get("tool_use_id") or ""))
            else:
                ids.append(str(getattr(call, "id", None) or getattr(call, "tool_use_id", "") or ""))
    else:
        for call in getattr(message, "tool_calls", None) or []:
            ids.append(str(getattr(call, "id", None) or getattr(call, "tool_use_id", "") or ""))
    content = _content(message)
    if isinstance(content, list):
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
            if block_type == "tool_use":
                value = block.get("id") if isinstance(block, dict) else getattr(block, "id", "")
                ids.append(str(value or ""))
    return tuple(ids)


def extract_tool_result_ids(message: Any) -> tuple[str, ...]:
    """Return ordered result IDs, retaining empty and duplicate values."""
    role = _role(message)
    ids: list[str] = []
    if role == "tool":
        value = message.get("tool_call_id", "") if isinstance(message, dict) else getattr(message, "tool_call_id", "")
        ids.append(str(value or ""))
    content = _content(message)
    if isinstance(content, list):
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
            if block_type == "tool_result":
                value = block.get("tool_use_id", "") if isinstance(block, dict) else getattr(block, "tool_use_id", "")
                ids.append(str(value or ""))
    return tuple(ids)


def is_tool_call_message(message: Any) -> bool:
    """Return whether *message* contains assistant tool calls."""
    return bool(extract_tool_call_ids(message)) or (
        _role(message) == "assistant"
        and isinstance(_content(message), list)
        and any(
            (block.get("type") if isinstance(block, dict) else getattr(block, "type", "")) == "tool_use"
            for block in _content(message)
        )
    )


def is_tool_result_message(message: Any) -> bool:
    """Return whether *message* contains tool results."""
    return bool(extract_tool_result_ids(message)) or (
        _role(message) in {"tool", "user"}
        and isinstance(_content(message), list)
        and any(
            (block.get("type") if isinstance(block, dict) else getattr(block, "type", "")) == "tool_result"
            for block in _content(message)
        )
    )


def redact_tool_ids(ids: Iterable[str]) -> tuple[str, ...]:
    """Return stable short hashes suitable for diagnostics and logs."""
    return tuple(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12] for value in ids if value)

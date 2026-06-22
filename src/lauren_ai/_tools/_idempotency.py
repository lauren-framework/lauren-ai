"""Turn-scoped idempotency ledger for tool execution (PRD-129 Phase 1).

Prevents a tool from being re-executed when an agent turn is retried after a
transient network failure.  Transport-retry rolls the whole turn back to its
pre-turn memory snapshot and re-runs it; without this ledger every
already-completed tool — including side-effecting ones (``write_file``,
``run_bash``, ``git_commit``) — would run a second time and duplicate its side
effect.

The ledger is **turn-scoped**: the caller creates one per user turn and passes
it (the same instance) to every ``run`` / ``run_stream`` attempt of that turn.
A successful tool result recorded in attempt *n* is replayed — not
re-executed — when the model requests the identical call in attempt *n+1*.

Only successful (non-error) results are recorded, so a tool that genuinely
failed is free to run again on the next attempt.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lauren_ai._tools import ToolResult

__all__ = ["IdempotencyLedger"]


def _canonical_key(name: str, tool_input: dict[str, object] | None) -> str:
    """Stable content key for a tool call: name + canonicalised input.

    Uses sorted-key JSON so that semantically identical inputs (regardless of
    dict ordering) map to the same key.  Falls back to a sorted ``repr`` for
    inputs that are not JSON-serialisable.
    """
    try:
        payload = json.dumps(tool_input or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(sorted((tool_input or {}).items()))
    return f"{name}\x00{payload}"


class IdempotencyLedger:
    """In-memory record of completed tool calls, keyed by ``(name, input)``.

    Lives for the duration of one agent turn (across all transport-retry
    attempts of that turn) and is discarded afterwards.  Lookups and records
    are O(1).
    """

    __slots__ = ("_results",)

    def __init__(self) -> None:
        self._results: dict[str, ToolResult] = {}

    def lookup(self, name: str, tool_input: dict[str, object] | None) -> ToolResult | None:
        """Return the recorded result for an identical prior call, or ``None``."""
        return self._results.get(_canonical_key(name, tool_input))

    def record(self, name: str, tool_input: dict[str, object] | None, result: ToolResult) -> None:
        """Record a successful tool result for replay on a later attempt."""
        self._results[_canonical_key(name, tool_input)] = result

    def __len__(self) -> int:
        return len(self._results)

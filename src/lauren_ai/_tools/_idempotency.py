"""Turn-scoped idempotency ledger for tool execution (PRD-129 Phase 1).

A transport-retry rolls a whole agent turn back to its pre-turn memory snapshot
and re-runs it.  Without idempotency every already-completed tool — including
side-effecting ones (write_file, run_bash, git_commit) — would run a second time
and duplicate its side effect.

This ledger replays a recorded result **only across a rollback**, never within a
single forward attempt:

- ``record`` appends a successful result to the *pending* set — the current
  attempt's executions, in order.
- ``promote`` (called when the turn rolls back for a retry) moves the pending
  results into the *committed* set, where they become replayable.
- ``lookup`` consumes one committed result for a matching ``(name, input)``.

So a *retried* call replays its earlier result (the side effect already
happened), while a *legitimate repeat* call within one attempt — reading a file
twice, or reading it again after writing it — still executes live and sees
fresh data.  Crucially, the caller must rekey the replayed result's
``tool_use_id`` to the current call: the model issues a fresh id every attempt,
so leaking the recorded id would corrupt the conversation with a ``tool_result``
that has no matching ``tool_use`` block.

Only successful (non-error) results are recorded, so a tool that genuinely
failed is free to run again on the next attempt.
"""

from __future__ import annotations

import json
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lauren_ai._tools import ToolResult

__all__ = ["IdempotencyLedger", "canonical_tool_key"]


def canonical_tool_key(name: str, tool_input: dict[str, object] | None) -> str:
    """Stable content key for a tool call: name + canonicalised input.

    Sorted-key JSON so semantically identical inputs (any dict ordering) map to
    the same key; falls back to a sorted ``repr`` for non-JSON inputs.
    """
    try:
        payload = json.dumps(tool_input or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(sorted((tool_input or {}).items()))
    return f"{name}\x00{payload}"


class IdempotencyLedger:
    """Records completed tool calls and replays them across a rollback only.

    Lives for the duration of one agent turn (across all transport-retry
    attempts of that turn) and is discarded afterwards.
    """

    __slots__ = ("_committed", "_pending")

    def __init__(self) -> None:
        # key -> FIFO queue of replayable results from prior (rolled-back) attempts.
        self._committed: dict[str, deque[ToolResult]] = {}
        # (key, result) pairs executed in the current attempt, in call order.
        self._pending: list[tuple[str, ToolResult]] = []

    def lookup(self, name: str, tool_input: dict[str, object] | None) -> ToolResult | None:
        """Consume and return a replayable result for an identical prior call."""
        q = self._committed.get(canonical_tool_key(name, tool_input))
        if q:
            return q.popleft()
        return None

    def record(self, name: str, tool_input: dict[str, object] | None, result: ToolResult) -> None:
        """Record a successful result executed in the current attempt."""
        self._pending.append((canonical_tool_key(name, tool_input), result))

    def promote(self) -> None:
        """Make the current attempt's results replayable.

        Called on a turn rollback (retry).  Pending results — whose side effects
        already happened — become committed so the next attempt replays them
        instead of re-executing.
        """
        for key, result in self._pending:
            self._committed.setdefault(key, deque()).append(result)
        self._pending.clear()

    def __len__(self) -> int:
        return len(self._pending) + sum(len(q) for q in self._committed.values())

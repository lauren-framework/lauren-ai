"""Tests for the turn-scoped tool idempotency ledger (PRD-129 Phase 1).

The ledger replays a recorded tool result **only across a rollback** (a retry),
never within a single forward attempt.  A retried call replays its earlier
result (the side effect already happened); a legitimate repeat call within one
attempt still executes live.  Replayed results are rekeyed to the current
``tool_use_id`` so the conversation never gets a ``tool_result`` without a
matching ``tool_use`` block.
"""

from __future__ import annotations

import pytest

from lauren_ai import IdempotencyLedger, ToolResult
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory import ShortTermMemory
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


def _tool_map(*fns: object) -> dict:
    return {getattr(f, TOOL_META).name: (f, getattr(f, TOOL_META)) for f in fns}


def _final(text: str = "done") -> Completion:
    return Completion(
        id="cf",
        model="mock",
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


def _tool_result_blocks(memory: ShortTermMemory) -> list[dict]:
    blocks: list[dict] = []
    for m in memory._messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            blocks.extend(b for b in content if isinstance(b, dict) and b.get("type") == "tool_result")
    return blocks


# ── Ledger unit — replay only after promote ──────────────────────────────────


class TestLedgerUnit:
    def test_no_replay_before_promote(self) -> None:
        led = IdempotencyLedger()
        r = ToolResult.ok("x", tool_use_id="t1")
        led.record("write", {"p": "a"}, r)
        # Within the same attempt the call is NOT replayed — it runs live.
        assert led.lookup("write", {"p": "a"}) is None

    def test_replay_after_promote_then_consumed(self) -> None:
        led = IdempotencyLedger()
        r = ToolResult.ok("x", tool_use_id="t1")
        led.record("write", {"p": "a"}, r)
        led.promote()  # rollback
        assert led.lookup("write", {"p": "a"}) is r  # replayed once
        assert led.lookup("write", {"p": "a"}) is None  # then consumed

    def test_key_is_order_independent(self) -> None:
        led = IdempotencyLedger()
        r = ToolResult.ok("x", tool_use_id="t1")
        led.record("write", {"a": 1, "b": 2}, r)
        led.promote()
        assert led.lookup("write", {"b": 2, "a": 1}) is r

    def test_promote_preserves_per_key_order(self) -> None:
        led = IdempotencyLedger()
        r1 = ToolResult.ok("1", tool_use_id="t1")
        r2 = ToolResult.ok("2", tool_use_id="t2")
        led.record("read", {"p": "x"}, r1)
        led.record("read", {"p": "x"}, r2)
        led.promote()
        assert led.lookup("read", {"p": "x"}) is r1
        assert led.lookup("read", {"p": "x"}) is r2

    def test_distinct_keys(self) -> None:
        led = IdempotencyLedger()
        led.record("t", {"a": 1}, ToolResult.ok("r1", tool_use_id="t1"))
        led.record("u", {"a": 1}, ToolResult.ok("r3", tool_use_id="t3"))
        assert len(led) == 2
        led.promote()
        assert led.lookup("t", {"a": 1}).content == "r1"
        assert led.lookup("u", {"a": 1}).content == "r3"


# ── Regression: the production 400 (PRD-129) ─────────────────────────────────


class TestWithinAttemptRepeat:
    @pytest.mark.asyncio
    async def test_same_input_twice_in_one_attempt_runs_live_each_time(self) -> None:
        """The reported crash: a repeat (name,input) call within ONE turn.

        Content-dedup must NOT fire inside a single attempt — both calls execute
        and each tool_result keeps its own tool_use_id (else the provider rejects
        the conversation with a 400 'tool_result without matching tool_use').
        """
        calls: list[str] = []

        @tool()
        async def get_info(path: str) -> str:
            """Read file info (idempotent, but may change after a write)."""
            calls.append(path)
            return f"info:{path}:{len(calls)}"

        tools = _tool_map(get_info)

        @agent(model="mock-model")
        @use_tools(get_info)
        class G:
            pass

        G.__lauren_ai_agent__.tools = tools
        mem = ShortTermMemory()
        ledger = IdempotencyLedger()

        mock = MockTransport()
        # Same tool + same input across two turns of ONE attempt, distinct ids.
        mock.queue_tool_use("get_info", {"path": "README.md"}, tool_use_id="call_AA")
        mock.queue_tool_use("get_info", {"path": "README.md"}, tool_use_id="call_BB")
        mock.queue_response(_final())
        await AgentRunner(transport=mock).run(G(), "verify", memory=mem, idempotency_ledger=ledger)

        assert calls == ["README.md", "README.md"], "repeat call within one attempt must run live"
        ids = {b["tool_use_id"] for b in _tool_result_blocks(mem)}
        assert ids == {"call_AA", "call_BB"}, "each tool_result keeps its own tool_use_id"


# ── Retry idempotency (rollback → replay, rekeyed) ───────────────────────────


class TestRetryIdempotency:
    @pytest.mark.asyncio
    async def test_retry_replays_and_rekeys_tool_use_id(self) -> None:
        executions: list[str] = []

        @tool()
        async def write_file(path: str) -> str:
            """Write a file (side effect)."""
            executions.append(path)
            return f"wrote {path}"

        tools = _tool_map(write_file)

        @agent(model="mock-model")
        @use_tools(write_file)
        class W:
            pass

        W.__lauren_ai_agent__.tools = tools
        mem = ShortTermMemory()
        ledger = IdempotencyLedger()

        # Attempt 1 — model uses tool_use_id call_OLD; tool executes.
        mock1 = MockTransport()
        mock1.queue_tool_use("write_file", {"path": "f"}, tool_use_id="call_OLD")
        mock1.queue_response(_final())
        await AgentRunner(transport=mock1).run(W(), "go", memory=mem, idempotency_ledger=ledger)
        assert executions == ["f"]

        # Rollback for the retry: restore pre-turn memory + promote the ledger.
        mem.restore([])
        ledger.promote()

        # Attempt 2 — model issues a FRESH tool_use_id call_NEW for the same call.
        mock2 = MockTransport()
        mock2.queue_tool_use("write_file", {"path": "f"}, tool_use_id="call_NEW")
        mock2.queue_response(_final())
        await AgentRunner(transport=mock2).run(W(), "go", memory=mem, idempotency_ledger=ledger)

        assert executions == ["f"], "side-effecting tool must NOT re-run on retry"
        blocks = _tool_result_blocks(mem)
        assert blocks, "the replayed tool_result must be committed to memory"
        assert all(b["tool_use_id"] == "call_NEW" for b in blocks), "replayed result must be rekeyed"
        assert not any(b["tool_use_id"] == "call_OLD" for b in blocks)

    @pytest.mark.asyncio
    async def test_without_ledger_tool_runs_each_attempt(self) -> None:
        calls: list[str] = []

        @tool()
        async def write_file(path: str) -> str:
            """Write a file (side effect)."""
            calls.append(path)
            return f"wrote {path}"

        tools = _tool_map(write_file)

        @agent(model="mock-model")
        @use_tools(write_file)
        class W2:
            pass

        W2.__lauren_ai_agent__.tools = tools

        for _ in range(2):
            mock = MockTransport()
            mock.queue_tool_use("write_file", {"path": "f.txt"})
            mock.queue_response(_final())
            await AgentRunner(transport=mock).run(W2(), "write it")  # no ledger

        assert len(calls) == 2, "without a ledger the tool re-runs every attempt"

    @pytest.mark.asyncio
    async def test_failed_tool_is_not_recorded_and_reruns(self) -> None:
        attempts: list[int] = []

        @tool()
        async def flaky(x: str) -> str:
            """Always raises."""
            attempts.append(1)
            raise RuntimeError("boom")

        tools = _tool_map(flaky)

        @agent(model="mock-model")
        @use_tools(flaky)
        class F:
            pass

        F.__lauren_ai_agent__.tools = tools
        ledger = IdempotencyLedger()

        for _ in range(2):
            mock = MockTransport()
            mock.queue_tool_use("flaky", {"x": "1"})
            mock.queue_response(_final())
            await AgentRunner(transport=mock).run(F(), "go", idempotency_ledger=ledger)
            ledger.promote()  # rollback between attempts

        assert len(attempts) == 2, "failed tools must re-run (not replayed)"
        assert len(ledger) == 0, "errors must not be recorded in the ledger"

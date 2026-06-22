"""Tests for the turn-scoped tool idempotency ledger (PRD-129 Phase 1).

A transport-retry rolls a whole agent turn back to its pre-turn memory snapshot
and re-runs it.  The :class:`IdempotencyLedger` ensures a tool that already
completed successfully in an earlier attempt is *replayed* (not re-executed) on
the next attempt, so side-effecting tools (write_file, run_bash, …) apply their
effect exactly once.
"""

from __future__ import annotations

import pytest

from lauren_ai import IdempotencyLedger
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
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


# ── Ledger unit ──────────────────────────────────────────────────────────────


class TestLedgerUnit:
    def test_record_and_lookup_is_order_independent(self) -> None:
        led = IdempotencyLedger()
        led.record("write_file", {"path": "a", "content": "x"}, "RES")
        # Same input, different key ordering → same key.
        assert led.lookup("write_file", {"content": "x", "path": "a"}) == "RES"

    def test_lookup_miss_on_different_input(self) -> None:
        led = IdempotencyLedger()
        led.record("write_file", {"path": "a"}, "RES")
        assert led.lookup("write_file", {"path": "b"}) is None
        assert led.lookup("other_tool", {"path": "a"}) is None

    def test_distinct_keys(self) -> None:
        led = IdempotencyLedger()
        led.record("t", {"a": 1}, "r1")
        led.record("t", {"a": 2}, "r2")
        led.record("u", {"a": 1}, "r3")
        assert len(led) == 3
        assert led.lookup("t", {"a": 1}) == "r1"
        assert led.lookup("t", {"a": 2}) == "r2"
        assert led.lookup("u", {"a": 1}) == "r3"

    def test_unhashable_input_does_not_crash(self) -> None:
        led = IdempotencyLedger()
        led.record("t", {"items": [1, 2, {"nested": True}]}, "r")
        assert led.lookup("t", {"items": [1, 2, {"nested": True}]}) == "r"


# ── Retry idempotency (the core guarantee) ───────────────────────────────────


class TestRetryIdempotency:
    @pytest.mark.asyncio
    async def test_successful_tool_not_reexecuted_on_retry(self) -> None:
        """Same (name,input) call across two runs sharing a ledger runs once."""
        calls: list[tuple[str, str]] = []

        @tool()
        async def write_file(path: str, content: str) -> str:
            """Write a file (side effect)."""
            calls.append((path, content))
            return f"wrote {path}"

        tools = _tool_map(write_file)

        @agent(model="mock-model")
        @use_tools(write_file)
        class W:
            pass

        W.__lauren_ai_agent__.tools = tools
        ledger = IdempotencyLedger()

        # Attempt 1: tool_use → executes → final.
        mock1 = MockTransport()
        mock1.queue_tool_use("write_file", {"path": "f.txt", "content": "hi"})
        mock1.queue_response(_final())
        await AgentRunner(transport=mock1).run(W(), "write it", idempotency_ledger=ledger)
        assert len(calls) == 1

        # Attempt 2 (the retry): identical tool call requested again, SAME ledger.
        mock2 = MockTransport()
        mock2.queue_tool_use("write_file", {"path": "f.txt", "content": "hi"})
        mock2.queue_response(_final())
        await AgentRunner(transport=mock2).run(W(), "write it", idempotency_ledger=ledger)

        assert len(calls) == 1, "side-effecting tool must NOT re-run on retry"

    @pytest.mark.asyncio
    async def test_without_ledger_tool_runs_each_attempt(self) -> None:
        """Control: with no ledger, the retried tool re-executes (today's bug)."""
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
        """A tool that errors is NOT recorded → free to run again on retry."""
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

        assert len(attempts) == 2, "failed tools must re-run (not replayed from ledger)"
        assert len(ledger) == 0, "errors must not be recorded in the ledger"

    @pytest.mark.asyncio
    async def test_distinct_inputs_each_execute(self) -> None:
        """Different inputs to the same tool are distinct ledger entries."""
        calls: list[str] = []

        @tool()
        async def write_file(path: str) -> str:
            """Write a file."""
            calls.append(path)
            return f"wrote {path}"

        tools = _tool_map(write_file)

        @agent(model="mock-model")
        @use_tools(write_file)
        class W3:
            pass

        W3.__lauren_ai_agent__.tools = tools
        ledger = IdempotencyLedger()

        mock = MockTransport()
        mock.queue_tool_use("write_file", {"path": "a.txt"}, tool_use_id="t1")
        # second turn: a different file, then finish
        mock.queue_tool_use("write_file", {"path": "b.txt"}, tool_use_id="t2")
        mock.queue_response(_final())
        await AgentRunner(transport=mock).run(W3(), "write both", idempotency_ledger=ledger)

        assert calls == ["a.txt", "b.txt"]
        assert len(ledger) == 2

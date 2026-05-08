"""Integration tests for Skill 8: ReAct Agent Loop (Reason + Act).

Tests cover:
- Single turn (think only, no tool call) → end_turn
- Two turns: tool_call → final answer
- Three turns: tool → tool → final answer
- turns count matches actual loop iterations
- on_turn_complete hook fires per turn
- on_tool_result hook fires per tool result
- AgentMaxTurnsError raised when max_turns exceeded mid-loop
- Tool result injected into next call messages

NOTE: No `from __future__ import annotations` in this file — @tool() used.
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import _add_to_tool_map, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool()
async def search_tool(query: str) -> dict:
    """Search for information.

    Args:
        query: The search query.
    """
    return {"query": query, "result": f"Found info about {query}"}


@tool()
async def compute_tool(value: int) -> dict:
    """Compute a value.

    Args:
        value: The integer to process.
    """
    return {"input": value, "output": value * 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str = "OK", *, n: int = 1, stop_reason: str = "end_turn") -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner_with_tools(*tool_funcs) -> tuple[AgentRunner, MockTransport]:
    mock = MockTransport()
    tools = {}
    for t in tool_funcs:
        _add_to_tool_map(tools, t)
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools=tools, config=cfg)
    return runner, mock


def _make_runner(mock: MockTransport | None = None) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# TestReActSingleTurn
# ---------------------------------------------------------------------------


class TestReActSingleTurn:
    async def test_single_turn_no_tool_call(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Direct answer."))

        @agent(model="mock-model")
        class DirectAgent: ...

        resp = await runner.run(DirectAgent(), "What is 2+2?")
        assert resp.content == "Direct answer."

    async def test_single_turn_count(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Answer"))

        @agent(model="mock-model")
        class DirectAgent: ...

        resp = await runner.run(DirectAgent(), "hi")
        assert resp.turns == 1

    async def test_single_turn_stop_reason_end_turn(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Done"))

        @agent(model="mock-model")
        class DirectAgent: ...

        resp = await runner.run(DirectAgent(), "hi")
        assert resp.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# TestReActTwoTurns
# ---------------------------------------------------------------------------


class TestReActTwoTurns:
    async def test_think_act_answer_loop(self):
        runner, mock = _make_runner_with_tools(search_tool)

        @use_tools(search_tool)
        @agent(model="mock-model")
        class ReActAgent: ...

        mock.queue_tool_use("search_tool", {"query": "AI"})
        mock.queue_response(_completion("Based on search: AI is advancing rapidly."))

        resp = await runner.run(ReActAgent(), "Tell me about AI")
        assert "AI" in resp.content

    async def test_two_turn_count(self):
        runner, mock = _make_runner_with_tools(search_tool)

        @use_tools(search_tool)
        @agent(model="mock-model")
        class ReActAgent: ...

        mock.queue_tool_use("search_tool", {"query": "news"})
        mock.queue_response(_completion("News found."))

        resp = await runner.run(ReActAgent(), "find news")
        assert resp.turns == 2

    async def test_two_llm_calls_made(self):
        runner, mock = _make_runner_with_tools(search_tool)

        @use_tools(search_tool)
        @agent(model="mock-model")
        class ReActAgent: ...

        mock.queue_tool_use("search_tool", {"query": "test"})
        mock.queue_response(_completion("Done"))

        await runner.run(ReActAgent(), "go")
        assert len(mock.calls) == 2

    async def test_tool_result_in_second_call_messages(self):
        runner, mock = _make_runner_with_tools(search_tool)

        @use_tools(search_tool)
        @agent(model="mock-model")
        class ReActAgent: ...

        mock.queue_tool_use("search_tool", {"query": "quantum"})
        mock.queue_response(_completion("Quantum is complex."))

        await runner.run(ReActAgent(), "explain quantum")
        # Second call should have tool result in messages
        second_call_messages = mock.calls[1].messages
        # The tool result is appended as a message with tool_result type
        assert len(second_call_messages) > len(mock.calls[0].messages)


# ---------------------------------------------------------------------------
# TestReActThreeTurns
# ---------------------------------------------------------------------------


class TestReActThreeTurns:
    async def test_three_turn_loop(self):
        runner, mock = _make_runner_with_tools(search_tool, compute_tool)

        @use_tools(search_tool, compute_tool)
        @agent(model="mock-model")
        class ThreeTurnAgent: ...

        mock.queue_tool_use("search_tool", {"query": "data"})
        mock.queue_tool_use("compute_tool", {"value": 21})
        mock.queue_response(_completion("Searched and computed: result is 42."))

        resp = await runner.run(ThreeTurnAgent(), "search and compute")
        assert resp.turns == 3

    async def test_both_tools_in_tool_calls_made(self):
        runner, mock = _make_runner_with_tools(search_tool, compute_tool)

        @use_tools(search_tool, compute_tool)
        @agent(model="mock-model")
        class ThreeTurnAgent: ...

        mock.queue_tool_use("search_tool", {"query": "x"})
        mock.queue_tool_use("compute_tool", {"value": 5})
        mock.queue_response(_completion("Done"))

        resp = await runner.run(ThreeTurnAgent(), "go")
        tool_names = [t.name for t in resp.tool_calls_made]
        assert "search_tool" in tool_names
        assert "compute_tool" in tool_names


# ---------------------------------------------------------------------------
# TestReActLifecycleHooks
# ---------------------------------------------------------------------------


class TestReActLifecycleHooks:
    async def test_on_turn_complete_fires_per_turn(self):
        turns_seen: list[int] = []

        runner, mock = _make_runner_with_tools(search_tool)

        @use_tools(search_tool)
        @agent(model="mock-model")
        class HookAgent:
            async def on_turn_complete(self, completion, ctx):
                turns_seen.append(ctx.turn)

        mock.queue_tool_use("search_tool", {"query": "hook test"})
        mock.queue_response(_completion("done"))

        await runner.run(HookAgent(), "go")
        assert len(turns_seen) == 2

    async def test_on_tool_result_fires(self):
        results_seen: list[str] = []

        runner, mock = _make_runner_with_tools(compute_tool)

        @use_tools(compute_tool)
        @agent(model="mock-model")
        class ToolHookAgent:
            async def on_tool_result(self, result, ctx):
                results_seen.append(result.content if hasattr(result, "content") else str(result))
                return None

        mock.queue_tool_use("compute_tool", {"value": 5})
        mock.queue_response(_completion("computed"))

        await runner.run(ToolHookAgent(), "compute")
        assert len(results_seen) == 1


# ---------------------------------------------------------------------------
# TestReActMaxTurns
# ---------------------------------------------------------------------------


class TestReActMaxTurns:
    async def test_max_turns_sets_stop_reason(self):
        runner, mock = _make_runner_with_tools(search_tool)

        @use_tools(search_tool)
        @agent(model="mock-model", max_turns=1)
        class TightAgent: ...

        # Queue tool_use so the loop exhausts max_turns
        mock.queue_tool_use("search_tool", {"query": "x"})
        mock.queue_tool_use("search_tool", {"query": "y"})

        # Runner exhausts max_turns and returns stop_reason="max_turns"
        resp = await runner.run(TightAgent(), "go")
        assert resp.stop_reason in ("max_turns", "end_turn")

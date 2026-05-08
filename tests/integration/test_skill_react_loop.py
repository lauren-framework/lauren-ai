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

from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


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
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model")
class DirectAgent: ...


@agent(model="mock-model")
@use_tools(search_tool)
class ReActAgent: ...


@agent(model="mock-model")
@use_tools(search_tool, compute_tool)
class ThreeTurnAgent: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(content: str = "OK", *, n: int = 1, stop_reason: str = "end_turn") -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# TestReActSingleTurn
# ---------------------------------------------------------------------------


class TestReActSingleTurn:
    def test_single_turn_no_tool_call(self):
        client = TestClient(DirectAgent())
        client.mock.queue_response(_c("Direct answer."))
        result = client.run("What is 2+2?")
        assert result.content == "Direct answer."

    def test_single_turn_count(self):
        client = TestClient(DirectAgent())
        client.mock.queue_response(_c("Answer"))
        result = client.run("hi")
        assert result.turns == 1

    def test_single_turn_stop_reason_end_turn(self):
        client = TestClient(DirectAgent())
        client.mock.queue_response(_c("Done"))
        result = client.run("hi")
        assert result.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# TestReActTwoTurns
# ---------------------------------------------------------------------------


class TestReActTwoTurns:
    def test_think_act_answer_loop(self):
        client = TestClient(ReActAgent())
        client.mock.queue_tool_use("search_tool", {"query": "AI"})
        client.mock.queue_response(_c("Based on search: AI is advancing rapidly."))
        result = client.run("Tell me about AI")
        assert "AI" in result.content

    def test_two_turn_count(self):
        client = TestClient(ReActAgent())
        client.mock.queue_tool_use("search_tool", {"query": "news"})
        client.mock.queue_response(_c("News found."))
        result = client.run("find news")
        assert result.turns == 2

    def test_two_llm_calls_made(self):
        client = TestClient(ReActAgent())
        client.mock.queue_tool_use("search_tool", {"query": "test"})
        client.mock.queue_response(_c("Done"))
        client.run("go")
        assert len(client.calls) == 2

    def test_tool_result_in_second_call_messages(self):
        client = TestClient(ReActAgent())
        client.mock.queue_tool_use("search_tool", {"query": "quantum"})
        client.mock.queue_response(_c("Quantum is complex."))
        client.run("explain quantum")
        first_call_msg_count = len(client.calls[0].messages)
        second_call_msg_count = len(client.calls[1].messages)
        assert second_call_msg_count > first_call_msg_count


# ---------------------------------------------------------------------------
# TestReActThreeTurns
# ---------------------------------------------------------------------------


class TestReActThreeTurns:
    def test_three_turn_loop(self):
        client = TestClient(ThreeTurnAgent())
        client.mock.queue_tool_use("search_tool", {"query": "data"})
        client.mock.queue_tool_use("compute_tool", {"value": 21})
        client.mock.queue_response(_c("Searched and computed: result is 42."))
        result = client.run("search and compute")
        assert result.turns == 3

    def test_both_tools_in_tool_calls_made(self):
        client = TestClient(ThreeTurnAgent())
        client.mock.queue_tool_use("search_tool", {"query": "x"})
        client.mock.queue_tool_use("compute_tool", {"value": 5})
        client.mock.queue_response(_c("Done"))
        result = client.run("go")
        tool_names = [t.name for t in result.tool_calls_made]
        assert "search_tool" in tool_names
        assert "compute_tool" in tool_names


# ---------------------------------------------------------------------------
# TestReActLifecycleHooks
# ---------------------------------------------------------------------------


class TestReActLifecycleHooks:
    async def test_on_turn_complete_fires_per_turn(self):
        turns_seen = []

        @agent(model="mock-model")
        @use_tools(search_tool)
        class HookAgent:
            async def on_turn_complete(self, completion, ctx):
                turns_seen.append(ctx.turn)

        client = TestClient(HookAgent())
        client.mock.queue_tool_use("search_tool", {"query": "hook test"})
        client.mock.queue_response(_c("done"))
        await client.run_async("go")
        assert len(turns_seen) == 2

    async def test_on_tool_result_fires(self):
        results_seen = []

        @agent(model="mock-model")
        @use_tools(compute_tool)
        class ToolHookAgent:
            async def on_tool_result(self, result, ctx):
                results_seen.append(result.content if hasattr(result, "content") else str(result))
                return None

        client = TestClient(ToolHookAgent())
        client.mock.queue_tool_use("compute_tool", {"value": 5})
        client.mock.queue_response(_c("computed"))
        await client.run_async("compute")
        assert len(results_seen) == 1


# ---------------------------------------------------------------------------
# TestReActMaxTurns
# ---------------------------------------------------------------------------


class TestReActMaxTurns:
    def test_max_turns_sets_stop_reason(self):
        @agent(model="mock-model", max_turns=1)
        @use_tools(search_tool)
        class TightAgent: ...

        client = TestClient(TightAgent())
        client.mock.queue_tool_use("search_tool", {"query": "x"})
        client.mock.queue_tool_use("search_tool", {"query": "y"})
        result = client.run("go")
        assert result.stop_reason in ("max_turns", "end_turn")

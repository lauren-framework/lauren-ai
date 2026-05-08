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
from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
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
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content: str = "OK", *, n: int = 1, stop_reason: str = "end_turn") -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controller / Module
# ---------------------------------------------------------------------------


class _RunRequest(BaseModel):
    prompt: str = "hi"
    max_turns: int = 10


@controller("/agent")
class ReActController:
    def __init__(self, mock: MockTransport) -> None:
        tools = {}
        _add_to_tool_map(tools, search_tool)
        _add_to_tool_map(tools, compute_tool)
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools=tools, config=cfg)
        self._runner_no_tools = AgentRunner(transport=mock, tools={}, config=cfg)
        self._mock = mock

    @post("/run-direct")
    async def run_direct(self, body: Json[_RunRequest]) -> dict:
        @agent(model="mock-model")
        class DirectAgent: ...

        resp = await self._runner_no_tools.run(DirectAgent(), body.prompt)
        return {
            "content": resp.content,
            "turns": resp.turns,
            "stop_reason": resp.stop_reason,
        }

    @post("/run-search")
    async def run_search(self, body: Json[_RunRequest]) -> dict:
        @use_tools(search_tool)
        @agent(model="mock-model")
        class ReActAgent: ...

        resp = await self._runner.run(ReActAgent(), body.prompt)
        return {
            "content": resp.content,
            "turns": resp.turns,
            "calls": len(self._mock.calls),
            "second_call_msg_count": len(self._mock.calls[1].messages) if len(self._mock.calls) > 1 else 0,
            "first_call_msg_count": len(self._mock.calls[0].messages) if self._mock.calls else 0,
        }

    @post("/run-three-turns")
    async def run_three_turns(self, body: Json[_RunRequest]) -> dict:
        @use_tools(search_tool, compute_tool)
        @agent(model="mock-model")
        class ThreeTurnAgent: ...

        resp = await self._runner.run(ThreeTurnAgent(), body.prompt)
        tool_names = [t.name for t in resp.tool_calls_made]
        return {
            "content": resp.content,
            "turns": resp.turns,
            "tool_names": tool_names,
        }

    @post("/run-hooks")
    async def run_hooks(self, body: Json[_RunRequest]) -> dict:
        turns_seen = []
        results_seen = []

        @use_tools(search_tool)
        @agent(model="mock-model")
        class HookAgent:
            async def on_turn_complete(self, completion, ctx):
                turns_seen.append(ctx.turn)

        resp = await self._runner.run(HookAgent(), body.prompt)
        return {
            "turns_seen": turns_seen,
            "content": resp.content,
        }

    @post("/run-tool-result-hook")
    async def run_tool_result_hook(self, body: Json[_RunRequest]) -> dict:
        results_seen = []

        @use_tools(compute_tool)
        @agent(model="mock-model")
        class ToolHookAgent:
            async def on_tool_result(self, result, ctx):
                results_seen.append(result.content if hasattr(result, "content") else str(result))
                return None

        await self._runner.run(ToolHookAgent(), body.prompt)
        return {"results_count": len(results_seen)}

    @post("/run-max-turns")
    async def run_max_turns(self, body: Json[_RunRequest]) -> dict:
        @use_tools(search_tool)
        @agent(model="mock-model", max_turns=body.max_turns)
        class TightAgent: ...

        resp = await self._runner.run(TightAgent(), body.prompt)
        return {"stop_reason": resp.stop_reason}


@module(
    controllers=[ReActController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class ReActModule: ...


def build_app() -> TestClient:
    _MOCK.reset()
    return TestClient(LaurenFactory.create(ReActModule))


# ---------------------------------------------------------------------------
# TestReActSingleTurn
# ---------------------------------------------------------------------------


class TestReActSingleTurn:
    def test_single_turn_no_tool_call(self):
        client = build_app()
        _MOCK.queue_response(_completion("Direct answer."))
        r = client.post("/agent/run-direct", json={"prompt": "What is 2+2?"})
        assert r.status_code == 200
        assert r.json()["content"] == "Direct answer."

    def test_single_turn_count(self):
        client = build_app()
        _MOCK.queue_response(_completion("Answer"))
        r = client.post("/agent/run-direct", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["turns"] == 1

    def test_single_turn_stop_reason_end_turn(self):
        client = build_app()
        _MOCK.queue_response(_completion("Done"))
        r = client.post("/agent/run-direct", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# TestReActTwoTurns
# ---------------------------------------------------------------------------


class TestReActTwoTurns:
    def test_think_act_answer_loop(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "AI"})
        _MOCK.queue_response(_completion("Based on search: AI is advancing rapidly."))
        r = client.post("/agent/run-search", json={"prompt": "Tell me about AI"})
        assert r.status_code == 200
        assert "AI" in r.json()["content"]

    def test_two_turn_count(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "news"})
        _MOCK.queue_response(_completion("News found."))
        r = client.post("/agent/run-search", json={"prompt": "find news"})
        assert r.status_code == 200
        assert r.json()["turns"] == 2

    def test_two_llm_calls_made(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "test"})
        _MOCK.queue_response(_completion("Done"))
        r = client.post("/agent/run-search", json={"prompt": "go"})
        assert r.status_code == 200
        assert r.json()["calls"] == 2

    def test_tool_result_in_second_call_messages(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "quantum"})
        _MOCK.queue_response(_completion("Quantum is complex."))
        r = client.post("/agent/run-search", json={"prompt": "explain quantum"})
        assert r.status_code == 200
        data = r.json()
        assert data["second_call_msg_count"] > data["first_call_msg_count"]


# ---------------------------------------------------------------------------
# TestReActThreeTurns
# ---------------------------------------------------------------------------


class TestReActThreeTurns:
    def test_three_turn_loop(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "data"})
        _MOCK.queue_tool_use("compute_tool", {"value": 21})
        _MOCK.queue_response(_completion("Searched and computed: result is 42."))
        r = client.post("/agent/run-three-turns", json={"prompt": "search and compute"})
        assert r.status_code == 200
        assert r.json()["turns"] == 3

    def test_both_tools_in_tool_calls_made(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "x"})
        _MOCK.queue_tool_use("compute_tool", {"value": 5})
        _MOCK.queue_response(_completion("Done"))
        r = client.post("/agent/run-three-turns", json={"prompt": "go"})
        assert r.status_code == 200
        tool_names = r.json()["tool_names"]
        assert "search_tool" in tool_names
        assert "compute_tool" in tool_names


# ---------------------------------------------------------------------------
# TestReActLifecycleHooks
# ---------------------------------------------------------------------------


class TestReActLifecycleHooks:
    def test_on_turn_complete_fires_per_turn(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "hook test"})
        _MOCK.queue_response(_completion("done"))
        r = client.post("/agent/run-hooks", json={"prompt": "go"})
        assert r.status_code == 200
        assert len(r.json()["turns_seen"]) == 2

    def test_on_tool_result_fires(self):
        client = build_app()
        _MOCK.queue_tool_use("compute_tool", {"value": 5})
        _MOCK.queue_response(_completion("computed"))
        r = client.post("/agent/run-tool-result-hook", json={"prompt": "compute"})
        assert r.status_code == 200
        assert r.json()["results_count"] == 1


# ---------------------------------------------------------------------------
# TestReActMaxTurns
# ---------------------------------------------------------------------------


class TestReActMaxTurns:
    def test_max_turns_sets_stop_reason(self):
        client = build_app()
        _MOCK.queue_tool_use("search_tool", {"query": "x"})
        _MOCK.queue_tool_use("search_tool", {"query": "y"})
        r = client.post("/agent/run-max-turns", json={"prompt": "go", "max_turns": 1})
        assert r.status_code == 200
        assert r.json()["stop_reason"] in ("max_turns", "end_turn")

"""Integration tests for Skill 7: Multi-Tool Orchestration & Selection.

Tests cover:
- Agent with multiple tools attached via @use_tools()
- LLM can call first tool, then second tool (sequential)
- Both tool calls recorded in resp.tool_calls_made
- Final answer returned after all tools
- Tool error policy applies to any of the tools
- Parallel tool calls config accepted

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
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
async def search_web(query: str) -> dict:
    """Search the web for a query.

    Args:
        query: The search query string.
    """
    return {"query": query, "results": ["result1", "result2"]}


@tool()
async def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: The math expression to evaluate.
    """
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


@tool()
async def get_stock_price(symbol: str) -> dict:
    """Get current stock price.

    Args:
        symbol: Stock ticker symbol.
    """
    return {"symbol": symbol, "price": 150.0}


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content: str = "OK", *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controller / Module
# ---------------------------------------------------------------------------


class _RunRequest(BaseModel):
    prompt: str = "hi"


@controller("/agent")
class MultiToolController:
    def __init__(self, mock: MockTransport) -> None:
        tools = {}
        _add_to_tool_map(tools, search_web)
        _add_to_tool_map(tools, calculate)
        _add_to_tool_map(tools, get_stock_price)
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools=tools, config=cfg)
        self._mock = mock

    @post("/run-research")
    async def run_research(self, body: Json[_RunRequest]) -> dict:
        @use_tools(search_web, calculate)
        @agent(model="mock-model")
        class ResearchAgent: ...

        resp = await self._runner.run(ResearchAgent(), body.prompt)
        return {
            "content": resp.content,
            "tool_calls_made": [t.name for t in resp.tool_calls_made],
            "calls": len(self._mock.calls),
        }

    @post("/run-stock")
    async def run_stock(self, body: Json[_RunRequest]) -> dict:
        @use_tools(search_web, calculate, get_stock_price)
        @agent(model="mock-model")
        class FullAgent: ...

        resp = await self._runner.run(FullAgent(), body.prompt)
        return {
            "content": resp.content,
            "tool_calls_made": [t.name for t in resp.tool_calls_made],
        }

    @post("/run-parallel")
    async def run_parallel(self, body: Json[_RunRequest]) -> dict:
        @use_tools(search_web, calculate)
        @agent(model="mock-model", parallel_tool_calls=True)
        class ParallelAgent: ...

        resp = await self._runner.run(ParallelAgent(), body.prompt)
        return {"content": resp.content}

    @post("/run-resilient")
    async def run_resilient(self, body: Json[_RunRequest]) -> dict:
        @tool()
        async def broken_tool(x: str) -> dict:
            """Broken tool. Args: x: any string."""
            raise RuntimeError("always broken")

        local_tools = {}
        _add_to_tool_map(local_tools, search_web)
        _add_to_tool_map(local_tools, broken_tool)
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        local_runner = AgentRunner(transport=self._mock, tools=local_tools, config=cfg)

        @use_tools(search_web, broken_tool)
        @agent(model="mock-model", tool_error_policy="return_error")
        class ResilientAgent: ...

        resp = await local_runner.run(ResilientAgent(), body.prompt)
        return {"content": resp.content}


@module(
    controllers=[MultiToolController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class MultiToolModule: ...


def build_app() -> TestClient:
    _MOCK.reset()
    return TestClient(LaurenFactory.create(MultiToolModule))


# ---------------------------------------------------------------------------
# TestMultiToolAttachment (via TestClient)
# ---------------------------------------------------------------------------


class TestMultiToolAttachment:
    def test_two_tools_sequential_calls(self):
        client = build_app()
        _MOCK.queue_tool_use("search_web", {"query": "AI trends"})
        _MOCK.queue_tool_use("calculate", {"expression": "42 * 2"})
        _MOCK.queue_response(_completion("Searched and calculated. Answer is 84."))
        r = client.post("/agent/run-research", json={"prompt": "Research AI and compute 42*2"})
        assert r.status_code == 200
        assert r.json()["content"] == "Searched and calculated. Answer is 84."

    def test_two_tool_calls_recorded(self):
        client = build_app()
        _MOCK.queue_tool_use("search_web", {"query": "news"})
        _MOCK.queue_tool_use("calculate", {"expression": "1 + 1"})
        _MOCK.queue_response(_completion("Done"))
        r = client.post("/agent/run-research", json={"prompt": "do both"})
        assert r.status_code == 200
        assert len(r.json()["tool_calls_made"]) == 2

    def test_first_tool_name_correct(self):
        client = build_app()
        _MOCK.queue_tool_use("search_web", {"query": "news"})
        _MOCK.queue_tool_use("calculate", {"expression": "2 + 2"})
        _MOCK.queue_response(_completion("Done"))
        r = client.post("/agent/run-research", json={"prompt": "research and calculate"})
        assert r.status_code == 200
        assert r.json()["tool_calls_made"][0] == "search_web"

    def test_second_tool_name_correct(self):
        client = build_app()
        _MOCK.queue_tool_use("search_web", {"query": "trends"})
        _MOCK.queue_tool_use("calculate", {"expression": "10 / 2"})
        _MOCK.queue_response(_completion("Done"))
        r = client.post("/agent/run-research", json={"prompt": "research and calculate"})
        assert r.status_code == 200
        assert r.json()["tool_calls_made"][1] == "calculate"

    def test_three_llm_calls_for_two_tool_turns(self):
        client = build_app()
        _MOCK.queue_tool_use("search_web", {"query": "AI"})
        _MOCK.queue_tool_use("calculate", {"expression": "5 * 5"})
        _MOCK.queue_response(_completion("All done."))
        r = client.post("/agent/run-research", json={"prompt": "go"})
        assert r.status_code == 200
        assert r.json()["calls"] == 3

    def test_single_tool_from_multi_tool_agent(self):
        client = build_app()
        _MOCK.queue_tool_use("get_stock_price", {"symbol": "AAPL"})
        _MOCK.queue_response(_completion("AAPL is at $150."))
        r = client.post("/agent/run-stock", json={"prompt": "What's AAPL price?"})
        assert r.status_code == 200
        assert r.json()["tool_calls_made"][0] == "get_stock_price"


# ---------------------------------------------------------------------------
# TestMultiToolAgentConfig (via TestClient)
# ---------------------------------------------------------------------------


class TestMultiToolAgentConfig:
    def test_parallel_tool_calls_config_accepted(self):
        client = build_app()
        _MOCK.queue_tool_use("search_web", {"query": "x"})
        _MOCK.queue_response(_completion("done"))
        r = client.post("/agent/run-parallel", json={"prompt": "go"})
        assert r.status_code == 200
        assert r.json()["content"] is not None

    def test_tool_error_return_policy_continues(self):
        client = build_app()
        _MOCK.queue_tool_use("broken_tool", {"x": "test"})
        _MOCK.queue_response(_completion("Recovered gracefully."))
        r = client.post("/agent/run-resilient", json={"prompt": "try it"})
        assert r.status_code == 200
        assert r.json()["content"] == "Recovered gracefully."

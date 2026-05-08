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
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str = "OK", *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
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


# ---------------------------------------------------------------------------
# TestMultiToolAttachment
# ---------------------------------------------------------------------------


class TestMultiToolAttachment:
    async def test_two_tools_sequential_calls(self):
        runner, mock = _make_runner_with_tools(search_web, calculate)

        @use_tools(search_web, calculate)
        @agent(model="mock-model")
        class ResearchAgent: ...

        mock.queue_tool_use("search_web", {"query": "AI trends"})
        mock.queue_tool_use("calculate", {"expression": "42 * 2"})
        mock.queue_response(_completion("Searched and calculated. Answer is 84."))

        resp = await runner.run(ResearchAgent(), "Research AI and compute 42*2")
        assert resp.content == "Searched and calculated. Answer is 84."

    async def test_two_tool_calls_recorded(self):
        runner, mock = _make_runner_with_tools(search_web, calculate)

        @use_tools(search_web, calculate)
        @agent(model="mock-model")
        class ResearchAgent: ...

        mock.queue_tool_use("search_web", {"query": "news"})
        mock.queue_tool_use("calculate", {"expression": "1 + 1"})
        mock.queue_response(_completion("Done"))

        resp = await runner.run(ResearchAgent(), "do both")
        assert len(resp.tool_calls_made) == 2

    async def test_first_tool_name_correct(self):
        runner, mock = _make_runner_with_tools(search_web, calculate)

        @use_tools(search_web, calculate)
        @agent(model="mock-model")
        class ResearchAgent: ...

        mock.queue_tool_use("search_web", {"query": "news"})
        mock.queue_tool_use("calculate", {"expression": "2 + 2"})
        mock.queue_response(_completion("Done"))

        resp = await runner.run(ResearchAgent(), "research and calculate")
        assert resp.tool_calls_made[0].name == "search_web"

    async def test_second_tool_name_correct(self):
        runner, mock = _make_runner_with_tools(search_web, calculate)

        @use_tools(search_web, calculate)
        @agent(model="mock-model")
        class ResearchAgent: ...

        mock.queue_tool_use("search_web", {"query": "trends"})
        mock.queue_tool_use("calculate", {"expression": "10 / 2"})
        mock.queue_response(_completion("Done"))

        resp = await runner.run(ResearchAgent(), "research and calculate")
        assert resp.tool_calls_made[1].name == "calculate"

    async def test_three_llm_calls_for_two_tool_turns(self):
        runner, mock = _make_runner_with_tools(search_web, calculate)

        @use_tools(search_web, calculate)
        @agent(model="mock-model")
        class ResearchAgent: ...

        mock.queue_tool_use("search_web", {"query": "AI"})
        mock.queue_tool_use("calculate", {"expression": "5 * 5"})
        mock.queue_response(_completion("All done."))

        await runner.run(ResearchAgent(), "go")
        assert len(mock.calls) == 3

    async def test_single_tool_from_multi_tool_agent(self):
        runner, mock = _make_runner_with_tools(search_web, calculate, get_stock_price)

        @use_tools(search_web, calculate, get_stock_price)
        @agent(model="mock-model")
        class FullAgent: ...

        mock.queue_tool_use("get_stock_price", {"symbol": "AAPL"})
        mock.queue_response(_completion("AAPL is at $150."))

        resp = await runner.run(FullAgent(), "What's AAPL price?")
        assert resp.tool_calls_made[0].name == "get_stock_price"


# ---------------------------------------------------------------------------
# TestMultiToolAgentConfig
# ---------------------------------------------------------------------------


class TestMultiToolAgentConfig:
    async def test_parallel_tool_calls_config_accepted(self):
        runner, mock = _make_runner_with_tools(search_web, calculate)

        @use_tools(search_web, calculate)
        @agent(model="mock-model", parallel_tool_calls=True)
        class ParallelAgent: ...

        mock.queue_tool_use("search_web", {"query": "x"})
        mock.queue_response(_completion("done"))

        # Should run without error; parallel config is accepted
        resp = await runner.run(ParallelAgent(), "go")
        assert resp is not None

    async def test_tool_error_return_policy_continues(self):
        from lauren_ai._tools import _add_to_tool_map

        @tool()
        async def broken_tool(x: str) -> dict:
            """Broken tool. Args: x: any string."""
            raise RuntimeError("always broken")

        tools = {}
        _add_to_tool_map(tools, search_web)
        _add_to_tool_map(tools, broken_tool)
        mock = MockTransport()
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools=tools, config=cfg)

        @use_tools(search_web, broken_tool)
        @agent(model="mock-model", tool_error_policy="return_error")
        class ResilientAgent: ...

        mock.queue_tool_use("broken_tool", {"x": "test"})
        mock.queue_response(_completion("Recovered gracefully."))

        resp = await runner.run(ResilientAgent(), "try it")
        assert resp.content == "Recovered gracefully."

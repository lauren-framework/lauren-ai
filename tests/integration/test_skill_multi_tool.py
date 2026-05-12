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

from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

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
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model")
@use_tools(search_web, calculate)
class ResearchAgent: ...


@agent(model="mock-model")
@use_tools(search_web, calculate, get_stock_price)
class FullAgent: ...


@agent(model="mock-model", parallel_tool_calls=True)
@use_tools(search_web, calculate)
class ParallelAgent: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(content: str = "OK", *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# TestMultiToolAttachment
# ---------------------------------------------------------------------------


class TestMultiToolAttachment:
    def test_two_tools_sequential_calls(self):
        client = TestClient(ResearchAgent())
        client.mock.queue_tool_use("search_web", {"query": "AI trends"})
        client.mock.queue_tool_use("calculate", {"expression": "42 * 2"})
        client.mock.queue_response(_c("Searched and calculated. Answer is 84."))
        result = client.run("Research AI and compute 42*2")
        assert result.content == "Searched and calculated. Answer is 84."

    def test_two_tool_calls_recorded(self):
        client = TestClient(ResearchAgent())
        client.mock.queue_tool_use("search_web", {"query": "news"})
        client.mock.queue_tool_use("calculate", {"expression": "1 + 1"})
        client.mock.queue_response(_c("Done"))
        result = client.run("do both")
        assert len(result.tool_calls_made) == 2

    def test_first_tool_name_correct(self):
        client = TestClient(ResearchAgent())
        client.mock.queue_tool_use("search_web", {"query": "news"})
        client.mock.queue_tool_use("calculate", {"expression": "2 + 2"})
        client.mock.queue_response(_c("Done"))
        result = client.run("research and calculate")
        assert result.tool_calls_made[0].name == "search_web"

    def test_second_tool_name_correct(self):
        client = TestClient(ResearchAgent())
        client.mock.queue_tool_use("search_web", {"query": "trends"})
        client.mock.queue_tool_use("calculate", {"expression": "10 / 2"})
        client.mock.queue_response(_c("Done"))
        result = client.run("research and calculate")
        assert result.tool_calls_made[1].name == "calculate"

    def test_three_llm_calls_for_two_tool_turns(self):
        client = TestClient(ResearchAgent())
        client.mock.queue_tool_use("search_web", {"query": "AI"})
        client.mock.queue_tool_use("calculate", {"expression": "5 * 5"})
        client.mock.queue_response(_c("All done."))
        client.run("go")
        assert len(client.calls) == 3

    def test_single_tool_from_multi_tool_agent(self):
        client = TestClient(FullAgent())
        client.mock.queue_tool_use("get_stock_price", {"symbol": "AAPL"})
        client.mock.queue_response(_c("AAPL is at $150."))
        result = client.run("What's AAPL price?")
        assert result.tool_calls_made[0].name == "get_stock_price"


# ---------------------------------------------------------------------------
# TestMultiToolAgentConfig
# ---------------------------------------------------------------------------


class TestMultiToolAgentConfig:
    def test_parallel_tool_calls_config_accepted(self):
        client = TestClient(ParallelAgent())
        client.mock.queue_tool_use("search_web", {"query": "x"})
        client.mock.queue_response(_c("done"))
        result = client.run("go")
        assert result.content is not None

    def test_tool_error_return_policy_continues(self):
        @tool()
        async def broken_tool(x: str) -> dict:
            """Broken tool. Args: x: any string."""
            raise RuntimeError("always broken")

        @agent(model="mock-model", tool_error_policy="return_error")
        @use_tools(search_web, broken_tool)
        class ResilientAgent: ...

        client = TestClient(ResilientAgent())
        client.mock.queue_tool_use("broken_tool", {"x": "test"})
        client.mock.queue_response(_c("Recovered gracefully."))
        result = client.run("try it")
        assert result.content == "Recovered gracefully."

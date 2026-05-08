"""Integration tests for the agent step-logging pattern (Skill 33).

Tests cover:
- on_start hook is called before the first LLM call
- on_turn_complete hook is called after each LLM call
- on_tool_result hook is called after tool execution
- on_finish hook is called with the final response
- Hooks can use a list-based log accumulator (no real logger needed)
- on_tool_result returning None leaves the result unchanged
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents import AgentContext, AgentResponse
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._tools import ToolResult
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(content: str = "Done", *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock: MockTransport, tools: dict | None = None) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools=tools or {}, config=cfg)


def _make_tool_map(*tool_funcs) -> dict:
    tools = {}
    for t in tool_funcs:
        m = getattr(t, TOOL_META)
        tools[m.name] = (t, m)
    return tools


# ---------------------------------------------------------------------------
# Tool definition (no from __future__ import annotations in this file)
# ---------------------------------------------------------------------------


@tool()
async def step_echo_tool(message: str) -> str:
    """Echo the message. Args: message: The message to echo."""
    return f"echo:{message}"


# ---------------------------------------------------------------------------
# Agents with lifecycle hooks
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="Step logging agent.")
class StepLoggingAgent:
    def __init__(self):
        self.log: list[str] = []

    async def on_start(self, ctx: AgentContext) -> None:
        self.log.append(f"on_start:turn={ctx.turn}")

    async def on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None:
        self.log.append(f"on_turn_complete:turn={ctx.turn}:stop={completion.stop_reason}")

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.log.append(f"on_tool_result:error={result.is_error}")
        return None

    async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
        self.log.append(f"on_finish:turns={response.turns}")


@agent(model="mock-model", system="Agent with no hooks.")
class NoHookAgent:
    pass


@agent(model="mock-model", system="Agent that modifies tool results.")
@use_tools(step_echo_tool)
class ResultModifyingAgent:
    def __init__(self):
        self.modified = False

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.modified = True
        # Return a modified result
        return ToolResult(
            tool_use_id=result.tool_use_id,
            content="modified_content",
            is_error=False,
        )


# ---------------------------------------------------------------------------
# Tests: on_start
# ---------------------------------------------------------------------------


class TestOnStartHook:
    @pytest.mark.asyncio
    async def test_on_start_called_once(self):
        """on_start is called exactly once before the first LLM call."""
        mock = MockTransport()
        runner = _make_runner(mock)
        mock.queue_response(_make_completion("Hello"))

        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Hi")

        start_events = [e for e in agent_instance.log if e.startswith("on_start")]
        assert len(start_events) == 1

    @pytest.mark.asyncio
    async def test_on_start_has_turn_zero(self):
        """on_start receives ctx.turn == 0."""
        mock = MockTransport()
        runner = _make_runner(mock)
        mock.queue_response(_make_completion("OK"))

        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Go")

        assert "on_start:turn=0" in agent_instance.log


# ---------------------------------------------------------------------------
# Tests: on_turn_complete
# ---------------------------------------------------------------------------


class TestOnTurnCompleteHook:
    @pytest.mark.asyncio
    async def test_on_turn_complete_called_after_llm(self):
        """on_turn_complete is called once for a single-turn run."""
        mock = MockTransport()
        runner = _make_runner(mock)
        mock.queue_response(_make_completion("Result"))

        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Test")

        tc_events = [e for e in agent_instance.log if e.startswith("on_turn_complete")]
        assert len(tc_events) == 1

    @pytest.mark.asyncio
    async def test_on_turn_complete_has_stop_reason(self):
        """on_turn_complete log entry includes the stop reason."""
        mock = MockTransport()
        runner = _make_runner(mock)
        mock.queue_response(_make_completion("Done"))

        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Run")

        assert any("stop=end_turn" in e for e in agent_instance.log)


# ---------------------------------------------------------------------------
# Tests: on_tool_result
# ---------------------------------------------------------------------------


class TestOnToolResultHook:
    @pytest.mark.asyncio
    async def test_on_tool_result_called_after_tool(self):
        """on_tool_result is called after a tool execution."""
        tools = _make_tool_map(step_echo_tool)
        mock = MockTransport()
        runner = _make_runner(mock, tools)
        mock.queue_tool_use("step_echo_tool", {"message": "ping"})
        mock.queue_response(_make_completion("echo done", id="c2"))

        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Echo ping")

        tool_events = [e for e in agent_instance.log if e.startswith("on_tool_result")]
        assert len(tool_events) == 1
        assert "error=False" in tool_events[0]

    @pytest.mark.asyncio
    async def test_on_tool_result_returning_none_leaves_result(self):
        """Returning None from on_tool_result does not replace the result."""
        tools = _make_tool_map(step_echo_tool)
        mock = MockTransport()
        runner = _make_runner(mock, tools)
        mock.queue_tool_use("step_echo_tool", {"message": "hello"})
        mock.queue_response(_make_completion("OK", id="c2"))

        agent_instance = StepLoggingAgent()
        response = await runner.run(agent_instance, "Echo hello")

        # Run succeeds — result was not interfered with
        assert response.turns == 2
        assert response.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Tests: on_finish
# ---------------------------------------------------------------------------


class TestOnFinishHook:
    @pytest.mark.asyncio
    async def test_on_finish_called_with_response(self):
        """on_finish is called once with the final AgentResponse."""
        mock = MockTransport()
        runner = _make_runner(mock)
        mock.queue_response(_make_completion("Goodbye"))

        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Finish")

        finish_events = [e for e in agent_instance.log if e.startswith("on_finish")]
        assert len(finish_events) == 1
        assert "turns=1" in finish_events[0]

    @pytest.mark.asyncio
    async def test_hook_ordering(self):
        """Hooks fire in order: on_start → on_turn_complete → on_finish."""
        mock = MockTransport()
        runner = _make_runner(mock)
        mock.queue_response(_make_completion("Result"))

        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Ordered run")

        log = agent_instance.log
        assert log.index(next(e for e in log if e.startswith("on_start"))) < \
               log.index(next(e for e in log if e.startswith("on_turn_complete")))
        assert log.index(next(e for e in log if e.startswith("on_turn_complete"))) < \
               log.index(next(e for e in log if e.startswith("on_finish")))


# ---------------------------------------------------------------------------
# Tests: agents without hooks
# ---------------------------------------------------------------------------


class TestNoHookAgent:
    @pytest.mark.asyncio
    async def test_agent_without_hooks_runs_normally(self):
        """An agent without lifecycle hooks completes successfully."""
        mock = MockTransport()
        runner = _make_runner(mock)
        mock.queue_response(_make_completion("No hooks"))

        agent_instance = NoHookAgent()
        response = await runner.run(agent_instance, "Hello")

        assert response.content == "No hooks"
        assert response.turns == 1

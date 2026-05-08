"""Integration tests for the agent step-logging pattern (Skill 33).

Tests cover:
- on_start hook is called before the first LLM call
- on_turn_complete hook is called after each LLM call
- on_tool_result hook is called after tool execution
- on_finish hook is called with the final response
- Hooks can use a list-based log accumulator (no real logger needed)
- on_tool_result returning None leaves the result unchanged
"""

from lauren_ai._agents import AgentContext, AgentResponse, agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import TOOL_META, ToolResult, tool
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


def _make_tool_map(*tool_funcs) -> dict:
    tools = {}
    for t in tool_funcs:
        m = getattr(t, TOOL_META)
        tools[m.name] = (t, m)
    return tools


def _make_runner(mock: MockTransport, tools: dict | None = None) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools=tools or {}, config=cfg)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@tool()
async def step_echo_tool(message: str) -> str:
    """Echo the message. Args: message: The message to echo."""
    return f"echo:{message}"


_TOOLS = _make_tool_map(step_echo_tool)


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


# ---------------------------------------------------------------------------
# Tests: on_start
# ---------------------------------------------------------------------------


class TestOnStartHook:
    async def test_on_start_called_once(self):
        """on_start is called exactly once before the first LLM call."""
        mock = MockTransport()
        mock.queue_response(_make_completion("Hello"))
        runner = _make_runner(mock)
        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Hi")
        start_events = [e for e in agent_instance.log if e.startswith("on_start")]
        assert len(start_events) == 1

    async def test_on_start_has_turn_zero(self):
        """on_start receives ctx.turn == 0."""
        mock = MockTransport()
        mock.queue_response(_make_completion("OK"))
        runner = _make_runner(mock)
        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Go")
        assert "on_start:turn=0" in agent_instance.log


# ---------------------------------------------------------------------------
# Tests: on_turn_complete
# ---------------------------------------------------------------------------


class TestOnTurnCompleteHook:
    async def test_on_turn_complete_called_after_llm(self):
        """on_turn_complete is called once for a single-turn run."""
        mock = MockTransport()
        mock.queue_response(_make_completion("Result"))
        runner = _make_runner(mock)
        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Test")
        tc_events = [e for e in agent_instance.log if e.startswith("on_turn_complete")]
        assert len(tc_events) == 1

    async def test_on_turn_complete_has_stop_reason(self):
        """on_turn_complete log entry includes the stop reason."""
        mock = MockTransport()
        mock.queue_response(_make_completion("Done"))
        runner = _make_runner(mock)
        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Run")
        assert any("stop=end_turn" in e for e in agent_instance.log)


# ---------------------------------------------------------------------------
# Tests: on_tool_result
# ---------------------------------------------------------------------------


class TestOnToolResultHook:
    async def test_on_tool_result_called_after_tool(self):
        """on_tool_result is called after a tool execution."""
        mock = MockTransport()
        mock.queue_tool_use("step_echo_tool", {"message": "ping"})
        mock.queue_response(_make_completion("echo done", id="c2"))
        runner = _make_runner(mock, tools=_TOOLS)
        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Echo ping")
        tool_events = [e for e in agent_instance.log if e.startswith("on_tool_result")]
        assert len(tool_events) == 1
        assert "error=False" in tool_events[0]

    async def test_on_tool_result_returning_none_leaves_result(self):
        """Returning None from on_tool_result does not replace the result."""
        mock = MockTransport()
        mock.queue_tool_use("step_echo_tool", {"message": "hello"})
        mock.queue_response(_make_completion("OK", id="c2"))
        runner = _make_runner(mock, tools=_TOOLS)
        agent_instance = StepLoggingAgent()
        resp = await runner.run(agent_instance, "Echo hello")
        assert resp.turns == 2
        assert resp.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Tests: on_finish
# ---------------------------------------------------------------------------


class TestOnFinishHook:
    async def test_on_finish_called_with_response(self):
        """on_finish is called once with the final AgentResponse."""
        mock = MockTransport()
        mock.queue_response(_make_completion("Goodbye"))
        runner = _make_runner(mock)
        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Finish")
        finish_events = [e for e in agent_instance.log if e.startswith("on_finish")]
        assert len(finish_events) == 1
        assert "turns=1" in finish_events[0]

    async def test_hook_ordering(self):
        """Hooks fire in order: on_start -> on_turn_complete -> on_finish."""
        mock = MockTransport()
        mock.queue_response(_make_completion("Result"))
        runner = _make_runner(mock)
        agent_instance = StepLoggingAgent()
        await runner.run(agent_instance, "Ordered run")
        log = agent_instance.log
        start_idx = log.index(next(e for e in log if e.startswith("on_start")))
        tc_idx = log.index(next(e for e in log if e.startswith("on_turn_complete")))
        finish_idx = log.index(next(e for e in log if e.startswith("on_finish")))
        assert start_idx < tc_idx < finish_idx


# ---------------------------------------------------------------------------
# Tests: agents without hooks
# ---------------------------------------------------------------------------


class TestNoHookAgent:
    async def test_agent_without_hooks_runs_normally(self):
        """An agent without lifecycle hooks completes successfully."""
        mock = MockTransport()
        mock.queue_response(_make_completion("No hooks"))
        runner = _make_runner(mock)
        agent_instance = NoHookAgent()
        resp = await runner.run(agent_instance, "Hello")
        assert resp.content == "No hooks"
        assert resp.turns == 1

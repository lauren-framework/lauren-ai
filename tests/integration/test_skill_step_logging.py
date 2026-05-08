"""Integration tests for the agent step-logging pattern (Skill 33).

Tests cover:
- on_start hook is called before the first LLM call
- on_turn_complete hook is called after each LLM call
- on_tool_result hook is called after tool execution
- on_finish hook is called with the final response
- Hooks can use a list-based log accumulator (no real logger needed)
- on_tool_result returning None leaves the result unchanged
"""

from __future__ import annotations

from lauren import LaurenFactory, controller, get, post, module, Json, use_value
from lauren.testing import TestClient
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


def _make_tool_map(*tool_funcs) -> dict:
    tools = {}
    for t in tool_funcs:
        m = getattr(t, TOOL_META)
        tools[m.name] = (t, m)
    return tools


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@tool()
async def step_echo_tool(message: str) -> str:
    """Echo the message. Args: message: The message to echo."""
    return f"echo:{message}"


# ---------------------------------------------------------------------------
# Agents with lifecycle hooks
# ---------------------------------------------------------------------------

# Module-level hook log so we can inspect from HTTP endpoints
_hook_log: list[str] = []


@agent(model="mock-model", system="Step logging agent.")
class StepLoggingAgent:
    def __init__(self):
        self.log: list[str] = []

    async def on_start(self, ctx: AgentContext) -> None:
        self.log.append(f"on_start:turn={ctx.turn}")
        _hook_log.append(f"on_start:turn={ctx.turn}")

    async def on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None:
        self.log.append(f"on_turn_complete:turn={ctx.turn}:stop={completion.stop_reason}")
        _hook_log.append(f"on_turn_complete:turn={ctx.turn}:stop={completion.stop_reason}")

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.log.append(f"on_tool_result:error={result.is_error}")
        _hook_log.append(f"on_tool_result:error={result.is_error}")
        return None

    async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
        self.log.append(f"on_finish:turns={response.turns}")
        _hook_log.append(f"on_finish:turns={response.turns}")


@agent(model="mock-model", system="Agent with no hooks.")
class NoHookAgent:
    pass


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()
_TOOLS = _make_tool_map(step_echo_tool)


# ---------------------------------------------------------------------------
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/agent")
class AgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner_plain = AgentRunner(transport=mock, tools={}, config=cfg)
        self._runner_tools = AgentRunner(transport=mock, tools=_TOOLS, config=cfg)

    @post("/run")
    async def run(self, body: Json[dict]) -> dict:
        agent_instance = StepLoggingAgent()
        resp = await self._runner_plain.run(agent_instance, body.get("prompt", "hi"))
        return {
            "log": agent_instance.log,
            "turns": resp.turns,
            "stop_reason": resp.stop_reason,
        }

    @post("/run-with-tool")
    async def run_with_tool(self, body: Json[dict]) -> dict:
        agent_instance = StepLoggingAgent()
        resp = await self._runner_tools.run(agent_instance, body.get("prompt", "hi"))
        return {
            "log": agent_instance.log,
            "turns": resp.turns,
            "stop_reason": resp.stop_reason,
        }

    @post("/run-no-hooks")
    async def run_no_hooks(self, body: Json[dict]) -> dict:
        agent_instance = NoHookAgent()
        resp = await self._runner_plain.run(agent_instance, body.get("prompt", "hi"))
        return {"content": resp.content, "turns": resp.turns}

    @get("/hook-log")
    async def hook_log(self) -> dict:
        return {"log": _hook_log}


@module(
    controllers=[AgentController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class StepLoggingModule: ...


def build_app(*completions: Completion) -> TestClient:
    _MOCK.reset()
    _hook_log.clear()
    for c in completions:
        _MOCK.queue_response(c)
    return TestClient(LaurenFactory.create(StepLoggingModule))


def build_app_with_tool(*completions) -> TestClient:
    _MOCK.reset()
    _hook_log.clear()
    for c in completions:
        if isinstance(c, tuple):
            _MOCK.queue_tool_use(*c)
        else:
            _MOCK.queue_response(c)
    return TestClient(LaurenFactory.create(StepLoggingModule))


# ---------------------------------------------------------------------------
# Tests: on_start
# ---------------------------------------------------------------------------


class TestOnStartHook:
    def test_on_start_called_once(self):
        """on_start is called exactly once before the first LLM call."""
        client = build_app(_make_completion("Hello"))
        r = client.post("/agent/run", json={"prompt": "Hi"})
        assert r.status_code == 200
        log = r.json()["log"]
        start_events = [e for e in log if e.startswith("on_start")]
        assert len(start_events) == 1

    def test_on_start_has_turn_zero(self):
        """on_start receives ctx.turn == 0."""
        client = build_app(_make_completion("OK"))
        r = client.post("/agent/run", json={"prompt": "Go"})
        assert r.status_code == 200
        log = r.json()["log"]
        assert "on_start:turn=0" in log


# ---------------------------------------------------------------------------
# Tests: on_turn_complete
# ---------------------------------------------------------------------------


class TestOnTurnCompleteHook:
    def test_on_turn_complete_called_after_llm(self):
        """on_turn_complete is called once for a single-turn run."""
        client = build_app(_make_completion("Result"))
        r = client.post("/agent/run", json={"prompt": "Test"})
        assert r.status_code == 200
        log = r.json()["log"]
        tc_events = [e for e in log if e.startswith("on_turn_complete")]
        assert len(tc_events) == 1

    def test_on_turn_complete_has_stop_reason(self):
        """on_turn_complete log entry includes the stop reason."""
        client = build_app(_make_completion("Done"))
        r = client.post("/agent/run", json={"prompt": "Run"})
        assert r.status_code == 200
        log = r.json()["log"]
        assert any("stop=end_turn" in e for e in log)


# ---------------------------------------------------------------------------
# Tests: on_tool_result
# ---------------------------------------------------------------------------


class TestOnToolResultHook:
    def test_on_tool_result_called_after_tool(self):
        """on_tool_result is called after a tool execution."""
        client = build_app_with_tool(
            ("step_echo_tool", {"message": "ping"}),
            _make_completion("echo done", id="c2"),
        )
        r = client.post("/agent/run-with-tool", json={"prompt": "Echo ping"})
        assert r.status_code == 200
        log = r.json()["log"]
        tool_events = [e for e in log if e.startswith("on_tool_result")]
        assert len(tool_events) == 1
        assert "error=False" in tool_events[0]

    def test_on_tool_result_returning_none_leaves_result(self):
        """Returning None from on_tool_result does not replace the result."""
        client = build_app_with_tool(
            ("step_echo_tool", {"message": "hello"}),
            _make_completion("OK", id="c2"),
        )
        r = client.post("/agent/run-with-tool", json={"prompt": "Echo hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["turns"] == 2
        assert data["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# Tests: on_finish
# ---------------------------------------------------------------------------


class TestOnFinishHook:
    def test_on_finish_called_with_response(self):
        """on_finish is called once with the final AgentResponse."""
        client = build_app(_make_completion("Goodbye"))
        r = client.post("/agent/run", json={"prompt": "Finish"})
        assert r.status_code == 200
        log = r.json()["log"]
        finish_events = [e for e in log if e.startswith("on_finish")]
        assert len(finish_events) == 1
        assert "turns=1" in finish_events[0]

    def test_hook_ordering(self):
        """Hooks fire in order: on_start → on_turn_complete → on_finish."""
        client = build_app(_make_completion("Result"))
        r = client.post("/agent/run", json={"prompt": "Ordered run"})
        assert r.status_code == 200
        log = r.json()["log"]
        start_idx = log.index(next(e for e in log if e.startswith("on_start")))
        tc_idx = log.index(next(e for e in log if e.startswith("on_turn_complete")))
        finish_idx = log.index(next(e for e in log if e.startswith("on_finish")))
        assert start_idx < tc_idx < finish_idx


# ---------------------------------------------------------------------------
# Tests: agents without hooks
# ---------------------------------------------------------------------------


class TestNoHookAgent:
    def test_agent_without_hooks_runs_normally(self):
        """An agent without lifecycle hooks completes successfully."""
        client = build_app(_make_completion("No hooks"))
        r = client.post("/agent/run-no-hooks", json={"prompt": "Hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "No hooks"
        assert data["turns"] == 1

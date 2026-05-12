"""Integration tests for Skill 10: Multi-Agent Delegation & Handoff.

Tests cover:
- Orchestrator calls a delegation tool that invokes SpecialistAgent
- Delegation tool result is fed back to orchestrator
- Orchestrator produces final answer after delegation
- Two turns: tool_call (delegate) → final answer
- Execution context passed through to specialist
- DelegateToSpecialist tool can be instantiated with specialist + runner

NOTE: No `from __future__ import annotations` — @tool() class used.
NOTE: Class-form tools must annotate ctx as ToolContext for context injection.
"""

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import TOOL_META, ToolContext, _add_to_tool_map, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient

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


def _make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, config=cfg)


# ---------------------------------------------------------------------------
# Specialist agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="You are a billing specialist.")
class BillingAgent: ...


@agent(model="mock-model", system="You are a technical support specialist.")
class TechSupportAgent: ...


# ---------------------------------------------------------------------------
# Delegation tool (class form — needs runner injected)
# NOTE: @tool() class form — no from __future__ import annotations
# ---------------------------------------------------------------------------


@tool()
class DelegateToBilling:
    """Delegate a billing task to the BillingAgent.

    Args:
        task: The billing task description.
    """

    def __init__(self, specialist_runner: AgentRunner) -> None:
        self._runner = specialist_runner
        self._specialist = BillingAgent()

    async def run(self, ctx: ToolContext, task: str) -> dict:
        response = await self._runner.run(self._specialist, task)
        return {"result": response.content}


@tool()
class DelegateToTechSupport:
    """Delegate a technical support task to the TechSupportAgent.

    Args:
        task: The technical support task description.
    """

    def __init__(self, specialist_runner: AgentRunner) -> None:
        self._runner = specialist_runner
        self._specialist = TechSupportAgent()

    async def run(self, ctx: ToolContext, task: str) -> dict:
        response = await self._runner.run(self._specialist, task)
        return {"result": response.content}


# ---------------------------------------------------------------------------
# Helper to build orchestrator TestClient with delegation tool
# ---------------------------------------------------------------------------


def build_orchestrator(
    orch_mock: MockTransport,
    spec_mock: MockTransport,
    *,
    specialist: str = "billing",
) -> TestClient:
    """Build an orchestrator TestClient wired to a specialist via a delegation tool."""
    spec_runner = _make_runner(spec_mock)
    orch_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")

    if specialist == "billing":
        delegate_tool = DelegateToBilling(specialist_runner=spec_runner)
        orch_tools: dict = {}
        _add_to_tool_map(orch_tools, DelegateToBilling, instance=delegate_tool)

        @agent(model="mock-model", system="Route to specialists.")
        @use_tools(DelegateToBilling)
        class OrchestratorAgent: ...

        OrchestratorAgent.__lauren_ai_agent__.tools = orch_tools
        orch_runner = AgentRunner(transport=orch_mock, config=orch_cfg)
        return TestClient(OrchestratorAgent(), runner=orch_runner)
    else:
        delegate_tool = DelegateToTechSupport(specialist_runner=spec_runner)
        orch_tools = {}
        _add_to_tool_map(orch_tools, DelegateToTechSupport, instance=delegate_tool)

        @agent(model="mock-model")
        @use_tools(DelegateToTechSupport)
        class TechOrchestratorAgent: ...

        TechOrchestratorAgent.__lauren_ai_agent__.tools = orch_tools
        orch_runner = AgentRunner(transport=orch_mock, config=orch_cfg)
        return TestClient(TechOrchestratorAgent(), runner=orch_runner)


# ---------------------------------------------------------------------------
# TestDelegationToolInstantiation
# ---------------------------------------------------------------------------


class TestDelegationToolInstantiation:
    def test_delegate_to_billing_has_tool_meta(self):
        assert hasattr(DelegateToBilling, TOOL_META)

    def test_delegate_to_tech_support_has_tool_meta(self):
        assert hasattr(DelegateToTechSupport, TOOL_META)

    def test_delegate_tool_instantiates_with_runner(self):
        mock = MockTransport()
        runner = _make_runner(mock)
        delegate = DelegateToBilling(specialist_runner=runner)
        assert delegate is not None


# ---------------------------------------------------------------------------
# TestOrchestratorDelegation
# ---------------------------------------------------------------------------


class TestOrchestratorDelegation:
    async def test_orchestrator_delegates_and_gets_result(self):
        orch_mock = MockTransport()
        spec_mock = MockTransport()
        orch_mock.queue_tool_use("delegate_to_billing", {"task": "Process refund for order #42"})
        orch_mock.queue_response(_c("Your refund has been processed."))
        spec_mock.queue_response(_c("Refund processed for order #42."))

        client = build_orchestrator(orch_mock, spec_mock, specialist="billing")
        result = await client.run_async("Process refund for order #42")
        assert result.content == "Your refund has been processed."

    async def test_orchestrator_two_llm_calls(self):
        orch_mock = MockTransport()
        spec_mock = MockTransport()
        orch_mock.queue_tool_use("delegate_to_tech_support", {"task": "Fix login bug"})
        orch_mock.queue_response(_c("Issue has been resolved."))
        spec_mock.queue_response(_c("Tech issue resolved."))

        client = build_orchestrator(orch_mock, spec_mock, specialist="tech")
        await client.run_async("Fix login bug")
        assert len(orch_mock.calls) == 2

    async def test_specialist_receives_task_message(self):
        orch_mock = MockTransport()
        spec_mock = MockTransport()
        orch_mock.queue_tool_use("delegate_to_billing", {"task": "Handle refund for order X"})
        orch_mock.queue_response(_c("Done."))
        spec_mock.queue_response(_c("Handled."))

        client = build_orchestrator(orch_mock, spec_mock, specialist="billing")
        await client.run_async("Handle refund for order X")
        assert len(spec_mock.calls) == 1
        specialist_messages_str = str(spec_mock.calls[0].messages)
        assert "Handle refund for order X" in specialist_messages_str

    async def test_delegation_result_in_tool_calls_made(self):
        orch_mock = MockTransport()
        spec_mock = MockTransport()
        orch_mock.queue_tool_use("delegate_to_billing", {"task": "Refund order 9"})
        orch_mock.queue_response(_c("Refund done."))
        spec_mock.queue_response(_c("Billing done."))

        client = build_orchestrator(orch_mock, spec_mock, specialist="billing")
        result = await client.run_async("Refund order 9")
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0].name == "delegate_to_billing"

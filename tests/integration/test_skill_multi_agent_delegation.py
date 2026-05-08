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

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import ToolContext, _add_to_tool_map, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


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
# TestDelegationToolInstantiation
# ---------------------------------------------------------------------------


class TestDelegationToolInstantiation:
    def test_delegate_to_billing_has_tool_meta(self):
        from lauren_ai._tools import TOOL_META
        assert hasattr(DelegateToBilling, TOOL_META)

    def test_delegate_to_tech_support_has_tool_meta(self):
        from lauren_ai._tools import TOOL_META
        assert hasattr(DelegateToTechSupport, TOOL_META)

    def test_delegate_tool_instantiates_with_runner(self):
        mock = MockTransport()
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools={}, config=cfg)
        delegate = DelegateToBilling(specialist_runner=runner)
        assert delegate is not None


# ---------------------------------------------------------------------------
# TestOrchestratorDelegation (two separate mocks: one for orchestrator, one for specialist)
# ---------------------------------------------------------------------------


class TestOrchestratorDelegation:
    async def test_orchestrator_delegates_and_gets_result(self):
        """
        Two separate mocks: one for the orchestrator runner, one for the specialist runner.
        The delegation tool uses the specialist runner internally.
        Note: @tool() class name DelegateToBilling is stored as 'delegate_to_billing' (snake_case).
        """
        # Specialist mock: queue the billing specialist response
        specialist_mock = MockTransport()
        specialist_mock.queue_response(_completion("Refund processed for order #42.", n=1))

        specialist_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        specialist_runner = AgentRunner(transport=specialist_mock, tools={}, config=specialist_cfg)

        # Build delegation tool instance
        delegate_tool = DelegateToBilling(specialist_runner=specialist_runner)

        # Orchestrator mock: tool_call with snake_case name → final answer
        orch_mock = MockTransport()
        orch_mock.queue_tool_use("delegate_to_billing", {"task": "Process refund for order #42"})
        orch_mock.queue_response(_completion("Your refund has been processed.", n=2))

        tools = {}
        _add_to_tool_map(tools, DelegateToBilling, instance=delegate_tool)
        orch_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        orch_runner = AgentRunner(transport=orch_mock, tools=tools, config=orch_cfg)

        @use_tools(DelegateToBilling)
        @agent(model="mock-model", system="Route to specialists.")
        class OrchestratorAgent: ...

        resp = await orch_runner.run(OrchestratorAgent(), "Process refund for order #42")
        assert resp.content == "Your refund has been processed."

    async def test_orchestrator_two_llm_calls(self):
        specialist_mock = MockTransport()
        specialist_mock.queue_response(_completion("Tech issue resolved.", n=1))

        specialist_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        specialist_runner = AgentRunner(transport=specialist_mock, tools={}, config=specialist_cfg)

        delegate_tool = DelegateToTechSupport(specialist_runner=specialist_runner)

        orch_mock = MockTransport()
        # Tool name is snake_case: delegate_to_tech_support
        orch_mock.queue_tool_use("delegate_to_tech_support", {"task": "Fix login bug"})
        orch_mock.queue_response(_completion("Issue has been resolved.", n=2))

        tools = {}
        _add_to_tool_map(tools, DelegateToTechSupport, instance=delegate_tool)
        orch_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        orch_runner = AgentRunner(transport=orch_mock, tools=tools, config=orch_cfg)

        @use_tools(DelegateToTechSupport)
        @agent(model="mock-model")
        class OrchestratorAgent: ...

        await orch_runner.run(OrchestratorAgent(), "Fix login bug")
        # Orchestrator: tool_use call + final answer call = 2
        assert len(orch_mock.calls) == 2

    async def test_specialist_receives_task_message(self):
        specialist_mock = MockTransport()
        specialist_mock.queue_response(_completion("Handled.", n=1))

        specialist_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        specialist_runner = AgentRunner(transport=specialist_mock, tools={}, config=specialist_cfg)

        delegate_tool = DelegateToBilling(specialist_runner=specialist_runner)

        orch_mock = MockTransport()
        orch_mock.queue_tool_use("delegate_to_billing", {"task": "Handle refund for order X"})
        orch_mock.queue_response(_completion("Done.", n=2))

        tools = {}
        _add_to_tool_map(tools, DelegateToBilling, instance=delegate_tool)
        orch_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        orch_runner = AgentRunner(transport=orch_mock, tools=tools, config=orch_cfg)

        @use_tools(DelegateToBilling)
        @agent(model="mock-model")
        class OrchestratorAgent: ...

        await orch_runner.run(OrchestratorAgent(), "Handle refund for order X")

        # Specialist mock should have received 1 call
        assert len(specialist_mock.calls) == 1
        specialist_messages = specialist_mock.calls[0].messages
        # The task should appear in the user message
        user_contents = [
            m["content"] for m in specialist_messages
            if isinstance(m.get("content"), str) and m.get("role") == "user"
        ]
        assert any("Handle refund for order X" in c for c in user_contents)

    async def test_delegation_result_in_tool_calls_made(self):
        specialist_mock = MockTransport()
        specialist_mock.queue_response(_completion("Billing done.", n=1))

        specialist_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        specialist_runner = AgentRunner(transport=specialist_mock, tools={}, config=specialist_cfg)
        delegate_tool = DelegateToBilling(specialist_runner=specialist_runner)

        orch_mock = MockTransport()
        orch_mock.queue_tool_use("delegate_to_billing", {"task": "Refund order 9"})
        orch_mock.queue_response(_completion("Refund done.", n=2))

        tools = {}
        _add_to_tool_map(tools, DelegateToBilling, instance=delegate_tool)
        orch_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        orch_runner = AgentRunner(transport=orch_mock, tools=tools, config=orch_cfg)

        @use_tools(DelegateToBilling)
        @agent(model="mock-model")
        class OrchestratorAgent: ...

        resp = await orch_runner.run(OrchestratorAgent(), "Refund order 9")
        assert len(resp.tool_calls_made) == 1
        assert resp.tool_calls_made[0].name == "delegate_to_billing"

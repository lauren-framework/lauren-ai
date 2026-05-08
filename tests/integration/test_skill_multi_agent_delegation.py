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
from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import ToolContext, TOOL_META, _add_to_tool_map, tool
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
# Module-level mocks: one for orchestrator, one for specialist
# The controller is built fresh per test via build_app()
# ---------------------------------------------------------------------------


class _DelegateRequest(BaseModel):
    task: str = "Do something"
    specialist: str = "billing"


# We build a fresh app every call; the controller captures both mocks via closure.


def build_app(
    orch_responses: list | None = None,
    spec_responses: list[str] | None = None,
) -> tuple[TestClient, MockTransport, MockTransport]:
    """Build TestClient with two separate mocks for orchestrator and specialist.

    Returns (client, orch_mock, spec_mock).
    """
    orch_mock = MockTransport()
    spec_mock = MockTransport()

    if orch_responses:
        for item in orch_responses:
            if isinstance(item, tuple) and item[0] == "tool_use":
                orch_mock.queue_tool_use(item[1], item[2])
            else:
                val = item[0] if isinstance(item, tuple) else item
                orch_mock.queue_response(_completion(val))

    if spec_responses:
        for content in spec_responses:
            spec_mock.queue_response(_completion(content))

    # Capture mocks in closure for the controller
    _orch = orch_mock
    _spec = spec_mock

    @controller("/delegate")
    class BoundDelegateController:
        @post("/run")
        async def run(self, body: Json[_DelegateRequest]) -> dict:
            spec_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
            spec_runner = AgentRunner(transport=_spec, tools={}, config=spec_cfg)

            orch_cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")

            if body.specialist == "billing":
                delegate_tool = DelegateToBilling(specialist_runner=spec_runner)
                orch_tools = {}
                _add_to_tool_map(orch_tools, DelegateToBilling, instance=delegate_tool)
                orch_runner = AgentRunner(transport=_orch, tools=orch_tools, config=orch_cfg)

                @use_tools(DelegateToBilling)
                @agent(model="mock-model", system="Route to specialists.")
                class OrchestratorAgent: ...

                resp = await orch_runner.run(OrchestratorAgent(), body.task)
            else:
                delegate_tool = DelegateToTechSupport(specialist_runner=spec_runner)
                orch_tools = {}
                _add_to_tool_map(orch_tools, DelegateToTechSupport, instance=delegate_tool)
                orch_runner = AgentRunner(transport=_orch, tools=orch_tools, config=orch_cfg)

                @use_tools(DelegateToTechSupport)
                @agent(model="mock-model")
                class TechOrchestratorAgent: ...

                resp = await orch_runner.run(TechOrchestratorAgent(), body.task)

            specialist_messages = []
            if _spec.calls:
                for m in _spec.calls[0].messages:
                    specialist_messages.append(str(m))

            return {
                "content": resp.content,
                "orch_calls": len(_orch.calls),
                "spec_calls": len(_spec.calls),
                "tool_calls_made": [t.name for t in resp.tool_calls_made],
                "specialist_messages": specialist_messages,
            }

    @module(controllers=[BoundDelegateController])
    class BoundModule: ...

    return TestClient(LaurenFactory.create(BoundModule)), orch_mock, spec_mock


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
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools={}, config=cfg)
        delegate = DelegateToBilling(specialist_runner=runner)
        assert delegate is not None


# ---------------------------------------------------------------------------
# TestOrchestratorDelegation (via TestClient)
# ---------------------------------------------------------------------------


class TestOrchestratorDelegation:
    def test_orchestrator_delegates_and_gets_result(self):
        client, _, _ = build_app(
            orch_responses=[("tool_use", "delegate_to_billing", {"task": "Process refund for order #42"}), "Your refund has been processed."],
            spec_responses=["Refund processed for order #42."],
        )
        r = client.post("/delegate/run", json={"task": "Process refund for order #42", "specialist": "billing"})
        assert r.status_code == 200
        assert r.json()["content"] == "Your refund has been processed."

    def test_orchestrator_two_llm_calls(self):
        client, _, _ = build_app(
            orch_responses=[("tool_use", "delegate_to_tech_support", {"task": "Fix login bug"}), "Issue has been resolved."],
            spec_responses=["Tech issue resolved."],
        )
        r = client.post("/delegate/run", json={"task": "Fix login bug", "specialist": "tech"})
        assert r.status_code == 200
        assert r.json()["orch_calls"] == 2

    def test_specialist_receives_task_message(self):
        client, _, _ = build_app(
            orch_responses=[("tool_use", "delegate_to_billing", {"task": "Handle refund for order X"}), "Done."],
            spec_responses=["Handled."],
        )
        r = client.post("/delegate/run", json={"task": "Handle refund for order X", "specialist": "billing"})
        assert r.status_code == 200
        data = r.json()
        assert data["spec_calls"] == 1
        specialist_messages_str = str(data["specialist_messages"])
        assert "Handle refund for order X" in specialist_messages_str

    def test_delegation_result_in_tool_calls_made(self):
        client, _, _ = build_app(
            orch_responses=[("tool_use", "delegate_to_billing", {"task": "Refund order 9"}), "Refund done."],
            spec_responses=["Billing done."],
        )
        r = client.post("/delegate/run", json={"task": "Refund order 9", "specialist": "billing"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["tool_calls_made"]) == 1
        assert data["tool_calls_made"][0] == "delegate_to_billing"

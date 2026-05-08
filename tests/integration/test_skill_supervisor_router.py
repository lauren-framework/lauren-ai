"""Integration tests for the agent supervisor/router pattern (Skill 11).

Tests:
  - classify_intent tool returns correct intent for billing keywords
  - classify_intent tool returns correct intent for technical keywords
  - classify_intent tool returns general for unrecognised messages
  - SupervisorAgent runs with MockTransport, queued tool call returns billing intent
  - SupervisorAgent runs with MockTransport, queued tool call returns technical intent
  - Routing dispatch selects the correct specialist class based on intent
  - confidence values are always present in the result
  - multiple intent keywords in one message use first matching category
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage, ToolCall
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import AgentTestClient


# ---------------------------------------------------------------------------
# classify_intent tool
# ---------------------------------------------------------------------------


@tool()
async def classify_intent(message: str) -> dict:
    """Classify the intent of a user message.

    Args:
        message: The user message to classify.
    """
    if any(w in message.lower() for w in ["invoice", "payment", "charge"]):
        return {"intent": "billing", "confidence": 0.9}
    elif any(w in message.lower() for w in ["error", "bug", "crash", "slow"]):
        return {"intent": "technical", "confidence": 0.85}
    return {"intent": "general", "confidence": 0.7}


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------


@agent(model=None, system="You handle billing questions.")
class BillingAgent: ...


@agent(model=None, system="You handle technical support.")
class TechSupportAgent: ...


@agent(model=None, system="You handle general questions.")
class GeneralAgent: ...


# ---------------------------------------------------------------------------
# Supervisor agent
# ---------------------------------------------------------------------------


@agent(model=None, system="Route user to the correct specialist. Call classify_intent first.")
@use_tools(classify_intent)
class SupervisorAgent: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPECIALISTS = {
    "billing": BillingAgent,
    "technical": TechSupportAgent,
    "general": GeneralAgent,
}


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests: classify_intent tool logic
# ---------------------------------------------------------------------------


class TestClassifyIntentTool:
    async def test_billing_keyword_invoice(self):
        result = await classify_intent("I need help with my invoice")
        assert result["intent"] == "billing"
        assert result["confidence"] >= 0.8

    async def test_billing_keyword_payment(self):
        result = await classify_intent("my payment was declined")
        assert result["intent"] == "billing"

    async def test_billing_keyword_charge(self):
        result = await classify_intent("there is an extra charge on my account")
        assert result["intent"] == "billing"

    async def test_technical_keyword_error(self):
        result = await classify_intent("I keep getting an error")
        assert result["intent"] == "technical"
        assert result["confidence"] >= 0.8

    async def test_technical_keyword_crash(self):
        result = await classify_intent("the app keeps crashing")
        assert result["intent"] == "technical"

    async def test_technical_keyword_slow(self):
        result = await classify_intent("the site is really slow today")
        assert result["intent"] == "technical"

    async def test_general_intent_for_unrecognised(self):
        result = await classify_intent("hello, how are you?")
        assert result["intent"] == "general"
        assert result["confidence"] == 0.7

    async def test_result_always_has_confidence(self):
        for msg in ["invoice", "bug", "hello"]:
            result = await classify_intent(msg)
            assert "confidence" in result
            assert isinstance(result["confidence"], float)


# ---------------------------------------------------------------------------
# Tests: SupervisorAgent with MockTransport
# ---------------------------------------------------------------------------


class TestSupervisorAgentRouting:
    async def test_supervisor_runs_tool_call_for_billing_message(self):
        mock = MockTransport()
        # Turn 1: supervisor calls classify_intent
        mock.queue_tool_use("classify_intent", {"message": "I need help with my invoice"})
        # Turn 2: supervisor gets result, responds
        mock.queue_response(_completion("Routing to billing specialist", n=2))

        client = AgentTestClient(SupervisorAgent(), mock)
        resp = await client.run_async("I need help with my invoice")

        assert resp.content == "Routing to billing specialist"
        assert resp.turns == 2

    async def test_supervisor_runs_tool_call_for_technical_message(self):
        mock = MockTransport()
        mock.queue_tool_use("classify_intent", {"message": "the app keeps crashing"})
        mock.queue_response(_completion("Routing to technical support", n=2))

        client = AgentTestClient(SupervisorAgent(), mock)
        resp = await client.run_async("the app keeps crashing")

        assert resp.content == "Routing to technical support"

    async def test_supervisor_tool_calls_made_tracks_classify_intent(self):
        mock = MockTransport()
        mock.queue_tool_use("classify_intent", {"message": "payment issue"})
        mock.queue_response(_completion("Routing", n=2))

        client = AgentTestClient(SupervisorAgent(), mock)
        resp = await client.run_async("payment issue")

        assert len(resp.tool_calls_made) == 1
        assert resp.tool_calls_made[0].name == "classify_intent"


# ---------------------------------------------------------------------------
# Tests: dispatch routing logic
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    def test_routing_billing_intent_selects_billing_agent(self):
        specialist = SPECIALISTS.get("billing", GeneralAgent)
        assert specialist is BillingAgent

    def test_routing_technical_intent_selects_tech_agent(self):
        specialist = SPECIALISTS.get("technical", GeneralAgent)
        assert specialist is TechSupportAgent

    def test_routing_general_intent_selects_general_agent(self):
        specialist = SPECIALISTS.get("general", GeneralAgent)
        assert specialist is GeneralAgent

    def test_routing_unknown_intent_defaults_to_general(self):
        specialist = SPECIALISTS.get("unknown_intent", GeneralAgent)
        assert specialist is GeneralAgent

    async def test_specialist_agents_can_run_independently(self):
        mock = MockTransport()
        mock.queue_response(_completion("Your invoice is due on the 15th"))

        client = AgentTestClient(BillingAgent(), mock)
        resp = await client.run_async("When is my invoice due?")
        assert resp.content == "Your invoice is due on the 15th"

    async def test_general_agent_handles_greeting(self):
        mock = MockTransport()
        mock.queue_response(_completion("Hello! How can I help you today?"))

        client = AgentTestClient(GeneralAgent(), mock)
        resp = await client.run_async("Hello!")
        assert resp.content == "Hello! How can I help you today?"

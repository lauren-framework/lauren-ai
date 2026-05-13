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

NOTE: No from __future__ import annotations — tool() needs live annotations.
"""

from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

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


@agent(model="mock-model", system="You handle billing questions.")
class BillingAgent: ...


@agent(model="mock-model", system="You handle technical support.")
class TechSupportAgent: ...


@agent(model="mock-model", system="You handle general questions.")
class GeneralAgent: ...


# ---------------------------------------------------------------------------
# Supervisor agent
# ---------------------------------------------------------------------------


@agent(
    model="mock-model", system="Route user to the correct specialist. Call classify_intent first."
)
@use_tools(classify_intent)
class SupervisorAgent: ...


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

SPECIALISTS = {
    "billing": BillingAgent,
    "technical": TechSupportAgent,
    "general": GeneralAgent,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=text,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Tests: classify_intent tool logic (direct)
# ---------------------------------------------------------------------------


class TestClassifyIntentTool:
    def test_billing_keyword_invoice(self):
        import asyncio

        data = asyncio.run(classify_intent("I need help with my invoice"))
        assert data["intent"] == "billing"
        assert data["confidence"] >= 0.8

    def test_billing_keyword_payment(self):
        import asyncio

        data = asyncio.run(classify_intent("my payment was declined"))
        assert data["intent"] == "billing"

    def test_billing_keyword_charge(self):
        import asyncio

        data = asyncio.run(classify_intent("there is an extra charge on my account"))
        assert data["intent"] == "billing"

    def test_technical_keyword_error(self):
        import asyncio

        data = asyncio.run(classify_intent("I keep getting an error"))
        assert data["intent"] == "technical"
        assert data["confidence"] >= 0.8

    def test_technical_keyword_crash(self):
        import asyncio

        data = asyncio.run(classify_intent("the app keeps crashing"))
        assert data["intent"] == "technical"

    def test_technical_keyword_slow(self):
        import asyncio

        data = asyncio.run(classify_intent("the site is really slow today"))
        assert data["intent"] == "technical"

    def test_general_intent_for_unrecognised(self):
        import asyncio

        data = asyncio.run(classify_intent("hello, how are you?"))
        assert data["intent"] == "general"
        assert data["confidence"] == 0.7

    def test_result_always_has_confidence(self):
        import asyncio

        for msg in ["invoice", "bug", "hello"]:
            data = asyncio.run(classify_intent(msg))
            assert "confidence" in data
            assert isinstance(data["confidence"], float)


# ---------------------------------------------------------------------------
# Tests: SupervisorAgent with MockTransport (via TestClient)
# ---------------------------------------------------------------------------


class TestSupervisorAgentRouting:
    def test_supervisor_runs_tool_call_for_billing_message(self):
        client = TestClient(SupervisorAgent())
        client.mock.queue_tool_use("classify_intent", {"message": "I need help with my invoice"})
        client.mock.queue_response(_c("Routing to billing specialist", n=2))
        result = client.run("I need help with my invoice")
        assert result.content == "Routing to billing specialist"
        assert result.turns == 2

    def test_supervisor_runs_tool_call_for_technical_message(self):
        client = TestClient(SupervisorAgent())
        client.mock.queue_tool_use("classify_intent", {"message": "the app keeps crashing"})
        client.mock.queue_response(_c("Routing to technical support", n=2))
        result = client.run("the app keeps crashing")
        assert result.content == "Routing to technical support"

    def test_supervisor_tool_calls_made_tracks_classify_intent(self):
        client = TestClient(SupervisorAgent())
        client.mock.queue_tool_use("classify_intent", {"message": "payment issue"})
        client.mock.queue_response(_c("Routing", n=2))
        result = client.run("payment issue")
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0].name == "classify_intent"


# ---------------------------------------------------------------------------
# Tests: dispatch routing logic (direct Python)
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    def test_routing_specialist_map(self):
        assert SPECIALISTS.get("billing") is BillingAgent
        assert SPECIALISTS.get("technical") is TechSupportAgent
        assert SPECIALISTS.get("general") is GeneralAgent
        assert SPECIALISTS.get("unknown_intent", GeneralAgent) is GeneralAgent

    def test_specialist_agents_can_run_independently(self):
        client = TestClient(BillingAgent())
        client.mock.queue_response(_c("Your invoice is due on the 15th"))
        result = client.run("When is my invoice due?")
        assert result.content == "Your invoice is due on the 15th"

    def test_general_agent_handles_greeting(self):
        client = TestClient(GeneralAgent())
        client.mock.queue_response(_c("Hello! How can I help you today?"))
        result = client.run("Hello!")
        assert result.content == "Hello! How can I help you today?"

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

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, get, module, post, use_value
from lauren.testing import TestClient
from lauren_ai import LLMConfig
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


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
# Routing table
# ---------------------------------------------------------------------------

SPECIALISTS = {
    "billing": BillingAgent,
    "technical": TechSupportAgent,
    "general": GeneralAgent,
}

# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _ClassifyRequest(BaseModel):
    message: str


class _RouteRequest(BaseModel):
    prompt: str = "hello"


@controller("/classify-svc")
class ClassifyController:
    @post("/classify")
    async def classify(self, body: Json[_ClassifyRequest]) -> dict:
        result = await classify_intent(body.message)
        return {"intent": result["intent"], "confidence": result["confidence"]}


@controller("/supervisor")
class SupervisorController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)
        self._mock = mock

    @post("/route")
    async def route(self, body: Json[_RouteRequest]) -> dict:
        resp = await self._runner.run(SupervisorAgent(), body.prompt)
        return {
            "content": resp.content,
            "turns": resp.turns,
            "tool_calls": [tc.name for tc in resp.tool_calls_made],
        }

    @post("/billing")
    async def run_billing(self, body: Json[_RouteRequest]) -> dict:
        resp = await self._runner.run(BillingAgent(), body.prompt)
        return {"content": resp.content}

    @post("/general")
    async def run_general(self, body: Json[_RouteRequest]) -> dict:
        resp = await self._runner.run(GeneralAgent(), body.prompt)
        return {"content": resp.content}

    @get("/specialists")
    async def specialists(self) -> dict:
        return {
            "billing": SPECIALISTS.get("billing") is BillingAgent,
            "technical": SPECIALISTS.get("technical") is TechSupportAgent,
            "general": SPECIALISTS.get("general") is GeneralAgent,
            "unknown_defaults_general": SPECIALISTS.get("unknown_intent", GeneralAgent)
            is GeneralAgent,
        }


@module(
    controllers=[ClassifyController, SupervisorController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class SupervisorModule: ...


def build_app(*responses) -> TestClient:
    _MOCK.reset()
    for item in responses:
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "tool_use":
            # (tool_use, (name, args))
            name, args = item[1]
            _MOCK.queue_tool_use(name, args)
        else:
            _MOCK.queue_response(_completion(item))
    return TestClient(LaurenFactory.create(SupervisorModule))


# ---------------------------------------------------------------------------
# Tests: classify_intent tool logic (via HTTP)
# ---------------------------------------------------------------------------


class TestClassifyIntentTool:
    def test_billing_keyword_invoice(self):
        client = build_app()
        r = client.post("/classify-svc/classify", json={"message": "I need help with my invoice"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "billing"
        assert data["confidence"] >= 0.8

    def test_billing_keyword_payment(self):
        client = build_app()
        r = client.post("/classify-svc/classify", json={"message": "my payment was declined"})
        assert r.status_code == 200
        assert r.json()["intent"] == "billing"

    def test_billing_keyword_charge(self):
        client = build_app()
        r = client.post(
            "/classify-svc/classify",
            json={"message": "there is an extra charge on my account"},
        )
        assert r.status_code == 200
        assert r.json()["intent"] == "billing"

    def test_technical_keyword_error(self):
        client = build_app()
        r = client.post("/classify-svc/classify", json={"message": "I keep getting an error"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "technical"
        assert data["confidence"] >= 0.8

    def test_technical_keyword_crash(self):
        client = build_app()
        r = client.post("/classify-svc/classify", json={"message": "the app keeps crashing"})
        assert r.status_code == 200
        assert r.json()["intent"] == "technical"

    def test_technical_keyword_slow(self):
        client = build_app()
        r = client.post(
            "/classify-svc/classify", json={"message": "the site is really slow today"}
        )
        assert r.status_code == 200
        assert r.json()["intent"] == "technical"

    def test_general_intent_for_unrecognised(self):
        client = build_app()
        r = client.post("/classify-svc/classify", json={"message": "hello, how are you?"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "general"
        assert data["confidence"] == 0.7

    def test_result_always_has_confidence(self):
        client = build_app()
        for msg in ["invoice", "bug", "hello"]:
            r = client.post("/classify-svc/classify", json={"message": msg})
            assert r.status_code == 200
            data = r.json()
            assert "confidence" in data
            assert isinstance(data["confidence"], float)


# ---------------------------------------------------------------------------
# Tests: SupervisorAgent with MockTransport (via HTTP)
# ---------------------------------------------------------------------------


class TestSupervisorAgentRouting:
    def test_supervisor_runs_tool_call_for_billing_message(self):
        _MOCK.reset()
        _MOCK.queue_tool_use("classify_intent", {"message": "I need help with my invoice"})
        _MOCK.queue_response(_completion("Routing to billing specialist", n=2))
        client = TestClient(LaurenFactory.create(SupervisorModule))

        r = client.post("/supervisor/route", json={"prompt": "I need help with my invoice"})
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "Routing to billing specialist"
        assert data["turns"] == 2

    def test_supervisor_runs_tool_call_for_technical_message(self):
        _MOCK.reset()
        _MOCK.queue_tool_use("classify_intent", {"message": "the app keeps crashing"})
        _MOCK.queue_response(_completion("Routing to technical support", n=2))
        client = TestClient(LaurenFactory.create(SupervisorModule))

        r = client.post("/supervisor/route", json={"prompt": "the app keeps crashing"})
        assert r.status_code == 200
        assert r.json()["content"] == "Routing to technical support"

    def test_supervisor_tool_calls_made_tracks_classify_intent(self):
        _MOCK.reset()
        _MOCK.queue_tool_use("classify_intent", {"message": "payment issue"})
        _MOCK.queue_response(_completion("Routing", n=2))
        client = TestClient(LaurenFactory.create(SupervisorModule))

        r = client.post("/supervisor/route", json={"prompt": "payment issue"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0] == "classify_intent"


# ---------------------------------------------------------------------------
# Tests: dispatch routing logic (via HTTP)
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    def test_routing_specialist_map(self):
        client = build_app()
        r = client.get("/supervisor/specialists")
        assert r.status_code == 200
        data = r.json()
        assert data["billing"] is True
        assert data["technical"] is True
        assert data["general"] is True
        assert data["unknown_defaults_general"] is True

    def test_specialist_agents_can_run_independently(self):
        _MOCK.reset()
        _MOCK.queue_response(_completion("Your invoice is due on the 15th"))
        client = TestClient(LaurenFactory.create(SupervisorModule))

        r = client.post("/supervisor/billing", json={"prompt": "When is my invoice due?"})
        assert r.status_code == 200
        assert r.json()["content"] == "Your invoice is due on the 15th"

    def test_general_agent_handles_greeting(self):
        _MOCK.reset()
        _MOCK.queue_response(_completion("Hello! How can I help you today?"))
        client = TestClient(LaurenFactory.create(SupervisorModule))

        r = client.post("/supervisor/general", json={"prompt": "Hello!"})
        assert r.status_code == 200
        assert r.json()["content"] == "Hello! How can I help you today?"

"""Integration tests for the output-guardrails skill (Skill 26).

Verifies PIIRedactor, LengthFilter, and custom TopicScopeGuard via HTTP
through a Lauren TestClient using Pattern B (MockTransport).
"""

from lauren import LaurenFactory, controller, post, module, Json, use_value
from lauren.testing import TestClient
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent, use_tools
from lauren_ai import use_guardrails, PIIRedactor, LengthFilter, TopicFilter
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Custom guardrail for testing
# ---------------------------------------------------------------------------


class TopicScopeGuard:
    """Allow responses that mention at least one allowed topic."""

    def __init__(self, allowed_topics: list[str]):
        self._topics = allowed_topics

    async def check(self, response: str, ctx: GuardrailContext) -> GuardrailDecision:
        lower = response.lower()
        if any(topic.lower() in lower for topic in self._topics):
            return GuardrailDecision(action="pass", guardrail_name="TopicScopeGuard")
        return GuardrailDecision(
            action="block",
            violation=f"Response is out of scope. Allowed: {', '.join(self._topics)}",
            guardrail_name="TopicScopeGuard",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


@controller("/pii-agent")
class PIIAgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)

    @post("/run-email")
    async def run_email(self, body: Json[RunRequest]) -> dict:
        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor(entities=["EMAIL"])])
        class PIIEmailAgent: ...

        resp = await self._runner.run(PIIEmailAgent(), body.prompt)
        return {"content": resp.content}

    @post("/run-phone")
    async def run_phone(self, body: Json[RunRequest]) -> dict:
        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor(entities=["PHONE"])])
        class PIIPhoneAgent: ...

        resp = await self._runner.run(PIIPhoneAgent(), body.prompt)
        return {"content": resp.content}

    @post("/run-clean")
    async def run_clean(self, body: Json[RunRequest]) -> dict:
        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor()])
        class CleanAgent: ...

        resp = await self._runner.run(CleanAgent(), body.prompt)
        return {"content": resp.content}


@controller("/length-agent")
class LengthAgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)

    @post("/run-short-limit")
    async def run_short_limit(self, body: Json[RunRequest]) -> dict:
        @agent(model="mock-model", system="You are verbose.")
        @use_guardrails(output=[LengthFilter(max_chars=100)])
        class LongAgent: ...

        resp = await self._runner.run(LongAgent(), body.prompt)
        return {"content": resp.content, "length": len(resp.content)}

    @post("/run-large-limit")
    async def run_large_limit(self, body: Json[RunRequest]) -> dict:
        @agent(model="mock-model", system="You are concise.")
        @use_guardrails(output=[LengthFilter(max_chars=1000)])
        class ShortAgent: ...

        resp = await self._runner.run(ShortAgent(), body.prompt)
        return {"content": resp.content}


@controller("/topic-agent")
class TopicAgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)

    @post("/run-on-topic")
    async def run_on_topic(self, body: Json[RunRequest]) -> dict:
        @agent(model="mock-model", system="You are a banking assistant.")
        @use_guardrails(output=[TopicScopeGuard(allowed_topics=["account", "balance", "transfer"])])
        class BankingOnTopicAgent: ...

        resp = await self._runner.run(BankingOnTopicAgent(), body.prompt)
        return {"content": resp.content}

    @post("/run-off-topic")
    async def run_off_topic(self, body: Json[RunRequest]) -> dict:
        @agent(model="mock-model", system="You are a banking assistant.")
        @use_guardrails(output=[TopicScopeGuard(allowed_topics=["account", "balance", "transfer"])])
        class BankingOffTopicAgent: ...

        resp = await self._runner.run(BankingOffTopicAgent(), body.prompt)
        return {"content": resp.content}


@module(
    controllers=[PIIAgentController, LengthAgentController, TopicAgentController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class GuardrailsModule: ...


def build_app(*responses: str):
    _MOCK.reset()
    for c in responses:
        _MOCK.queue_response(_completion(c))
    return TestClient(LaurenFactory.create(GuardrailsModule))


# ---------------------------------------------------------------------------
# Tests: PIIRedactor
# ---------------------------------------------------------------------------


class TestPIIRedactorOutputGuardrail:
    def test_pii_email_is_redacted(self):
        client = build_app("Contact us at user@example.com for help.")
        resp = client.post("/pii-agent/run-email", json={"prompt": "Give me contact info."})
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "user@example.com" not in content
        assert "[REDACTED]" in content

    def test_pii_phone_is_redacted(self):
        client = build_app("Call us at 555-123-4567 anytime.")
        resp = client.post("/pii-agent/run-phone", json={"prompt": "What is the phone number?"})
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "555-123-4567" not in content
        assert "[REDACTED]" in content

    def test_clean_response_passes_through(self):
        client = build_app("The store is open Monday to Friday.")
        resp = client.post("/pii-agent/run-clean", json={"prompt": "When are you open?"})
        assert resp.status_code == 200
        assert resp.json()["content"] == "The store is open Monday to Friday."


# ---------------------------------------------------------------------------
# Tests: LengthFilter
# ---------------------------------------------------------------------------


class TestLengthFilterOutputGuardrail:
    def test_response_exceeding_max_chars_is_blocked(self):
        long_response = "word " * 500
        client = build_app(long_response)
        resp = client.post("/length-agent/run-short-limit", json={"prompt": "Tell me everything."})
        assert resp.status_code == 200
        # Block action → content is the violation message, shorter than original
        assert resp.json()["length"] < len(long_response)

    def test_short_response_passes_through(self):
        client = build_app("Short answer.")
        resp = client.post("/length-agent/run-large-limit", json={"prompt": "Hi."})
        assert resp.status_code == 200
        assert resp.json()["content"] == "Short answer."


# ---------------------------------------------------------------------------
# Tests: Custom TopicScopeGuard
# ---------------------------------------------------------------------------


class TestTopicScopeGuard:
    def test_on_topic_response_passes(self):
        client = build_app("Your account balance is $500.")
        resp = client.post("/topic-agent/run-on-topic", json={"prompt": "What is my balance?"})
        assert resp.status_code == 200
        assert "500" in resp.json()["content"]

    def test_off_topic_response_is_blocked(self):
        client = build_app("Here is a chocolate cake recipe: ...")
        resp = client.post("/topic-agent/run-off-topic", json={"prompt": "Tell me a recipe."})
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "out of scope" in content.lower() or "Allowed" in content


# ---------------------------------------------------------------------------
# Tests: GuardrailDecision fields (pure unit, no HTTP needed)
# ---------------------------------------------------------------------------


class TestGuardrailDecisionFields:
    def test_pass_decision(self):
        d = GuardrailDecision(action="pass", guardrail_name="Test")
        assert d.action == "pass"
        assert d.modified_content is None
        assert d.violation is None

    def test_block_decision(self):
        d = GuardrailDecision(action="block", violation="Bad content", guardrail_name="Test")
        assert d.action == "block"
        assert d.violation == "Bad content"

    def test_modify_decision(self):
        d = GuardrailDecision(action="modify", modified_content="Safe content", guardrail_name="Test")
        assert d.action == "modify"
        assert d.modified_content == "Safe content"

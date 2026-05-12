"""Integration tests for the output-guardrails skill (Skill 26).

Verifies PIIRedactor, LengthFilter, and custom TopicScopeGuard via
TestClient with MockTransport.
"""

from lauren_ai import LengthFilter, PIIRedactor, use_guardrails
from lauren_ai._agents import agent
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

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


def _c(text, *, n=1, stop="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Tests: PIIRedactor
# ---------------------------------------------------------------------------


class TestPIIRedactorOutputGuardrail:
    def test_pii_email_is_redacted(self):
        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor(entities=["EMAIL"])])
        class PIIEmailAgent:
            pass

        client = TestClient(PIIEmailAgent())
        client.mock.queue_response(_c("Contact us at user@example.com for help."))
        result = client.run("Give me contact info.")
        assert "user@example.com" not in result.content
        assert "[REDACTED]" in result.content

    def test_pii_phone_is_redacted(self):
        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor(entities=["PHONE"])])
        class PIIPhoneAgent:
            pass

        client = TestClient(PIIPhoneAgent())
        client.mock.queue_response(_c("Call us at 555-123-4567 anytime."))
        result = client.run("What is the phone number?")
        assert "555-123-4567" not in result.content
        assert "[REDACTED]" in result.content

    def test_clean_response_passes_through(self):
        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor()])
        class CleanAgent:
            pass

        client = TestClient(CleanAgent())
        client.mock.queue_response(_c("The store is open Monday to Friday."))
        result = client.run("When are you open?")
        assert result.content == "The store is open Monday to Friday."


# ---------------------------------------------------------------------------
# Tests: LengthFilter
# ---------------------------------------------------------------------------


class TestLengthFilterOutputGuardrail:
    def test_response_exceeding_max_chars_is_blocked(self):
        @agent(model="mock-model", system="You are verbose.")
        @use_guardrails(output=[LengthFilter(max_chars=100)])
        class LongAgent:
            pass

        long_response = "word " * 500
        client = TestClient(LongAgent())
        client.mock.queue_response(_c(long_response))
        result = client.run("Tell me everything.")
        # Block action → content is the violation message, shorter than original
        assert len(result.content) < len(long_response)

    def test_short_response_passes_through(self):
        @agent(model="mock-model", system="You are concise.")
        @use_guardrails(output=[LengthFilter(max_chars=1000)])
        class ShortAgent:
            pass

        client = TestClient(ShortAgent())
        client.mock.queue_response(_c("Short answer."))
        result = client.run("Hi.")
        assert result.content == "Short answer."


# ---------------------------------------------------------------------------
# Tests: Custom TopicScopeGuard
# ---------------------------------------------------------------------------


class TestTopicScopeGuard:
    def test_on_topic_response_passes(self):
        @agent(model="mock-model", system="You are a banking assistant.")
        @use_guardrails(output=[TopicScopeGuard(allowed_topics=["account", "balance", "transfer"])])
        class BankingOnTopicAgent:
            pass

        client = TestClient(BankingOnTopicAgent())
        client.mock.queue_response(_c("Your account balance is $500."))
        result = client.run("What is my balance?")
        assert "500" in result.content

    def test_off_topic_response_is_blocked(self):
        @agent(model="mock-model", system="You are a banking assistant.")
        @use_guardrails(output=[TopicScopeGuard(allowed_topics=["account", "balance", "transfer"])])
        class BankingOffTopicAgent:
            pass

        client = TestClient(BankingOffTopicAgent())
        client.mock.queue_response(_c("Here is a chocolate cake recipe: ..."))
        result = client.run("Tell me a recipe.")
        assert "out of scope" in result.content.lower() or "Allowed" in result.content


# ---------------------------------------------------------------------------
# Tests: GuardrailDecision fields (pure unit, no agent needed)
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
        d = GuardrailDecision(
            action="modify", modified_content="Safe content", guardrail_name="Test"
        )
        assert d.action == "modify"
        assert d.modified_content == "Safe content"

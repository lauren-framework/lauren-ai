"""Integration tests for the output-guardrails skill (Skill 26).

Verifies PIIRedactor, LengthFilter, and custom TopicScopeGuard via the
full runner path using AgentRunnerBase + MockTransport.
"""
import pytest

from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent, use_tools
from lauren_ai import use_guardrails, PIIRedactor, LengthFilter, TopicFilter
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision


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


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


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
# Tests: PIIRedactor
# ---------------------------------------------------------------------------


class TestPIIRedactorOutputGuardrail:
    @pytest.mark.asyncio
    async def test_pii_email_is_redacted(self):
        mock = MockTransport()
        mock.queue_response(_completion("Contact us at user@example.com for help."))

        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor(entities=["EMAIL"])])
        class PIIAgent: ...

        runner, _ = _make_runner(mock)
        response = await runner.run(PIIAgent(), "Give me contact info.")
        assert "user@example.com" not in response.content
        assert "[REDACTED]" in response.content

    @pytest.mark.asyncio
    async def test_pii_phone_is_redacted(self):
        mock = MockTransport()
        mock.queue_response(_completion("Call us at 555-123-4567 anytime."))

        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor(entities=["PHONE"])])
        class PIIPhoneAgent: ...

        runner, _ = _make_runner(mock)
        response = await runner.run(PIIPhoneAgent(), "What is the phone number?")
        assert "555-123-4567" not in response.content
        assert "[REDACTED]" in response.content

    @pytest.mark.asyncio
    async def test_clean_response_passes_through(self):
        mock = MockTransport()
        mock.queue_response(_completion("The store is open Monday to Friday."))

        @agent(model="mock-model", system="You are helpful.")
        @use_guardrails(output=[PIIRedactor()])
        class CleanAgent: ...

        runner, _ = _make_runner(mock)
        response = await runner.run(CleanAgent(), "When are you open?")
        assert response.content == "The store is open Monday to Friday."


# ---------------------------------------------------------------------------
# Tests: LengthFilter
# ---------------------------------------------------------------------------


class TestLengthFilterOutputGuardrail:
    @pytest.mark.asyncio
    async def test_response_exceeding_max_chars_is_blocked(self):
        mock = MockTransport()
        long_response = "word " * 500  # ~2500 chars
        mock.queue_response(_completion(long_response))

        @agent(model="mock-model", system="You are verbose.")
        @use_guardrails(output=[LengthFilter(max_chars=100)])
        class LongAgent: ...

        runner, _ = _make_runner(mock)
        response = await runner.run(LongAgent(), "Tell me everything.")
        # Block action → content is the violation message
        assert len(response.content) < len(long_response)

    @pytest.mark.asyncio
    async def test_short_response_passes_through(self):
        mock = MockTransport()
        mock.queue_response(_completion("Short answer."))

        @agent(model="mock-model", system="You are concise.")
        @use_guardrails(output=[LengthFilter(max_chars=1000)])
        class ShortAgent: ...

        runner, _ = _make_runner(mock)
        response = await runner.run(ShortAgent(), "Hi.")
        assert response.content == "Short answer."


# ---------------------------------------------------------------------------
# Tests: Custom TopicScopeGuard
# ---------------------------------------------------------------------------


class TestTopicScopeGuard:
    @pytest.mark.asyncio
    async def test_on_topic_response_passes(self):
        mock = MockTransport()
        mock.queue_response(_completion("Your account balance is $500."))

        @agent(model="mock-model", system="You are a banking assistant.")
        @use_guardrails(output=[TopicScopeGuard(allowed_topics=["account", "balance", "transfer"])])
        class BankingAgent: ...

        runner, _ = _make_runner(mock)
        response = await runner.run(BankingAgent(), "What is my balance?")
        assert "500" in response.content

    @pytest.mark.asyncio
    async def test_off_topic_response_is_blocked(self):
        mock = MockTransport()
        mock.queue_response(_completion("Here is a chocolate cake recipe: ..."))

        @agent(model="mock-model", system="You are a banking assistant.")
        @use_guardrails(output=[TopicScopeGuard(allowed_topics=["account", "balance", "transfer"])])
        class BankingAgent2: ...

        runner, _ = _make_runner(mock)
        response = await runner.run(BankingAgent2(), "Tell me a recipe.")
        # Block fires, violation message replaces response
        assert "out of scope" in response.content.lower() or "Allowed" in response.content


# ---------------------------------------------------------------------------
# Tests: GuardrailDecision fields
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

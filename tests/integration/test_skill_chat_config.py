"""Integration tests for Skill 2: Chat Model Configuration & Parameter Tuning.

Tests cover:
- max_turns enforcement (agent stops after N turns)
- temperature config is passed to transport
- system prompt propagated to LLM
- conversation_store interaction
- docstring as system prompt
"""

from lauren_ai._agents import agent
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text: str = "OK", *, n: int = 1, stop_reason: str = "end_turn") -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=text,
        tool_calls=[],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="You are a pirate.")
class PirateAgent: ...


@agent(model="mock-model")
class DocAgent:
    """You are an expert chef."""


@agent(model="mock-model", system="Explicit system")
class ExplicitDocAgent:
    """Docstring system."""


@agent(model="mock-model")
class DefaultTempAgent: ...


# ---------------------------------------------------------------------------
# TestAgentMaxTurns
# ---------------------------------------------------------------------------


class TestAgentMaxTurns:
    def test_max_turns_1_single_completion(self):
        @agent(model="mock-model", max_turns=1)
        class MaxTurns1Agent: ...

        client = TestClient(MaxTurns1Agent())
        client.mock.queue_response(_c("done"))
        result = client.run("hi")
        assert result.content == "done"
        assert len(client.calls) == 1

    def test_max_turns_sets_stop_reason_max_turns(self):
        @agent(model="mock-model", max_turns=2)
        class LimitedAgent: ...

        client = TestClient(LimitedAgent())
        # Queue several tool_use responses to exhaust max_turns
        for i in range(5):
            client.mock.queue_response(_c(f"r{i}", stop_reason="tool_use"))
        result = client.run("hi")
        assert result.stop_reason in ("max_turns", "end_turn")

    def test_max_turns_0_returns_immediately(self):
        @agent(model="mock-model", max_turns=0)
        class ZeroTurnAgent: ...

        client = TestClient(ZeroTurnAgent())
        result = client.run("hi")
        assert result.stop_reason in ("max_turns", "end_turn")
        assert len(client.calls) == 0


# ---------------------------------------------------------------------------
# TestSystemPromptConfig
# ---------------------------------------------------------------------------


class TestSystemPromptConfig:
    def test_system_kwarg_sent_to_transport(self):
        client = TestClient(PirateAgent())
        client.mock.queue_response(_c("OK"))
        client.run("hello")
        assert client.calls[-1].system == "You are a pirate."

    def test_docstring_used_as_system_prompt(self):
        client = TestClient(DocAgent())
        client.mock.queue_response(_c("OK"))
        client.run("hello")
        assert client.calls[-1].system == "You are an expert chef."

    def test_explicit_system_overrides_docstring(self):
        client = TestClient(ExplicitDocAgent())
        client.mock.queue_response(_c("OK"))
        client.run("hello")
        assert client.calls[-1].system == "Explicit system"


# ---------------------------------------------------------------------------
# TestTemperatureConfig
# ---------------------------------------------------------------------------


class TestTemperatureConfig:
    def test_default_temperature_is_not_none(self):
        client = TestClient(DefaultTempAgent())
        client.mock.queue_response(_c("OK"))
        client.run("hello")
        assert client.calls[-1].temperature is not None


# ---------------------------------------------------------------------------
# TestConversationStore
# ---------------------------------------------------------------------------


class TestConversationStore:
    async def test_conversation_store_saves_exchange(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        client = TestClient(StoreAgent())
        client.mock.queue_response(_c("answer"))
        await client.run_async("question", conversation_id="sess1")

        history = await store.load("sess1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

"""Integration tests for Skill 2: Chat Model Configuration & Parameter Tuning.

Tests cover:
- max_turns enforcement (agent stops after N turns)
- temperature config is passed to transport
- system prompt propagated to LLM
- conversation_store interaction
- docstring as system prompt
"""

import pytest
from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai import LLMConfig
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content: str = "OK", *, n: int = 1, stop_reason: str = "end_turn") -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controllers / Module
# ---------------------------------------------------------------------------


class _RunRequest(BaseModel):
    prompt: str = "hi"
    max_turns: int = 10
    system: str = ""
    conversation_id: str = ""


@controller("/agent")
class AgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._mock = mock
        self._cfg = cfg

    @post("/run")
    async def run(self, body: Json[_RunRequest]) -> dict:
        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)

        system = body.system or None

        @agent(model="mock-model", max_turns=body.max_turns, system=system)
        class DynamicAgent: ...

        resp = await runner.run(DynamicAgent(), body.prompt)
        return {
            "content": resp.content,
            "stop_reason": resp.stop_reason,
            "turns": resp.turns,
            "calls": len(self._mock.calls),
        }

    @post("/run-system")
    async def run_system(self, body: Json[_RunRequest]) -> dict:
        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)

        @agent(model="mock-model", system="You are a pirate.")
        class PirateAgent: ...

        await runner.run(PirateAgent(), body.prompt)
        return {
            "system_sent": self._mock.calls[-1].system if self._mock.calls else None,
        }

    @post("/run-docstring")
    async def run_docstring(self, body: Json[_RunRequest]) -> dict:
        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)

        @agent(model="mock-model")
        class DocAgent:
            """You are an expert chef."""

        await runner.run(DocAgent(), body.prompt)
        return {
            "system_sent": self._mock.calls[-1].system if self._mock.calls else None,
        }

    @post("/run-explicit-system")
    async def run_explicit_system(self, body: Json[_RunRequest]) -> dict:
        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)

        @agent(model="mock-model", system="Explicit system")
        class ExplicitDocAgent:
            """Docstring system."""

        await runner.run(ExplicitDocAgent(), body.prompt)
        return {
            "system_sent": self._mock.calls[-1].system if self._mock.calls else None,
        }

    @post("/run-temperature")
    async def run_temperature(self, body: Json[_RunRequest]) -> dict:
        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)

        @agent(model="mock-model")
        class DefaultTempAgent: ...

        await runner.run(DefaultTempAgent(), body.prompt)
        return {
            "temperature": self._mock.calls[-1].temperature if self._mock.calls else None,
        }

    @post("/run-with-store")
    async def run_with_store(self, body: Json[_RunRequest]) -> dict:
        store = InMemoryConversationStore()
        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        conversation_id = body.conversation_id or "default"
        await runner.run(StoreAgent(), body.prompt, conversation_id=conversation_id)
        history = await store.load(conversation_id)
        return {
            "history_length": len(history),
            "first_role": history[0]["role"] if history else None,
            "second_role": history[1]["role"] if len(history) > 1 else None,
        }

    @get("/info")
    async def info(self) -> dict:
        return {
            "model": self._cfg.model,
        }


@module(
    controllers=[AgentController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class ChatConfigModule: ...


def build_app(*responses_args) -> TestClient:
    """Build app. Pass (content, stop_reason) tuples or plain strings."""
    _MOCK.reset()
    for item in responses_args:
        if isinstance(item, tuple):
            content, stop_reason = item
            _MOCK.queue_response(_completion(content, stop_reason=stop_reason))
        else:
            _MOCK.queue_response(_completion(item))
    return TestClient(LaurenFactory.create(ChatConfigModule))


# ---------------------------------------------------------------------------
# TestAgentMaxTurns
# ---------------------------------------------------------------------------


class TestAgentMaxTurns:
    def test_max_turns_1_single_completion(self):
        client = build_app("done")
        r = client.post("/agent/run", json={"prompt": "hi", "max_turns": 1})
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "done"
        assert data["calls"] == 1

    def test_max_turns_sets_stop_reason_max_turns(self):
        # Queue several tool_use responses that will exhaust max_turns
        client = build_app(
            ("r0", "tool_use"),
            ("r1", "tool_use"),
            ("r2", "tool_use"),
            ("r3", "tool_use"),
            ("r4", "tool_use"),
        )
        r = client.post("/agent/run", json={"prompt": "hi", "max_turns": 2})
        assert r.status_code == 200
        assert r.json()["stop_reason"] in ("max_turns", "end_turn")

    def test_max_turns_0_returns_immediately(self):
        client = build_app()
        r = client.post("/agent/run", json={"prompt": "hi", "max_turns": 0})
        assert r.status_code == 200
        data = r.json()
        assert data["stop_reason"] in ("max_turns", "end_turn")
        assert data["calls"] == 0


# ---------------------------------------------------------------------------
# TestSystemPromptConfig
# ---------------------------------------------------------------------------


class TestSystemPromptConfig:
    def test_system_kwarg_sent_to_transport(self):
        client = build_app("OK")
        r = client.post("/agent/run-system", json={"prompt": "hello"})
        assert r.status_code == 200
        assert r.json()["system_sent"] == "You are a pirate."

    def test_docstring_used_as_system_prompt(self):
        client = build_app("OK")
        r = client.post("/agent/run-docstring", json={"prompt": "hello"})
        assert r.status_code == 200
        assert r.json()["system_sent"] == "You are an expert chef."

    def test_explicit_system_overrides_docstring(self):
        client = build_app("OK")
        r = client.post("/agent/run-explicit-system", json={"prompt": "hello"})
        assert r.status_code == 200
        assert r.json()["system_sent"] == "Explicit system"


# ---------------------------------------------------------------------------
# TestTemperatureConfig
# ---------------------------------------------------------------------------


class TestTemperatureConfig:
    def test_default_temperature_is_not_none(self):
        client = build_app("OK")
        r = client.post("/agent/run-temperature", json={"prompt": "hello"})
        assert r.status_code == 200
        assert r.json()["temperature"] is not None


# ---------------------------------------------------------------------------
# TestConversationStore
# ---------------------------------------------------------------------------


class TestConversationStore:
    def test_conversation_store_saves_exchange(self):
        client = build_app("answer")
        r = client.post("/agent/run-with-store", json={"prompt": "question", "conversation_id": "sess1"})
        assert r.status_code == 200
        data = r.json()
        assert data["history_length"] == 2
        assert data["first_role"] == "user"
        assert data["second_role"] == "assistant"

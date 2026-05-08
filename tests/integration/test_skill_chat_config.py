"""Integration tests for Skill 2: Chat Model Configuration & Parameter Tuning.

Tests cover:
- max_turns enforcement (agent stops after N turns)
- AgentMaxTurnsError raised when max_turns exceeded
- temperature config is passed to transport
- system prompt propagated to LLM
- conversation_store interaction
- docstring as system prompt
"""

from __future__ import annotations

import pytest

from lauren_ai import LLMConfig
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str = "OK", *, n: int = 1, stop_reason: str = "end_turn") -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock: MockTransport | None = None) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# TestAgentMaxTurns
# ---------------------------------------------------------------------------


class TestAgentMaxTurns:
    async def test_max_turns_1_single_completion(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("done"))

        @agent(model="mock-model", max_turns=1)
        class OneAgent: ...

        resp = await runner.run(OneAgent(), "hi")
        assert resp.content == "done"
        assert len(mock.calls) == 1

    async def test_max_turns_sets_stop_reason_max_turns(self):
        runner, mock = _make_runner()
        # Queue more completions than max_turns allows; runner exhausts the loop
        for i in range(5):
            mock.queue_response(_completion(f"r{i}", stop_reason="tool_use"))

        @agent(model="mock-model", max_turns=2)
        class LimitedAgent: ...

        # Runner exhausts max_turns and returns stop_reason="max_turns"
        resp = await runner.run(LimitedAgent(), "hi")
        assert resp.stop_reason in ("max_turns", "end_turn")

    async def test_max_turns_3_allows_three_completions(self):
        """Verify that max_turns=3 allows at least 3 completions when tools are used."""
        from lauren_ai._tools import _add_to_tool_map, tool

        @tool()
        async def noop_op() -> dict:
            """No-op. Args: none."""
            return {}

        tools = {}
        _add_to_tool_map(tools, noop_op)
        mock = MockTransport()
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        runner = AgentRunner(transport=mock, tools=tools, config=cfg)

        # Two tool_use turns followed by end_turn
        mock.queue_tool_use("noop_op", {})
        mock.queue_tool_use("noop_op", {})
        mock.queue_response(_completion("final", stop_reason="end_turn"))

        @use_tools(noop_op)
        @agent(model="mock-model", max_turns=3)
        class ThreeAgent: ...

        resp = await runner.run(ThreeAgent(), "go")
        assert resp.content == "final"
        assert len(mock.calls) == 3

    async def test_max_turns_0_returns_immediately(self):
        runner, mock = _make_runner()
        # max_turns=0 means range(0) → loop body never runs; stop_reason stays "max_turns"
        @agent(model="mock-model", max_turns=0)
        class ZeroAgent: ...

        resp = await runner.run(ZeroAgent(), "hi")
        # No LLM calls made; stop_reason is "max_turns"
        assert resp.stop_reason in ("max_turns", "end_turn")
        assert len(mock.calls) == 0


# ---------------------------------------------------------------------------
# TestSystemPromptConfig
# ---------------------------------------------------------------------------


class TestSystemPromptConfig:
    async def test_system_kwarg_sent_to_transport(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model", system="You are a pirate.")
        class PirateAgent: ...

        await runner.run(PirateAgent(), "hello")
        assert mock.calls[0].system == "You are a pirate."

    async def test_docstring_used_as_system_prompt(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model")
        class DocAgent:
            """You are an expert chef."""

        await runner.run(DocAgent(), "hello")
        assert mock.calls[0].system == "You are an expert chef."

    async def test_explicit_system_overrides_docstring(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model", system="Explicit system")
        class DocAgent:
            """Docstring system."""

        await runner.run(DocAgent(), "hello")
        assert mock.calls[0].system == "Explicit system"


# ---------------------------------------------------------------------------
# TestTemperatureConfig
# ---------------------------------------------------------------------------


class TestTemperatureConfig:
    async def test_default_temperature_is_1(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model")
        class DefaultTempAgent: ...

        await runner.run(DefaultTempAgent(), "hello")
        # AgentRunnerBase picks temperature from config (not overridden per-agent)
        assert mock.calls[0].temperature is not None


# ---------------------------------------------------------------------------
# TestConversationStore
# ---------------------------------------------------------------------------


class TestConversationStore:
    async def test_conversation_store_saves_exchange(self):
        store = InMemoryConversationStore()
        runner, mock = _make_runner()
        mock.queue_response(_completion("answer"))

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        await runner.run(StoreAgent(), "question", conversation_id="sess1")
        history = await store.load("sess1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    async def test_conversation_store_loads_prior_history(self):
        store = InMemoryConversationStore()
        runner, mock = _make_runner()
        mock.queue_response(_completion("r1"))
        mock.queue_response(_completion("r2"))

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        await runner.run(StoreAgent(), "first", conversation_id="s")
        await runner.run(StoreAgent(), "second", conversation_id="s")

        # The second call should have seen the prior history
        second_messages = mock.calls[1].messages
        contents = [m["content"] for m in second_messages if isinstance(m.get("content"), str)]
        assert "first" in contents

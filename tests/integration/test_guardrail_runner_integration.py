"""Integration tests: output guardrails wired through AgentRunnerBase.

These tests verify the FULL path — agent class decorated with
``@use_guardrails(output=[...])`` → runner reads the metadata → guardrail
``check()`` is called → decision applied — in both sync (``run()``) and
streaming (``run_stream()``) modes.
"""

from __future__ import annotations

import pytest

from lauren_ai import agent, use_guardrails
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision
from lauren_ai._transport import Completion, CompletionChunk, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str, *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _stream_chunks(*parts: str) -> list[CompletionChunk]:
    chunks = [CompletionChunk(delta=p) for p in parts]
    chunks.append(
        CompletionChunk(
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=len(parts) or 1),
        )
    )
    return chunks


class _BlockInputGuard:
    async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
        return GuardrailDecision(
            action="block",
            violation="Input blocked.",
            guardrail_name="BlockInputGuard",
        )


class _ModifyInputGuard:
    async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
        return GuardrailDecision(
            action="modify",
            modified_content="[SAFE] " + message,
            guardrail_name="ModifyInputGuard",
        )


class _BlockOutputGuard:
    async def check(self, response: str, ctx: GuardrailContext) -> GuardrailDecision:
        if "HALLUCINATION" in response.upper():
            return GuardrailDecision(
                action="block",
                violation="Output contains hallucination.",
                guardrail_name="BlockOutputGuard",
            )
        return GuardrailDecision(action="pass", guardrail_name="BlockOutputGuard")


class _ModifyOutputGuard:
    async def check(self, response: str, ctx: GuardrailContext) -> GuardrailDecision:
        if "HALLUCINATION" in response.upper():
            return GuardrailDecision(
                action="modify",
                modified_content="I can't help with that. Redirecting you.",
                guardrail_name="ModifyOutputGuard",
            )
        return GuardrailDecision(action="pass", guardrail_name="ModifyOutputGuard")


def _make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, config=cfg)


# ---------------------------------------------------------------------------
# Sync run() tests
# ---------------------------------------------------------------------------


class TestInputGuardrailRun:
    @pytest.mark.asyncio
    async def test_input_guardrail_blocks_run(self):
        """Block action prevents LLM call and returns the violation text."""
        mock = MockTransport()

        @use_guardrails(input=[_BlockInputGuard()])
        @agent(model="mock-model")
        class BlockedAgent: ...

        runner = _make_runner(mock)
        response = await runner.run(BlockedAgent(), "anything")

        assert response.content == "Input blocked."
        assert response.turns == 0

    @pytest.mark.asyncio
    async def test_input_guardrail_modifies_message(self):
        """Modify action replaces the user message before the LLM sees it."""
        mock = MockTransport()
        mock.queue_response(_completion("ok"))

        received: list[str] = []
        orig = mock.complete

        async def spy(messages, **kw):
            received.append(messages[-1]["content"])
            return await orig(messages, **kw)

        mock.complete = spy

        @use_guardrails(input=[_ModifyInputGuard()])
        @agent(model="mock-model")
        class ModifiedAgent: ...

        runner = _make_runner(mock)
        await runner.run(ModifiedAgent(), "hello")

        assert received[0] == "[SAFE] hello"


class TestOutputGuardrailRun:
    @pytest.mark.asyncio
    async def test_output_guardrail_modifies_run_response(self):
        """Modify action replaces hallucinated content."""
        mock = MockTransport()
        mock.queue_response(_completion("This is a HALLUCINATION response."))

        @use_guardrails(output=[_ModifyOutputGuard()])
        @agent(model="mock-model")
        class ModifiedOutputAgent: ...

        runner = _make_runner(mock)
        response = await runner.run(ModifiedOutputAgent(), "question")

        assert response.content == "I can't help with that. Redirecting you."

    @pytest.mark.asyncio
    async def test_output_guardrail_blocks_run_response(self):
        """Block action ends the run with the violation message."""
        mock = MockTransport()
        mock.queue_response(_completion("HALLUCINATION detected here"))

        @use_guardrails(output=[_BlockOutputGuard()])
        @agent(model="mock-model")
        class BlockedOutputAgent: ...

        runner = _make_runner(mock)
        response = await runner.run(BlockedOutputAgent(), "question")

        assert response.content == "Output contains hallucination."

    @pytest.mark.asyncio
    async def test_output_guardrail_passes_clean_response(self):
        """Guardrail does not interfere with a clean response."""
        mock = MockTransport()
        mock.queue_response(_completion("Your balance is $1,234."))

        @use_guardrails(output=[_ModifyOutputGuard()])
        @agent(model="mock-model")
        class CleanAgent: ...

        runner = _make_runner(mock)
        response = await runner.run(CleanAgent(), "question")

        assert response.content == "Your balance is $1,234."


# ---------------------------------------------------------------------------
# Streaming run_stream() tests
# ---------------------------------------------------------------------------


class TestOutputGuardrailStream:
    @pytest.mark.asyncio
    async def test_output_guardrail_override_emitted_in_stream(self):
        """After all chunks stream, guardrail fires and emits guardrail_override."""
        mock = MockTransport()
        mock.queue_stream(_stream_chunks("This ", "is ", "a HALLUCINATION."))

        @use_guardrails(output=[_ModifyOutputGuard()])
        @agent(model="mock-model")
        class GuardedStreamAgent: ...

        runner = _make_runner(mock)
        chunks: list[CompletionChunk] = []
        async for chunk in await runner.run_stream(GuardedStreamAgent(), "q"):
            chunks.append(chunk)

        # Normal token chunks come first
        deltas = [c.delta for c in chunks if c.delta]
        assert deltas == ["This ", "is ", "a HALLUCINATION."]

        # The final sentinel has guardrail_override set
        override_chunks = [c for c in chunks if c.guardrail_override is not None]
        assert len(override_chunks) == 1
        assert override_chunks[0].guardrail_override == "I can't help with that. Redirecting you."

    @pytest.mark.asyncio
    async def test_no_guardrail_stream_is_real_time(self):
        """Agents without guardrails: chunks yield normally, no override chunk."""
        mock = MockTransport()
        mock.queue_stream(_stream_chunks("Hello ", "world."))

        @agent(model="mock-model")
        class PlainStreamAgent: ...

        runner = _make_runner(mock)
        chunks: list[CompletionChunk] = []
        async for chunk in await runner.run_stream(PlainStreamAgent(), "q"):
            chunks.append(chunk)

        deltas = [c.delta for c in chunks if c.delta]
        assert deltas == ["Hello ", "world."]

        override_chunks = [c for c in chunks if c.guardrail_override is not None]
        assert override_chunks == []

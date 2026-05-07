"""Unit tests for the enhanced ``LLMGuardrail``.

These tests verify both backward-compatible behaviour (``action="block"``) and
the new parameters added in this version (``action``, ``system``, ``max_tokens``,
``temperature``, ``guardrail_name``).

All tests use a minimal async mock for ``llm`` so no network calls are made.
"""

from __future__ import annotations

import pytest

from lauren_ai._guardrails._base import GuardrailContext
from lauren_ai._guardrails._llm import LLMGuardrail
from lauren_ai._transport import Completion, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str) -> Completion:
    return Completion(
        id="mock",
        model="mock",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=1),
    )


class _MockLLM:
    """Minimal mock that records the kwargs passed to complete() and returns a
    preset Completion."""

    def __init__(self, response_text: str) -> None:
        self._response = response_text
        self.last_messages: list = []
        self.last_kwargs: dict = {}

    async def complete(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return _completion(self._response)


_CTX = GuardrailContext(agent_name="TestAgent")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMGuardrailBlockAction:
    @pytest.mark.asyncio
    async def test_block_action_returns_block_decision(self):
        """Existing behaviour: action='block' (default) raises block on YES."""
        mock = _MockLLM("YES")
        guard = LLMGuardrail(
            llm=mock,
            prompt="Is this bad? {content}\nYES or NO.",
            block_if="YES",
            violation_message="Blocked!",
        )
        decision = await guard.check("bad content", _CTX)
        assert decision.action == "block"
        assert decision.violation == "Blocked!"

    @pytest.mark.asyncio
    async def test_block_action_passes_on_no(self):
        mock = _MockLLM("NO")
        guard = LLMGuardrail(
            llm=mock,
            prompt="Is this bad? {content}",
            block_if="YES",
        )
        decision = await guard.check("fine content", _CTX)
        assert decision.action == "pass"


class TestLLMGuardrailModifyAction:
    @pytest.mark.asyncio
    async def test_modify_action_returns_modify_decision(self):
        mock = _MockLLM("YES")
        guard = LLMGuardrail(
            llm=mock,
            prompt="Out of scope? {content}",
            block_if="YES",
            violation_message="Please redirect.",
            action="modify",
        )
        decision = await guard.check("off topic", _CTX)
        assert decision.action == "modify"

    @pytest.mark.asyncio
    async def test_modify_uses_violation_message_as_modified_content(self):
        mock = _MockLLM("YES")
        redirect = "I can't help with that — redirecting you."
        guard = LLMGuardrail(
            llm=mock,
            prompt="{content}",
            block_if="YES",
            violation_message=redirect,
            action="modify",
        )
        decision = await guard.check("x", _CTX)
        assert decision.modified_content == redirect

    @pytest.mark.asyncio
    async def test_modify_passes_on_no(self):
        mock = _MockLLM("NO")
        guard = LLMGuardrail(
            llm=mock,
            prompt="{content}",
            block_if="YES",
            action="modify",
        )
        decision = await guard.check("fine", _CTX)
        assert decision.action == "pass"


class TestLLMGuardrailNewParams:
    @pytest.mark.asyncio
    async def test_custom_system_prompt_passed_to_llm(self):
        mock = _MockLLM("NO")
        guard = LLMGuardrail(
            llm=mock,
            prompt="{content}",
            block_if="YES",
            system="Answer YES or NO only.",
        )
        await guard.check("anything", _CTX)
        assert mock.last_kwargs.get("system") == "Answer YES or NO only."

    @pytest.mark.asyncio
    async def test_max_tokens_and_temperature_passed_to_llm(self):
        mock = _MockLLM("NO")
        guard = LLMGuardrail(
            llm=mock,
            prompt="{content}",
            block_if="YES",
            max_tokens=5,
            temperature=0.0,
        )
        await guard.check("anything", _CTX)
        assert mock.last_kwargs.get("max_tokens") == 5
        assert mock.last_kwargs.get("temperature") == 0.0

    @pytest.mark.asyncio
    async def test_guardrail_name_used_in_decision(self):
        mock = _MockLLM("YES")
        guard = LLMGuardrail(
            llm=mock,
            prompt="{content}",
            block_if="YES",
            guardrail_name="MyCustomGuard",
        )
        decision = await guard.check("x", _CTX)
        assert decision.guardrail_name == "MyCustomGuard"

    @pytest.mark.asyncio
    async def test_omitted_optional_params_not_passed_to_llm(self):
        """When system/max_tokens/temperature are None, they must not appear
        in the kwargs forwarded to llm.complete() — callers may not accept them."""
        mock = _MockLLM("NO")
        guard = LLMGuardrail(
            llm=mock,
            prompt="{content}",
            block_if="YES",
        )
        await guard.check("anything", _CTX)
        assert "system" not in mock.last_kwargs
        assert "max_tokens" not in mock.last_kwargs
        assert "temperature" not in mock.last_kwargs

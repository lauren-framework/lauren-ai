"""Integration tests for the response-validation-retry skill (Skill 28).

Verifies ResponseValidator rejects invalid first responses, retries, and
accepts the second valid response; also tests validator factory functions.
"""
import asyncio
import json
import pytest

from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------

from typing import Any, Callable


class ResponseValidator:
    def __init__(self, validators: list[Callable[[str], bool]], max_retries: int = 3):
        self._validators = validators
        self._max_retries = max_retries

    async def validate_and_retry(self, run_fn: Callable, prompt: str, **kwargs) -> Any:
        last_response = None
        for attempt in range(self._max_retries):
            response = await run_fn(prompt, **kwargs)
            content = response.content if hasattr(response, "content") else str(response)
            if all(v(content) for v in self._validators):
                return response
            last_response = response
            if attempt < self._max_retries - 1:
                correction = (
                    f"\nPrevious response was invalid. Please try again.\n"
                    f"Previous: {content}"
                )
                prompt = prompt + correction
                await asyncio.sleep(0)  # yield without real delay in tests
        return last_response


def is_non_empty(response: str) -> bool:
    return bool(response.strip())


def is_valid_json(response: str) -> bool:
    try:
        json.loads(response.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def max_length(n: int) -> Callable[[str], bool]:
    return lambda r: len(r) <= n


def contains_required_keys(*keys: str) -> Callable[[str], bool]:
    def check(r: str) -> bool:
        try:
            data = json.loads(r)
            return all(k in data for k in keys)
        except Exception:
            return False
    return check


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
# Tests: Validator functions
# ---------------------------------------------------------------------------


class TestValidatorFunctions:
    def test_is_non_empty_with_content(self):
        assert is_non_empty("hello") is True

    def test_is_non_empty_with_empty_string(self):
        assert is_non_empty("") is False

    def test_is_non_empty_with_whitespace(self):
        assert is_non_empty("   ") is False

    def test_is_valid_json_with_valid_json(self):
        assert is_valid_json('{"key": "value"}') is True

    def test_is_valid_json_with_array(self):
        assert is_valid_json('[1, 2, 3]') is True

    def test_is_valid_json_with_invalid_json(self):
        assert is_valid_json("not json") is False

    def test_is_valid_json_with_plain_text(self):
        assert is_valid_json("The answer is 42") is False

    def test_max_length_within_limit(self):
        assert max_length(100)("short") is True

    def test_max_length_exceeds_limit(self):
        assert max_length(5)("too long string here") is False

    def test_contains_required_keys_all_present(self):
        checker = contains_required_keys("result", "confidence")
        assert checker('{"result": "yes", "confidence": 0.9}') is True

    def test_contains_required_keys_missing_key(self):
        checker = contains_required_keys("result", "confidence")
        assert checker('{"result": "yes"}') is False

    def test_contains_required_keys_invalid_json(self):
        checker = contains_required_keys("result")
        assert checker("not json") is False


# ---------------------------------------------------------------------------
# Tests: ResponseValidator
# ---------------------------------------------------------------------------


class TestResponseValidator:
    @pytest.mark.asyncio
    async def test_accepts_first_valid_response(self):
        mock = MockTransport()
        valid_json = '{"result": "ok", "confidence": 0.95}'
        mock.queue_response(_completion(valid_json))

        @agent(model="mock-model", system="Return JSON only.")
        class JsonAgent: ...

        runner, _ = _make_runner(mock)
        validator = ResponseValidator(
            validators=[is_non_empty, is_valid_json, contains_required_keys("result", "confidence")],
        )
        response = await validator.validate_and_retry(
            lambda prompt: runner.run(JsonAgent(), prompt),
            "Give me JSON",
        )
        assert response.content == valid_json
        assert len(mock.calls) == 1

    @pytest.mark.asyncio
    async def test_retries_on_invalid_first_response(self):
        mock = MockTransport()
        mock.queue_response(_completion("Sorry, I cannot help."))  # invalid JSON
        valid_json = '{"result": "ok", "confidence": 0.9}'
        mock.queue_response(_completion(valid_json))

        @agent(model="mock-model", system="Return JSON only.")
        class JsonAgent2: ...

        runner, _ = _make_runner(mock)
        validator = ResponseValidator(
            validators=[is_valid_json],
            max_retries=3,
        )
        response = await validator.validate_and_retry(
            lambda prompt: runner.run(JsonAgent2(), prompt),
            "Give me JSON",
        )
        # Should have retried exactly once
        assert len(mock.calls) == 2
        assert response.content == valid_json

    @pytest.mark.asyncio
    async def test_returns_last_response_after_exhausting_retries(self):
        mock = MockTransport()
        for i in range(3):
            mock.queue_response(_completion(f"invalid response {i}"))

        @agent(model="mock-model", system="Return JSON.")
        class JsonAgent3: ...

        runner, _ = _make_runner(mock)
        validator = ResponseValidator(
            validators=[is_valid_json],
            max_retries=3,
        )
        response = await validator.validate_and_retry(
            lambda prompt: runner.run(JsonAgent3(), prompt),
            "Give me JSON",
        )
        assert len(mock.calls) == 3
        assert "invalid response" in response.content

    @pytest.mark.asyncio
    async def test_corrective_prompt_appended_on_retry(self):
        mock = MockTransport()
        mock.queue_response(_completion("not json"))
        mock.queue_response(_completion('{"ok": true}'))

        received_prompts = []
        orig_complete = mock.complete

        async def spy(messages, **kw):
            received_prompts.append(messages[-1]["content"])
            return await orig_complete(messages, **kw)

        mock.complete = spy

        @agent(model="mock-model", system="Return JSON.")
        class JsonAgent4: ...

        runner, _ = _make_runner(mock)
        validator = ResponseValidator(validators=[is_valid_json], max_retries=2)
        await validator.validate_and_retry(
            lambda prompt: runner.run(JsonAgent4(), prompt),
            "Initial prompt",
        )
        assert len(received_prompts) == 2
        assert "invalid" in received_prompts[1].lower()
        assert "Please try again" in received_prompts[1]

    @pytest.mark.asyncio
    async def test_multiple_validators_all_must_pass(self):
        mock = MockTransport()
        # First: invalid JSON → fails is_valid_json
        # Second: valid JSON but missing required keys → fails contains_required_keys
        # Third: valid JSON with required keys → passes all
        mock.queue_response(_completion("not json"))
        mock.queue_response(_completion('{"other": "value"}'))
        mock.queue_response(_completion('{"result": "yes", "confidence": 0.8}'))

        @agent(model="mock-model", system="Return JSON.")
        class JsonAgent5: ...

        runner, _ = _make_runner(mock)
        validator = ResponseValidator(
            validators=[is_valid_json, contains_required_keys("result", "confidence")],
            max_retries=3,
        )
        response = await validator.validate_and_retry(
            lambda prompt: runner.run(JsonAgent5(), prompt),
            "Give me structured JSON",
        )
        assert len(mock.calls) == 3
        data = json.loads(response.content)
        assert "result" in data
        assert "confidence" in data

"""Integration tests for the response-validation-retry skill (Skill 28).

Verifies ResponseValidator rejects invalid first responses, retries, and
accepts the second valid response; also tests validator factory functions.
"""

import asyncio
import json
from typing import Any, Callable

import pytest

from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------


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
                    f"\nPrevious response was invalid. Please try again.\nPrevious: {content}"
                )
                prompt = prompt + correction
                await asyncio.sleep(0)
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
# Tests: Validator functions (direct calls)
# ---------------------------------------------------------------------------


class TestValidatorFunctions:
    def test_non_empty_with_content(self):
        assert is_non_empty("hello") is True

    def test_non_empty_with_empty_string(self):
        assert is_non_empty("") is False

    def test_non_empty_with_whitespace(self):
        assert is_non_empty("   ") is False

    def test_valid_json_with_valid_json(self):
        assert is_valid_json('{"key": "value"}') is True

    def test_valid_json_with_invalid_json(self):
        assert is_valid_json("not json") is False

    def test_max_length_within_limit(self):
        assert max_length(100)("short") is True

    def test_max_length_exceeds_limit(self):
        assert max_length(5)("too long string here") is False

    def test_contains_keys_all_present(self):
        assert (
            contains_required_keys("result", "confidence")('{"result": "yes", "confidence": 0.9}')
            is True
        )

    def test_contains_keys_missing_key(self):
        assert contains_required_keys("result", "confidence")('{"result": "yes"}') is False


# ---------------------------------------------------------------------------
# Tests: ResponseValidator via TestClient
# ---------------------------------------------------------------------------


class TestResponseValidator:
    def test_accepts_first_valid_response(self):
        @agent(model="mock-model", system="Return JSON only.")
        class ValidatedAgent:
            pass

        valid_json = '{"result": "ok", "confidence": 0.95}'
        client = TestClient(ValidatedAgent())
        client.mock.queue_response(_c(valid_json))

        validator = ResponseValidator(
            validators=[
                is_non_empty,
                is_valid_json,
                contains_required_keys("result", "confidence"),
            ],
            max_retries=3,
        )
        response = asyncio.run(
            validator.validate_and_retry(
                lambda prompt: client.run_async(prompt),
                "Give me JSON",
            )
        )
        assert response.content == valid_json
        assert len(client.calls) == 1

    def test_retries_on_invalid_first_response(self):
        @agent(model="mock-model", system="Return JSON only.")
        class ValidatedAgent2:
            pass

        valid_json = '{"result": "ok", "confidence": 0.9}'
        client = TestClient(ValidatedAgent2())
        client.mock.queue_response(_c("Sorry, I cannot help."))
        client.mock.queue_response(_c(valid_json))

        validator = ResponseValidator(validators=[is_valid_json], max_retries=3)
        response = asyncio.run(
            validator.validate_and_retry(
                lambda prompt: client.run_async(prompt),
                "Give me JSON",
            )
        )
        assert len(client.calls) == 2
        assert response.content == valid_json

    def test_returns_last_response_after_exhausting_retries(self):
        @agent(model="mock-model", system="Return JSON only.")
        class ValidatedAgent3:
            pass

        client = TestClient(ValidatedAgent3())
        for i in range(3):
            client.mock.queue_response(_c(f"invalid {i}"))

        validator = ResponseValidator(validators=[is_valid_json], max_retries=3)
        response = asyncio.run(
            validator.validate_and_retry(
                lambda prompt: client.run_async(prompt),
                "Give me JSON",
            )
        )
        assert len(client.calls) == 3
        assert "invalid" in response.content

    def test_multiple_validators_all_must_pass(self):
        @agent(model="mock-model", system="Return JSON only.")
        class ValidatedAgent4:
            pass

        client = TestClient(ValidatedAgent4())
        client.mock.queue_response(_c("not json"))
        client.mock.queue_response(_c('{"other": "value"}'))
        client.mock.queue_response(_c('{"result": "yes", "confidence": 0.8}'))

        validator = ResponseValidator(
            validators=[is_valid_json, contains_required_keys("result", "confidence")],
            max_retries=3,
        )
        response = asyncio.run(
            validator.validate_and_retry(
                lambda prompt: client.run_async(prompt),
                "Give me structured JSON",
            )
        )
        assert len(client.calls) == 3
        parsed = json.loads(response.content)
        assert "result" in parsed
        assert "confidence" in parsed

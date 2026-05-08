"""Integration tests for the response-validation-retry skill (Skill 28).

Verifies ResponseValidator rejects invalid first responses, retries, and
accepts the second valid response; also tests validator factory functions,
via HTTP through a Lauren TestClient.
"""

import asyncio
import json

from lauren import LaurenFactory, controller, post, module, Json, use_value
from lauren.testing import TestClient
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent
from pydantic import BaseModel
from typing import Any, Callable


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
                    f"\nPrevious response was invalid. Please try again.\n"
                    f"Previous: {content}"
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
# Request models
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    response: str
    rules: list[str]  # e.g. ["non_empty", "valid_json", "max_length:100"]


class RunValidatedRequest(BaseModel):
    prompt: str
    rules: list[str]
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Helper: parse rule strings into validator functions
# ---------------------------------------------------------------------------


def _parse_rules(rules: list[str]) -> list[Callable[[str], bool]]:
    validators: list[Callable[[str], bool]] = []
    for rule in rules:
        if rule == "non_empty":
            validators.append(is_non_empty)
        elif rule == "valid_json":
            validators.append(is_valid_json)
        elif rule.startswith("max_length:"):
            n = int(rule.split(":")[1])
            validators.append(max_length(n))
        elif rule.startswith("contains_keys:"):
            keys = rule.split(":")[1].split(",")
            validators.append(contains_required_keys(*keys))
    return validators


# ---------------------------------------------------------------------------
# Controller / Module / build_app
# ---------------------------------------------------------------------------


@controller("/validate")
class ValidateController:
    @post("/check")
    async def check(self, body: Json[ValidateRequest]) -> dict:
        validators = _parse_rules(body.rules)
        valid = all(v(body.response) for v in validators)
        return {"valid": valid}


@controller("/agent")
class ValidatedAgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)
        self._mock = mock

    @post("/run-validated")
    async def run_validated(self, body: Json[RunValidatedRequest]) -> dict:
        @agent(model="mock-model", system="Return JSON only.")
        class ValidatedAgent: ...

        validators = _parse_rules(body.rules)
        validator = ResponseValidator(validators=validators, max_retries=body.max_retries)

        response = await validator.validate_and_retry(
            lambda prompt: self._runner.run(ValidatedAgent(), prompt),
            body.prompt,
        )
        return {
            "content": response.content,
            "calls": len(self._mock.calls),
        }


@module(
    controllers=[ValidateController, ValidatedAgentController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class ValidationModule: ...


def build_app(*responses: str):
    _MOCK.reset()
    for c in responses:
        _MOCK.queue_response(_completion(c))
    return TestClient(LaurenFactory.create(ValidationModule))


# ---------------------------------------------------------------------------
# Tests: Validator functions (via /validate/check)
# ---------------------------------------------------------------------------


class TestValidatorFunctions:
    def test_non_empty_with_content(self):
        client = build_app()
        resp = client.post("/validate/check", json={"response": "hello", "rules": ["non_empty"]})
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_non_empty_with_empty_string(self):
        client = build_app()
        resp = client.post("/validate/check", json={"response": "", "rules": ["non_empty"]})
        assert resp.status_code == 200
        assert resp.json()["valid"] is False

    def test_non_empty_with_whitespace(self):
        client = build_app()
        resp = client.post("/validate/check", json={"response": "   ", "rules": ["non_empty"]})
        assert resp.status_code == 200
        assert resp.json()["valid"] is False

    def test_valid_json_with_valid_json(self):
        client = build_app()
        resp = client.post("/validate/check", json={
            "response": '{"key": "value"}', "rules": ["valid_json"],
        })
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_valid_json_with_invalid_json(self):
        client = build_app()
        resp = client.post("/validate/check", json={"response": "not json", "rules": ["valid_json"]})
        assert resp.status_code == 200
        assert resp.json()["valid"] is False

    def test_max_length_within_limit(self):
        client = build_app()
        resp = client.post("/validate/check", json={"response": "short", "rules": ["max_length:100"]})
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_max_length_exceeds_limit(self):
        client = build_app()
        resp = client.post("/validate/check", json={
            "response": "too long string here", "rules": ["max_length:5"],
        })
        assert resp.status_code == 200
        assert resp.json()["valid"] is False

    def test_contains_keys_all_present(self):
        client = build_app()
        resp = client.post("/validate/check", json={
            "response": '{"result": "yes", "confidence": 0.9}',
            "rules": ["contains_keys:result,confidence"],
        })
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_contains_keys_missing_key(self):
        client = build_app()
        resp = client.post("/validate/check", json={
            "response": '{"result": "yes"}',
            "rules": ["contains_keys:result,confidence"],
        })
        assert resp.status_code == 200
        assert resp.json()["valid"] is False


# ---------------------------------------------------------------------------
# Tests: ResponseValidator via /agent/run-validated
# ---------------------------------------------------------------------------


class TestResponseValidator:
    def test_accepts_first_valid_response(self):
        valid_json = '{"result": "ok", "confidence": 0.95}'
        client = build_app(valid_json)
        resp = client.post("/agent/run-validated", json={
            "prompt": "Give me JSON",
            "rules": ["non_empty", "valid_json", "contains_keys:result,confidence"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == valid_json
        assert data["calls"] == 1

    def test_retries_on_invalid_first_response(self):
        valid_json = '{"result": "ok", "confidence": 0.9}'
        client = build_app("Sorry, I cannot help.", valid_json)
        resp = client.post("/agent/run-validated", json={
            "prompt": "Give me JSON",
            "rules": ["valid_json"],
            "max_retries": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["calls"] == 2
        assert data["content"] == valid_json

    def test_returns_last_response_after_exhausting_retries(self):
        client = build_app("invalid 0", "invalid 1", "invalid 2")
        resp = client.post("/agent/run-validated", json={
            "prompt": "Give me JSON",
            "rules": ["valid_json"],
            "max_retries": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["calls"] == 3
        assert "invalid" in data["content"]

    def test_multiple_validators_all_must_pass(self):
        client = build_app(
            "not json",
            '{"other": "value"}',
            '{"result": "yes", "confidence": 0.8}',
        )
        resp = client.post("/agent/run-validated", json={
            "prompt": "Give me structured JSON",
            "rules": ["valid_json", "contains_keys:result,confidence"],
            "max_retries": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["calls"] == 3
        parsed = json.loads(data["content"])
        assert "result" in parsed
        assert "confidence" in parsed

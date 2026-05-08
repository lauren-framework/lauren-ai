"""Integration tests for Skill 48: Agent Health Check & Timeout Handling.

Tests cover:
- run_with_timeout with fast response → status 'ok', content correct
- run_with_timeout with asyncio.TimeoutError → status 'timeout'
- run_with_timeout with generic exception → status 'error'
- CircuitBreaker starts closed
- CircuitBreaker opens after threshold failures
- CircuitBreaker can_call returns False when open
- CircuitBreaker can_call returns True when closed
- record_success resets failures and closes circuit

NOTE: from __future__ import annotations is safe here.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai._agents import agent


# ---------------------------------------------------------------------------
# Timeout + circuit breaker implementations (inline for test file)
# ---------------------------------------------------------------------------


async def run_with_timeout(
    runner: AgentRunner,
    agent_instance,
    prompt: str,
    timeout: float = 30.0,
) -> dict:
    try:
        response = await asyncio.wait_for(
            runner.run(agent_instance, prompt),
            timeout=timeout,
        )
        return {
            "status": "ok",
            "content": response.content,
            "turns": response.turns,
        }
    except asyncio.TimeoutError:
        return {"status": "timeout", "error": f"Agent timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self._failures = 0
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._state = "closed"
        self._last_failure = 0.0

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure = time.monotonic()
        if self._failures >= self._threshold:
            self._state = "open"

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def can_call(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - self._last_failure > self._reset_timeout:
                self._state = "half-open"
                return True
            return False
        return True  # half-open

    @property
    def state(self) -> str:
        return self._state


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model=None, system="You are helpful.")
class HealthAgent: ...


# ---------------------------------------------------------------------------
# Module-level mock and circuit breaker
# ---------------------------------------------------------------------------

_MOCK = MockTransport()
_circuit: dict = {}


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools={}, config=cfg)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _RunRequest(BaseModel):
    prompt: str = "Hi"
    timeout: float = 30.0


@controller("/agent")
class AgentController:
    def __init__(self, mock: MockTransport) -> None:
        self._mock = mock

    @post("/run-with-timeout")
    async def run_with_timeout_endpoint(self, body: Json[_RunRequest]) -> dict:
        runner = _make_runner(self._mock)
        return await run_with_timeout(runner, HealthAgent(), body.prompt, timeout=body.timeout)


@controller("/circuit")
class CircuitController:
    @post("/record-failure")
    async def record_failure(self) -> dict:
        cb = _circuit["cb"]
        cb.record_failure()
        return {"failures": cb._failures, "state": cb.state}

    @post("/record-success")
    async def record_success(self) -> dict:
        cb = _circuit["cb"]
        cb.record_success()
        return {"failures": cb._failures, "state": cb.state}

    @get("/status")
    async def status(self) -> dict:
        cb = _circuit["cb"]
        return {"state": cb.state, "failures": cb._failures}

    @post("/can-call")
    async def can_call(self) -> dict:
        cb = _circuit["cb"]
        return {"can_call": cb.can_call()}


@module(
    controllers=[AgentController, CircuitController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class HealthModule: ...


# ---------------------------------------------------------------------------
# Build app helper
# ---------------------------------------------------------------------------


def build_app(
    *responses: str,
    failure_threshold: int = 3,
    reset_timeout: float = 60.0,
) -> TestClient:
    _MOCK.reset()
    for c in responses:
        _MOCK.queue_response(_completion(c))
    _circuit["cb"] = CircuitBreaker(failure_threshold=failure_threshold, reset_timeout=reset_timeout)
    return TestClient(LaurenFactory.create(HealthModule))


# ---------------------------------------------------------------------------
# Tests: run_with_timeout
# ---------------------------------------------------------------------------


class TestRunWithTimeout:
    def test_successful_run_returns_ok_status(self):
        client = build_app("Hello!")
        r = client.post("/agent/run-with-timeout", json={"prompt": "Hello", "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["content"] == "Hello!"

    def test_successful_run_returns_turns(self):
        client = build_app("Done")
        r = client.post("/agent/run-with-timeout", json={"prompt": "Go", "timeout": 5.0})
        assert r.status_code == 200
        assert "turns" in r.json()
        assert r.json()["turns"] >= 1

    async def test_timeout_error_returns_timeout_status(self):
        runner, mock = MockTransport(), None
        mock = MockTransport()
        mock.queue_response(_completion("too slow"))
        runner = _make_runner(mock)

        async def always_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=always_timeout):
            result = await run_with_timeout(runner, HealthAgent(), "slow prompt", timeout=0.001)

        assert result["status"] == "timeout"
        assert "timed out" in result["error"].lower()

    async def test_timeout_error_includes_timeout_value(self):
        mock = MockTransport()
        runner = _make_runner(mock)

        async def always_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=always_timeout):
            result = await run_with_timeout(runner, HealthAgent(), "prompt", timeout=15.5)

        assert "15.5" in result["error"]

    async def test_generic_exception_returns_error_status(self):
        mock = MockTransport()
        runner = _make_runner(mock)

        async def always_raises(*args, **kwargs):
            raise ValueError("Something went wrong")

        with patch("asyncio.wait_for", side_effect=always_raises):
            result = await run_with_timeout(runner, HealthAgent(), "prompt", timeout=5.0)

        assert result["status"] == "error"
        assert "Something went wrong" in result["error"]

    def test_default_timeout_parameter_exists(self):
        client = build_app("Default timeout test")
        r = client.post("/agent/run-with-timeout", json={"prompt": "Hello"})
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests: CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        client = build_app()
        r = client.get("/circuit/status")
        assert r.json()["state"] == "closed"

    def test_can_call_when_closed(self):
        client = build_app()
        r = client.post("/circuit/can-call")
        assert r.json()["can_call"] is True

    def test_single_failure_stays_closed_below_threshold(self):
        client = build_app(failure_threshold=3)
        client.post("/circuit/record-failure")
        r = client.get("/circuit/status")
        assert r.json()["state"] == "closed"
        r2 = client.post("/circuit/can-call")
        assert r2.json()["can_call"] is True

    def test_opens_after_threshold_failures(self):
        client = build_app(failure_threshold=3)
        client.post("/circuit/record-failure")
        client.post("/circuit/record-failure")
        client.post("/circuit/record-failure")
        r = client.get("/circuit/status")
        assert r.json()["state"] == "open"

    def test_cannot_call_when_open(self):
        client = build_app(failure_threshold=2, reset_timeout=600.0)
        client.post("/circuit/record-failure")
        client.post("/circuit/record-failure")
        r = client.post("/circuit/can-call")
        assert r.json()["can_call"] is False

    def test_record_success_resets_failures(self):
        client = build_app(failure_threshold=5)
        client.post("/circuit/record-failure")
        client.post("/circuit/record-failure")
        client.post("/circuit/record-success")
        r = client.get("/circuit/status")
        assert r.json()["failures"] == 0

    def test_record_success_closes_circuit(self):
        client = build_app(failure_threshold=2)
        client.post("/circuit/record-failure")
        client.post("/circuit/record-failure")  # opens
        r1 = client.get("/circuit/status")
        assert r1.json()["state"] == "open"
        client.post("/circuit/record-success")
        r2 = client.get("/circuit/status")
        assert r2.json()["state"] == "closed"

    def test_can_call_after_reset_success(self):
        client = build_app(failure_threshold=2)
        client.post("/circuit/record-failure")
        client.post("/circuit/record-failure")
        client.post("/circuit/record-success")
        r = client.post("/circuit/can-call")
        assert r.json()["can_call"] is True

    def test_half_open_state_after_reset_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
        cb.record_failure()  # opens immediately
        # reset_timeout=0.0 means it transitions to half-open immediately
        assert cb.can_call() is True  # half-open: allows one probe
        assert cb.state == "half-open"

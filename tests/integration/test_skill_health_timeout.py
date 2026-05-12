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

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient

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
    except TimeoutError:
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


def _make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, config=cfg)


# ---------------------------------------------------------------------------
# Tests: run_with_timeout
# ---------------------------------------------------------------------------


class TestRunWithTimeout:
    async def test_successful_run_returns_ok_status(self):
        mock = MockTransport()
        mock.queue_response(_completion("Hello!"))
        runner = _make_runner(mock)
        result = await run_with_timeout(runner, HealthAgent(), "Hello", timeout=5.0)
        assert result["status"] == "ok"
        assert result["content"] == "Hello!"

    async def test_successful_run_returns_turns(self):
        mock = MockTransport()
        mock.queue_response(_completion("Done"))
        runner = _make_runner(mock)
        result = await run_with_timeout(runner, HealthAgent(), "Go", timeout=5.0)
        assert "turns" in result
        assert result["turns"] >= 1

    async def test_timeout_error_returns_timeout_status(self):
        mock = MockTransport()
        mock.queue_response(_completion("too slow"))
        runner = _make_runner(mock)

        async def always_timeout(*args, **kwargs):
            raise TimeoutError()

        with patch("asyncio.wait_for", side_effect=always_timeout):
            result = await run_with_timeout(runner, HealthAgent(), "slow prompt", timeout=0.001)

        assert result["status"] == "timeout"
        assert "timed out" in result["error"].lower()

    async def test_timeout_error_includes_timeout_value(self):
        mock = MockTransport()
        runner = _make_runner(mock)

        async def always_timeout(*args, **kwargs):
            raise TimeoutError()

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

    async def test_default_timeout_parameter_exists(self):
        mock = MockTransport()
        mock.queue_response(_completion("Default timeout test"))
        runner = _make_runner(mock)
        result = await run_with_timeout(runner, HealthAgent(), "Hello")
        assert result["status"] == "ok"

    def test_successful_run_with_test_client(self):
        client = TestClient(HealthAgent())
        client.mock.queue_response(_completion("Hello!"))
        result = client.run("Hello")
        assert result.content == "Hello!"


# ---------------------------------------------------------------------------
# Tests: CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"

    def test_can_call_when_closed(self):
        cb = CircuitBreaker()
        assert cb.can_call() is True

    def test_single_failure_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.can_call() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_cannot_call_when_open(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=600.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_call() is False

    def test_record_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failures == 0

    def test_record_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()  # opens
        assert cb.state == "open"
        cb.record_success()
        assert cb.state == "closed"

    def test_can_call_after_reset_success(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.can_call() is True

    def test_half_open_state_after_reset_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
        cb.record_failure()  # opens immediately
        # reset_timeout=0.0 means it transitions to half-open immediately
        assert cb.can_call() is True  # half-open: allows one probe
        assert cb.state == "half-open"

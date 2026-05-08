---
name: agent-health-timeout
description: Implements timeout wrappers and a circuit breaker pattern for agent runs. Use when protecting services from slow or failing agent calls, adding deadline enforcement via asyncio.wait_for, or implementing automatic circuit breaking to stop calling a degraded agent.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Agent Health Check & Timeout Handling

## Overview

Two utilities are provided:

1. `run_with_timeout` — wraps `AgentRunnerBase.run()` with `asyncio.wait_for`
   and maps `TimeoutError` to a structured error dict.
2. `CircuitBreaker` — tracks consecutive failures and opens the circuit after
   `failure_threshold` failures, preventing calls during `reset_timeout`.

---

## Timeout wrapper

```python
# resilience/timeout.py
import asyncio
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner

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
```

---

## Circuit breaker

```python
# resilience/circuit_breaker.py
import time

class CircuitBreaker:
    """Three-state circuit breaker: closed → open → half-open."""

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self._failures = 0
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._state = "closed"   # closed | open | half-open
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
            elapsed = time.monotonic() - self._last_failure
            if elapsed > self._reset_timeout:
                self._state = "half-open"
                return True
            return False
        return True  # half-open: allow one probe call

    @property
    def state(self) -> str:
        return self._state
```

---

## Combined usage

```python
async def resilient_run(runner, agent, prompt, *, cb: CircuitBreaker) -> dict:
    if not cb.can_call():
        return {"status": "circuit_open", "error": "Agent circuit is open"}
    result = await run_with_timeout(runner, agent, prompt, timeout=30.0)
    if result["status"] == "ok":
        cb.record_success()
    else:
        cb.record_failure()
    return result
```

---

## Lauren controller health endpoint

```python
# controllers/health.py
from lauren import controller, get
from lauren_ai import AgentRunner

@controller("/health")
class HealthController:
    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    @get("/agent")
    async def agent_health(self) -> dict:
        # Quick ping — one-turn, low max_tokens
        result = await run_with_timeout(self._runner, PingAgent(), "ping", timeout=5.0)
        return {"status": result["status"]}
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_agents/_runner.py` | `AgentRunnerBase.run()` |
| `src/lauren_ai/_exceptions.py` | `AgentMaxTurnsError`, `AgentBudgetExceededError` |

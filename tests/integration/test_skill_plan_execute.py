"""Integration tests for Skill 9: Plan-and-Execute Agent Pattern.

Tests cover:
- Planner agent returns JSON plan
- plan_and_execute runs planner then one executor per step
- Multiple steps each trigger a runner.run() call
- Results are collected from each step
- Empty plan (no steps) returns empty result
- Planner with single step
- Planner with three steps
- JSON parse failure handled gracefully

NOTE: No `from __future__ import annotations` — @tool() used.
"""

import json

from lauren_ai._agents import agent
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=10),
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="Return JSON plan", max_turns=1)
class PlannerAgent: ...


@agent(model="mock-model", system="Execute step")
class ExecutorAgent: ...


# ---------------------------------------------------------------------------
# plan_and_execute helper (uses two clients sharing a mock)
# ---------------------------------------------------------------------------


async def plan_and_execute(mock: MockTransport, request: str) -> tuple[list[str], int]:
    """Run the plan-and-execute pattern using a shared mock transport.

    Returns (step_results, total_llm_calls).
    """
    planner_client = TestClient(PlannerAgent(), mock)
    plan_resp = await planner_client.run_async(request)
    try:
        plan = json.loads(plan_resp.content)
    except json.JSONDecodeError:
        return [], len(mock.calls)

    executor_client = TestClient(ExecutorAgent(), mock)
    results = []
    for step in plan.get("steps", []):
        result = await executor_client.run_async(step)
        results.append(result.content)
    return results, len(mock.calls)


# ---------------------------------------------------------------------------
# TestPlannerPhase
# ---------------------------------------------------------------------------


class TestPlannerPhase:
    async def test_planner_returns_valid_json(self):
        mock = MockTransport()
        mock.queue_response(_c('{"steps": ["step1", "step2"]}'))
        mock.queue_response(_c("step1 done"))
        mock.queue_response(_c("step2 done"))
        results, _ = await plan_and_execute(mock, "Do two things")
        assert len(results) == 2

    async def test_planner_single_step(self):
        mock = MockTransport()
        mock.queue_response(_c('{"steps": ["only step"]}'))
        mock.queue_response(_c("only step done"))
        results, _ = await plan_and_execute(mock, "Do one thing")
        assert len(results) == 1

    async def test_planner_empty_steps(self):
        mock = MockTransport()
        mock.queue_response(_c('{"steps": []}'))
        results, _ = await plan_and_execute(mock, "Do nothing")
        assert results == []

    async def test_planner_invalid_json_returns_empty(self):
        mock = MockTransport()
        mock.queue_response(_c("Not valid JSON at all"))
        results, _ = await plan_and_execute(mock, "do it")
        assert results == []


# ---------------------------------------------------------------------------
# TestExecutorPhase
# ---------------------------------------------------------------------------


class TestExecutorPhase:
    async def test_executor_runs_for_each_step(self):
        mock = MockTransport()
        mock.queue_response(_c('{"steps": ["A", "B", "C"]}'))
        mock.queue_response(_c("result A"))
        mock.queue_response(_c("result B"))
        mock.queue_response(_c("result C"))
        results, _ = await plan_and_execute(mock, "Three steps")
        assert len(results) == 3

    async def test_executor_result_content_matches(self):
        mock = MockTransport()
        mock.queue_response(_c('{"steps": ["fetch data"]}'))
        mock.queue_response(_c("Data fetched successfully."))
        results, _ = await plan_and_execute(mock, "fetch something")
        assert results[0] == "Data fetched successfully."

    async def test_total_llm_calls_equals_steps_plus_one(self):
        mock = MockTransport()
        mock.queue_response(_c('{"steps": ["s1", "s2"]}'))
        mock.queue_response(_c("r1"))
        mock.queue_response(_c("r2"))
        _, total_calls = await plan_and_execute(mock, "do two steps")
        # 1 planner call + 2 executor calls = 3
        assert total_calls == 3

    async def test_executor_receives_step_as_message(self):
        mock = MockTransport()
        mock.queue_response(_c('{"steps": ["analyze metrics"]}'))
        mock.queue_response(_c("Metrics analyzed."))
        _, total_calls = await plan_and_execute(mock, "analyze")
        assert total_calls == 2
        executor_call = mock.calls[1]
        assert "analyze metrics" in str(executor_call.messages)

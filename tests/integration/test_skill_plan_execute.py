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

from __future__ import annotations

import json

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import _add_to_tool_map, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=10),
    )


def _make_runner(mock: MockTransport | None = None) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


async def plan_and_execute(
    runner: AgentRunner,
    request: str,
    planner,
    executor,
) -> list[str]:
    """Run the plan-and-execute pattern. Returns a list of step results."""
    plan_response = await runner.run(planner, request)
    try:
        plan = json.loads(plan_response.content)
    except json.JSONDecodeError:
        return []

    results = []
    for step in plan.get("steps", []):
        result = await runner.run(executor, step)
        results.append(result.content)
    return results


# ---------------------------------------------------------------------------
# TestPlannerPhase
# ---------------------------------------------------------------------------


class TestPlannerPhase:
    async def test_planner_returns_valid_json(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"steps": ["step1", "step2"]}'))

        @agent(model="mock-model", system="Return JSON plan", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model", system="Execute step")
        class ExecutorAgent: ...

        mock.queue_response(_completion("step1 done"))
        mock.queue_response(_completion("step2 done"))

        results = await plan_and_execute(runner, "Do two things", PlannerAgent(), ExecutorAgent())
        assert len(results) == 2

    async def test_planner_single_step(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"steps": ["only step"]}'))

        @agent(model="mock-model", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model")
        class ExecutorAgent: ...

        mock.queue_response(_completion("only step done"))

        results = await plan_and_execute(runner, "Do one thing", PlannerAgent(), ExecutorAgent())
        assert len(results) == 1

    async def test_planner_empty_steps(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"steps": []}'))

        @agent(model="mock-model", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model")
        class ExecutorAgent: ...

        results = await plan_and_execute(runner, "Do nothing", PlannerAgent(), ExecutorAgent())
        assert results == []

    async def test_planner_invalid_json_returns_empty(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Not valid JSON at all"))

        @agent(model="mock-model", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model")
        class ExecutorAgent: ...

        results = await plan_and_execute(runner, "do it", PlannerAgent(), ExecutorAgent())
        assert results == []


# ---------------------------------------------------------------------------
# TestExecutorPhase
# ---------------------------------------------------------------------------


class TestExecutorPhase:
    async def test_executor_runs_for_each_step(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"steps": ["A", "B", "C"]}'))

        @agent(model="mock-model", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model")
        class ExecutorAgent: ...

        mock.queue_response(_completion("result A"))
        mock.queue_response(_completion("result B"))
        mock.queue_response(_completion("result C"))

        results = await plan_and_execute(runner, "Three steps", PlannerAgent(), ExecutorAgent())
        assert len(results) == 3

    async def test_executor_result_content_matches(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"steps": ["fetch data"]}'))

        @agent(model="mock-model", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model")
        class ExecutorAgent: ...

        mock.queue_response(_completion("Data fetched successfully."))

        results = await plan_and_execute(runner, "fetch something", PlannerAgent(), ExecutorAgent())
        assert results[0] == "Data fetched successfully."

    async def test_executor_receives_step_as_message(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"steps": ["analyze metrics"]}'))

        @agent(model="mock-model", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model")
        class ExecutorAgent: ...

        mock.queue_response(_completion("Metrics analyzed."))

        await plan_and_execute(runner, "analyze", PlannerAgent(), ExecutorAgent())
        # Second mock call (index 1) is for executor; its first message should be the step
        executor_call = mock.calls[1]
        assert "analyze metrics" in str(executor_call.messages)

    async def test_total_llm_calls_equals_steps_plus_one(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"steps": ["s1", "s2"]}'))

        @agent(model="mock-model", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model")
        class ExecutorAgent: ...

        mock.queue_response(_completion("r1"))
        mock.queue_response(_completion("r2"))

        await plan_and_execute(runner, "do two steps", PlannerAgent(), ExecutorAgent())
        # 1 planner call + 2 executor calls = 3
        assert len(mock.calls) == 3

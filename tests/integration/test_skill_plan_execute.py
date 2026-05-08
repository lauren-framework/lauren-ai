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

import pytest
from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=10),
    )


# ---------------------------------------------------------------------------
# plan_and_execute helper (pure async utility)
# ---------------------------------------------------------------------------


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
# Controller / Module
# ---------------------------------------------------------------------------


class _PlanRequest(BaseModel):
    request: str = "Do something"


@controller("/plan-execute")
class PlanExecuteController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)
        self._mock = mock

    @post("/run")
    async def run(self, body: Json[_PlanRequest]) -> dict:
        @agent(model="mock-model", system="Return JSON plan", max_turns=1)
        class PlannerAgent: ...

        @agent(model="mock-model", system="Execute step")
        class ExecutorAgent: ...

        results = await plan_and_execute(
            self._runner, body.request, PlannerAgent(), ExecutorAgent()
        )
        return {
            "results": results,
            "total_calls": len(self._mock.calls),
        }


@module(
    controllers=[PlanExecuteController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class PlanExecuteModule: ...


def build_app(*responses: str) -> TestClient:
    _MOCK.reset()
    for content in responses:
        _MOCK.queue_response(_completion(content))
    return TestClient(LaurenFactory.create(PlanExecuteModule))


# ---------------------------------------------------------------------------
# TestPlannerPhase
# ---------------------------------------------------------------------------


class TestPlannerPhase:
    def test_planner_returns_valid_json(self):
        client = build_app(
            '{"steps": ["step1", "step2"]}',
            "step1 done",
            "step2 done",
        )
        r = client.post("/plan-execute/run", json={"request": "Do two things"})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2

    def test_planner_single_step(self):
        client = build_app(
            '{"steps": ["only step"]}',
            "only step done",
        )
        r = client.post("/plan-execute/run", json={"request": "Do one thing"})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1

    def test_planner_empty_steps(self):
        client = build_app('{"steps": []}')
        r = client.post("/plan-execute/run", json={"request": "Do nothing"})
        assert r.status_code == 200
        assert r.json()["results"] == []

    def test_planner_invalid_json_returns_empty(self):
        client = build_app("Not valid JSON at all")
        r = client.post("/plan-execute/run", json={"request": "do it"})
        assert r.status_code == 200
        assert r.json()["results"] == []


# ---------------------------------------------------------------------------
# TestExecutorPhase
# ---------------------------------------------------------------------------


class TestExecutorPhase:
    def test_executor_runs_for_each_step(self):
        client = build_app(
            '{"steps": ["A", "B", "C"]}',
            "result A",
            "result B",
            "result C",
        )
        r = client.post("/plan-execute/run", json={"request": "Three steps"})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 3

    def test_executor_result_content_matches(self):
        client = build_app(
            '{"steps": ["fetch data"]}',
            "Data fetched successfully.",
        )
        r = client.post("/plan-execute/run", json={"request": "fetch something"})
        assert r.status_code == 200
        assert r.json()["results"][0] == "Data fetched successfully."

    def test_total_llm_calls_equals_steps_plus_one(self):
        client = build_app(
            '{"steps": ["s1", "s2"]}',
            "r1",
            "r2",
        )
        r = client.post("/plan-execute/run", json={"request": "do two steps"})
        assert r.status_code == 200
        # 1 planner call + 2 executor calls = 3
        assert r.json()["total_calls"] == 3

    def test_executor_receives_step_as_message(self):
        client = build_app(
            '{"steps": ["analyze metrics"]}',
            "Metrics analyzed.",
        )
        r = client.post("/plan-execute/run", json={"request": "analyze"})
        assert r.status_code == 200
        # Verify the step text appeared in mock calls
        assert len(_MOCK.calls) == 2
        executor_call = _MOCK.calls[1]
        assert "analyze metrics" in str(executor_call.messages)

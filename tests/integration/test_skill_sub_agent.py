"""Integration tests for Skill 45: Sub-Agent Spawning & Result Aggregation.

Tests cover:
- Sequential sub-agent spawning returns correct results
- Parallel sub-agent spawning returns all results
- Result order preserved for sequential
- Parallel results all collected (unordered check)
- Single sub-agent run
- Multiple agents with different mock responses

NOTE: No `from __future__ import annotations` here at the top.
"""

import asyncio

import pytest
from pydantic import BaseModel

from lauren import LaurenFactory, controller, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai._agents import agent


# ---------------------------------------------------------------------------
# Agent definitions (module level)
# ---------------------------------------------------------------------------

@agent(model=None, system="You are a research specialist.")
class ResearchSubAgent: ...


@agent(model=None, system="You are a writing specialist.")
class WritingSubAgent: ...


@agent(model=None, system="You are a summary specialist.")
class SummarySubAgent: ...


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


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


class _SpawnRequest(BaseModel):
    tasks: list[str]


@controller("/spawn")
class SpawnController:
    def __init__(self, mock: MockTransport) -> None:
        self._mock = mock

    @post("/sequential")
    async def sequential(self, body: Json[_SpawnRequest]) -> dict:
        runner = _make_runner(self._mock)
        results = []
        for task in body.tasks:
            resp = await runner.run(ResearchSubAgent(), task)
            results.append(resp.content)
        return {"results": results}

    @post("/parallel")
    async def parallel(self, body: Json[_SpawnRequest]) -> dict:
        runner = _make_runner(self._mock)
        coros = [runner.run(ResearchSubAgent(), task) for task in body.tasks]
        responses = await asyncio.gather(*coros)
        return {"results": [r.content for r in responses]}

    @post("/run-once")
    async def run_once(self, body: Json[dict]) -> dict:
        runner = _make_runner(self._mock)
        resp = await runner.run(ResearchSubAgent(), body.get("prompt", ""))
        return {"content": resp.content, "turns": resp.turns}


@module(
    controllers=[SpawnController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class SpawnModule: ...


# ---------------------------------------------------------------------------
# build_app helper
# ---------------------------------------------------------------------------


def build_app(*responses: str) -> TestClient:
    _MOCK.reset()
    for c in responses:
        _MOCK.queue_response(_completion(c))
    return TestClient(LaurenFactory.create(SpawnModule))


# ---------------------------------------------------------------------------
# Tests: Sequential sub-agents
# ---------------------------------------------------------------------------


class TestSequentialSubAgents:
    def test_sequential_single_agent(self):
        client = build_app("Research result")
        r = client.post("/spawn/sequential", json={"tasks": ["Research topic A"]})
        assert r.status_code == 200
        assert r.json()["results"] == ["Research result"]

    def test_sequential_two_agents_correct_order(self):
        client = build_app("First result", "Second result")
        r = client.post("/spawn/sequential", json={"tasks": ["Topic A", "Write about A"]})
        assert r.status_code == 200
        results = r.json()["results"]
        assert results[0] == "First result"
        assert results[1] == "Second result"

    def test_sequential_three_agents(self):
        client = build_app("Result 0", "Result 1", "Result 2")
        r = client.post("/spawn/sequential", json={"tasks": ["Research", "Write", "Summarise"]})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 3
        assert results == ["Result 0", "Result 1", "Result 2"]

    def test_sequential_preserves_response_content(self):
        client = build_app("Detailed research findings about AI")
        r = client.post("/spawn/sequential", json={"tasks": ["Research AI"]})
        assert r.status_code == 200
        assert "AI" in r.json()["results"][0]


class TestParallelSubAgents:
    def test_parallel_single_agent(self):
        client = build_app("Parallel result")
        r = client.post("/spawn/parallel", json={"tasks": ["Topic"]})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 1
        assert results[0] == "Parallel result"

    def test_parallel_two_agents_all_results_collected(self):
        client = build_app("Result A", "Result B")
        r = client.post("/spawn/parallel", json={"tasks": ["Topic A", "Topic B"]})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 2
        assert set(results) == {"Result A", "Result B"}

    def test_parallel_results_are_strings(self):
        client = build_app("content one", "content two")
        r = client.post("/spawn/parallel", json={"tasks": ["Q1", "Q2"]})
        assert r.status_code == 200
        for res in r.json()["results"]:
            assert isinstance(res, str)


class TestSubAgentRunnerApi:
    def test_runner_run_returns_agent_response(self):
        client = build_app("Hello from sub-agent")
        r = client.post("/spawn/run-once", json={"prompt": "Hello"})
        assert r.status_code == 200
        assert r.json()["content"] == "Hello from sub-agent"

    def test_runner_records_turns(self):
        client = build_app("Done")
        r = client.post("/spawn/run-once", json={"prompt": "Summarise this"})
        assert r.status_code == 200
        assert r.json()["turns"] >= 1

    def test_runner_run_multiple_times_sequential(self):
        client = build_app("First", "Second")
        r1 = client.post("/spawn/run-once", json={"prompt": "First prompt"})
        r2 = client.post("/spawn/run-once", json={"prompt": "Second prompt"})
        assert r1.json()["content"] == "First"
        assert r2.json()["content"] == "Second"

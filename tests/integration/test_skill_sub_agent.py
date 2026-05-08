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

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient


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
# Helpers
# ---------------------------------------------------------------------------


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools={}, config=cfg)


# ---------------------------------------------------------------------------
# Tests: Sequential sub-agents
# ---------------------------------------------------------------------------


class TestSequentialSubAgents:
    def test_sequential_single_agent(self):
        mock = MockTransport()
        mock.queue_response(_completion("Research result"))
        runner = _make_runner(mock)
        result = asyncio.run(runner.run(ResearchSubAgent(), "Research topic A"))
        assert result.content == "Research result"

    def test_sequential_two_agents_correct_order(self):
        mock = MockTransport()
        mock.queue_response(_completion("First result"))
        mock.queue_response(_completion("Second result"))
        runner = _make_runner(mock)

        r1 = asyncio.run(runner.run(ResearchSubAgent(), "Topic A"))
        r2 = asyncio.run(runner.run(ResearchSubAgent(), "Write about A"))
        assert r1.content == "First result"
        assert r2.content == "Second result"

    def test_sequential_three_agents(self):
        mock = MockTransport()
        mock.queue_response(_completion("Result 0"))
        mock.queue_response(_completion("Result 1"))
        mock.queue_response(_completion("Result 2"))
        runner = _make_runner(mock)

        results = []
        for task in ["Research", "Write", "Summarise"]:
            resp = asyncio.run(runner.run(ResearchSubAgent(), task))
            results.append(resp.content)
        assert results == ["Result 0", "Result 1", "Result 2"]

    def test_sequential_preserves_response_content(self):
        mock = MockTransport()
        mock.queue_response(_completion("Detailed research findings about AI"))
        runner = _make_runner(mock)
        result = asyncio.run(runner.run(ResearchSubAgent(), "Research AI"))
        assert "AI" in result.content


class TestParallelSubAgents:
    async def test_parallel_single_agent(self):
        mock = MockTransport()
        mock.queue_response(_completion("Parallel result"))
        runner = _make_runner(mock)
        result = await runner.run(ResearchSubAgent(), "Topic")
        assert result.content == "Parallel result"

    async def test_parallel_two_agents_all_results_collected(self):
        mock = MockTransport()
        mock.queue_response(_completion("Result A"))
        mock.queue_response(_completion("Result B"))
        runner = _make_runner(mock)

        responses = await asyncio.gather(
            runner.run(ResearchSubAgent(), "Topic A"),
            runner.run(ResearchSubAgent(), "Topic B"),
        )
        results = [r.content for r in responses]
        assert len(results) == 2
        assert set(results) == {"Result A", "Result B"}

    async def test_parallel_results_are_strings(self):
        mock = MockTransport()
        mock.queue_response(_completion("content one"))
        mock.queue_response(_completion("content two"))
        runner = _make_runner(mock)

        responses = await asyncio.gather(
            runner.run(ResearchSubAgent(), "Q1"),
            runner.run(ResearchSubAgent(), "Q2"),
        )
        for resp in responses:
            assert isinstance(resp.content, str)


class TestSubAgentRunnerApi:
    def test_runner_run_returns_agent_response(self):
        mock = MockTransport()
        mock.queue_response(_completion("Hello from sub-agent"))
        runner = _make_runner(mock)
        result = asyncio.run(runner.run(ResearchSubAgent(), "Hello"))
        assert result.content == "Hello from sub-agent"

    def test_runner_records_turns(self):
        mock = MockTransport()
        mock.queue_response(_completion("Done"))
        runner = _make_runner(mock)
        result = asyncio.run(runner.run(ResearchSubAgent(), "Summarise this"))
        assert result.turns >= 1

    def test_runner_run_multiple_times_sequential(self):
        mock = MockTransport()
        mock.queue_response(_completion("First"))
        mock.queue_response(_completion("Second"))
        runner = _make_runner(mock)
        r1 = asyncio.run(runner.run(ResearchSubAgent(), "First prompt"))
        r2 = asyncio.run(runner.run(ResearchSubAgent(), "Second prompt"))
        assert r1.content == "First"
        assert r2.content == "Second"


class TestSubAgentWithTestClient:
    def test_test_client_sequential_spawning(self):
        client1 = TestClient(ResearchSubAgent())
        client1.mock.queue_response(_completion("Research result 1"))
        result1 = client1.run("Research task 1")

        client2 = TestClient(WritingSubAgent())
        client2.mock.queue_response(_completion("Writing result 2"))
        result2 = client2.run("Writing task 2")

        assert result1.content == "Research result 1"
        assert result2.content == "Writing result 2"

    async def test_test_client_async_run(self):
        client = TestClient(SummarySubAgent())
        client.mock.queue_response(_completion("Summary done"))
        result = await client.run_async("Summarise this")
        assert result.content == "Summary done"

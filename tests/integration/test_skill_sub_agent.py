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
# Helpers
# ---------------------------------------------------------------------------

def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Sequential aggregation
# ---------------------------------------------------------------------------

async def spawn_and_aggregate_sequential(runner, agents_prompts: list) -> list:
    """Run sub-agents sequentially, collect results."""
    results = []
    for sub_agent, prompt in agents_prompts:
        response = await runner.run(sub_agent, prompt)
        results.append(response.content)
    return results


async def spawn_and_aggregate_parallel(runner, agents_prompts: list) -> list:
    """Run sub-agents in parallel, collect results."""
    tasks = [runner.run(sub_agent, prompt) for sub_agent, prompt in agents_prompts]
    responses = await asyncio.gather(*tasks)
    return [r.content for r in responses]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSequentialSubAgents:
    async def test_sequential_single_agent(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Research result"))

        results = await spawn_and_aggregate_sequential(
            runner, [(ResearchSubAgent(), "Research topic A")]
        )
        assert results == ["Research result"]

    async def test_sequential_two_agents_correct_order(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("First result"))
        mock.queue_response(_completion("Second result"))

        results = await spawn_and_aggregate_sequential(
            runner, [
                (ResearchSubAgent(), "Topic A"),
                (WritingSubAgent(), "Write about A"),
            ]
        )
        assert results[0] == "First result"
        assert results[1] == "Second result"

    async def test_sequential_three_agents(self):
        runner, mock = _make_runner()
        for i in range(3):
            mock.queue_response(_completion(f"Result {i}"))

        results = await spawn_and_aggregate_sequential(
            runner, [
                (ResearchSubAgent(), "Research"),
                (WritingSubAgent(), "Write"),
                (SummarySubAgent(), "Summarise"),
            ]
        )
        assert len(results) == 3
        assert results == ["Result 0", "Result 1", "Result 2"]

    async def test_sequential_preserves_response_content(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Detailed research findings about AI"))

        results = await spawn_and_aggregate_sequential(
            runner, [(ResearchSubAgent(), "Research AI")]
        )
        assert "AI" in results[0]


class TestParallelSubAgents:
    async def test_parallel_single_agent(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Parallel result"))

        results = await spawn_and_aggregate_parallel(
            runner, [(ResearchSubAgent(), "Topic")]
        )
        assert len(results) == 1
        assert results[0] == "Parallel result"

    async def test_parallel_two_agents_all_results_collected(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Result A"))
        mock.queue_response(_completion("Result B"))

        results = await spawn_and_aggregate_parallel(
            runner, [
                (ResearchSubAgent(), "Topic A"),
                (WritingSubAgent(), "Topic B"),
            ]
        )
        assert len(results) == 2
        assert set(results) == {"Result A", "Result B"}

    async def test_parallel_results_are_strings(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("content one"))
        mock.queue_response(_completion("content two"))

        results = await spawn_and_aggregate_parallel(
            runner, [
                (ResearchSubAgent(), "Q1"),
                (WritingSubAgent(), "Q2"),
            ]
        )
        for r in results:
            assert isinstance(r, str)


class TestSubAgentRunnerApi:
    async def test_runner_run_returns_agent_response(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Hello from sub-agent"))

        response = await runner.run(ResearchSubAgent(), "Hello")
        assert response.content == "Hello from sub-agent"

    async def test_runner_records_turns(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Done"))

        response = await runner.run(SummarySubAgent(), "Summarise this")
        assert response.turns >= 1

    async def test_runner_run_multiple_times_sequential(self):
        """Same runner can be called multiple times."""
        runner, mock = _make_runner()
        mock.queue_response(_completion("First"))
        mock.queue_response(_completion("Second"))

        r1 = await runner.run(ResearchSubAgent(), "First prompt")
        r2 = await runner.run(WritingSubAgent(), "Second prompt")
        assert r1.content == "First"
        assert r2.content == "Second"

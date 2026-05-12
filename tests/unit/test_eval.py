"""Unit tests for the evaluation framework."""

from __future__ import annotations

import pytest

from lauren_ai._agents import AgentResponse
from lauren_ai._eval import (
    AccuracyEval,
    EvalDataset,
    EvalExample,
    EvalReport,
    EvalResult,
    TrajectoryEval,
)
from lauren_ai._transport import TokenUsage


class FakeAgentClient:
    """Minimal agent client returning a fixed response."""

    def __init__(self, response: str, tool_names: list[str] | None = None):
        self._response = response
        self._tool_names = tool_names or []

    async def run(self, message: str):
        class FakeToolCall:
            def __init__(self, name):
                self.name = name

        return AgentResponse(
            content=self._response,
            turns=1,
            total_usage=TokenUsage(input_tokens=10, output_tokens=5),
            tool_calls_made=[FakeToolCall(n) for n in self._tool_names],
            stop_reason="end_turn",
        )


class TestEvalDataset:
    def test_len(self):
        ds = EvalDataset(
            [
                EvalExample(input="q1", expected="a1"),
                EvalExample(input="q2", expected="a2"),
            ]
        )
        assert len(ds) == 2

    def test_iter(self):
        ds = EvalDataset([EvalExample(input="q1")])
        examples = list(ds)
        assert len(examples) == 1


class TestAccuracyEval:
    @pytest.mark.asyncio
    async def test_exact_match_pass(self):
        client = FakeAgentClient("Paris")
        evaluator = AccuracyEval(exact_match=True)
        dataset = EvalDataset([EvalExample(input="Capital of France?", expected="Paris")])
        report = await evaluator.run(client, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_fail(self):
        client = FakeAgentClient("Rome")
        evaluator = AccuracyEval(exact_match=True)
        dataset = EvalDataset([EvalExample(input="Capital of France?", expected="Paris")])
        report = await evaluator.run(client, dataset)
        assert report.pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_substring_match(self):
        client = FakeAgentClient("The capital of France is Paris, a beautiful city.")
        evaluator = AccuracyEval(exact_match=False)
        dataset = EvalDataset([EvalExample(input="Capital?", expected="Paris")])
        report = await evaluator.run(client, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_none_expected_always_passes(self):
        client = FakeAgentClient("anything")
        evaluator = AccuracyEval()
        dataset = EvalDataset([EvalExample(input="Tell me something", expected=None)])
        report = await evaluator.run(client, dataset)
        assert report.pass_rate == 1.0


class TestEvalReport:
    def test_pass_rate_empty(self):
        report = EvalReport(results=[])
        assert report.pass_rate == 0.0

    def test_pass_rate_all_pass(self):
        results = [EvalResult(example=EvalExample(input="x"), passed=True) for _ in range(3)]
        report = EvalReport(results=results)
        assert report.pass_rate == 1.0

    def test_pass_rate_mixed(self):
        results = [
            EvalResult(example=EvalExample(input="x"), passed=True),
            EvalResult(example=EvalExample(input="y"), passed=False),
        ]
        report = EvalReport(results=results)
        assert report.pass_rate == pytest.approx(0.5)

    def test_assert_pass_rate_passes(self):
        results = [EvalResult(example=EvalExample(input="x"), passed=True)]
        report = EvalReport(results=results)
        report.assert_pass_rate(0.5)  # Should not raise

    def test_assert_pass_rate_fails(self):
        results = [EvalResult(example=EvalExample(input="x"), passed=False)]
        report = EvalReport(results=results)
        with pytest.raises(AssertionError):
            report.assert_pass_rate(0.5)


class TestTrajectoryEval:
    @pytest.mark.asyncio
    async def test_exact_tool_order(self):
        client = FakeAgentClient("result", tool_names=["search", "summarise"])
        evaluator = TrajectoryEval(strict_order=True)
        dataset = EvalDataset(
            [
                EvalExample(
                    input="Research and summarise",
                    expected_tools=["search", "summarise"],
                )
            ]
        )
        report = await evaluator.run(client, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_wrong_order_fails_strict(self):
        client = FakeAgentClient("result", tool_names=["summarise", "search"])
        evaluator = TrajectoryEval(strict_order=True)
        dataset = EvalDataset(
            [
                EvalExample(
                    input="Research and summarise",
                    expected_tools=["search", "summarise"],
                )
            ]
        )
        report = await evaluator.run(client, dataset)
        assert report.pass_rate == 0.0

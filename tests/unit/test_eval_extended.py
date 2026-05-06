"""Extended tests for _eval/__init__.py — covers missing branches."""

from __future__ import annotations

import pytest

from lauren_ai._eval import (
    AccuracyEval,
    EvalDataset,
    EvalExample,
    EvalReport,
    EvalResult,
    PerformanceEval,
    TrajectoryEval,
)

# ---------------------------------------------------------------------------
# EvalReport extended tests
# ---------------------------------------------------------------------------


class TestEvalReportExtended:
    def test_avg_latency_empty(self):
        report = EvalReport(results=[])
        assert report.avg_latency_ms == pytest.approx(0.0)

    def test_avg_latency_with_results(self):
        results = [
            EvalResult(example=EvalExample(input="q1"), latency_ms=100.0),
            EvalResult(example=EvalExample(input="q2"), latency_ms=200.0),
        ]
        report = EvalReport(results=results)
        assert report.avg_latency_ms == pytest.approx(150.0)

    def test_avg_score_none_when_no_scores(self):
        results = [
            EvalResult(example=EvalExample(input="q1"), score=None),
        ]
        report = EvalReport(results=results)
        assert report.avg_score is None

    def test_avg_score_computed(self):
        results = [
            EvalResult(example=EvalExample(input="q1"), score=0.8),
            EvalResult(example=EvalExample(input="q2"), score=0.6),
        ]
        report = EvalReport(results=results)
        assert report.avg_score == pytest.approx(0.7)

    def test_avg_score_mixed_none(self):
        results = [
            EvalResult(example=EvalExample(input="q1"), score=0.8),
            EvalResult(example=EvalExample(input="q2"), score=None),
        ]
        report = EvalReport(results=results)
        assert report.avg_score == pytest.approx(0.8)

    def test_summary_string(self):
        results = [
            EvalResult(example=EvalExample(input="q1"), passed=True, latency_ms=100.0),
            EvalResult(example=EvalExample(input="q2"), passed=False, latency_ms=200.0),
        ]
        report = EvalReport(results=results, dataset_name="my_dataset", evaluator_name="acc")
        summary = report.summary()
        assert "my_dataset" in summary
        assert "acc" in summary
        assert "50.0%" in summary

    def test_summary_with_avg_score(self):
        results = [
            EvalResult(example=EvalExample(input="q1"), passed=True, score=0.9),
        ]
        report = EvalReport(results=results, evaluator_name="acc")
        summary = report.summary()
        assert "0.900" in summary or "score" in summary.lower()

    def test_assert_pass_rate_with_failed_examples(self):
        results = [
            EvalResult(example=EvalExample(input="What is 2+2?"), passed=False),
            EvalResult(example=EvalExample(input="What is 3+3?"), passed=False),
        ]
        report = EvalReport(results=results)
        with pytest.raises(AssertionError) as exc_info:
            report.assert_pass_rate(0.5)
        # Should mention failed examples
        assert "What is 2+2?" in str(exc_info.value)

    def test_assert_pass_rate_truncates_long_input(self):
        long_input = "A very long question that exceeds forty characters in length for sure"
        results = [EvalResult(example=EvalExample(input=long_input), passed=False)]
        report = EvalReport(results=results)
        with pytest.raises(AssertionError) as exc_info:
            report.assert_pass_rate(0.5)
        # The input should be truncated to 40 chars in the message
        assert long_input[:40] in str(exc_info.value)


# ---------------------------------------------------------------------------
# AccuracyEval extended tests
# ---------------------------------------------------------------------------


class TestAccuracyEvalExtended:
    @pytest.mark.asyncio
    async def test_callable_agent_client(self):
        """Test with a direct coroutine function as agent_client."""

        async def agent_fn(message: str) -> str:
            return f"Response to: {message}"

        evaluator = AccuracyEval(exact_match=False)
        dataset = EvalDataset([EvalExample(input="hello", expected="Response to: hello")])
        report = await evaluator.run(agent_fn, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_sync_callable_agent_client(self):
        """Test with a sync callable."""

        def sync_agent(message: str) -> str:
            return "fixed response"

        evaluator = AccuracyEval(exact_match=True)
        dataset = EvalDataset([EvalExample(input="hi", expected="fixed response")])
        report = await evaluator.run(sync_agent, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_agent_with_sync_run_method(self):
        """Test with a client that has a sync run() method."""

        class SyncClient:
            def run(self, message: str):
                return "sync result"

        evaluator = AccuracyEval(exact_match=True)
        dataset = EvalDataset([EvalExample(input="hi", expected="sync result")])
        report = await evaluator.run(SyncClient(), dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_agent_exception_marks_failed(self):
        async def failing_agent(message: str):
            raise RuntimeError("Agent error")

        evaluator = AccuracyEval()
        dataset = EvalDataset([EvalExample(input="hi", expected="anything")])
        report = await evaluator.run(failing_agent, dataset)
        assert report.pass_rate == 0.0
        assert report.results[0].error is not None

    @pytest.mark.asyncio
    async def test_latency_recorded(self):
        async def agent(message: str):
            return "ok"

        evaluator = AccuracyEval()
        dataset = EvalDataset([EvalExample(input="hi", expected=None)])
        report = await evaluator.run(agent, dataset)
        assert report.results[0].latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_dataset_name_in_report(self):
        async def agent(message: str):
            return "ok"

        evaluator = AccuracyEval(name="my_eval")
        dataset = EvalDataset([EvalExample(input="hi")], name="my_dataset")
        report = await evaluator.run(agent, dataset)
        assert report.dataset_name == "my_dataset"
        assert report.evaluator_name == "my_eval"

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self):
        async def agent(message: str):
            return "PARIS"

        evaluator = AccuracyEval(exact_match=True)
        dataset = EvalDataset([EvalExample(input="Capital?", expected="paris")])
        report = await evaluator.run(agent, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_substring_match_case_insensitive(self):
        async def agent(message: str):
            return "The capital city is PARIS."

        evaluator = AccuracyEval(exact_match=False)
        dataset = EvalDataset([EvalExample(input="Capital?", expected="paris")])
        report = await evaluator.run(agent, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_agent_response_with_content_attr(self):
        class Response:
            def __init__(self, content):
                self.content = content

        async def agent(message: str):
            return Response("Paris")

        evaluator = AccuracyEval(exact_match=True)
        dataset = EvalDataset([EvalExample(input="Capital?", expected="Paris")])
        report = await evaluator.run(agent, dataset)
        assert report.pass_rate == 1.0


# ---------------------------------------------------------------------------
# TrajectoryEval extended tests
# ---------------------------------------------------------------------------


class TestTrajectoryEvalExtended:
    @pytest.mark.asyncio
    async def test_non_strict_subset_match(self):
        class AgentClient:
            async def run(self, message: str):
                class FakeToolCall:
                    name = "search"

                class FakeToolCall2:
                    name = "summarise"

                class FakeResp:
                    tool_calls_made = [FakeToolCall(), FakeToolCall2()]

                return FakeResp()

        evaluator = TrajectoryEval(strict_order=False)
        dataset = EvalDataset([EvalExample(input="q", expected_tools=["search"])])
        report = await evaluator.run(AgentClient(), dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_no_expected_tools_always_passes(self):
        class AgentClient:
            async def run(self, message: str):
                class FakeResp:
                    tool_calls_made = []

                return FakeResp()

        evaluator = TrajectoryEval()
        dataset = EvalDataset([EvalExample(input="q", expected_tools=None)])
        report = await evaluator.run(AgentClient(), dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_callable_agent_no_run_method(self):
        async def agent_fn(message: str):
            class FakeResp:
                tool_calls_made = []

            return FakeResp()

        evaluator = TrajectoryEval()
        dataset = EvalDataset([EvalExample(input="q", expected_tools=[])])
        report = await evaluator.run(agent_fn, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_error_fails_result(self):
        async def failing_agent(message: str):
            raise RuntimeError("oops")

        # TrajectoryEval uses hasattr(agent_client, "run") check
        class FailingClient:
            async def run(self, message: str):
                raise RuntimeError("oops")

        evaluator = TrajectoryEval()
        dataset = EvalDataset([EvalExample(input="q", expected_tools=["search"])])
        report = await evaluator.run(FailingClient(), dataset)
        assert report.pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_actual_tools_recorded(self):
        class AgentClient:
            async def run(self, message: str):
                class TC:
                    name = "tool_a"

                class FakeResp:
                    tool_calls_made = [TC()]

                return FakeResp()

        evaluator = TrajectoryEval(strict_order=True)
        dataset = EvalDataset([EvalExample(input="q", expected_tools=["tool_a"])])
        report = await evaluator.run(AgentClient(), dataset)
        assert "tool_a" in report.results[0].actual

    @pytest.mark.asyncio
    async def test_tool_call_no_name_attr_uses_str(self):
        class AgentClient:
            async def run(self, message: str):
                class TC:
                    pass  # No name attr

                class FakeResp:
                    tool_calls_made = [TC()]

                return FakeResp()

        evaluator = TrajectoryEval(strict_order=True)
        dataset = EvalDataset([EvalExample(input="q", expected_tools=["something"])])
        report = await evaluator.run(AgentClient(), dataset)
        assert report.pass_rate == 0.0  # str(TC()) won't match "something"


# ---------------------------------------------------------------------------
# PerformanceEval tests
# ---------------------------------------------------------------------------


class TestPerformanceEval:
    @pytest.mark.asyncio
    async def test_all_pass_within_latency(self):
        class AgentClient:
            async def run(self, message: str):
                class FakeResp:
                    content = "ok"
                    total_usage = None

                return FakeResp()

        evaluator = PerformanceEval(max_latency_ms=10000)  # Very generous
        dataset = EvalDataset([EvalExample(input="q")])
        report = await evaluator.run(AgentClient(), dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_no_max_latency_always_passes(self):
        class AgentClient:
            async def run(self, message: str):
                class FakeResp:
                    content = "ok"
                    total_usage = None

                return FakeResp()

        evaluator = PerformanceEval(max_latency_ms=None)
        dataset = EvalDataset([EvalExample(input="q")])
        report = await evaluator.run(AgentClient(), dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_with_token_usage(self):
        class FakeUsage:
            input_tokens = 100
            output_tokens = 50

        class AgentClient:
            async def run(self, message: str):
                class FakeResp:
                    content = "ok"
                    total_usage = FakeUsage()

                return FakeResp()

        evaluator = PerformanceEval(max_latency_ms=None)
        dataset = EvalDataset([EvalExample(input="q")])
        report = await evaluator.run(AgentClient(), dataset)
        # Score should be set to total tokens
        assert report.results[0].score == pytest.approx(150.0)

    @pytest.mark.asyncio
    async def test_error_fails(self):
        class AgentClient:
            async def run(self, message: str):
                raise RuntimeError("error")

        evaluator = PerformanceEval()
        dataset = EvalDataset([EvalExample(input="q")])
        report = await evaluator.run(AgentClient(), dataset)
        assert report.pass_rate == 0.0
        assert report.results[0].error is not None

    @pytest.mark.asyncio
    async def test_callable_client(self):
        async def agent_fn(message: str):
            class FakeResp:
                content = "ok"
                total_usage = None

            return FakeResp()

        evaluator = PerformanceEval(max_latency_ms=None)
        dataset = EvalDataset([EvalExample(input="q")])
        report = await evaluator.run(agent_fn, dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_sync_run_method(self):
        class SyncClient:
            def run(self, message: str):
                class FakeResp:
                    content = "ok"
                    total_usage = None

                return FakeResp()

        evaluator = PerformanceEval(max_latency_ms=None)
        dataset = EvalDataset([EvalExample(input="q")])
        report = await evaluator.run(SyncClient(), dataset)
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_no_total_usage_score_is_none(self):
        class AgentClient:
            async def run(self, message: str):
                class FakeResp:
                    content = "ok"
                    total_usage = None

                return FakeResp()

        evaluator = PerformanceEval(max_latency_ms=None)
        dataset = EvalDataset([EvalExample(input="q")])
        report = await evaluator.run(AgentClient(), dataset)
        assert report.results[0].score is None

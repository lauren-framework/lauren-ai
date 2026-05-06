"""Agent evaluation framework for ``lauren-ai``.

Provides utilities for evaluating agent accuracy, tool-call trajectories,
and performance benchmarks.

Example::

    from lauren_ai.eval import EvalDataset, AccuracyEval, EvalExample

    dataset = EvalDataset([
        EvalExample(
            input="What is the capital of France?",
            expected="Paris",
        ),
        EvalExample(
            input="Translate 'hello' to Spanish.",
            expected="hola",
        ),
    ])

    evaluator = AccuracyEval(exact_match=False)
    report = await evaluator.run(agent_client, dataset)
    report.assert_pass_rate(min_pass_rate=0.9)
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EvalExample",
    "EvalDataset",
    "EvalResult",
    "EvalReport",
    "AccuracyEval",
    "TrajectoryEval",
    "PerformanceEval",
]


# ---------------------------------------------------------------------------
# EvalExample / EvalDataset
# ---------------------------------------------------------------------------


@dataclass
class EvalExample:
    """A single evaluation test case.

    :param input: The input message sent to the agent.
    :type input: str
    :param expected: The expected output (for exact/semantic matching).
    :type expected: str | None
    :param expected_tools: Expected tool names to be called, in order.
    :type expected_tools: list[str] | None
    :param metadata: Arbitrary metadata for filtering/grouping.
    :type metadata: dict[str, Any]
    """

    input: str
    expected: str | None = None
    expected_tools: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """A collection of :class:`EvalExample` objects.

    :param examples: The test cases.
    :type examples: list[EvalExample]
    :param name: Dataset name.
    :type name: str
    """

    examples: list[EvalExample]
    name: str = "eval_dataset"

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):  # type: ignore[override]
        return iter(self.examples)


# ---------------------------------------------------------------------------
# EvalResult / EvalReport
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of evaluating a single :class:`EvalExample`.

    :param example: The test case.
    :type example: EvalExample
    :param actual: The agent's actual output.
    :type actual: str
    :param passed: Whether the evaluation passed.
    :type passed: bool
    :param score: Optional numeric score (0.0–1.0).
    :type score: float | None
    :param latency_ms: Response latency in milliseconds.
    :type latency_ms: float
    :param error: Exception raised during evaluation, or ``None``.
    :type error: Exception | None
    """

    example: EvalExample
    actual: str = ""
    passed: bool = False
    score: float | None = None
    latency_ms: float = 0.0
    error: Exception | None = None


@dataclass
class EvalReport:
    """Summary of an evaluation run.

    :param results: Individual :class:`EvalResult` objects.
    :type results: list[EvalResult]
    :param dataset_name: The name of the evaluated dataset.
    :type dataset_name: str
    :param evaluator_name: The evaluator's name.
    :type evaluator_name: str
    """

    results: list[EvalResult]
    dataset_name: str = ""
    evaluator_name: str = ""

    @property
    def pass_rate(self) -> float:
        """Fraction of results that passed (0.0–1.0).

        :rtype: float
        """
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds across all results.

        :rtype: float
        """
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    @property
    def avg_score(self) -> float | None:
        """Average score when numeric scores are available.

        :rtype: float | None
        """
        scores = [r.score for r in self.results if r.score is not None]
        return sum(scores) / len(scores) if scores else None

    def assert_pass_rate(self, min_pass_rate: float = 0.8) -> None:
        """Assert that the pass rate meets a minimum threshold.

        :param min_pass_rate: Minimum required pass rate (0.0–1.0).
        :type min_pass_rate: float
        :raises AssertionError: When ``pass_rate < min_pass_rate``.
        """
        actual = self.pass_rate
        assert actual >= min_pass_rate, (
            f"Pass rate {actual:.1%} is below the minimum {min_pass_rate:.1%}. "
            f"Failed examples: "
            + "; ".join(f"[{r.example.input[:40]!r}]" for r in self.results if not r.passed)
        )

    def summary(self) -> str:
        """Return a human-readable summary string.

        :rtype: str
        """
        lines = [
            f"Eval: {self.evaluator_name!r}  Dataset: {self.dataset_name!r}",
            f"Pass rate: {self.pass_rate:.1%}  ({sum(r.passed for r in self.results)}/{len(self.results)})",
            f"Avg latency: {self.avg_latency_ms:.0f}ms",
        ]
        if self.avg_score is not None:
            lines.append(f"Avg score: {self.avg_score:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AccuracyEval
# ---------------------------------------------------------------------------


class AccuracyEval:
    """Evaluator that checks agent output against expected answers.

    :param exact_match: When ``True``, checks for exact string equality
        (case-insensitive and stripped).  When ``False``, checks that
        *expected* appears as a substring of *actual*.
    :type exact_match: bool
    :param name: Evaluator name.
    :type name: str
    """

    def __init__(
        self,
        exact_match: bool = False,
        name: str = "accuracy",
    ) -> None:
        self._exact = exact_match
        self.name = name

    async def run(self, agent_client: Any, dataset: EvalDataset) -> EvalReport:
        """Run all examples in *dataset* through *agent_client* and evaluate.

        :param agent_client: An object with a ``run(message: str) -> AgentResponse``
            method or any callable.
        :type agent_client: Any
        :param dataset: The dataset to evaluate.
        :type dataset: EvalDataset
        :return: The evaluation report.
        :rtype: EvalReport
        """
        results: list[EvalResult] = []
        for ex in dataset.examples:
            start = time.monotonic()
            error: Exception | None = None
            actual = ""
            try:
                if inspect.iscoroutinefunction(agent_client):
                    resp = await agent_client(ex.input)
                elif hasattr(agent_client, "run"):
                    if inspect.iscoroutinefunction(agent_client.run):
                        resp = await agent_client.run(ex.input)
                    else:
                        resp = agent_client.run(ex.input)
                else:
                    resp = agent_client(ex.input)
                actual = getattr(resp, "content", str(resp))
            except Exception as exc:
                error = exc

            latency = (time.monotonic() - start) * 1000.0
            passed = self._check(ex.expected, actual) if error is None else False
            results.append(
                EvalResult(
                    example=ex,
                    actual=actual,
                    passed=passed,
                    latency_ms=latency,
                    error=error,
                )
            )

        return EvalReport(
            results=results,
            dataset_name=dataset.name,
            evaluator_name=self.name,
        )

    def _check(self, expected: str | None, actual: str) -> bool:
        if expected is None:
            return True  # No expected value — always pass
        if self._exact:
            return expected.strip().lower() == actual.strip().lower()
        return expected.lower() in actual.lower()


# ---------------------------------------------------------------------------
# TrajectoryEval
# ---------------------------------------------------------------------------


class TrajectoryEval:
    """Evaluator that checks tool call trajectories.

    Verifies that the agent called the correct tools in the expected order.

    :param strict_order: When ``True``, tool order must exactly match
        *expected_tools*.  When ``False``, only set membership is checked.
    :type strict_order: bool
    :param name: Evaluator name.
    :type name: str
    """

    def __init__(self, strict_order: bool = True, name: str = "trajectory") -> None:
        self._strict = strict_order
        self.name = name

    async def run(self, agent_client: Any, dataset: EvalDataset) -> EvalReport:
        """Evaluate tool-call trajectories.

        :param agent_client: Agent client with a ``run()`` method returning
            an :class:`~lauren_ai._agents.AgentResponse`.
        :type agent_client: Any
        :param dataset: Dataset with examples that have ``expected_tools``.
        :type dataset: EvalDataset
        :return: The evaluation report.
        :rtype: EvalReport
        """
        results: list[EvalResult] = []
        for ex in dataset.examples:
            start = time.monotonic()
            error: Exception | None = None
            actual_tools: list[str] = []
            try:
                if hasattr(agent_client, "run"):
                    if inspect.iscoroutinefunction(agent_client.run):
                        resp = await agent_client.run(ex.input)
                    else:
                        resp = agent_client.run(ex.input)
                else:
                    resp = await agent_client(ex.input)
                actual_tools = [
                    getattr(tc, "name", str(tc)) for tc in getattr(resp, "tool_calls_made", [])
                ]
            except Exception as exc:
                error = exc

            latency = (time.monotonic() - start) * 1000.0
            passed = False
            if error is None and ex.expected_tools is not None:
                if self._strict:
                    passed = actual_tools == ex.expected_tools
                else:
                    passed = set(ex.expected_tools).issubset(set(actual_tools))
            elif error is None and ex.expected_tools is None:
                passed = True

            results.append(
                EvalResult(
                    example=ex,
                    actual=str(actual_tools),
                    passed=passed,
                    latency_ms=latency,
                    error=error,
                )
            )

        return EvalReport(
            results=results,
            dataset_name=dataset.name,
            evaluator_name=self.name,
        )


# ---------------------------------------------------------------------------
# PerformanceEval
# ---------------------------------------------------------------------------


class PerformanceEval:
    """Evaluator that measures agent latency and token efficiency.

    :param max_latency_ms: Maximum acceptable average latency in ms.
        ``None`` disables the latency check.
    :type max_latency_ms: float | None
    :param name: Evaluator name.
    :type name: str
    """

    def __init__(
        self,
        max_latency_ms: float | None = None,
        name: str = "performance",
    ) -> None:
        self._max_latency = max_latency_ms
        self.name = name

    async def run(self, agent_client: Any, dataset: EvalDataset) -> EvalReport:
        """Measure performance across the dataset.

        :param agent_client: Agent client.
        :type agent_client: Any
        :param dataset: Evaluation dataset.
        :type dataset: EvalDataset
        :return: The evaluation report.
        :rtype: EvalReport
        """
        results: list[EvalResult] = []
        for ex in dataset.examples:
            start = time.monotonic()
            error: Exception | None = None
            actual = ""
            total_tokens = 0
            try:
                if hasattr(agent_client, "run"):
                    if inspect.iscoroutinefunction(agent_client.run):
                        resp = await agent_client.run(ex.input)
                    else:
                        resp = agent_client.run(ex.input)
                else:
                    resp = await agent_client(ex.input)
                actual = getattr(resp, "content", str(resp))
                usage = getattr(resp, "total_usage", None)
                if usage:
                    total_tokens = getattr(usage, "input_tokens", 0) + getattr(
                        usage, "output_tokens", 0
                    )
            except Exception as exc:
                error = exc

            latency = (time.monotonic() - start) * 1000.0
            passed = error is None and (self._max_latency is None or latency <= self._max_latency)
            results.append(
                EvalResult(
                    example=ex,
                    actual=actual,
                    passed=passed,
                    score=float(total_tokens) if total_tokens else None,
                    latency_ms=latency,
                    error=error,
                )
            )

        return EvalReport(
            results=results,
            dataset_name=dataset.name,
            evaluator_name=self.name,
        )

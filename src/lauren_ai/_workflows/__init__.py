"""Structured workflows for multi-step AI pipelines.

Provides composable step primitives for building deterministic, multi-agent
pipelines where the control flow (sequence, parallel, conditional, loop) is
defined in code rather than left to the LLM.

Example::

    from lauren_ai.workflows import Workflow, Step, Parallel, Condition

    workflow = Workflow([
        Step("fetch", fetch_data),
        Parallel([
            Step("summarise", summarise),
            Step("classify", classify),
        ]),
        Condition(
            predicate=lambda ctx: ctx["classify"] == "urgent",
            if_true=Step("escalate", escalate),
            if_false=Step("log", log_result),
        ),
    ])

    result = await workflow.run({"input": "raw data"})
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Workflow",
    "Step",
    "Parallel",
    "Condition",
    "Loop",
    "StepResult",
    "WorkflowResult",
]


# ---------------------------------------------------------------------------
# StepResult / WorkflowResult
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single workflow step.

    :param name: Step name.
    :type name: str
    :param output: The step's output value.
    :type output: Any
    :param error: Exception raised during the step, or ``None``.
    :type error: Exception | None
    :param duration_ms: Step wall-clock time in milliseconds.
    :type duration_ms: float
    """

    name: str
    output: Any
    error: Exception | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """``True`` when the step completed without error."""
        return self.error is None


@dataclass
class WorkflowResult:
    """Result of a complete workflow run.

    :param steps: Ordered list of :class:`StepResult` objects.
    :type steps: list[StepResult]
    :param context: Final workflow context dict (accumulated outputs).
    :type context: dict[str, Any]
    """

    steps: list[StepResult] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """``True`` when all steps completed without error."""
        return all(s.ok for s in self.steps)

    def get(self, key: str, default: Any = None) -> Any:
        """Return context value for *key*.

        :param key: Context key.
        :type key: str
        :param default: Fallback value.
        :type default: Any
        :return: The context value or *default*.
        :rtype: Any
        """
        return self.context.get(key, default)


# ---------------------------------------------------------------------------
# Workflow primitives
# ---------------------------------------------------------------------------


class Step:
    """A single named workflow step.

    :param name: Step name (used as context key for the output).
    :type name: str
    :param fn: The async (or sync) callable to run.
    :type fn: Callable[..., Any]
    :param kwargs: Additional keyword arguments forwarded to *fn*.
    :type kwargs: Any
    """

    def __init__(self, name: str, fn: Callable[..., Any], **kwargs: Any) -> None:
        self.name = name
        self._fn = fn
        self._kwargs = kwargs

    async def run(self, context: dict[str, Any]) -> StepResult:
        """Execute the step with the current *context*.

        :param context: Accumulated workflow context.
        :type context: dict[str, Any]
        :return: The step result.
        :rtype: StepResult
        """
        import time  # noqa: PLC0415

        start = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(self._fn):
                output = await self._fn(context, **self._kwargs)
            else:
                output = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._fn(context, **self._kwargs)
                )
            return StepResult(
                name=self.name,
                output=output,
                duration_ms=(time.monotonic() - start) * 1000.0,
            )
        except Exception as exc:
            return StepResult(
                name=self.name,
                output=None,
                error=exc,
                duration_ms=(time.monotonic() - start) * 1000.0,
            )


class Parallel:
    """Run a list of steps (or sub-workflows) concurrently.

    :param steps: Steps to execute in parallel.
    :type steps: list[Step | Parallel | Condition | Loop]
    :param name: Name for this parallel group (used in the result context).
    :type name: str
    """

    def __init__(
        self,
        steps: list[Any],
        name: str = "parallel",
    ) -> None:
        self.name = name
        self._steps = steps

    async def run(self, context: dict[str, Any]) -> StepResult:
        """Execute all steps concurrently and collect results.

        :param context: Accumulated workflow context.
        :type context: dict[str, Any]
        :return: A :class:`StepResult` whose ``output`` is a list of
            :class:`StepResult` objects.
        :rtype: StepResult
        """
        import time  # noqa: PLC0415

        start = time.monotonic()
        results = await asyncio.gather(
            *[s.run(dict(context)) for s in self._steps],
            return_exceptions=True,
        )
        step_results: list[StepResult] = []
        for s, r in zip(self._steps, results):
            if isinstance(r, BaseException):
                step_results.append(StepResult(name=getattr(s, "name", "?"), output=None, error=r))
            else:
                step_results.append(r)
        return StepResult(
            name=self.name,
            output=step_results,
            duration_ms=(time.monotonic() - start) * 1000.0,
        )


class Condition:
    """Conditionally execute one of two branches.

    :param predicate: Function receiving the context dict; returns ``True``
        to run *if_true*, ``False`` to run *if_false*.
    :type predicate: Callable[[dict[str, Any]], bool]
    :param if_true: Step or sub-workflow to run when *predicate* is truthy.
    :type if_true: Any
    :param if_false: Step or sub-workflow to run when *predicate* is falsy.
        ``None`` skips execution.
    :type if_false: Any | None
    :param name: Name for this conditional.
    :type name: str
    """

    def __init__(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        if_true: Any,
        if_false: Any = None,
        name: str = "condition",
    ) -> None:
        self.name = name
        self._predicate = predicate
        self._if_true = if_true
        self._if_false = if_false

    async def run(self, context: dict[str, Any]) -> StepResult:
        """Evaluate *predicate* and run the appropriate branch.

        :param context: Accumulated workflow context.
        :type context: dict[str, Any]
        :return: The branch :class:`StepResult`, or an empty result when the
            false branch is ``None``.
        :rtype: StepResult
        """
        try:
            if asyncio.iscoroutinefunction(self._predicate):
                cond = await self._predicate(context)
            else:
                cond = self._predicate(context)
        except Exception as exc:
            return StepResult(name=self.name, output=None, error=exc)

        branch = self._if_true if cond else self._if_false
        if branch is None:
            return StepResult(name=self.name, output=None)
        return await branch.run(context)


class Loop:
    """Repeat a step until *condition* returns ``False`` or ``max_iterations``.

    :param step: The step to repeat.
    :type step: Any
    :param condition: Function receiving the context; returns ``True`` to
        continue looping.
    :type condition: Callable[[dict[str, Any]], bool]
    :param max_iterations: Safety cap on iterations.
    :type max_iterations: int
    :param name: Name for this loop.
    :type name: str
    """

    def __init__(
        self,
        step: Any,
        condition: Callable[[dict[str, Any]], bool],
        max_iterations: int = 10,
        name: str = "loop",
    ) -> None:
        self.name = name
        self._step = step
        self._condition = condition
        self._max = max_iterations

    async def run(self, context: dict[str, Any]) -> StepResult:
        """Execute the loop.

        :param context: Accumulated workflow context.
        :type context: dict[str, Any]
        :return: A :class:`StepResult` whose ``output`` is the list of per-
            iteration results.
        :rtype: StepResult
        """
        import time  # noqa: PLC0415

        start = time.monotonic()
        iteration_results: list[StepResult] = []

        for _ in range(self._max):
            try:
                if asyncio.iscoroutinefunction(self._condition):
                    keep_going = await self._condition(context)
                else:
                    keep_going = self._condition(context)
            except Exception as exc:
                return StepResult(
                    name=self.name,
                    output=iteration_results,
                    error=exc,
                    duration_ms=(time.monotonic() - start) * 1000.0,
                )
            if not keep_going:
                break

            result = await self._step.run(context)
            iteration_results.append(result)
            if result.error:
                break
            if result.output is not None:
                context[self._step.name] = result.output

        return StepResult(
            name=self.name,
            output=iteration_results,
            duration_ms=(time.monotonic() - start) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class Workflow:
    """A sequential workflow composed of steps, parallel groups, and conditions.

    :param steps: Ordered list of workflow steps.
    :type steps: list[Any]
    :param name: Workflow name (for logging/tracing).
    :type name: str
    """

    def __init__(self, steps: list[Any], *, name: str = "workflow") -> None:
        self._steps = steps
        self.name = name

    async def run(self, initial_context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute all steps in order, accumulating context.

        :param initial_context: Optional initial context dict.
        :type initial_context: dict[str, Any] | None
        :return: The :class:`WorkflowResult` with all step results.
        :rtype: WorkflowResult
        :raises WorkflowError: On step failure when error propagation is
            configured.
        """
        from lauren_ai._exceptions import WorkflowError  # noqa: PLC0415

        context: dict[str, Any] = dict(initial_context or {})
        step_results: list[StepResult] = []

        for step in self._steps:
            result = await step.run(context)
            step_results.append(result)
            if result.error:
                raise WorkflowError(
                    f"Workflow {self.name!r} failed at step {result.name!r}: "
                    f"{result.error}",
                    cause=result.error,
                )
            if result.output is not None:
                context[result.name] = result.output

        return WorkflowResult(steps=step_results, context=context)

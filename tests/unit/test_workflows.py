"""Unit tests for the workflow primitives."""
from __future__ import annotations

import pytest

from lauren_ai._workflows import (
    Condition,
    Loop,
    Parallel,
    Step,
    StepResult,
    Workflow,
    WorkflowResult,
)
from lauren_ai._exceptions import WorkflowError


class TestStep:
    @pytest.mark.asyncio
    async def test_basic_step(self):
        async def add_greeting(ctx):
            return f"Hello, {ctx.get('name', 'World')}!"

        step = Step("greet", add_greeting)
        result = await step.run({"name": "Alice"})
        assert result.ok
        assert result.output == "Hello, Alice!"
        assert result.name == "greet"

    @pytest.mark.asyncio
    async def test_step_error_captured(self):
        async def failing_step(ctx):
            raise ValueError("Step failed!")

        step = Step("fail", failing_step)
        result = await step.run({})
        assert not result.ok
        assert isinstance(result.error, ValueError)

    @pytest.mark.asyncio
    async def test_sync_step(self):
        def sync_fn(ctx):
            return ctx.get("x", 0) * 2

        step = Step("double", sync_fn)
        result = await step.run({"x": 5})
        assert result.ok
        assert result.output == 10


class TestParallel:
    @pytest.mark.asyncio
    async def test_parallel_runs_all(self):
        calls = []

        async def fn_a(ctx):
            calls.append("a")
            return "a"

        async def fn_b(ctx):
            calls.append("b")
            return "b"

        group = Parallel([Step("a", fn_a), Step("b", fn_b)])
        result = await group.run({})
        assert result.ok
        assert len(result.output) == 2
        assert set(calls) == {"a", "b"}


class TestCondition:
    @pytest.mark.asyncio
    async def test_true_branch(self):
        async def true_step(ctx):
            return "true path"

        async def false_step(ctx):
            return "false path"

        cond = Condition(
            predicate=lambda ctx: ctx.get("flag") is True,
            if_true=Step("true", true_step),
            if_false=Step("false", false_step),
        )
        result = await cond.run({"flag": True})
        assert result.output == "true path"

    @pytest.mark.asyncio
    async def test_false_branch(self):
        async def true_step(ctx):
            return "true path"

        async def false_step(ctx):
            return "false path"

        cond = Condition(
            predicate=lambda ctx: ctx.get("flag") is True,
            if_true=Step("true", true_step),
            if_false=Step("false", false_step),
        )
        result = await cond.run({"flag": False})
        assert result.output == "false path"

    @pytest.mark.asyncio
    async def test_no_false_branch_skips(self):
        cond = Condition(
            predicate=lambda ctx: False,
            if_true=Step("x", lambda ctx: "x"),
        )
        result = await cond.run({})
        assert result.output is None


class TestLoop:
    @pytest.mark.asyncio
    async def test_runs_until_condition_false(self):
        count = [0]

        async def increment(ctx):
            count[0] += 1
            return count[0]

        step = Step("increment", increment)
        loop = Loop(step, condition=lambda ctx: count[0] < 3, max_iterations=10)
        result = await loop.run({})
        assert result.ok
        assert count[0] == 3

    @pytest.mark.asyncio
    async def test_max_iterations_cap(self):
        count = [0]

        async def increment(ctx):
            count[0] += 1
            return count[0]

        step = Step("increment", increment)
        loop = Loop(step, condition=lambda ctx: True, max_iterations=5)
        result = await loop.run({})
        assert count[0] == 5


class TestWorkflow:
    @pytest.mark.asyncio
    async def test_sequential_steps(self):
        async def step1(ctx):
            return ctx.get("input", "") + " -> step1"

        async def step2(ctx):
            return ctx.get("step1", "") + " -> step2"

        wf = Workflow([Step("step1", step1), Step("step2", step2)])
        result = await wf.run({"input": "start"})
        assert result.ok
        assert "step1" in result.context
        assert "step2" in result.context
        assert result.context["step2"] == "start -> step1 -> step2"

    @pytest.mark.asyncio
    async def test_failed_step_raises(self):
        async def failing(ctx):
            raise RuntimeError("Boom!")

        wf = Workflow([Step("boom", failing)])
        with pytest.raises(WorkflowError):
            await wf.run({})

    @pytest.mark.asyncio
    async def test_result_ok_true(self):
        wf = Workflow([Step("noop", lambda ctx: None)])
        result = await wf.run({})
        assert result.ok

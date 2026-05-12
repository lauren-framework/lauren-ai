"""Unit tests for agent teams."""

from __future__ import annotations

import pytest

from lauren_ai._config import LLMConfig
from lauren_ai._exceptions import DecoratorUsageError
from lauren_ai._module import LLMService
from lauren_ai._teams._decorator import TEAM_META, TeamConfigError, TeamMeta, team
from lauren_ai._teams._memory import TeamMemory
from lauren_ai._teams._runner import TeamResult, TeamRunner
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


def _make_completion(content: str) -> Completion:
    """Build a minimal Completion object for use in tests."""
    return Completion(
        id="mock-id",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=10),
    )


class TestTeamDecorator:
    def test_attaches_metadata(self):
        @team(name="research", mode="coordinator", model="claude-haiku-4-5")
        class MyTeam:
            pass

        meta: TeamMeta = getattr(MyTeam, TEAM_META)
        assert meta.name == "research"
        assert meta.mode == "coordinator"
        assert meta.model == "claude-haiku-4-5"

    def test_defaults_name_to_classname(self):
        @team(mode="coordinator", model="m")
        class AwesomeTeam:
            pass

        meta: TeamMeta = getattr(AwesomeTeam, TEAM_META)
        assert meta.name == "AwesomeTeam"

    def test_bare_usage_raises(self):
        with pytest.raises(DecoratorUsageError, match="parentheses"):

            @team
            class Bad:
                pass

    def test_invalid_mode_raises(self):
        with pytest.raises(TeamConfigError, match="mode"):

            @team(name="t", mode="invalid", model="m")  # type: ignore[arg-type]
            class Bad:
                pass

    def test_collaborate_mode(self):
        @team(mode="collaborate", model="m")
        class ColabTeam:
            pass

        meta: TeamMeta = getattr(ColabTeam, TEAM_META)
        assert meta.mode == "collaborate"

    def test_max_rounds_stored(self):
        @team(mode="coordinator", model="m", max_rounds=10)
        class T:
            pass

        assert getattr(T, TEAM_META).max_rounds == 10

    def test_custom_coordinator_prompt_stored(self):
        @team(mode="coordinator", model="m", coordinator_prompt="custom prompt")
        class T:
            pass

        assert getattr(T, TEAM_META).coordinator_prompt == "custom prompt"

    def test_does_not_mutate_class(self):
        set(object.__dict__.keys())

        @team(mode="coordinator", model="m")
        class MyTeam:
            def my_method(self) -> None:
                pass

        # class still has its method
        assert hasattr(MyTeam, "my_method")
        # TEAM_META was added
        assert hasattr(MyTeam, TEAM_META)


class TestTeamMemory:
    async def test_set_and_get(self):
        mem = TeamMemory()
        await mem.set("researcher", "found some data")
        result = await mem.get("researcher")
        assert result == "found some data"

    async def test_get_missing_returns_default(self):
        mem = TeamMemory()
        assert await mem.get("missing") is None
        assert await mem.get("missing", "default") == "default"

    async def test_get_all_returns_all(self):
        mem = TeamMemory()
        await mem.set("a", "1")
        await mem.set("b", "2")
        all_data = await mem.get_all()
        assert all_data == {"a": "1", "b": "2"}

    async def test_clear(self):
        mem = TeamMemory()
        await mem.set("x", "y")
        await mem.clear()
        assert len(mem) == 0

    def test_len(self):
        mem = TeamMemory()
        assert len(mem) == 0

    async def test_overwrite(self):
        mem = TeamMemory()
        await mem.set("k", "v1")
        await mem.set("k", "v2")
        assert await mem.get("k") == "v2"

    async def test_get_all_is_copy(self):
        mem = TeamMemory()
        await mem.set("k", "v")
        snapshot = await mem.get_all()
        snapshot["k"] = "mutated"
        # Original should be unchanged
        assert await mem.get("k") == "v"


def _make_llm(responses: list[str]) -> tuple[LLMService, MockTransport]:
    """Build an LLMService backed by a MockTransport with queued responses."""
    transport = MockTransport()
    for r in responses:
        transport.queue_response(_make_completion(r))
    config, _ = LLMConfig.for_testing()
    llm = LLMService(transport=transport, config=config)
    return llm, transport


class TestTeamRunner:
    def test_runner_requires_team_decorator(self):
        class NotATeam:
            pass

        llm, _ = _make_llm([])
        with pytest.raises(TeamConfigError, match="@team"):
            TeamRunner(team_cls=NotATeam, llm=llm, agent_runner=None)

    async def test_coordinator_mode_runs(self):
        @team(name="t", mode="coordinator", model="m")
        class MyTeam:
            def __init__(self, researcher: object, writer: object) -> None:
                pass

        # Responses: coordinator says ROUTE researcher, then worker output, then DONE
        llm, _ = _make_llm(
            [
                "ROUTE: researcher",  # coordinator decision round 0
                "Research findings",  # researcher worker output
                "DONE: Final answer",  # coordinator decision round 1
            ]
        )
        runner = TeamRunner(team_cls=MyTeam, llm=llm, agent_runner=None)
        result = await runner.run("Research AI trends")
        assert isinstance(result, TeamResult)
        assert result.final_answer == "Final answer"
        assert result.rounds >= 1

    async def test_collaborate_mode_runs(self):
        @team(name="t", mode="collaborate", model="m")
        class MyTeam:
            def __init__(self, analyst: object) -> None:
                pass

        llm, _ = _make_llm(
            [
                "Analysis result",  # analyst output
                "Final synthesis",  # synthesis
            ]
        )
        runner = TeamRunner(team_cls=MyTeam, llm=llm, agent_runner=None)
        result = await runner.run("Analyse this data")
        assert result.final_answer == "Final synthesis"
        assert result.worker_outputs == {"analyst": "Analysis result"}

    async def test_stream_emits_events(self):
        @team(name="t", mode="collaborate", model="m")
        class MyTeam:
            def __init__(self, worker: object) -> None:
                pass

        llm, _ = _make_llm(
            [
                "Worker output",
                "Final answer",
            ]
        )
        runner = TeamRunner(team_cls=MyTeam, llm=llm, agent_runner=None)
        events: list[object] = []
        async for event in runner.run_stream("Task"):
            events.append(event)

        event_types = [type(e).__name__ for e in events]
        assert "TeamWorkerStarted" in event_types
        assert "TeamWorkerFinished" in event_types
        assert "TeamFinalAnswer" in event_types

    async def test_coordinator_stream_emits_decision_events(self):
        @team(name="t", mode="coordinator", model="m")
        class MyTeam:
            def __init__(self, worker: object) -> None:
                pass

        llm, _ = _make_llm(
            [
                "DONE: All done",  # coordinator immediately done
            ]
        )
        runner = TeamRunner(team_cls=MyTeam, llm=llm, agent_runner=None)
        events: list[object] = []
        async for event in runner.run_stream("Task"):
            events.append(event)

        event_types = [type(e).__name__ for e in events]
        assert "TeamCoordinatorDecision" in event_types
        assert "TeamFinalAnswer" in event_types

    async def test_worker_names_discovered_from_init(self):
        @team(name="t", mode="collaborate", model="m")
        class MyTeam:
            def __init__(self, alpha: object, beta: object) -> None:
                pass

        llm, _ = _make_llm(["out_a", "out_b", "synth"])
        runner = TeamRunner(team_cls=MyTeam, llm=llm, agent_runner=None)
        assert runner._worker_names == ["alpha", "beta"]

    async def test_max_rounds_respected(self):
        @team(name="t", mode="coordinator", model="m", max_rounds=2)
        class MyTeam:
            def __init__(self, w: object) -> None:
                pass

        # Always routes to worker, never says DONE — should hit max_rounds
        llm, _ = _make_llm(
            [
                "ROUTE: w",
                "worker result 1",
                "ROUTE: w",
                "worker result 2",
            ]
        )
        runner = TeamRunner(team_cls=MyTeam, llm=llm, agent_runner=None)
        result = await runner.run("Task")
        assert result.rounds == 2

"""Stream events emitted during TeamRunner.run_stream()."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TeamEvent:
    """Base class for team stream events."""

    ...


@dataclass
class TeamWorkerStarted(TeamEvent):
    """Emitted when a worker agent begins its sub-task."""

    worker_name: str = ""
    task: str = ""
    round: int = 0


@dataclass
class TeamWorkerFinished(TeamEvent):
    """Emitted when a worker agent completes its sub-task."""

    worker_name: str = ""
    result_content: str = ""
    round: int = 0


@dataclass
class TeamCoordinatorDecision(TeamEvent):
    """Emitted when the coordinator decides next action."""

    decision: str = ""  # "ROUTE: <name>" or "DONE"
    reasoning: str | None = None
    round: int = 0


@dataclass
class TeamFinalAnswer(TeamEvent):
    """Emitted when the team produces its final answer."""

    content: str = ""
    rounds: int = 0


__all__ = [
    "TeamEvent",
    "TeamWorkerStarted",
    "TeamWorkerFinished",
    "TeamCoordinatorDecision",
    "TeamFinalAnswer",
]

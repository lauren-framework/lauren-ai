"""@team() decorator for multi-agent coordination."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from lauren_ai._exceptions import DecoratorUsageError, LaurenAIError

TEAM_META = "__lauren_ai_team__"
C = TypeVar("C", bound=type)

VALID_MODES = ("coordinator", "collaborate")


class TeamConfigError(LaurenAIError):
    """Raised at startup when a @team() class is misconfigured."""


@dataclass
class TeamMeta:
    """Attached to the class by @team()."""

    name: str
    mode: Literal["coordinator", "collaborate"]
    model: str
    max_rounds: int = 5
    coordinator_prompt: str | None = None


def team(
    *args: Any,
    name: str = "",
    mode: Literal["coordinator", "collaborate"] = "coordinator",
    model: str = "claude-haiku-4-5",
    max_rounds: int = 5,
    coordinator_prompt: str | None = None,
) -> Callable[[C], C]:
    """Mark a class as a multi-agent team.

    Must be called with parentheses::

        @team(name="research_team", mode="coordinator", model="claude-haiku-4-5")
        class ResearchTeam:
            def __init__(
                self,
                researcher: WebResearcherAgent,
                writer: WriterAgent,
            ) -> None: ...

    The constructor declares worker agents as typed parameters; they are
    resolved from the DI container at startup.

    :param name: Human-readable team name (defaults to class name).
    :param mode: "coordinator" routes sub-tasks one at a time; "collaborate"
                 runs all workers sequentially and synthesises.
    :param model: Model for the coordinator LLM calls.
    :param max_rounds: Maximum coordinator-worker rounds before stopping.
    :param coordinator_prompt: Override the default routing prompt.
    """
    if args:
        raise DecoratorUsageError("@team must be called with parentheses: @team(name=..., mode=..., model=...)")

    def _apply(cls: C) -> C:
        effective_name = name or cls.__name__
        if mode not in VALID_MODES:
            raise TeamConfigError(f"@team() mode must be one of {VALID_MODES}, got {mode!r}")
        meta = TeamMeta(
            name=effective_name,
            mode=mode,
            model=model,
            max_rounds=max_rounds,
            coordinator_prompt=coordinator_prompt,
        )
        setattr(cls, TEAM_META, meta)
        return cls

    return _apply

# Agent Teams

Multi-agent coordination via coordinator and collaborate modes.

## Decorator

### `team`

Mark a class as a multi-agent team.

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

## Metadata

### `TeamMeta`

Attached to the class by @team().

## Runner & result

### `TeamRunner`

Orchestrates a @team() class through its workers.

Usage::

    runner = TeamRunner(team_cls=ResearchTeam, llm=llm_service)
    result = await runner.run("Summarise recent news about AI.")

### `TeamResult`

Final result of a team run.

## Events

### `TeamWorkerStarted`

Emitted when a worker agent begins its sub-task.

### `TeamWorkerFinished`

Emitted when a worker agent completes its sub-task.

### `TeamCoordinatorDecision`

Emitted when the coordinator decides next action.

### `TeamFinalAnswer`

Emitted when the team produces its final answer.


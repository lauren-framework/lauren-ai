---
name: building-teams
description: Builds multi-agent teams in lauren-ai using @team and TeamRunner for coordinator and collaborate modes. Use when implementing a @team() class with multiple workers, running TeamRunner.run() or run_stream(), handling TeamWorkerStarted/Finished/TeamFinalAnswer events, or streaming team progress over SSE in a web controller.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.


# Building Multi-Agent Teams

## Quick start

Define a team and run it:

```python
from __future__ import annotations

from lauren_ai import team, TeamRunner
from .agents import ResearchAgent, WriterAgent

@team(
    name="content-team",
    mode="collaborate",
    model="claude-opus-4-6",
)
class ContentTeam:
    def __init__(self, researcher: ResearchAgent, writer: WriterAgent) -> None:
        self.researcher = researcher
        self.writer = writer


# Run
runner = TeamRunner(
    team_cls=ContentTeam,
    llm=llm_service,
    agent_runner=agent_runner,
)
result = await runner.run("Write a report on quantum computing.")
print(result.final_answer)
```

**Rules:**
- `@team()` must use parentheses.
- Worker parameters in `__init__` must be type-annotated with their agent class.
- `TeamRunner` discovers workers by reading `__init__.__annotations__`.

---

## Modes

| Mode | Behaviour | Use when |
|------|-----------|----------|
| `coordinator` | LLM coordinator decides after each round: `ROUTE: <worker>` or `DONE: <answer>`. Runs up to `max_rounds` rounds. | Tasks need dynamic routing between specialists |
| `collaborate` | All workers run sequentially in annotation order, then a synthesis call produces the final answer. | Tasks need all workers to contribute in a fixed sequence |

---

## Streaming events

`TeamRunner.run_stream()` yields `TeamEvent` subclass instances:

| Event class | Fields | Meaning |
|-------------|--------|---------|
| `TeamWorkerStarted` | `worker_name`, `task`, `round` | A worker is about to run |
| `TeamWorkerFinished` | `worker_name`, `result_content`, `round` | A worker completed |
| `TeamCoordinatorDecision` | `decision`, `round` | Coordinator chose next step |
| `TeamFinalAnswer` | `content`, `rounds` | Team produced its final answer |

```python
from lauren_ai import (
    TeamWorkerStarted,
    TeamWorkerFinished,
    TeamCoordinatorDecision,
    TeamFinalAnswer,
)

async for event in runner.run_stream("Research quantum computing"):
    if isinstance(event, TeamWorkerStarted):
        print(f"[Round {event.round}] Starting: {event.worker_name}")
    elif isinstance(event, TeamWorkerFinished):
        print(f"[Round {event.round}] {event.worker_name} done")
    elif isinstance(event, TeamCoordinatorDecision):
        print(f"[Coordinator] {event.decision}")
    elif isinstance(event, TeamFinalAnswer):
        print(f"[Final] {event.content}")
```

---

## `TeamResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | `str` | Synthesised final output |
| `worker_outputs` | `dict[str, str]` | Per-worker outputs keyed by worker name |
| `rounds` | `int` | Number of coordinator rounds taken |
| `total_input_tokens` | `int` | Cumulative input tokens (if tracked) |
| `total_output_tokens` | `int` | Cumulative output tokens (if tracked) |

---

## Reference

See [teams.md](teams.md) for:

- Full `@team()` parameter reference (`coordinator_prompt`, `max_rounds`)
- Custom coordinator prompt with `{worker_descriptions}`, `{task}`, `{prior_outputs}` placeholders
- SSE streaming in a web controller with `EventStream` and `ServerSentEvent`
- `collaborate` vs `coordinator` mode detailed behaviour

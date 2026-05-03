# Multi-Agent Teams

Teams allow multiple specialist agents to collaborate on a task.  The
`@team()` decorator marks a class as a team definition; `TeamRunner`
orchestrates its workers.

---

## Defining a team

```python
from __future__ import annotations

from lauren_ai import team
from .agents import ResearchAgent, AnalysisAgent, WriterAgent

@team(
    name="content-team",
    mode="coordinator",          # "coordinator" | "collaborate"
    model="claude-opus-4-6",     # Model for coordinator LLM calls
    max_rounds=5,                # Max coordinator-worker rounds
    coordinator_prompt=None,     # Override default routing prompt
)
class ContentTeam:
    def __init__(
        self,
        researcher: ResearchAgent,
        analyst: AnalysisAgent,
        writer: WriterAgent,
    ) -> None:
        self.researcher = researcher
        self.analyst = analyst
        self.writer = writer
```

**Rules:**
- `@team()` must use parentheses.
- Worker parameters in `__init__` must be type-annotated with the agent class.
- `TeamRunner` discovers workers by reading `__init__.__annotations__`.

---

## Modes

### `coordinator` mode

The coordinator LLM decides after each round which worker to call next,
or declares the task done.  Responses must follow the format:

```
ROUTE: <worker_name>   — hand off to the named worker
DONE: <final answer>   — task is complete, provide the answer
```

The default `coordinator_prompt` template accepts `{worker_descriptions}`,
`{task}`, and `{prior_outputs}` placeholders.

### `collaborate` mode

All workers run sequentially (in `__init__` annotation order).  After all
workers complete, the coordinator synthesises a final answer from all outputs.

---

## Custom coordinator prompt

```python
MY_PROMPT = """\
You coordinate a research team.

Available workers:
{worker_descriptions}

Task: {task}

Work completed:
{prior_outputs}

Next step (ROUTE: <worker> or DONE: <answer>):
"""

@team(
    name="research-team",
    mode="coordinator",
    model="claude-opus-4-6",
    coordinator_prompt=MY_PROMPT,
)
class ResearchTeam:
    def __init__(self, researcher: ResearchAgent, writer: WriterAgent) -> None:
        ...
```

---

## Running a team

```python
from lauren_ai import TeamRunner

runner = TeamRunner(
    team_cls=ResearchTeam,
    llm=llm_service,          # raw LLMService (from LLMProvider.llm_service_instance)
    agent_runner=agent_runner, # shared AgentRunner singleton
)

# Blocking
result = await runner.run("Summarise the latest AI safety research.")
print(result.final_answer)
print(result.worker_outputs)  # {"researcher": "...", "writer": "..."}
print(result.rounds)          # int
```

---

## Streaming team events

```python
from lauren_ai import (
    TeamWorkerStarted,
    TeamWorkerFinished,
    TeamCoordinatorDecision,
    TeamFinalAnswer,
)

async for event in runner.run_stream("Research quantum computing"):
    if isinstance(event, TeamWorkerStarted):
        print(f"[Round {event.round}] Starting worker: {event.worker_name}")

    elif isinstance(event, TeamWorkerFinished):
        print(f"[Round {event.round}] {event.worker_name} finished:")
        print(f"  Preview: {event.result_content[:200]}")

    elif isinstance(event, TeamCoordinatorDecision):
        print(f"[Coordinator] {event.decision}")

    elif isinstance(event, TeamFinalAnswer):
        print(f"[Final Answer] ({event.rounds} rounds)")
        print(event.content)
```

---

## SSE streaming in a web controller

```python
import json
from lauren import EventStream, ServerSentEvent, controller, post

@controller("/api/team")
class TeamController:
    def __init__(self, runner: TeamRunner) -> None:
        self._runner = runner

    @post("/")
    async def run(self, body: Json[TeamRequest]) -> EventStream:
        async def generate():
            async for event in self._runner.run_stream(body.task):
                if isinstance(event, TeamWorkerStarted):
                    yield ServerSentEvent(
                        event="worker_started",
                        data=json.dumps({"worker": event.worker_name, "round": event.round}),
                    )
                elif isinstance(event, TeamWorkerFinished):
                    yield ServerSentEvent(
                        event="worker_finished",
                        data=json.dumps({"worker": event.worker_name, "preview": event.result_content[:200]}),
                    )
                elif isinstance(event, TeamCoordinatorDecision):
                    yield ServerSentEvent(event="coordinator", data=event.decision)
                elif isinstance(event, TeamFinalAnswer):
                    yield ServerSentEvent(event="team_done", data=event.content)
            yield ServerSentEvent(event="done", data="")

        return EventStream(generate(), keep_alive=30.0)
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

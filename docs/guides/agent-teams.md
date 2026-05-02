# Agent Teams

Agent teams let you compose multiple specialist agents into a coordinated unit.
`lauren-ai` provides two coordination strategies and a shared memory bus so
workers can build on each other's results.

## Quick start

```python
from lauren_ai import agent
from lauren_ai._teams import team, TeamRunner

@agent(model="claude-haiku-4-5", system="You are a web researcher.")
class ResearchAgent:
    pass

@agent(model="claude-haiku-4-5", system="You are a concise writer.")
class WriterAgent:
    pass

@team(name="content_team", mode="coordinator", model="claude-haiku-4-5")
class ContentTeam:
    def __init__(
        self,
        researcher: ResearchAgent,
        writer: WriterAgent,
    ) -> None: ...
```

The constructor's typed parameters declare the workers the team can call.
Names matter — the coordinator references them by name when routing.

## Modes

### `mode="coordinator"`

The coordinator LLM calls `_coordinator_decide()` each round.  It responds with:

```
ROUTE: <worker_name>   # delegate next step to this worker
DONE: <final answer>   # synthesis complete
```

The loop continues until `DONE` is returned or `max_rounds` is exhausted.
This mode is best when tasks naturally decompose into sequential hand-offs.

### `mode="collaborate"`

Every worker runs in sequence on the same task; their outputs accumulate in
`TeamMemory`.  A final synthesis call summarises all outputs.
Use this for tasks where every specialist should weigh in, such as code review
or multi-angle analysis.

## Running a team

```python
from lauren_ai._teams import TeamRunner

runner = TeamRunner(
    team_cls=ContentTeam,
    llm=llm_service,    # LLMService instance
    agent_runner=None,  # pass AgentRunner for nested agent execution
)

result = await runner.run("Write a short guide on asyncio.")
print(result.final_answer)
print(result.worker_outputs)   # per-worker raw output
print(result.rounds)           # number of coordinator rounds
```

### Streaming

```python
async for event in runner.run_stream("Analyse this data"):
    match event:
        case TeamWorkerStarted():
            print(f"Starting {event.worker_name} (round {event.round})")
        case TeamWorkerFinished():
            print(f"Done: {event.result_content[:80]}")
        case TeamCoordinatorDecision():
            print(f"Coordinator: {event.decision}")
        case TeamFinalAnswer():
            print(f"Final: {event.content}")
```

## Stream events

| Event class | When emitted |
|---|---|
| `TeamWorkerStarted` | Before a worker begins its sub-task |
| `TeamWorkerFinished` | After a worker completes |
| `TeamCoordinatorDecision` | When coordinator chooses ROUTE or DONE |
| `TeamFinalAnswer` | When the team produces its final answer |

## TeamMemory

Workers share a `TeamMemory` instance for the duration of one `run()` call.
The runner writes each worker's output automatically; you can also inject and
read from it if workers are real agents that accept DI dependencies.

```python
memory = TeamMemory()
await memory.set("context", {"user_id": 42})
ctx = await memory.get("context")
all_data = await memory.get_all()
```

Memory is ephemeral — it is created fresh for each `run()` call.

## Custom coordinator prompt

Override the default routing prompt via `coordinator_prompt`:

```python
@team(
    name="qa_team",
    mode="coordinator",
    model="claude-haiku-4-5",
    coordinator_prompt=(
        "Workers: {worker_descriptions}\n"
        "Task: {task}\n"
        "Prior: {prior_outputs}\n"
        "Reply with ROUTE: <name> or DONE: <answer>"
    ),
)
class QATeam:
    def __init__(self, checker: CheckerAgent, fixer: FixerAgent) -> None: ...
```

Available template variables: `{worker_descriptions}`, `{task}`, `{prior_outputs}`.

## `TeamResult`

```python
@dataclass
class TeamResult:
    final_answer: str
    worker_outputs: dict[str, str]   # keyed by worker parameter name
    rounds: int
    total_input_tokens: int
    total_output_tokens: int
```

## Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `name` | class name | Human-readable team identifier |
| `mode` | `"coordinator"` | `"coordinator"` or `"collaborate"` |
| `model` | `"claude-haiku-4-5"` | Model for coordinator/synthesis calls |
| `max_rounds` | `5` | Maximum coordinator rounds before stopping |
| `coordinator_prompt` | built-in | Override the routing prompt template |

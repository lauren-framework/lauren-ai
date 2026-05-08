---
name: plan-and-execute-agent
description: Implements a two-phase Plan-and-Execute agent pattern where a Planner agent produces a step-by-step plan and an Executor agent runs each step using tools. Use when a user request is too complex for a single ReAct loop and benefits from explicit planning before execution.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Plan-and-Execute Agent Pattern

## Architecture

```
User request
  → PlannerAgent  → JSON plan {"steps": ["step1", "step2", ...]}
  → for each step:
      ExecutorAgent + tools → result string
  → collect results → final answer
```

---

## Quick start

```python
# agents.py — from __future__ import annotations IS safe here
from __future__ import annotations

import json
from lauren_ai import agent, use_tools
from .tools import search_web, calculate

PLANNER_SYSTEM = """Break the user request into numbered execution steps.
Respond ONLY with valid JSON: {"steps": ["step 1", "step 2", ...]}
No extra text outside the JSON."""

EXECUTOR_SYSTEM = """Execute the given step using available tools.
Return a concise result."""

@agent(model="claude-opus-4-6", system=PLANNER_SYSTEM, max_turns=1)
class PlannerAgent: ...

@agent(model="claude-opus-4-6", system=EXECUTOR_SYSTEM, max_turns=5)
@use_tools(search_web, calculate)
class ExecutorAgent: ...


async def plan_and_execute(
    runner,
    request: str,
    planner: PlannerAgent,
    executor: ExecutorAgent,
) -> str:
    """Run the two-phase plan-and-execute loop."""
    # Phase 1: plan
    plan_response = await runner.run(planner, request)
    plan = json.loads(plan_response.content)

    # Phase 2: execute each step
    results = []
    for step in plan.get("steps", []):
        result = await runner.run(executor, step)
        results.append(f"Step: {step}\nResult: {result.content}")

    return "\n\n".join(results)
```

---

## Testing

Queue the planner JSON response, then one executor response per step:

```python
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()

# Planner returns a 2-step plan
mock.queue_response(Completion(
    id="p1", model="mock-model",
    content='{"steps": ["Search for Python trends", "Summarize findings"]}',
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=20, output_tokens=15),
))

# Executor step 1
mock.queue_response(Completion(
    id="e1", model="mock-model",
    content="Python remains top language in 2025.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=15, output_tokens=10),
))

# Executor step 2
mock.queue_response(Completion(
    id="e2", model="mock-model",
    content="Summary: Python is widely used for AI and data science.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=15, output_tokens=12),
))
```

---

## Shared state pattern

Pass a shared `dict` through `metadata=` for state that spans planner and executor:

```python
state: dict = {}
plan_resp = await runner.run(planner, request, metadata={"state": state})
for step in steps:
    await runner.run(executor, step, metadata={"state": state})
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_agents/_runner.py` | `AgentRunnerBase.run()` — single-agent loop |
| `src/lauren_ai/_agents/__init__.py` | `@agent()`, `AgentMeta`, `AgentResponse` |
| `src/lauren_ai/_output_parsers/` | `JSONOutputParser` — parse plan JSON safely |

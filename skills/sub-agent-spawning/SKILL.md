---
name: sub-agent-spawning
description: Shows how an orchestrator agent spawns multiple sub-agents and aggregates their results. Use when building hierarchical multi-agent systems where a coordinator delegates specialised tasks to sub-agents, either sequentially or in parallel via asyncio.gather.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Sub-Agent Spawning & Result Aggregation

## Overview

`AgentRunnerBase.run()` can be called multiple times with different `@agent`
classes.  An orchestrator function calls `runner.run(sub_agent, prompt)` for
each delegated task and collects `AgentResponse.content` strings.

Sequential and parallel patterns are both straightforward:

---

## Sequential sub-agent spawning

```python
# orchestrator.py — from __future__ import annotations is safe here
from __future__ import annotations
import asyncio
from lauren_ai import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner

@agent(model=None, system="You are a research specialist.")
class ResearchSubAgent: ...

@agent(model=None, system="You are a writing specialist.")
class WritingSubAgent: ...

async def spawn_sequential(runner: AgentRunner, prompts: list[tuple]) -> list[str]:
    """Run sub-agents one at a time and collect results."""
    results = []
    for sub_agent, prompt in prompts:
        response = await runner.run(sub_agent, prompt)
        results.append(response.content)
    return results
```

---

## Parallel sub-agent spawning

```python
async def spawn_parallel(runner: AgentRunner, prompts: list[tuple]) -> list[str]:
    """Run sub-agents concurrently and collect all results."""
    tasks = [runner.run(sub_agent, prompt) for sub_agent, prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return [r.content for r in responses]
```

> `asyncio.gather` runs all coroutines on the same event loop.  Because
> `MockTransport` uses a shared dequeue, parallel tests must queue enough
> responses for every concurrent call.

---

## Runner setup for tests

```python
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock)
    return runner, mock

def _completion(content="OK", *, n=1):
    return Completion(
        id=f"c{n}", model="mock-model", content=content,
        tool_calls=[], stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
```

---

## Cross-module AgentRunner injection

When sub-agents live in different `AgentModule`s, inject via
`AgentRunner[SubAgentClass]`:

```python
class OrchestratorService:
    def __init__(self, research: AgentRunner[ResearchSubAgent]) -> None:
        self._research = research

    async def run_task(self, topic: str) -> str:
        response = await self._research.run(ResearchSubAgent(), topic)
        return response.content
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_agents/_runner.py` | `AgentRunnerBase.run()` |
| `src/lauren_ai/_agents/__init__.py` | `@agent()`, `AgentResponse` |
| `src/lauren_ai/_transport/_mock.py` | `MockTransport` |

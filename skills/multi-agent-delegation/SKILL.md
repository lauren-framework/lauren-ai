---
name: multi-agent-delegation
description: Implements multi-agent delegation where an orchestrator agent hands off tasks to specialist agents via a tool call. Use when building a system where a triage or coordinator agent routes requests to domain-specific agents (e.g. billing, support, technical).
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Multi-Agent Delegation & Handoff

## Architecture

```
User → OrchestratorAgent
         → DelegateToSpecialist tool call (task="...")
             → SpecialistAgent.run(task)
             → returns {result: "..."}
         → OrchestratorAgent sees result
         → Final answer to user
```

---

## Define the handoff tool

```python
# tools.py — future annotations are allowed; keep tool types importable

from lauren_ai import tool, ToolContext
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner

@tool()
class DelegateToSpecialist:
    """Delegate a task to the SpecialistAgent.

    Args:
        task: The full task description for the specialist.
    """

    def __init__(self, specialist, runner: AgentRunner) -> None:
        self._specialist = specialist
        self._runner = runner

    async def run(self, ctx: ToolContext, task: str) -> dict:
        response = await self._runner.run(
            self._specialist,
            task,
            execution_context=ctx.execution_context,
        )
        return {"result": response.content}
```

**Important:** The `ctx` parameter MUST be annotated as `ToolContext` so the framework detects
`reads_context=True` and injects it automatically.  Without the annotation, `ctx` appears in the
LLM's JSON schema as a required string parameter, which prevents execution.

**Tool naming:** `@tool()` on a class converts the class name to `snake_case`.
`DelegateToSpecialist` → `delegate_to_specialist`.  Use this name when queuing mock responses:

```python
mock.queue_tool_use("delegate_to_specialist", {"task": "..."})
```

---

## Define agents

```python
# agents.py — from __future__ import annotations IS safe here
from __future__ import annotations

from lauren_ai import agent, use_tools
from .tools import DelegateToSpecialist

@agent(model="claude-opus-4-6", system="You are a billing specialist.")
class BillingAgent: ...

@agent(model="claude-opus-4-6", system="Route requests to the right specialist.")
@use_tools(DelegateToSpecialist)
class OrchestratorAgent: ...
```

---

## Testing handoff

```python
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()

# Orchestrator decides to delegate
mock.queue_tool_use("DelegateToSpecialist", {"task": "Process refund for order #42"})

# Specialist's response (consumed by runner for the specialist)
mock.queue_response(Completion(
    id="s1", model="mock-model",
    content="Refund processed for order #42.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=10, output_tokens=8),
))

# Orchestrator's final answer after seeing delegate result
mock.queue_response(Completion(
    id="o2", model="mock-model",
    content="Your refund has been processed.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=20, output_tokens=10),
))
```

---

## Lauren DI wiring

```python
from lauren_ai import LLMModule, AgentModule
from lauren import module

LLMProvider = LLMModule.for_root(cfg)
AIModule = AgentModule.for_root(
    agents=[OrchestratorAgent, BillingAgent],
    tools=[DelegateToSpecialist],
)

@module(imports=[LLMProvider, AIModule])
class AppModule: ...
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_agents/_runner.py` | `AgentRunnerBase.run()` — accepts `execution_context=` |
| `src/lauren_ai/_tools/__init__.py` | Class-form `@tool()`, `ToolContext` |
| `examples/banking-chatbot/` | Full handoff example with `HandoffTo` pattern |

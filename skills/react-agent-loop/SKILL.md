---
name: react-agent-loop
description: Implements the ReAct (Reason + Act) agentic loop where the model alternates between thinking, calling tools, observing results, and producing a final answer. Use when building an agent that must iteratively use tools over multiple turns before arriving at a response.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# ReAct Agent Loop (Reason + Act)

## How the loop works

```
User message
  → Think (LLM reasoning, optional tool call)
  → Act   (tool execution if stop_reason == "tool_use")
  → Observe (tool result injected as next message)
  → Think (LLM sees tool result, decides next action)
  → ...
  → Final answer (stop_reason == "end_turn")
```

`AgentRunnerBase` implements this loop automatically. Each iteration is one
"turn". The loop ends when:

- `stop_reason == "end_turn"` (model is done)
- `turn == max_turns` → raises `AgentMaxTurnsError`
- Cost exceeds `max_cost_usd` → raises `AgentBudgetExceededError`

---

## Quick start

```python
# agent.py — from __future__ import annotations IS safe here
from __future__ import annotations

from lauren_ai import agent, use_tools
from .tools import search_web, run_python

@agent(
    model="claude-opus-4-6",
    system="""You are a research assistant.
Think through problems step-by-step.
Use tools to gather information before answering.""",
    max_turns=10,
)
@use_tools(search_web, run_python)
class ReActAgent: ...
```

---

## Lifecycle hooks for observability

```python
@agent(model="claude-opus-4-6", max_turns=10)
@use_tools(search_web)
class ObservableAgent:
    async def on_start(self, ctx):
        print(f"Starting run {ctx.run_id} — message: {ctx.message}")

    async def on_turn_complete(self, completion, ctx):
        print(f"Turn {ctx.turn}: stop_reason={completion.stop_reason}")

    async def on_tool_result(self, result, ctx):
        print(f"Tool result: {result.content[:80]}")

    async def on_finish(self, response, ctx):
        print(f"Finished in {response.turns} turns")
```

---

## Inspecting turn count

```python
resp = await runner.run(ReActAgent(), "Research quantum computing trends")
print(f"Used {resp.turns} turns")
print(f"Tools called: {[t.name for t in resp.tool_calls_made]}")
print(f"Cost: ${resp.total_usage.cost_usd('claude-opus-4-6'):.4f}")
```

---

## Testing the ReAct loop

```python
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()
# Turn 1: think → tool call
mock.queue_tool_use("search_web", {"query": "quantum computing"})
# Turn 2: observe → think → tool call
mock.queue_tool_use("run_python", {"code": "print(42)"})
# Turn 3: observe → final answer
mock.queue_response(Completion(
    id="c3", model="mock-model",
    content="Quantum computing is advancing rapidly. 42.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=50, output_tokens=20),
))
# Runner drives the loop automatically
resp = await runner.run(ReActAgent(), "Tell me about quantum computing")
assert resp.turns == 3
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_agents/_runner.py` | `AgentRunnerBase` — the full ReAct loop |
| `src/lauren_ai/_agents/__init__.py` | `AgentContext` — `ctx.turn`, `ctx.run_id` |
| `src/lauren_ai/_exceptions.py` | `AgentMaxTurnsError`, `AgentBudgetExceededError` |

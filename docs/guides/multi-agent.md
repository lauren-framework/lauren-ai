# Multi-Agent Systems

`lauren-ai` supports two delegation patterns and a full team orchestration
system.  This page covers the two delegation patterns.  For coordinated teams
with multiple worker agents see [agent-teams.md](agent-teams.md).

---

## Pattern 1 — ctx.delegate (hook-based)

The simplest way to hand off to another agent is to call `ctx.delegate()` from
any lifecycle hook.  The runner catches the internal `DelegateToAgent` signal,
recursively runs the target agent with the supplied message, and returns its
result with `stop_reason="delegated"`.

```python
from lauren_ai import agent, AgentContext

@agent(model="claude-opus-4-6", system="You are a routing assistant.")
class RouterAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        message = ctx.memory.messages()[-1]["content"]
        if "legal" in message.lower():
            await ctx.delegate(LegalAgent, message)
        elif "medical" in message.lower():
            await ctx.delegate(MedicalAgent, message)
        # Otherwise the router handles it itself
```

Use this pattern when:
- The routing decision is simple and deterministic.
- You do not need the delegation to appear in the model's tool call log.
- A single handoff is sufficient (no iterative routing).

---

## Pattern 2 — tool-based delegation (observable)

For more complex routing, give the coordinator agent a tool that calls
`AgentRunner.run()` directly.  The LLM decides which specialist to call, the
delegation appears in the tool call log, and the result flows back through the
normal tool result channel.

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
from lauren_ai import tool, agent, use_tools, AgentRunner
from lauren_ai._tools import ToolContext

@tool()
async def ask_researcher(question: str, ctx: ToolContext) -> dict:
    """Delegate a research question to the ResearchAgent.

    Args:
        question: The research question to answer.
    """
    runner: AgentRunner = ctx.agent_context.get_metadata("runner")
    researcher = ctx.agent_context.get_metadata("researcher")
    response = await runner.run(researcher, question)
    return {"answer": response.content, "turns": response.turns}


@agent(
    model="claude-opus-4-6",
    system="You are a coordinator. Delegate research questions to ask_researcher.",
)
@use_tools(ask_researcher)
class CoordinatorAgent: ...
```

Pass the runner and specialist agent into the coordinator's context via
`metadata=` when calling `runner.run()`:

```python
response = await runner.run(
    coordinator,
    "Research the history of the Eiffel Tower.",
    metadata={
        "runner": runner,
        "researcher": researcher_instance,
    },
)
```

Use this pattern when:
- The routing decision should be made by the LLM itself.
- You want the delegation visible in `AgentResponse.tool_calls_made`.
- Multiple rounds of delegation may occur.

---

## Choosing a pattern

| | ctx.delegate | Tool-based |
|---|---|---|
| Decision made by | Your code | The LLM |
| Delegation in tool log | No | Yes |
| Multiple handoffs | One | Many (tool can be called repeatedly) |
| Complexity | Low | Medium |

---

## Team orchestration

For full coordinator/collaborate team workflows — where multiple worker agents
run in parallel or sequence and a coordinator synthesises their results — see
[agent-teams.md](agent-teams.md).

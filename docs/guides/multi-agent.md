# Multi-Agent Systems

`lauren-ai` supports tool-based agent delegation and a full team orchestration
system.  This page covers tool-based delegation.  For coordinated teams with
multiple worker agents see [agent-teams.md](agent-teams.md).

Agent handoff is **always observable**: every delegation appears in the
caller's tool-call log, can be traced through `AgentResponse.tool_calls_made`,
and composes cleanly with `run_stream()`.  Hidden control-flow handoff (the
old `ctx.delegate()` exception pattern) was removed in favour of this single
explicit pattern.

---

## Tool-based delegation (observable)

Give the coordinator agent a tool that calls
`AgentRunner.run()` directly.  The LLM decides which specialist to call, the
delegation appears in the tool call log, and the result flows back through the
normal tool result channel.

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
from lauren_ai import tool, agent, use_tools, AgentRunnerBase
from lauren_ai._tools import ToolContext
from lauren import injectable, Scope

@injectable(scope=Scope.SINGLETON)
class ResearchAgentRunner(AgentRunnerBase):
    """Distinct DI token for the Research module's runner."""

@tool()
class AskResearcher:
    """Delegate a research question to the ResearchAgent.

    Args:
        question: The research question to answer.
    """
    def __init__(self, researcher: ResearchAgent, runner: ResearchAgentRunner) -> None:
        self._researcher = researcher
        self._runner = runner   # named subclass — unambiguous

    async def run(self, ctx: ToolContext, question: str) -> dict:
        response = await self._runner.run(self._researcher, question)
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

For deterministic routing (where *your code* — not the LLM — decides which
agent runs), do the dispatch in your own controller / handler before calling
`runner.run()` or `runner.run_stream()`.  See the banking-chatbot example for
a controller that polls an `ActiveAgentStore` between turns and emits SSE
`break` events on each handoff.

---

## Team orchestration

For full coordinator/collaborate team workflows — where multiple worker agents
run in parallel or sequence and a coordinator synthesises their results — see
[agent-teams.md](agent-teams.md).

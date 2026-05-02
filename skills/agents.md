# Writing Agents

---

## Minimal agent

```python
# agent.py — from __future__ import annotations IS allowed here (no @tool)
from __future__ import annotations

from lauren_ai import agent, use_tools
from .tools import my_tool

@use_tools(my_tool)
@agent(model="openai/gpt-4o-mini", system="You are a helpful assistant.")
class MyAgent: ...
```

---

## Full lifecycle hooks

```python
from lauren_ai import agent, AgentContext, AgentResponse, Completion, ToolResult, use_tools

@use_tools(my_tool)
@agent(model="openai/gpt-4o-mini", system="You are helpful.", max_turns=8)
class FullAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        """Called once before the first LLM call.

        Use for: initialising state, loading user context, setting metadata.
        Can raise DelegateToAgent to immediately hand off before the first turn.
        """
        ctx.metadata["start_time"] = __import__("time").monotonic()

    async def on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None:
        """Called after each LLM turn.

        Use for: logging turn content, inspecting intermediate reasoning.
        """
        print(f"Turn {ctx.turn}: {completion.content[:80]}")

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        """Called after each tool execution.

        Return a modified ToolResult to replace the original, or None to keep it.
        Use for: filtering sensitive data, logging, adding context.
        """
        if result.is_error:
            # Wrap error message with friendly context
            from lauren_ai._tools import ToolResult as TR
            return TR(
                tool_use_id=result.tool_use_id,
                content=f"Tool error (will retry): {result.content}",
                is_error=True,
            )
        return None

    async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
        """Called after the loop terminates.

        Use for: cleanup, logging final cost, storing conversation.
        """
        elapsed = __import__("time").monotonic() - ctx.metadata.get("start_time", 0)
        print(f"Done in {elapsed:.2f}s, turns={response.turns}, "
              f"cost=${response.total_usage.cost_usd(ctx.config.system_prompt):.6f}")
```

---

## Streaming

```python
from lauren_ai._agents._runner import AgentRunner

async def stream_agent(runner: AgentRunner, agent_inst, message: str) -> None:
    async for chunk in await runner.run_stream(agent_inst, message):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()  # newline at end
```

---

## Delegation pattern

**Context-based** (via `ctx.delegate()`):

```python
@agent(model="openai/gpt-4o-mini")
class RouterAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        message = ctx.memory.messages()[-1].content if ctx.memory.messages() else ""
        if "code" in message.lower():
            await ctx.delegate(CodeAgent, message)
        # else falls through to normal LLM handling
```

**Tool-based delegation** (recommended for multi-agent systems — see AGENTS.md):
Delegation tools close over `AgentRunner` singletons and are called by the LLM
through the normal tool-use mechanism.  This approach is more observable and
auditable because tool calls appear in the conversation history.

---

## AgentConfig parameters

Set via kwargs on `@agent()`:

```python
@agent(
    model="openai/gpt-4o-mini",
    system="...",
    max_turns=10,                # Loop iteration limit
    max_tokens_per_turn=4096,    # Max output tokens per LLM call
    temperature=0.7,             # Sampling temperature
    memory_window_tokens=40_000, # Rolling context window
    max_cost_usd=0.50,          # Hard cost budget — raises AgentBudgetExceededError
    parallel_tool_calls=False,   # Execute tools serially (True = concurrent)
    tool_error_policy="return_error",  # "raise" | "return_error" | "skip"
)
class MyAgent: ...
```

---

## Decorator order (mandatory)

```
@agent()          ← outermost
@remember()       ← optional, between agent and guardrail
@guardrail()      ← optional, between remember and use_tools
@use_tools(...)   ← innermost
class MyAgent: ...
```

Python applies decorators bottom-up, so `@use_tools` runs first (sets
`USE_TOOLS_META`), then `@guardrail`, `@remember`, and finally `@agent` which
reads all the metadata and builds `AgentMeta`.

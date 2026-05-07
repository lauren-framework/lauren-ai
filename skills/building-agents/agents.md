# Writing Agents

Full reference for the `@agent` decorator, lifecycle hooks, streaming,
delegation, and `AgentConfig`.

---

## Contents

- [Minimal agent](#minimal-agent)
- [Full lifecycle hooks](#full-lifecycle-hooks)
- [Streaming](#streaming)
- [Delegation patterns](#delegation-patterns)
- [AgentConfig — all parameters](#agentconfig--all-parameters)
- [Decorator order](#decorator-order)

---

## Minimal agent

```python
# agent.py — from __future__ import annotations IS allowed here (no @tool)
from __future__ import annotations

from lauren_ai import agent, use_tools
from .tools import my_tool

@agent(model="claude-opus-4-6", system="You are a helpful assistant.")
@use_tools(my_tool)
class MyAgent: ...
```

Run it with `AgentRunnerBase` (for scripting/testing) or via `AgentModule.for_root()` (production):

```python
from lauren_ai import AgentRunnerBase, LLMConfig

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
runner = AgentRunnerBase(transport=transport, tools={}, config=cfg)
result = await runner.run(MyAgent(), "What can you help me with?")
print(result.content)
print(f"turns={result.turns}, cost=${result.total_usage.total_tokens}")
```

If you have no tools, omit `@use_tools` entirely:

```python
@agent(model="claude-opus-4-6", system="You summarise text concisely.")
class SummaryAgent: ...
```

---

## Full lifecycle hooks

The agent class can define any subset of these four hooks.  All are optional.

```python
from lauren_ai import (
    agent,
    use_tools,
    AgentContext,
    AgentResponse,
    Completion,
    ToolResult,
)

@agent(model="claude-opus-4-6", system="You are helpful.", max_turns=8)
@use_tools(my_tool)
class FullAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        """Called once before the first LLM call.

        Use for: initialising state, loading user context, setting metadata.
        Can raise DelegateToAgent to immediately hand off before the first turn.
        """
        ctx.metadata["start_time"] = __import__("time").monotonic()

    async def on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None:
        """Called after each LLM turn (before tool execution, if any).

        Use for: logging turn content, inspecting intermediate reasoning,
        checking intermediate cost.
        """
        print(f"Turn {ctx.turn}: {completion.content[:80]!r}")

    async def on_tool_result(
        self, result: ToolResult, ctx: AgentContext
    ) -> ToolResult | None:
        """Called after each tool execution.

        Return a modified ToolResult to replace the original, or None to keep it.
        Use for: filtering sensitive data, logging, adding context.
        """
        if result.is_error:
            from lauren_ai._tools import ToolResult as TR
            return TR(
                tool_use_id=result.tool_use_id,
                content=f"Tool error (will retry): {result.content}",
                is_error=True,
            )
        return None  # keep unchanged

    async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
        """Called after the agentic loop terminates (all turns done).

        Use for: cleanup, logging final cost, storing conversation results.
        """
        elapsed = __import__("time").monotonic() - ctx.metadata.get("start_time", 0)
        print(
            f"Done in {elapsed:.2f}s, turns={response.turns}, "
            f"stop_reason={response.stop_reason}"
        )
```

### Hook call order

```
AgentRunner.run()
  └─ on_start(ctx)
  └─ for each turn:
       ├─ transport.complete(...)         ← LLM call
       ├─ on_turn_complete(completion, ctx)
       └─ if tool_use:
            ├─ tool.run(ctx, ...)
            └─ on_tool_result(result, ctx)
  └─ on_finish(response, ctx)
```

---

## Streaming

`AgentRunner.run_stream()` returns an async generator of `CompletionChunk`
objects.  Each chunk has a `delta` attribute with the streamed text fragment.

```python
from lauren_ai import AgentRunner

async def stream_response(runner: AgentRunner, agent_inst, message: str) -> None:
    async for chunk in await runner.run_stream(agent_inst, message):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()  # trailing newline
```

Streaming is compatible with all lifecycle hooks — `on_start`, `on_turn_complete`,
`on_tool_result`, and `on_finish` all fire normally during a streamed run.

---

## Delegation patterns

### Exception-based delegation (`ctx.delegate`)

Any lifecycle hook can call `ctx.delegate(TargetAgent, message)` to immediately
hand the current task to another agent.  The runner catches `DelegateToAgent`
and runs the target, returning a response with `stop_reason="delegated"`.

```python
from lauren_ai import agent, AgentContext, DelegateToAgent

@agent(model="claude-opus-4-6", system="Route requests to specialist agents.")
class RouterAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        messages = ctx.memory.messages()
        last = messages[-1].content if messages else ""
        if "code" in last.lower():
            await ctx.delegate(CodeAgent, last)
        elif "legal" in last.lower():
            await ctx.delegate(LegalAgent, last)
        # else: falls through to normal LLM routing
```

`ctx.delegate()` raises `DelegateToAgent` internally — you do not need to
`raise` it yourself.

### Tool-based delegation (recommended for production systems)

The LLM decides when to delegate by calling a delegation tool.  This approach
keeps all delegation decisions visible in conversation history and is easier
to audit and debug.

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
from lauren_ai import AgentRunner, tool, ToolContext

@tool()
class DelegateToResearcher:
    """Hand a research task to the Researcher agent.

    Args:
        task: Detailed description of what to research.
    """

    def __init__(
        self,
        researcher: ResearchAgent,
        runner: AgentRunner[ResearchAgent],   # ← parameterized DI token — no boilerplate subclass
    ) -> None:
        self._agent = researcher
        self._runner = runner

    async def run(self, ctx: ToolContext, task: str) -> dict:
        response = await self._runner.run(
            self._agent,
            task,
            execution_context=ctx.execution_context,  # forward auth context
        )
        return {"result": response.content}
```

Wire the delegation tool in the **calling (coordinator) module's `tools=`**; the
coordinator module imports the researcher module so `AgentRunner[ResearchAgent]`
is visible:

```python
from lauren_ai import AgentModule, agent, use_tools

@agent(model="claude-opus-4-6", system="You are a coordinator.")
@use_tools(DelegateToResearcher)
class CoordinatorAgent: ...

ResearchMod = AgentModule.for_root(
    agents=[ResearchAgent], tools=[ResearchTool],
    imports=[LLMProvider],
    # AgentRunner[ResearchAgent] auto-registered — no runner= needed
)

CoordinatorMod = AgentModule.for_root(
    agents=[CoordinatorAgent],
    tools=[DelegateToResearcher],          # ← delegation tool lives in calling module
    imports=[LLMProvider, ResearchMod],    # ← makes AgentRunner[ResearchAgent] visible
)
```

The `execution_context` is always forwarded verbatim so that downstream tools
in the sub-agent can still read the authenticated user identity.

---

## DI integration with Lauren

`@agent()` automatically applies `@injectable(scope=Scope.SINGLETON)`.
Register agents with `AgentModule.for_root()` — this adds each agent class to
`providers` **and** exports it so controllers in the parent module can inject it:

```python
# ai_module.py
from lauren_ai import AgentModule, LLMModule, LLMConfig
from .my_agent import MyAgent

LLMProvider = LLMModule.for_root(LLMConfig.for_anthropic())

AgentProvider = AgentModule.for_root(
    agents=[MyAgent],
    imports=LLMProvider,
)
```

Then inject the agent **instance** into a controller:

```python
from lauren_ai import AgentRunner  # @runtime_checkable Protocol

class ChatController:
    # runner: AgentRunner works when only one AgentModule is in scope.
    # When two AgentModules are in scope, use AgentRunner[AgentClass] (parameterized form).
    def __init__(self, runner: AgentRunner, agent: MyAgent) -> None:
        self._runner = runner
        self._agent = agent   # DI-resolved singleton

    async def chat(self, message: str) -> str:
        response = await self._runner.run(self._agent, message)
        return response.content

# Multiple modules — parameterized form, no boilerplate subclasses:
class MultiAgentController:
    def __init__(
        self,
        researcher_runner: AgentRunner[ResearchAgent],
        writer_runner:     AgentRunner[WriterAgent],
    ) -> None: ...
```

**Critical:** always pass the instance — never the class — to `runner.run()`.
Passing the class breaks all lifecycle hooks (`on_start`, `on_turn_complete`,
`on_tool_result`, `on_finish`) because Python cannot bind `self` for an unbound
method retrieved from a class object.  The symptom is:

```
TypeError: MyAgent.on_start() missing 1 required positional argument: 'ctx'
```

---

## AgentConfig — all parameters

All parameters are keyword arguments on `@agent()`.  Parameters not listed
in `@agent()` fall back to the `AgentConfig` defaults or the `LLMConfig`
passed to `AgentRunner`.

```python
@agent(
    model="claude-opus-4-6",         # required — model string
    system="You are helpful.",        # system prompt
    max_turns=10,                     # agentic loop iteration limit
    max_tokens_per_turn=4096,         # max output tokens per LLM call
    temperature=0.7,                  # sampling temperature (0.0–1.0)
    memory_window_tokens=40_000,      # rolling context window in tokens
    max_cost_usd=0.50,                # hard cost cap — raises AgentBudgetExceededError
    parallel_tool_calls=True,         # True = concurrent tool calls
    tool_error_policy="return_error", # "raise" | "return_error" | "skip"
    memory=ShortTermMemory(max_tokens=60_000),      # optional — reused across run() calls
    conversation_store=InMemoryConversationStore(), # optional — auto-created if omitted
)
class MyAgent: ...
```

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | `str` | required | Provider-prefixed model string |
| `system` | `str` | `""` | System prompt |
| `max_turns` | `int` | `10` | Prevents infinite loops |
| `max_tokens_per_turn` | `int` | `4096` | Per-call output token limit |
| `temperature` | `float` | `1.0` | Higher = more creative, lower = more deterministic |
| `memory_window_tokens` | `int` | `40_000` | Oldest messages dropped when limit reached |
| `max_cost_usd` | `float \| None` | `None` | Raises `AgentBudgetExceededError` when exceeded |
| `parallel_tool_calls` | `bool` | `True` | `False` = serial execution (useful for order-dependent tools) |
| `tool_error_policy` | `str` | `"return_error"` | What to do when a tool raises an exception |
| `memory` | `ShortTermMemory \| None` | `None` | Instance reused across **every** `run()` call; fresh per turn when `None` |
| `conversation_store` | `ConversationStore \| None` | `None` | Per-agent store; `AgentModule.for_root()` auto-creates if `None` |

### `tool_error_policy` values

| Value | Behaviour |
|-------|-----------|
| `"return_error"` | Tool result carries the error message; LLM sees it and can adapt |
| `"raise"` | Propagates the exception out of `AgentRunner.run()` |
| `"skip"` | Silently drops the failed tool result; LLM does not see the error |

---

## Decorator order

Python applies decorators bottom-up, so write them top-to-bottom in this order:

```
@agent()           ← outermost (top)   — applied last, reads USE_TOOLS_META,
                                          GUARDRAIL_META, REMEMBER_META
@remember()        ← optional — reads USE_TOOLS_META, sets REMEMBER_META
@use_guardrails()  ← optional — reads USE_TOOLS_META, sets GUARDRAIL_META
@use_tools(...)    ← innermost (bottom) — applied first, sets USE_TOOLS_META
class MyAgent: ...
```

**Why this matters:**

1. `@use_tools(my_tool)` runs first and attaches `USE_TOOLS_META` to the class.
2. `@use_guardrails(...)` runs next, reads `USE_TOOLS_META`, and attaches
   `GUARDRAIL_META`.
3. `@remember(...)` runs next, reads existing metadata, and attaches
   `REMEMBER_META`.
4. `@agent(...)` runs last, reads all three metadata attributes, and builds
   the final `AgentMeta` that `AgentRunner` uses.

Swapping any pair causes the reader to run before the writer — the metadata
is missing and the feature silently vanishes (tools not registered, guardrails
not applied, memory not injected).

### Concrete example

```python
from lauren_ai import (
    agent,
    remember,
    use_guardrails,
    use_tools,
    PromptInjectionFilter,
    PIIRedactor,
    LengthFilter,
    InMemoryUserMemoryStore,
)
from .tools import search_kb, draft_reply

_memory = InMemoryUserMemoryStore()

@agent(
    model="claude-opus-4-6",
    system="You are a customer support assistant.",
    max_turns=12,
    max_cost_usd=0.20,
)
@remember(store=_memory, extract=True, inject=True, top_k=5)
@use_guardrails(
    input=[PromptInjectionFilter(), PIIRedactor()],
    output=[LengthFilter(max_chars=8000)],
)
@use_tools(search_kb, draft_reply)
class SupportAgent: ...
```

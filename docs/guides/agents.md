# Agents

`lauren-ai` agents are classes decorated with `@agent()`.  The agentic loop —
call the LLM, execute tools, call the LLM again — is managed automatically by
`AgentRunner`.

---

## @agent() — defining an agent

```python
from lauren_ai import agent, AgentContext, AgentResponse

@agent(
    model="claude-opus-4-6",
    system="You are a helpful research assistant.",
    max_turns=10,
    temperature=0.7,
)
class ResearchAgent:
    """Fallback system prompt if system= is omitted."""
```

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `model` | `str \| None` | inherited from `LLMConfig` | LLM model identifier |
| `system` | `str \| None` | class docstring or `AgentConfig.system_prompt` | System prompt |
| `max_turns` | `int \| None` | `10` | Maximum agentic loop iterations |
| `temperature` | `float \| None` | `1.0` | Sampling temperature |
| `thinking` | `bool` | `False` | Enable Anthropic extended thinking |
| `thinking_budget_tokens` | `int` | `8_000` | Token ceiling for the thinking phase |
| `reasoning_effort` | `str \| None` | `None` | OpenAI o-series effort: `"low"` / `"medium"` / `"high"` |
| `include_reasoning_in_response` | `bool` | `False` | Expose OpenAI reasoning text in `AgentResponse` |

`@agent()` must be called **with parentheses**.  Bare `@agent` raises
`DecoratorUsageError`.

For a full treatment of extended thinking — including `Completion.thinking_blocks`,
streaming thinking deltas, lifecycle hook access, and testing — see the
[extended thinking guide](extended-thinking.md).

`@agent()` automatically applies `@injectable(scope=Scope.SINGLETON)` unless
the class is already `@injectable`.

---

## @use_tools() — attaching tools

`@use_tools()` is the **innermost** decorator (closest to the class).
`@agent()` must be the **outermost** (farthest from the class):

```python
from lauren_ai import agent, use_tools
from lauren_ai.skills import WebSearchTool, HttpFetchTool

@agent(model="claude-opus-4-6", system="You are a research assistant.")
@use_tools(WebSearchTool, HttpFetchTool)
class ResearchAgent: ...
```

`None` entries are silently dropped, which is useful for conditional tools:

```python
@agent(model="claude-opus-4-6")
@use_tools(
    WebSearchTool,
    CodeExecutionTool if settings.code_tools_enabled else None,
)
class FlexibleAgent: ...
```

---

## Decorator ordering — mandatory

Decorators are applied bottom-up in Python.  The required order is:

```
@agent(...)            ← outermost: reads USE_TOOLS_META, sets AGENT_META
@remember()            ← optional
@use_guardrails(...)   ← optional
@use_tools(...)        ← innermost: sets USE_TOOLS_META on class
class MyAgent: ...
```

Swapping the order causes metadata written by inner decorators to be read
before it exists, resulting in silently missing tools or guardrails.

---

## Lifecycle hooks

Define any of these methods on the agent class.  They are called by
`AgentRunner` at the matching point in the loop.  All hooks may be sync or
async and are optional.

### on_start

Called once before the first LLM turn.

```python
from lauren_ai import AgentContext

async def on_start(self, ctx: AgentContext) -> None:
    print(f"Run {ctx.agent_run_id} started")
```

### on_turn_complete

Called after each LLM response, before tool execution.

```python
from lauren_ai._transport import Completion

async def on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None:
    print(f"Turn {ctx.turn}: {completion.stop_reason}, tokens={completion.usage.total_tokens}")
```

### on_tool_result

Called after each tool executes.  Return a modified `ToolResult` to override
what the model sees, or return `None` to use the original.

```python
from lauren_ai._tools import ToolResult

async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
    if result.is_error:
        print(f"Tool error: {result.content}")
    return result
```

### on_finish

Called once after the agentic loop terminates.

```python
from lauren_ai import AgentResponse

async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
    print(f"Done in {response.turns} turns, stop_reason={response.stop_reason}")
```

---

## AgentContext

`AgentContext` is passed to every lifecycle hook:

| Field | Type | Description |
|---|---|---|
| `agent_id` | `str` | Agent instance identifier (random hex) |
| `agent_run_id` | `str` | Unique identifier for this specific run |
| `agent_class` | `type` | The `@agent()`-decorated class |
| `config` | `AgentConfig` | Effective configuration for this run |
| `memory` | `ShortTermMemory` | Sliding-window conversation buffer |
| `turn` | `int` | Current loop iteration (0-based) |
| `metadata` | `dict[str, Any]` | Caller-supplied or decorator-set metadata |
| `request` | `Any \| None` | Originating HTTP request, if any |

```python
# Read metadata set by the caller
user_id = ctx.get_metadata("user_id", default="anonymous")
```

---

## AgentResponse

`AgentRunner.run()` returns an `AgentResponse`:

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Final text output |
| `turns` | `int` | Number of loop iterations executed |
| `total_usage` | `TokenUsage` | Cumulative token usage across all turns |
| `tool_calls_made` | `list[ToolCall]` | All tool calls executed during the run |
| `stop_reason` | `str` | `"end_turn"`, `"max_turns"`, `"budget_exceeded"`, `"delegated"`, or `"error"` |
| `reasoning_traces` | `list[str]` | Extended-thinking blocks (Anthropic only) |

---

## Running an agent — AgentRunner.run()

`AgentRunner` is a `@runtime_checkable Protocol`; `AgentRunnerBase` is the concrete
implementation. `AgentModule.for_root()` generates a unique `AgentRunnerBase` subclass
and registers it as the module's runner. Inject it via `runner: AgentRunner` (Protocol)
when only one AgentModule is in scope, or via the named concrete subclass when two or
more modules are in scope (see `runner=MyRunner` in [lauren-integration.md](lauren-integration.md)):

```python
from lauren_ai import AgentRunner  # @runtime_checkable Protocol

class ChatController:
    # runner: AgentRunner works when exactly one AgentModule is in scope.
    def __init__(self, runner: AgentRunner, agent: ResearchAgent) -> None:
        self._runner = runner
        self._agent = agent

    @post("/chat")
    async def chat(self, body: ChatBody) -> dict:
        response = await self._runner.run(
            self._agent,
            body.message,
            conversation_id=body.session_id,
            metadata={"user_id": body.user_id},
        )
        return {"content": response.content, "turns": response.turns}
```

`run()` keyword arguments:

| Argument | Type | Description |
|---|---|---|
| `conversation_id` | `str \| None` | Links to a `ConversationStore` for history persistence |
| `metadata` | `dict[str, Any] \| None` | Injected into `AgentContext.metadata` |
| `run_id` | `str \| None` | Optional explicit run identifier |

---

## Streaming — AgentRunner.run_stream()

`run_stream()` returns an async iterator of `CompletionChunk` objects.  Tool
calls execute silently between turns; only text deltas are yielded.

```python
async for chunk in await runner.run_stream(agent, "Tell me a story"):
    print(chunk.delta, end="", flush=True)
```

See the [streaming guide](streaming.md) for SSE controller examples.

---

## Delegation

### ctx.delegate — hook-based handoff

Any lifecycle hook can call `ctx.delegate()` to hand off to another agent.
The runner catches the internal `DelegateToAgent` signal, recursively runs
the target agent, and returns its result with `stop_reason="delegated"`.

```python
from lauren_ai import AgentContext

async def on_start(self, ctx: AgentContext) -> None:
    if is_legal_question(ctx):
        await ctx.delegate(LegalAgent, ctx.memory.messages()[-1]["content"])
```

### Tool-based delegation — observable multi-agent systems

For more complex routing, give the coordinator a tool that calls
`AgentRunner.run()` directly inside its body.  The delegation appears in the
tool call log and is visible to the model.  See [multi-agent](multi-agent.md)
for full examples.

---

## Guardrails on agents

Use `@use_guardrails()` to attach input and output safety checks.  It sits
between `@agent()` and `@use_tools()` in the stack:

```python
from lauren_ai import agent, use_guardrails, use_tools, TopicFilter, PIIRedactor

@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[TopicFilter(allowed_topics=["cooking"])],
    output=[PIIRedactor(entities=["EMAIL"])],
)
@use_tools(WebSearchTool)
class CookingAgent: ...
```

See the [guardrails guide](guardrails.md) for details.

---

## Module wiring

Register agents with `AgentModule.for_root()`:

```python
import os
from lauren_ai import LLMConfig
from lauren_ai._module import LLMModule, AgentModule

LLMProvider = LLMModule.for_root(
    LLMConfig.for_anthropic(
        model="claude-opus-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
)

AIModule = AgentModule.for_root(
    agents=[ResearchAgent],
    tools=[WebSearchTool],
    imports=LLMProvider,   # required — makes Transport and LLMConfig visible
)

@module(
    controllers=[ChatController],
    imports=[LLMProvider, AIModule],
)
class AppModule: ...
```

The `imports=LLMProvider` argument is required so the generated agent module
can see the `Transport` and `LLMConfig` tokens exported by `LLMModule`.
Without it, `AgentRunner` cannot be wired and startup raises
`MissingProviderError`.

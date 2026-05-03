# Agents Reference

## `@agent()` decorator

Marks a class as an AI agent. Must be called **with parentheses** — bare `@agent` raises `DecoratorUsageError`.

The decorated class automatically receives `@injectable(scope=Scope.SINGLETON)` unless already marked injectable.

```python
from lauren_ai import agent, use_tools, AgentContext, AgentResponse

@agent(
    model="claude-opus-4-6",
    system="You are a research assistant.",
    max_turns=10,
    temperature=0.7,
)
@use_tools(WebSearchTool, CitationTool)
class ResearchAgent:
    async def on_start(self, ctx: AgentContext) -> None: ...
    async def on_turn_complete(self, completion, ctx: AgentContext) -> None: ...
    async def on_tool_result(self, result, ctx: AgentContext) -> None: ...
    async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None: ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| None` | `None` | LLM model identifier. `None` inherits from `LLMConfig` at runtime. |
| `system` | `str \| None` | `None` | System prompt. `None` falls back to the class docstring, then `AgentConfig.system_prompt`. |
| `max_turns` | `int \| None` | `None` | Maximum agentic loop iterations. Forwarded to `AgentConfig`. |
| `temperature` | `float \| None` | `None` | Sampling temperature override. Forwarded to `AgentConfig`. |
| `**config_kwargs` | `Any` | — | Additional keyword arguments forwarded verbatim to `AgentConfig`. |

### Lifecycle hooks

| Method | Signature | Called when |
|--------|-----------|-------------|
| `on_start` | `(self, ctx: AgentContext) -> None` | Before the first LLM call |
| `on_turn_complete` | `(self, completion, ctx: AgentContext) -> None` | After each LLM completion |
| `on_tool_result` | `(self, result, ctx: AgentContext) -> None` | After each tool executes |
| `on_finish` | `(self, response: AgentResponse, ctx: AgentContext) -> None` | After the loop terminates |

All hooks are optional. All may be `async`.

---

## `@use_tools()` decorator

Attaches tool classes or functions to an `@agent()`-decorated class. `None` entries are silently dropped. Multiple `@use_tools()` calls on the same class accumulate — their tool lists are merged.

```python
@agent(model="claude-opus-4-6")
@use_tools(WebSearchTool, get_weather, None)   # None is silently dropped
class ResearchAgent: ...
```

### Decorator ordering

Decorators apply bottom-up in Python. The correct order for a full agent:

```python
@agent()           # outermost — reads USE_TOOLS_META, writes AGENT_META
@remember()        # optional — must be below @agent, above @use_guardrails
@use_guardrails()  # optional — must be below @remember, above @use_tools
@use_tools(...)    # innermost — writes USE_TOOLS_META
class MyAgent: ...
```

---

## `AgentMeta`

Metadata stored on `@agent()`-decorated classes under the `AGENT_META` attribute (`"__lauren_ai_agent__"`). Populated at decoration time; read by `AgentRunner` at runtime.

```python
from lauren_ai._agents import AgentMeta, AGENT_META

meta: AgentMeta = getattr(MyAgent, AGENT_META)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str \| None` | LLM model identifier, or `None` to inherit from `LLMConfig`. |
| `system` | `str \| None` | System prompt, or `None` to use class docstring / `AgentConfig` default. |
| `config` | `AgentConfig` | Runtime behaviour configuration built from `@agent()` kwargs. |
| `tool_classes` | `tuple[Any, ...]` | Tool classes/functions attached via `@use_tools()`. |

---

## `USE_TOOLS_META`

String sentinel `"__lauren_ai_tools__"` — the attribute name that `@use_tools()` sets on the decorated class to store the tuple of attached tool classes. Read by `@agent()` during decoration.

---

## `AgentContext`

One instance is created per `AgentRunner.run()` call and passed to all lifecycle hooks and tools that declare `ctx: ToolContext`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Unique identifier for the agent instance (random hex). |
| `agent_run_id` | `str` | Unique identifier for this specific run (distinct from `agent_id`). |
| `agent_class` | `type` | The `@agent()`-decorated class. |
| `config` | `AgentConfig` | Effective merged config for this run. |
| `memory` | `ShortTermMemory` | Short-term conversation memory for this run. |
| `turn` | `int` | Current agentic loop iteration (0-based). |
| `metadata` | `dict[str, Any]` | Key-value metadata bag. |
| `request` | `Any \| None` | Originating HTTP request, or `None` outside a web handler. |
| `execution_context` | `Any \| None` | Lauren `ExecutionContext` when invoked from a route handler, else `None`. |
| `signals` | `Any \| None` | `SignalBus` instance when one is registered, else `None`. |

### Methods

#### `get_metadata(key, default=None)`

Returns the metadata value for `key`, or `default` if absent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Metadata key to look up. |
| `default` | `Any` | Fallback when the key is absent. |

#### `async delegate(agent, message, *, metadata=None)`

Hands off execution to another agent by raising `DelegateToAgent`. The runner catches the exception and calls `AgentRunner.run()` recursively on the target agent, then wraps the result with `stop_reason="delegated"`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Any` | The `@agent()`-decorated class or instance to delegate to. |
| `message` | `str` | Message passed to the delegated agent. |
| `metadata` | `dict[str, Any] \| None` | Optional additional metadata for the delegated run. |

Raises `DelegateToAgent` — always intercepted by the runner.

---

## `AgentResponse`

The result returned by `AgentRunner.run()` after the agentic loop terminates.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Final text output from the agent (last assistant message text). |
| `turns` | `int` | Number of agentic loop iterations executed. |
| `total_usage` | `TokenUsage` | Cumulative token usage across all turns. |
| `tool_calls_made` | `list[ToolCall]` | All tool calls executed during the run. |
| `stop_reason` | `Literal[...]` | Why the loop terminated (see values below). |
| `metadata` | `dict[str, Any]` | Arbitrary metadata attached to the response. Defaults to `{}`. |
| `reasoning_traces` | `list[str]` | Extended-thinking traces collected across all turns (Anthropic only). Defaults to `[]`. |

### `stop_reason` values

| Value | Meaning |
|-------|---------|
| `"end_turn"` | Model indicated a natural end of response. |
| `"max_turns"` | `max_turns` limit was reached. |
| `"budget_exceeded"` | Cost or token budget was crossed mid-run. |
| `"delegated"` | Execution was handed off to another agent via `ctx.delegate()`. |
| `"error"` | An unrecoverable error occurred. |

### Methods

#### `async as_stream()`

Wraps `content` as a single-item `AsyncIterator[str]`. Useful for handlers that expect an async generator regardless of whether streaming was used.

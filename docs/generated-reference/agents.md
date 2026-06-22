# Agents

Decorators and types for building AI agents.

## Decorators

### `agent`

```python
def agent(args: Any = (), name: str | None = None, model: str | None = None, system: str | None = None, max_turns: int | None = None, temperature: float | None = None, memory: Any | None = None, conversation_store: Any | None = None, config_kwargs: Any = {}) -> Callable[[type[C]], type[C]]
```

Decorator that marks a class as an AI agent.

Must be called **with parentheses**: `@agent()`.  Using the bare form
`@agent` (without parentheses) raises
`DecoratorUsageError`.

The decorated class:

* Gets the `AGENT_META` attribute set to an `AgentMeta`
  instance.
* Is automatically registered as `@injectable(scope=Scope.SINGLETON)`
  unless `@injectable` is already applied.
* Can define optional lifecycle hooks: `on_start`, `on_tool_result`,
  `on_turn_complete`, `on_finish`.

Example::

    @use_tools(WebSearchTool, CitationTool)
    @agent(
        name="Research Agent",
        model="claude-opus-4-6",
        system="You are a research assistant.",
        max_turns=10,
        temperature=0.7,
    )
    class ResearchAgent:
        async def on_start(self, ctx: AgentContext) -> None: ...
        async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None: ...

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str | None` | Human-readable agent name exposed via
`AgentMeta.name` and `AgentContext.agent_name`.  When
`None` the decorated class `__name__` is used. |
| `model` | `str | None` | LLM model identifier override.  When `None` the model is
taken from `LLMConfig` at runtime. |
| `system` | `str | None` | System prompt.  When `None` the class docstring is used,
falling back to `AgentConfig.system_prompt`. |
| `max_turns` | `int | None` | Maximum agentic loop iterations.  Forwarded to
`AgentConfig`. |
| `temperature` | `float | None` | Sampling temperature override.  Forwarded to
`AgentConfig`. |
| `memory` | `Any | None` | Per-agent memory instance.  When set, the **same
instance is reused across every** `run()` **call** — that is
what "agent memory" means semantically.  When `None` the runner
builds a fresh `ShortTermMemory` per
turn.  Per-call `runner.run(agent, …, memory=…)` always wins. |
| `conversation_store` | `Any | None` | Per-agent conversation store.  When
`None`, `AgentModule.for_root()` auto-creates an
`InMemoryConversationStore` and writes
it back to AgentMeta.  Per-call `runner.run(agent, …,
conversation_store=…)` always wins. |
| `config_kwargs` | `Any` | Additional keyword arguments forwarded to
`AgentConfig`. |

**Returns:** `Callable[[type], type]` — A class decorator.

**Raises:**

| Exception | Description |
|---|---|
| `DecoratorUsageError` | When called without parentheses (bare
`@agent`). |

### `use_tools`

```python
def use_tools(tools: Any = ()) -> Callable[[type[C]], type[C]]
```

Attach tool classes or functions to an `@agent()`-decorated class.

`None` entries are silently dropped (consistent with `use_guards`
behaviour in the lauren framework).

Typically stacked above `@agent()`::

    @use_tools(WebSearchTool, get_weather, None)  # None is dropped
    @agent(model="claude-opus-4-6")
    class ResearchAgent: ...

At compile time, `validate_agent_class()` resolves each entry from the
DI container and builds the `ToolSchema` list.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `tools` | `Any` | Tool classes or callables decorated with `@tool()`.
`None` entries are ignored. |

**Returns:** `Callable[[type], type]` — A class decorator that attaches the tools to the class.

### `use_knowledge_sources`

```python
def use_knowledge_sources(sources: Any = ()) -> Callable[[type[C]], type[C]]
```

Restrict an agent's knowledge-base tool visibility to the listed sources.

Without this decorator an agent has **no** knowledge-base tools — KB
visibility is **opt-in** even when the enclosing
`AgentModule` declares `knowledge=[…]`.
With it, only the listed `KnowledgeSource`
instances are attached to the agent's tool schema at runtime.

Stores the *tool names* (strings) on the class, not the
`KnowledgeSource` instances — matching against the module's declared
sources is a string-set comparison at module-build time.

Stacking is allowed (decorator concatenates names with any already
present), and **strict-inheritance applies** (mirrors Lauren's
framework golden-rule #3): a subclass that inherits the metadata from
a parent without redeclaring `@use_knowledge_sources` raises
`MetadataInheritanceError` at module
construction time.

Typically stacked above `@agent()`::

    from app.knowledge_sources import PUBLIC_KB_SOURCE

    @use_knowledge_sources(PUBLIC_KB_SOURCE)
    @agent(name="UnauthCRM", model="...")
    class UnauthenticatedCRMAgent: ...

The `AgentModule.for_root()` validation step checks every name
here against the module's `knowledge=` list and raises
`DecoratorUsageError` if any source is
not declared at module level.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `sources` | `Any` | One or more `KnowledgeSource` instances.  Their
`tool_name` strings are stored on the class. |

**Returns:** `Callable[[type], type]` — A class decorator.

**Raises:**

| Exception | Description |
|---|---|
| `DecoratorUsageError` | When called with no sources. |

## Agent types

### `AgentMeta`

```python
class AgentMeta(model: str | None, system: str | None, config: AgentConfig, tool_classes: tuple[Any, ...] = tuple(), name: str = '', memory: Any | None = None, conversation_store: Any | None = None, knowledge_source_filter: tuple[str, ...] | None = None, runner_class: type | None = None, tools: dict[str, Any] = dict())
```

Metadata attached to a class decorated with `@agent()`.

Stored under the `AGENT_META` attribute on the decorated class.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `str | None` | LLM model identifier.  `None` means "inherit from
`LLMConfig` at runtime". |
| `system` | `str | None` | System prompt for this agent.  `None` falls back to the
class docstring (if any) or the `AgentConfig.system_prompt` default. |
| `config` | `AgentConfig` | Runtime behaviour configuration. |
| `tool_classes` | `tuple[Any, ...]` | Tool classes/functions attached via `@use_tools()`.
Resolved at compile time; `None` entries are already removed. |
| `name` | `str` | Human-readable agent name.  Defaults to the decorated class
name when not supplied explicitly to `@agent()`. |
| `memory` | `Any | None` | Per-agent memory instance supplied via
`@agent(memory=…)`.  When set, the **same instance is reused
across every** `~lauren_ai._agents._runner.AgentRunnerBase.run()`
**call** — that is what "agent memory" means.  When `None` the
runner builds a fresh
`ShortTermMemory` per turn. |
| `conversation_store` | `Any | None` | Per-agent conversation store supplied via
`@agent(conversation_store=…)`.  When `None`,
`AgentModule.for_root()` auto-creates an
`InMemoryConversationStore` and writes it
back here. |
| `knowledge_source_filter` | `tuple[str, ...] | None` | Tuple of tool names from
`@use_knowledge_sources(…)`.  `None` means **no KB tools** for
this agent — opt-in only.  Set during `AgentModule.for_root` from
the decorated class's `__dict__` (strict-inheritance — never
inherited from parent classes). |
| `runner_class` | `type | None` | The concrete `AgentRunnerBase` subclass for the
`AgentModule` this agent belongs to.  Set
by `AgentModule.for_root`.  Used by `AgentRunner[X]` resolution. |
| `tools` | `dict[str, Any]` | Per-agent resolved tool map `{name: (callable, ToolMeta)}`.
Populated by `AgentModule.for_root` at startup.  Mirrors exactly the
subset of the module's tool dict this agent is allowed to use (via
`@use_tools()` + opted-in KB tools).  Empty dict until `for_root` runs. |

### `AgentContext`

```python
class AgentContext(agent_id: str, agent_run_id: str, agent_class: type, config: AgentConfig, memory: Any, turn: int, metadata: dict[str, Any], request: Any | None = None, execution_context: Any | None = None, signals: Any | None = None)
```

Runtime context for a single agent run.

One `AgentContext` is created at the start of every
`~lauren_ai._agents._runner.AgentRunner.run()` call and passed to
all lifecycle hooks.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent_id` | `str` | Unique identifier for this agent instance (random hex). |
| `agent_run_id` | `str` | Unique identifier for *this specific run* (random hex).
Distinct from `agent_id` — the same agent instance may be run
multiple times. |
| `agent_class` | `type` | The `@agent()`-decorated class. |
| `config` | `AgentConfig` | Effective `AgentConfig` for this
run (merged from module-level defaults and per-agent overrides). |
| `memory` | `Any` | Short-term conversation memory for this run. |
| `turn` | `int` | Current agentic loop iteration (0-based). |
| `metadata` | `dict[str, Any]` | Key-value metadata bag.  Populated from
`@set_metadata()` decorators and caller-supplied `metadata=`
arguments. |
| `request` | `Any | None` | Originating HTTP `Request`, or
`None` when the agent is not invoked from a web handler. |
| `execution_context` | `Any | None` | The lauren `ExecutionContext`
(carries `request`, `handler_class`, `handler_func`,
`route_template`, and `metadata`) when the agent is invoked
from a route handler.  `None` otherwise. |
| `signals` | `Any | None` | Signal bus for emitting lifecycle events.  `None` in
environments where no `SignalBus` is registered. |

#### `AgentContext.get_metadata`

```python
def get_metadata(self, key: str, default: Any = None) -> Any
```

Return metadata value for *key*, or *default* if absent.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `key` | `str` | Metadata key to look up. |
| `default` | `Any` | Fallback value when the key is not present. |

**Returns:** `Any` — The metadata value or *default*.

### `AgentResponse`

```python
class AgentResponse(content: str, turns: int, total_usage: Any, tool_calls_made: list[Any], stop_reason: Literal['end_turn', 'max_turns', 'budget_exceeded', 'error'], metadata: dict[str, Any] = dict(), reasoning_traces: list[str] = list())
```

The result of a completed agent run.

Returned by `~lauren_ai._agents._runner.AgentRunner.run()` after the
agentic loop terminates.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `content` | `str` | Final text output from the agent (the last assistant
message's text content). |
| `turns` | `int` | Number of agentic loop iterations executed. |
| `total_usage` | `Any` | Cumulative `TokenUsage`
across all turns. |
| `tool_calls_made` | `list[Any]` | All `ToolCall`
instances that were executed during the run (across all turns). |
| `stop_reason` | `Literal['end_turn', 'max_turns', 'budget_exceeded', 'error']` | Why the agent loop terminated:

* `"end_turn"` — the model indicated a natural end.
* `"max_turns"` — the `max_turns` limit was reached.
* `"budget_exceeded"` — cost or token budget was crossed.
* `"error"` — an unrecoverable error occurred. |
| `metadata` | `dict[str, Any]` | Arbitrary metadata attached to the response. |
| `reasoning_traces` | `list[str]` | Extended-thinking / reasoning traces collected
across all turns (Anthropic only). |

#### `AgentResponse.as_stream`

```python
def as_stream(self) -> AsyncIterator[str]
```

Wrap the response content as a single-item async iterator.

Useful for handlers that expect an async generator regardless of
whether the response was produced via streaming or not.

**Returns:** `AsyncIterator[str]` — An async iterator yielding the single content string.

## Runner

### `AgentRunner`

```python
class AgentRunner
```

Structural interface for agent runner implementations.

In-module DI
------------
Declare `runner: AgentRunner` in any service or tool inside the same
`AgentModule` and the DI container injects
that module's runner automatically.

Cross-module DI — `AgentRunner[AgentX]`
-----------------------------------------
For controllers in other modules that need a *specific* agent's
runner, subscript `AgentRunner` with the agent class:

    class MyController:
        def __init__(
            self,
            unauth_runner: AgentRunner[UnauthenticatedCRMAgent],
            auth_runner:   AgentRunner[AuthenticatedCRMAgent],
        ): ...

`AgentRunner[X]` returns a fresh, **cached** real subclass — so
`AgentRunner[X] is AgentRunner[X]` and the parameterized form is a
valid DI token.  `AgentModule.for_root` registers
`use_existing(provide=AgentRunner[agent_cls],
existing=<module's runner>)` for every agent in `agents=`, so the
container can resolve cross-module references by agent class.

The mechanism mirrors `HandoffTo` / `HandoffBackTo`'s
`__class_getitem__` precedent — the parameterized form is a real
class (not `_GenericAlias`), so the framework's `_looks_injectable`
check accepts it as a constructor annotation.

Static-typing note
------------------
Because subscript returns a real subclass via `__class_getitem__`,
static type-checkers (mypy, pyright) see `AgentRunner[X]` as bare
`AgentRunner` — the type parameter `X` is *not* preserved for
static analysis.  This is the same limitation as `HandoffTo[X, Y]`.
Runtime DI resolution is unaffected.

#### `AgentRunner.run`

```python
def run(self, agent: Any, message: str, conversation_id: str | None = None, metadata: dict[str, Any] | None = None, request: Any | None = None, execution_context: Any | None = None, run_id: str | None = None) -> AgentResponse
```

#### `AgentRunner.run_stream`

```python
def run_stream(self, agent: Any, message: str, conversation_id: str | None = None, metadata: dict[str, Any] | None = None, request: Any | None = None, execution_context: Any | None = None, run_id: str | None = None) -> AsyncIterator[CompletionChunk]
```

#### `AgentRunner.approve_tool`

```python
def approve_tool(self, agent_run_id: str, tool_use_id: str) -> None
```

#### `AgentRunner.reject_tool`

```python
def reject_tool(self, agent_run_id: str, tool_use_id: str, reason: str = '') -> None
```

### `AgentRunnerBase`

```python
class AgentRunnerBase(transport: Any, signals: Any | None = None, cache_backend: CacheBackend | None = None)
```

Concrete implementation of the `AgentRunner` Protocol.

Owns the observe → think → act → observe loop.  Resolves agent meta from
the decorated class, creates per-run state (`AgentContext`
and `ShortTermMemory`), calls the LLM transport,
dispatches tool calls, and aggregates results into an
`AgentResponse`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `transport` | `Any` | Provider-agnostic LLM transport. |
| `signals` | `Any | None` | Optional signal bus for emitting lifecycle events. |
| `cache_backend` | `CacheBackend | None` | Optional cache backend for tool result caching. |

#### `AgentRunnerBase.run`

```python
def run(self, agent: Any, message: str, conversation_id: str | None = None, metadata: dict[str, Any] | None = None, request: Any | None = None, execution_context: Any | None = None, run_id: str | None = None, conversation_store: Any | None = None, memory: Any | None = None) -> AgentResponse
```

Run an `@agent()`-decorated instance through the agentic loop.

Returns once the loop terminates (end of turn, max turns, budget
exceeded, or delegation).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent` | `Any` | A resolved `@agent()`-decorated class instance (from
the DI container) or the class itself (auto-resolved when a
container is set). |
| `message` | `str` | The initial user message to seed the conversation. |
| `conversation_id` | `str | None` | Optional conversation session identifier.
When provided, initial history is loaded from the effective
conversation store. |
| `metadata` | `dict[str, Any] | None` | Additional key-value metadata injected into
`AgentContext`. |
| `request` | `Any | None` | Originating HTTP request, if any. |
| `execution_context` | `Any | None` | The lauren `ExecutionContext` (route
metadata, handler class/func, authenticated user via
`request.state`) when invoked from a route handler. |
| `run_id` | `str | None` | Optional explicit run identifier.  A random hex string
is generated when `None`. |
| `conversation_store` | `Any | None` | Per-request override of the agent's
conversation store.  Wins over `meta.conversation_store`. |
| `memory` | `Any | None` | Per-request override of the agent's memory instance.
Wins over `meta.memory`.  When neither is supplied, a fresh
`ShortTermMemory` is constructed for
this turn. |

**Returns:** `AgentResponse` — The aggregated result of the agentic run.

**Raises:**

| Exception | Description |
|---|---|
| `AgentConfigError` | When *agent* is not decorated with
`@agent()`. |
| `AgentMaxTurnsError` | When the `max_turns` limit is reached
and `tool_error_policy` is `"raise"`. |
| `AgentBudgetExceededError` | When the cost / token budget is
crossed mid-run. |

#### `AgentRunnerBase.run_stream`

```python
def run_stream(self, agent: Any, message: str, conversation_id: str | None = None, metadata: dict[str, Any] | None = None, request: Any | None = None, execution_context: Any | None = None, run_id: str | None = None, conversation_store: Any | None = None, memory: Any | None = None) -> AsyncIterator[CompletionChunk]
```

Run an agent with streaming output.

Yields `CompletionChunk` items as they
arrive from the transport.  Tool calls are executed silently between
turns (their results are **not** yielded to the caller).

Functionally at parity with `run()` — fires the same lifecycle
hooks (`on_start` / `on_turn_complete` / `on_finish`), emits the
same signals (`ModelCallStarted`, `ModelCallComplete`,
`AgentTurnComplete`, `ToolCall*`, `AgentRunComplete`), enforces
`max_cost_usd`, and loads / saves conversation history through the
agent's `meta.conversation_store` (with per-request override).

Usage::

    async for chunk in await runner.run_stream(agent, "Hello"):
        print(chunk.delta, end="", flush=True)

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent` | `Any` | A resolved `@agent()`-decorated instance. |
| `message` | `str` | The initial user message. |
| `conversation_id` | `str | None` | Optional conversation session identifier. |
| `metadata` | `dict[str, Any] | None` | Additional key-value metadata for the context. |
| `request` | `Any | None` | Originating HTTP request, if any. |
| `execution_context` | `Any | None` | The lauren `ExecutionContext` (route
metadata, handler class/func, authenticated user via
`request.state`) when invoked from a route handler.  Threaded
into `ToolContext.agent_context.execution_context` for every tool call. |
| `run_id` | `str | None` | Optional explicit run identifier. |
| `conversation_store` | `Any | None` | Per-request override of the agent's
conversation store.  Wins over `meta.conversation_store`. |
| `memory` | `Any | None` | Per-request override of the agent's memory instance.
Wins over `meta.memory`. |

**Returns:** `AsyncIterator[CompletionChunk]` — An async iterator of completion chunks.

#### `AgentRunnerBase.approve_tool`

```python
def approve_tool(self, agent_run_id: str, tool_use_id: str) -> None
```

Approve a pending HITL tool call.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent_run_id` | `str` | The run identifier returned by `run()`. |
| `tool_use_id` | `str` | The provider-assigned tool call identifier to
approve. |

#### `AgentRunnerBase.reject_tool`

```python
def reject_tool(self, agent_run_id: str, tool_use_id: str, reason: str = '') -> None
```

Reject a pending HITL tool call.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agent_run_id` | `str` | The run identifier. |
| `tool_use_id` | `str` | The tool call identifier to reject. |
| `reason` | `str` | Optional human-readable rejection reason. |

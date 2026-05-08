# Agents

Decorators and types for building AI agents.

## Decorators

### `agent`

```python
def agent(args: Any = (), name: str | None = None, model: str | None = None, system: str | None = None, max_turns: int | None = None, temperature: float | None = None, memory: Any | None = None, conversation_store: Any | None = None, config_kwargs: Any = {}) -> Callable[[type[C]], type[C]]
```

Decorator that marks a class as an AI agent.

Must be called **with parentheses**: ``@agent()``.  Using the bare form
``@agent`` (without parentheses) raises
:class:`~lauren_ai._exceptions.DecoratorUsageError`.

The decorated class:

* Gets the :data:`AGENT_META` attribute set to an :class:`AgentMeta`
  instance.
* Is automatically registered as ``@injectable(scope=Scope.SINGLETON)``
  unless ``@injectable`` is already applied.
* Can define optional lifecycle hooks: ``on_start``, ``on_tool_result``,
  ``on_turn_complete``, ``on_finish``.

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

:param name: Human-readable agent name exposed via
    :attr:`AgentMeta.name` and :attr:`AgentContext.agent_name`.  When
    ``None`` the decorated class ``__name__`` is used.
:type name: str | None
:param model: LLM model identifier override.  When ``None`` the model is
    taken from :class:`~lauren_ai._config.LLMConfig` at runtime.
:type model: str | None
:param system: System prompt.  When ``None`` the class docstring is used,
    falling back to ``AgentConfig.system_prompt``.
:type system: str | None
:param max_turns: Maximum agentic loop iterations.  Forwarded to
    :class:`~lauren_ai._config.AgentConfig`.
:type max_turns: int | None
:param temperature: Sampling temperature override.  Forwarded to
    :class:`~lauren_ai._config.AgentConfig`.
:type temperature: float | None
:param memory: Per-agent memory instance.  When set, the **same
    instance is reused across every** :meth:`run` **call** — that is
    what "agent memory" means semantically.  When ``None`` the runner
    builds a fresh :class:`~lauren_ai._memory.ShortTermMemory` per
    turn.  Per-call ``runner.run(agent, …, memory=…)`` always wins.
:type memory: Any | None
:param conversation_store: Per-agent conversation store.  When
    ``None``, :meth:`AgentModule.for_root` auto-creates an
    :class:`~lauren_ai._memory.InMemoryConversationStore` and writes
    it back to AgentMeta.  Per-call ``runner.run(agent, …,
    conversation_store=…)`` always wins.
:type conversation_store: Any | None
:param config_kwargs: Additional keyword arguments forwarded to
    :class:`~lauren_ai._config.AgentConfig`.
:return: A class decorator.
:rtype: Callable[[type], type]
:raises DecoratorUsageError: When called without parentheses (bare
    ``@agent``).

### `use_tools`

```python
def use_tools(tools: Any = ()) -> Callable[[type[C]], type[C]]
```

Attach tool classes or functions to an ``@agent()``-decorated class.

``None`` entries are silently dropped (consistent with ``use_guards``
behaviour in the lauren framework).

Typically stacked above ``@agent()``::

    @use_tools(WebSearchTool, get_weather, None)  # None is dropped
    @agent(model="claude-opus-4-6")
    class ResearchAgent: ...

At compile time, :func:`validate_agent_class` resolves each entry from the
DI container and builds the :class:`~lauren_ai._transport.ToolSchema` list.

:param tools: Tool classes or callables decorated with ``@tool()``.
    ``None`` entries are ignored.
:type tools: Any
:return: A class decorator that attaches the tools to the class.
:rtype: Callable[[type], type]

### `use_knowledge_sources`

```python
def use_knowledge_sources(sources: Any = ()) -> Callable[[type[C]], type[C]]
```

Restrict an agent's knowledge-base tool visibility to the listed sources.

Without this decorator an agent has **no** knowledge-base tools — KB
visibility is **opt-in** even when the enclosing
:class:`~lauren_ai._module.AgentModule` declares ``knowledge=[…]``.
With it, only the listed :class:`~lauren_ai._knowledge.KnowledgeSource`
instances are attached to the agent's tool schema at runtime.

Stores the *tool names* (strings) on the class, not the
``KnowledgeSource`` instances — matching against the module's declared
sources is a string-set comparison at module-build time.

Stacking is allowed (decorator concatenates names with any already
present), and **strict-inheritance applies** (mirrors Lauren's
framework golden-rule #3): a subclass that inherits the metadata from
a parent without redeclaring ``@use_knowledge_sources`` raises
:class:`~lauren_ai._exceptions.MetadataInheritanceError` at module
construction time.

Typically stacked above ``@agent()``::

    from app.knowledge_sources import PUBLIC_KB_SOURCE

    @use_knowledge_sources(PUBLIC_KB_SOURCE)
    @agent(name="UnauthCRM", model="...")
    class UnauthenticatedCRMAgent: ...

The :meth:`AgentModule.for_root` validation step checks every name
here against the module's ``knowledge=`` list and raises
:class:`~lauren_ai._exceptions.DecoratorUsageError` if any source is
not declared at module level.

:param sources: One or more :class:`KnowledgeSource` instances.  Their
    ``tool_name`` strings are stored on the class.
:type sources: KnowledgeSource
:return: A class decorator.
:rtype: Callable[[type], type]
:raises DecoratorUsageError: When called with no sources.

## Agent types

### `AgentMeta`

```python
class AgentMeta(model: str | None, system: str | None, config: AgentConfig, tool_classes: tuple[Any, ...] = tuple(), name: str = '', memory: Any | None = None, conversation_store: Any | None = None, knowledge_source_filter: tuple[str, ...] | None = None, runner_class: type | None = None)
```

Metadata attached to a class decorated with ``@agent()``.

Stored under the :data:`AGENT_META` attribute on the decorated class.

:param model: LLM model identifier.  ``None`` means "inherit from
    :class:`~lauren_ai._config.LLMConfig` at runtime".
:type model: str | None
:param system: System prompt for this agent.  ``None`` falls back to the
    class docstring (if any) or the ``AgentConfig.system_prompt`` default.
:type system: str | None
:param config: Runtime behaviour configuration.
:type config: AgentConfig
:param tool_classes: Tool classes/functions attached via ``@use_tools()``.
    Resolved at compile time; ``None`` entries are already removed.
:type tool_classes: tuple[Any, ...]
:param name: Human-readable agent name.  Defaults to the decorated class
    name when not supplied explicitly to ``@agent()``.
:type name: str
:param memory: Per-agent memory instance supplied via
    ``@agent(memory=…)``.  When set, the **same instance is reused
    across every** :meth:`~lauren_ai._agents._runner.AgentRunnerBase.run`
    **call** — that is what "agent memory" means.  When ``None`` the
    runner builds a fresh
    :class:`~lauren_ai._memory.ShortTermMemory` per turn.
:type memory: Any | None
:param conversation_store: Per-agent conversation store supplied via
    ``@agent(conversation_store=…)``.  When ``None``,
    :meth:`AgentModule.for_root` auto-creates an
    :class:`~lauren_ai._memory.InMemoryConversationStore` and writes it
    back here.
:type conversation_store: Any | None
:param knowledge_source_filter: Tuple of tool names from
    ``@use_knowledge_sources(…)``.  ``None`` means **no KB tools** for
    this agent — opt-in only.  Set during ``AgentModule.for_root`` from
    the decorated class's ``__dict__`` (strict-inheritance — never
    inherited from parent classes).
:type knowledge_source_filter: tuple[str, ...] | None
:param runner_class: The concrete ``AgentRunnerBase`` subclass for the
    :class:`~lauren_ai._module.AgentModule` this agent belongs to.  Set
    by ``AgentModule.for_root``.  Used by ``AgentRunner[X]`` resolution.
:type runner_class: type | None

### `AgentContext`

```python
class AgentContext(agent_id: str, agent_run_id: str, agent_class: type, config: AgentConfig, memory: Any, turn: int, metadata: dict[str, Any], request: Any | None = None, execution_context: Any | None = None, signals: Any | None = None)
```

Runtime context for a single agent run.

One :class:`AgentContext` is created at the start of every
:meth:`~lauren_ai._agents._runner.AgentRunner.run` call and passed to
all lifecycle hooks.

:param agent_id: Unique identifier for this agent instance (random hex).
:type agent_id: str
:param agent_run_id: Unique identifier for *this specific run* (random hex).
    Distinct from ``agent_id`` — the same agent instance may be run
    multiple times.
:type agent_run_id: str
:param agent_class: The ``@agent()``-decorated class.
:type agent_class: type
:param config: Effective :class:`~lauren_ai._config.AgentConfig` for this
    run (merged from module-level defaults and per-agent overrides).
:type config: AgentConfig
:param memory: Short-term conversation memory for this run.
:type memory: ShortTermMemory
:param turn: Current agentic loop iteration (0-based).
:type turn: int
:param metadata: Key-value metadata bag.  Populated from
    ``@set_metadata()`` decorators and caller-supplied ``metadata=``
    arguments.
:type metadata: dict[str, Any]
:param request: Originating HTTP :class:`~lauren.types.Request`, or
    ``None`` when the agent is not invoked from a web handler.
:type request: Any | None
:param execution_context: The lauren :class:`~lauren.types.ExecutionContext`
    (carries ``request``, ``handler_class``, ``handler_func``,
    ``route_template``, and ``metadata``) when the agent is invoked
    from a route handler.  ``None`` otherwise.
:type execution_context: Any | None
:param signals: Signal bus for emitting lifecycle events.  ``None`` in
    environments where no :class:`SignalBus` is registered.
:type signals: Any | None

#### `AgentContext.get_metadata`

```python
def get_metadata(self, key: str, default: Any = None) -> Any
```

Return metadata value for *key*, or *default* if absent.

:param key: Metadata key to look up.
:type key: str
:param default: Fallback value when the key is not present.
:type default: Any
:return: The metadata value or *default*.
:rtype: Any

### `AgentResponse`

```python
class AgentResponse(content: str, turns: int, total_usage: Any, tool_calls_made: list[Any], stop_reason: Literal['end_turn', 'max_turns', 'budget_exceeded', 'error'], metadata: dict[str, Any] = dict(), reasoning_traces: list[str] = list())
```

The result of a completed agent run.

Returned by :meth:`~lauren_ai._agents._runner.AgentRunner.run` after the
agentic loop terminates.

:param content: Final text output from the agent (the last assistant
    message's text content).
:type content: str
:param turns: Number of agentic loop iterations executed.
:type turns: int
:param total_usage: Cumulative :class:`~lauren_ai._transport.TokenUsage`
    across all turns.
:type total_usage: TokenUsage
:param tool_calls_made: All :class:`~lauren_ai._transport.ToolCall`
    instances that were executed during the run (across all turns).
:type tool_calls_made: list[ToolCall]
:param stop_reason: Why the agent loop terminated:

    * ``"end_turn"`` — the model indicated a natural end.
    * ``"max_turns"`` — the ``max_turns`` limit was reached.
    * ``"budget_exceeded"`` — cost or token budget was crossed.
    * ``"error"`` — an unrecoverable error occurred.
:type stop_reason: Literal["end_turn", "max_turns", "budget_exceeded", "error"]
:param metadata: Arbitrary metadata attached to the response.
:type metadata: dict[str, Any]
:param reasoning_traces: Extended-thinking / reasoning traces collected
    across all turns (Anthropic only).
:type reasoning_traces: list[str]

#### `AgentResponse.as_stream`

```python
def as_stream(self) -> AsyncIterator[str]
```

Wrap the response content as a single-item async iterator.

Useful for handlers that expect an async generator regardless of
whether the response was produced via streaming or not.

:return: An async iterator yielding the single content string.
:rtype: AsyncIterator[str]

## Runner

### `AgentRunner`

```python
class AgentRunner
```

Structural interface for agent runner implementations.

In-module DI
------------
Declare ``runner: AgentRunner`` in any service or tool inside the same
:class:`~lauren_ai._module.AgentModule` and the DI container injects
that module's runner automatically.

Cross-module DI — ``AgentRunner[AgentX]``
-----------------------------------------
For controllers in other modules that need a *specific* agent's
runner, subscript ``AgentRunner`` with the agent class:

    class MyController:
        def __init__(
            self,
            unauth_runner: AgentRunner[UnauthenticatedCRMAgent],
            auth_runner:   AgentRunner[AuthenticatedCRMAgent],
        ): ...

``AgentRunner[X]`` returns a fresh, **cached** real subclass — so
``AgentRunner[X] is AgentRunner[X]`` and the parameterized form is a
valid DI token.  ``AgentModule.for_root`` registers
``use_existing(provide=AgentRunner[agent_cls],
existing=<module's runner>)`` for every agent in ``agents=``, so the
container can resolve cross-module references by agent class.

The mechanism mirrors :class:`HandoffTo` / :class:`HandoffBackTo`'s
``__class_getitem__`` precedent — the parameterized form is a real
class (not ``_GenericAlias``), so the framework's ``_looks_injectable``
check accepts it as a constructor annotation.

Static-typing note
------------------
Because subscript returns a real subclass via ``__class_getitem__``,
static type-checkers (mypy, pyright) see ``AgentRunner[X]`` as bare
``AgentRunner`` — the type parameter ``X`` is *not* preserved for
static analysis.  This is the same limitation as ``HandoffTo[X, Y]``.
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
class AgentRunnerBase(transport: Any, tools: dict[str, tuple[Any, ToolMeta]], config: LLMConfig, signals: Any | None = None, cache_backend: CacheBackend | None = None, knowledge_tool_names: set[str] | None = None)
```

Concrete implementation of the :class:`AgentRunner` Protocol.

Owns the observe → think → act → observe loop.  Resolves agent meta from
the decorated class, creates per-run state (:class:`~lauren_ai._agents.AgentContext`
and :class:`~lauren_ai._memory.ShortTermMemory`), calls the LLM transport,
dispatches tool calls, and aggregates results into an
:class:`~lauren_ai._agents.AgentResponse`.

:param transport: Provider-agnostic LLM transport.
:type transport: Any
:param tools: Mapping of tool name to ``(callable_or_instance, ToolMeta)``.
    Built by ``AgentModule.for_root()`` or ``AgentTestClient``.
:type tools: dict[str, tuple[Any, ToolMeta]]
:param config: Application-level LLM configuration (model, max_tokens, etc.).
:type config: LLMConfig
:param signals: Optional signal bus for emitting lifecycle events.
:type signals: Any | None
:param cache_backend: Optional cache backend for tool result caching.
:type cache_backend: CacheBackend | None

#### `AgentRunnerBase.run`

```python
def run(self, agent: Any, message: str, conversation_id: str | None = None, metadata: dict[str, Any] | None = None, request: Any | None = None, execution_context: Any | None = None, run_id: str | None = None, conversation_store: Any | None = None, memory: Any | None = None) -> AgentResponse
```

Run an ``@agent()``-decorated instance through the agentic loop.

Returns once the loop terminates (end of turn, max turns, budget
exceeded, or delegation).

:param agent: A resolved ``@agent()``-decorated class instance (from
    the DI container) or the class itself (auto-resolved when a
    container is set).
:type agent: Any
:param message: The initial user message to seed the conversation.
:type message: str
:param conversation_id: Optional conversation session identifier.
    When provided, initial history is loaded from the effective
    conversation store.
:type conversation_id: str | None
:param metadata: Additional key-value metadata injected into
    :class:`~lauren_ai._agents.AgentContext`.
:type metadata: dict[str, Any] | None
:param request: Originating HTTP request, if any.
:type request: Any | None
:param execution_context: The lauren ``ExecutionContext`` (route
    metadata, handler class/func, authenticated user via
    ``request.state``) when invoked from a route handler.
:type execution_context: Any | None
:param run_id: Optional explicit run identifier.  A random hex string
    is generated when ``None``.
:type run_id: str | None
:param conversation_store: Per-request override of the agent's
    conversation store.  Wins over ``meta.conversation_store``.
:type conversation_store: Any | None
:param memory: Per-request override of the agent's memory instance.
    Wins over ``meta.memory``.  When neither is supplied, a fresh
    :class:`~lauren_ai._memory.ShortTermMemory` is constructed for
    this turn.
:type memory: Any | None
:return: The aggregated result of the agentic run.
:rtype: AgentResponse
:raises AgentConfigError: When *agent* is not decorated with
    ``@agent()``.
:raises AgentMaxTurnsError: When the ``max_turns`` limit is reached
    and ``tool_error_policy`` is ``"raise"``.
:raises AgentBudgetExceededError: When the cost / token budget is
    crossed mid-run.

#### `AgentRunnerBase.run_stream`

```python
def run_stream(self, agent: Any, message: str, conversation_id: str | None = None, metadata: dict[str, Any] | None = None, request: Any | None = None, execution_context: Any | None = None, run_id: str | None = None, conversation_store: Any | None = None, memory: Any | None = None) -> AsyncIterator[CompletionChunk]
```

Run an agent with streaming output.

Yields :class:`~lauren_ai._transport.CompletionChunk` items as they
arrive from the transport.  Tool calls are executed silently between
turns (their results are **not** yielded to the caller).

Functionally at parity with :meth:`run` — fires the same lifecycle
hooks (``on_start`` / ``on_turn_complete`` / ``on_finish``), emits the
same signals (``ModelCallStarted``, ``ModelCallComplete``,
``AgentTurnComplete``, ``ToolCall*``, ``AgentRunComplete``), enforces
``max_cost_usd``, and loads / saves conversation history through the
agent's ``meta.conversation_store`` (with per-request override).

Usage::

    async for chunk in await runner.run_stream(agent, "Hello"):
        print(chunk.delta, end="", flush=True)

:param agent: A resolved ``@agent()``-decorated instance.
:type agent: Any
:param message: The initial user message.
:type message: str
:param conversation_id: Optional conversation session identifier.
:type conversation_id: str | None
:param metadata: Additional key-value metadata for the context.
:type metadata: dict[str, Any] | None
:param request: Originating HTTP request, if any.
:type request: Any | None
:param execution_context: The lauren ``ExecutionContext`` (route
    metadata, handler class/func, authenticated user via
    ``request.state``) when invoked from a route handler.  Threaded
    into ``ToolContext.execution_context`` for every tool call.
:type execution_context: Any | None
:param run_id: Optional explicit run identifier.
:type run_id: str | None
:param conversation_store: Per-request override of the agent's
    conversation store.  Wins over ``meta.conversation_store``.
:type conversation_store: Any | None
:param memory: Per-request override of the agent's memory instance.
    Wins over ``meta.memory``.
:type memory: Any | None
:return: An async iterator of completion chunks.
:rtype: AsyncIterator[CompletionChunk]

#### `AgentRunnerBase.approve_tool`

```python
def approve_tool(self, agent_run_id: str, tool_use_id: str) -> None
```

Approve a pending HITL tool call.

:param agent_run_id: The run identifier returned by ``run()``.
:type agent_run_id: str
:param tool_use_id: The provider-assigned tool call identifier to
    approve.
:type tool_use_id: str

#### `AgentRunnerBase.reject_tool`

```python
def reject_tool(self, agent_run_id: str, tool_use_id: str, reason: str = '') -> None
```

Reject a pending HITL tool call.

:param agent_run_id: The run identifier.
:type agent_run_id: str
:param tool_use_id: The tool call identifier to reject.
:type tool_use_id: str
:param reason: Optional human-readable rejection reason.
:type reason: str


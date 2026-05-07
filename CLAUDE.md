# CLAUDE.md — Developer & Contributor Guide for `lauren-ai`

This file is loaded automatically by Claude Code when working inside the
`lauren-ai` repository.  It records the architectural invariants, testing
conventions, and contribution patterns that every AI coding agent should
follow.

---

## Project overview

`lauren-ai` is the first-party AI/LLM companion for the Lauren web framework.
It adds:

- A decorator-first agent system (`@agent`, `@use_tools`, `@guardrail`, `@remember`, `@team`, `@traced`)
- A provider-agnostic transport layer (Anthropic, OpenAI, Ollama, LiteLLM)
- A tool system with automatic JSON-schema generation
- A multi-agent delegation and team orchestration system
- Memory tiers: short-term, conversation, vector, and user memory (`@remember`)
- Built-in guardrails, prompt templates, chains, and output parsers
- Cost tracking, rate limiting, and token budgets
- A `SignalBus` for lifecycle telemetry

---

## Repository layout

```
src/lauren_ai/
├── __init__.py             # Public API — all public symbols re-exported here
├── _agents/
│   ├── __init__.py         # @agent(), @use_tools(), AgentMeta, AgentContext, AgentResponse
│   ├── _compile.py         # Agent validation at startup
│   └── _runner.py          # AgentRunner (Protocol) / AgentRunnerBase (concrete) — the agentic loop
├── _chains/                # Chain / Runnable / RunnableLambda
├── _config.py              # LLMConfig, AgentConfig (frozen dataclasses)
├── _cost/                  # CostTracker, TokenBudget, RateLimiter, PricingTable
├── _exceptions.py          # All exception classes
├── _extractors.py          # DI extractors: Agent[T], Embed, StreamCompletion
├── _guardrails/
│   ├── _base.py            # InputGuardrail / OutputGuardrail protocols
│   ├── _builtin.py         # PromptInjectionFilter, PIIRedactor, LengthFilter, TopicFilter
│   ├── _decorator.py       # @guardrail() decorator
│   └── _llm.py             # LLMGuardrail (uses LLM to evaluate content)
├── _guards.py              # HTTP guard factories: token_budget_guard, safety_guard, etc.
├── _memory/
│   ├── __init__.py         # ShortTermMemory (exported)
│   ├── _in_memory_user.py  # InMemoryUserMemoryStore
│   ├── _remember.py        # @remember() decorator, RememberMeta, extract_facts()
│   ├── _stores.py          # InMemoryConversationStore, ConversationStore protocol
│   ├── _user.py            # UserMemoryStore protocol, MemoryFact
│   └── _vector.py          # InMemoryVectorStore
├── _module.py              # LLMModule, LLMService, AgentModule, EmbedService
├── _output_parsers/        # StrOutputParser, JSONOutputParser, PydanticOutputParser, etc.
├── _prompts/               # PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
├── _routing/               # SemanticRouter, Route
├── _signals.py             # SignalBus, ModelCallComplete, AgentRunComplete, etc.
├── _skills/
│   └── __init__.py         # WebSearchTool, HttpFetchTool, CodeExecutionTool, DelegateToAgentTool
├── _teams/
│   ├── _decorator.py       # @team() decorator, TeamMeta
│   ├── _events.py          # TeamWorkerStarted/Finished, TeamCoordinatorDecision, TeamFinalAnswer
│   ├── _memory.py          # TeamMemory (shared state across workers)
│   └── _runner.py          # TeamRunner — coordinator and collaborate modes
├── _tools/
│   ├── __init__.py         # @tool(), ToolContext, ToolResult
│   ├── _executor.py        # ToolExecutor with cache support
│   ├── _registry.py        # ToolRegistry
│   ├── _schema.py          # JSON-schema generation from function signatures
│   └── _types_compat.py    # Python 3.9 / 3.10 type annotation compat
├── _tracing/               # @traced(), TraceStore, exporters
├── _transport/
│   ├── __init__.py         # Message, Completion, CompletionChunk, TokenUsage, ToolCall, ToolSchema
│   ├── _anthropic.py       # Anthropic transport
│   ├── _mock.py            # MockTransport for testing
│   ├── _multimodal.py      # ImageContent, AudioContent, DocumentContent
│   ├── _ollama.py          # Ollama transport
│   ├── _openai.py          # OpenAI-compatible transport (also used for OpenRouter)
│   └── _structured.py      # StructuredLLM
└── testing.py              # AgentTestClient, MockTransport re-export
```

---

## Critical invariants

### 1. No `from __future__ import annotations` in tool files

The `@tool()` decorator calls `inspect.signature()` and reads `__annotations__`
**at decoration time** to build the JSON schema.  PEP 563 lazy evaluation (enabled
by `from __future__ import annotations`) causes all annotations to become strings
instead of live types, which breaks schema generation silently.

**Rule:** Never add `from __future__ import annotations` to any file that defines
`@tool()`-decorated functions.  Add the comment at the top of every such file:

```python

# The @tool() decorator uses inspect.signature() at decoration time to build
# the JSON schema, and PEP 563 lazy evaluation breaks that introspection.
```

### 2. Decorator ordering

Decorators are applied bottom-up in Python.  The correct order for a full
agent definition is:

```
@agent()          ← outermost: reads USE_TOOLS_META from class, sets AGENT_META
@remember()       ← reads/sets REMEMBER_META
@guardrail()      ← reads/sets GUARDRAIL_META
@use_tools(...)   ← innermost: sets USE_TOOLS_META on class
class MyAgent: ...
```

Swapping the order causes the decorator that reads metadata to run before the
decorator that writes it, resulting in silently missing tools or guardrails.

### 3. Parentheses on all AI decorators

Every decorator in `lauren-ai` **must** be called with parentheses:
`@agent()`, `@tool()`, `@guardrail()`, `@remember()`, `@team()`.
Using bare form (e.g. `@agent`) raises `DecoratorUsageError` at decoration time.

### 4. `@team()` constructor declares workers via type hints

`TeamRunner._discover_workers()` reads `__init__.__annotations__` to find worker
names.  Always annotate worker parameters with their agent class:

```python
@team(name="my-team", mode="coordinator", model="openai/gpt-4o-mini")
class MyTeam:
    def __init__(self, researcher: ResearchAgent, writer: WriterAgent) -> None:
        self.researcher = researcher
        self.writer = writer
```

---

## Transport layer — adding a new provider

1. Create `src/lauren_ai/_transport/_myprovider.py`.
2. Implement `async def complete(messages, *, model, system, tools, max_tokens, temperature, stream)`.
3. Implement `async def embed(texts, *, model, dimensions)`.
4. Register the provider string in `LLMModule.for_root()` inside `_module.py`.
5. Add tests in `tests/transports/test_myprovider.py` using `MockTransport` for
   network-free CI runs.

---

## Tool system — schema generation

`_tools/_schema.py` builds the JSON schema by:

1. Calling `inspect.signature(fn)` to get parameter names and annotations.
2. Parsing the Google-style docstring (`Args:` section) for parameter descriptions.
3. Mapping Python types → JSON Schema types (str→string, int/float→number,
   bool→boolean, dict→object, list→array, Optional[X] marks field as not required).
4. Setting `required` to all non-optional parameters excluding `ctx: ToolContext`.

The `ToolContext` parameter is **never** included in the JSON schema — it is injected
internally by `ToolExecutor` and never exposed to the LLM.

---

## Agent DI — auto-injectable singletons

`@agent()` automatically applies `@injectable(scope=Scope.SINGLETON)` unless
the class is already `@injectable`.  This means every `@agent()`-decorated class
is a registered DI provider once it is listed in `AgentModule.for_root()`.

`AgentModule.for_root()` adds each agent class to **both** `providers` and
`exports`, so controllers in the parent module can declare it as a constructor
argument and receive the fully-resolved singleton.

### `@agent()` — per-agent state parameters

```python
@agent(
    model="claude-opus-4-6",
    system="...",
    memory=ShortTermMemory(max_tokens=60_000),    # optional — reused across run() calls
    conversation_store=InMemoryConversationStore(), # optional — auto-created if None
)
class MyAgent: ...
```

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `memory` | `ShortTermMemory \| None` | `None` | Instance reused across every `run()` call (per-agent persistent context window). When `None`, a fresh `ShortTermMemory` is built per turn. |
| `conversation_store` | `ConversationStore \| None` | `None` | Per-agent conversation store. When `None`, `AgentModule.for_root()` auto-creates an `InMemoryConversationStore` and writes it back to the agent's `AgentMeta`. |

`AgentModule.for_root()` **no longer accepts `memory=` or `conversation_store=`**
at the module level — these are per-agent concerns.  Each agent gets its own
store, guaranteeing isolation even when two agents share a module.

### Cross-module DI with `AgentRunner[X]`

When a controller can see runners from **two or more** `AgentModule`s, use
the parameterized form `AgentRunner[AgentClass]` — a cached real subclass that
functions as an unambiguous DI token.

```python
from lauren_ai import AgentRunner  # @runtime_checkable Protocol

class BankingChatController:
    def __init__(
        self,
        unauth_runner: AgentRunner[UnauthenticatedCRMAgent],
        auth_runner:   AgentRunner[AuthenticatedCRMAgent],
        transfer_runner: AgentRunner[BankTransferAgent],
    ) -> None:
        ...
```

`AgentModule.for_root()` automatically registers `use_factory(provide=AgentRunner[agent_cls], ...)` for every agent in `agents=`.  Named `AgentRunnerBase` subclasses are no longer needed for cross-module DI.

Single-module scope still works as before: `runner: AgentRunner` in a tool or service inside the same module resolves via the structural Protocol fallback.

**Critical invariant: pass instances, not classes, to `runner.run()`.**

```python
# WRONG — passes the class itself, not an instance
response = await self._runner.run(ChatAgent, message)
```

Passing the class bypasses DI and **breaks lifecycle hooks** (`on_start`,
`on_turn_complete`, `on_tool_result`, `on_finish`).  When `runner.run()` receives
a class, `getattr(cls, 'on_start')` returns an unbound function; calling
`hook(ctx)` treats `ctx` as `self`, raising:

```
TypeError: ChatAgent.on_start() missing 1 required positional argument: 'ctx'
```

Always obtain the agent instance from DI (constructor injection) and pass that
instance to `runner.run()`.

---

## Agent runner — agentic loop

```
AgentRunner.run(agent_instance, message)
    │
    ├─ _get_meta() → AgentMeta from AGENT_META attribute
    ├─ _merge_config() → effective AgentConfig
    ├─ ShortTermMemory(max_tokens=...)
    ├─ hook: agent.on_start(ctx)
    │
    └─ for turn in range(max_turns):
           ├─ transport.complete(messages, model, system, tools, stream=False)
           ├─ emit ModelCallComplete signal
           ├─ hook: agent.on_turn_complete(completion, ctx)
           ├─ budget check (AgentConfig.max_cost_usd)
           ├─ if stop_reason == "end_turn" → break
           └─ if stop_reason == "tool_use" → _execute_tools() → memory.add_tool_result()

    ├─ catch DelegateToAgent → recursive AgentRunner.run(delegated_agent, message)
    ├─ hook: agent.on_finish(response, ctx)
    ├─ emit AgentRunComplete signal
    └─ return AgentResponse
```

**Delegation via `DelegateToAgent` exception:**  Any lifecycle hook (or tool)
can call `ctx.delegate(TargetAgent, message)` which raises `DelegateToAgent`.
The runner catches it and recursively runs the target agent, then wraps the
result with `stop_reason="delegated"`.

Tool-based delegation (as used in the chatbot example) calls
`AgentRunner.run(specialist, task)` directly and returns the result as a plain
dict tool result — no exception involved.

---

## Teams — coordinator vs collaborate mode

| Mode | Behaviour |
|------|-----------|
| `coordinator` | LLM coordinator decides after each round: `ROUTE: <worker>` or `DONE: <answer>`. Runs up to `max_rounds` rounds. |
| `collaborate`  | All workers run sequentially, then a synthesis call produces the final answer. |

`TeamRunner.run_stream()` yields `TeamEvent` subclass instances:
- `TeamWorkerStarted(worker_name, task, round)`
- `TeamWorkerFinished(worker_name, result_content, round)`
- `TeamCoordinatorDecision(decision, round)`
- `TeamFinalAnswer(content, rounds)`

---

## Memory — 4-tier architecture

| Tier | Class | Scope | Purpose |
|------|-------|-------|---------|
| Short-term | `ShortTermMemory` | Per run | Rolling message window passed to LLM each turn |
| Conversation | `ConversationStore` / `InMemoryConversationStore` | Per session | Persists message history across runs within a conversation |
| User memory | `UserMemoryStore` / `InMemoryUserMemoryStore` | Per user | Long-term facts extracted by `@remember()` |
| Vector | `InMemoryVectorStore` | Application | Semantic retrieval for RAG patterns |

`@remember()` must be placed between `@agent()` and `@guardrail()`.  It attaches
`RememberMeta` (with `store_token`, `extract`, `inject`, `top_k`).  The runner
reads this metadata to inject memory context and extract new facts after each turn.

### Conversation memory — automatic load/save

When an agent has a `conversation_store` (set via `@agent(conversation_store=...)`)
**and** `runner.run()` is called with a `conversation_id`, the runner automatically:

1. **Loads** prior messages from the agent's store and seeds `ShortTermMemory`
   before adding the new user message.
2. **Saves** the full updated history back to the store after `on_finish`.

Per-agent declaration (preferred):

```python
from lauren_ai import agent, InMemoryConversationStore

@agent(model="claude-opus-4-6", conversation_store=InMemoryConversationStore())
class MyAgent: ...

# AgentModule.for_root() also auto-creates a store for agents that omit it
AgentModule.for_root(agents=[MyAgent], imports=LLMProvider)
```

Per-request override via `runner.run(agent, ..., conversation_store=other_store)` wins over the agent's stored value — useful for testing or per-request stores.

Without a store the runner creates a fresh `ShortTermMemory` on every call and
the `conversation_id` is accepted but unused.  Without a `conversation_id` the
store is never touched even if one is configured.

---

## Signals — `SignalBus` lifecycle events

```python
signal_bus = SignalBus()

@signal_bus.on(ModelCallComplete)
async def my_handler(event: ModelCallComplete) -> None:
    print(event.model, event.cost_usd, event.duration_ms)
```

Available signals:
- `ModelCallStarted` — before the LLM call
- `ModelCallComplete` — after the LLM call (includes usage, cost, duration)
- `ToolCallStarted` — before a tool executes
- `ToolCallComplete` — after a tool executes (includes success/error)
- `AgentRunComplete` — after the full agentic loop finishes (includes total_cost_usd, turns)

---

## Testing patterns

Use `LLMConfig.for_testing()` to get a `MockTransport` that makes zero network
calls.  Queue responses before running code under test:

```python
from lauren_ai import LLMConfig
from lauren_ai.testing import AgentTestClient

cfg, mock = LLMConfig.for_testing()
mock.queue_response(Completion(id="1", model="mock", content="42", ...))

client = AgentTestClient(agent=MyAgent, config=cfg, mock_transport=mock)
result = await client.run("What is 6 * 7?")
assert result.content == "42"
```

---

## Adding a new built-in skill

1. Add a `@tool()`-decorated function to `src/lauren_ai/_skills/__init__.py`.
2. Add its name to `__all__`.
3. Write unit tests in `tests/skills/test_myskill.py`.
4. Document it in `skills/tools.md` and `AGENTS.md`.

---

## Production security invariants (from the banking chatbot)

The following invariants are derived from `lauren-examples/lauren-ai-chatbot/backend/`
and must be preserved in any production agent that handles privileged operations.

### 1. Identity must come from `execution_context.request.state`, never from the LLM

The authenticated sender is always read from the server-side execution context
that a guard populated **before** the LLM ran.  The LLM only supplies
non-identity parameters such as the transfer recipient and amount.

```python
# banking_tools.py — TransferFundsTool.run()
async def run(self, ctx: ToolContext, to_user: str, amount: float) -> dict:
    # CORRECT: read the sender from the guard-verified state
    exec_ctx = ctx.execution_context
    auth_uid = exec_ctx.request.state.get("user_id")  # set by SignatureGuard
    if not auth_uid:
        return {"error": "Security error: no authenticated user in ExecutionContext."}

    # `to_user` and `amount` are LLM-supplied — that is fine, they are not identity.
    result = self._db.transfer(from_user=auth_uid, to_user=to_user, amount=amount)
    ...
```

The full trust chain is:

```
SignatureGuard.can_activate(ExecutionContext)
    └─ request.state.user_id = <HMAC-verified value>
            │
AgentRunner.run(..., execution_context=ExecutionContext(request=request))
    └─ AgentContext.execution_context  (same ExecutionContext)
            │
ToolExecutor._execute_single_tool(...)
    └─ ToolContext.execution_context  (forwarded from AgentContext)
            │
tool.run(ctx: ToolContext, ...)
    └─ ctx.execution_context.request.state.get("user_id")  ← read here
```

At no point does the LLM supply or influence the sender identity.

### 2. The ContextVar routing pattern

A `ContextVar` set **before** `AgentRunner.run()` is visible inside every
`asyncio.gather` task that `SignalBus.emit` spawns, because `gather` copies the
current context into each spawned task automatically.

```python
# ws/context.py
from contextvars import ContextVar

current_user_id: ContextVar[str | None] = ContextVar("ws_current_user_id", default=None)

# chat_banking_controller.py — generate() async generator
async def generate():
    current_user_id.set(account.user_id)   # set BEFORE runner.run()
    response = await self._runner.run(
        self._crm_agent,
        full_prompt,
        execution_context=exec_ctx,
    )
    ...

# event_forwarder.py — signal handler called via asyncio.gather inside SignalBus.emit
async def _on_model_complete(self, event: ModelCallComplete) -> None:
    user_id = current_user_id.get()        # visible because gather copies the context
    if user_id:
        await self.send_to_user(user_id, {...})
```

### 3. Never `reset()` a ContextVar across task boundaries

`EventStream`'s keep-alive implementation wraps each `__anext__()` call in a
separate `asyncio.Task`.  Each task receives its own copy of the ContextVar
context at creation time.  Calling `ContextVar.reset(token)` with a `Token`
that was created in a *different* task raises `ValueError: <Token> was created
in a different Context`.

The correct pattern is to call `.set()` once at the top of the generator and
never call `.reset()`.  The task's context copy is discarded automatically when
the task completes:

```python
async def generate():
    # CORRECT: set once, never reset
    current_user_id.set(account.user_id)
    try:
        response = await self._runner.run(...)
        ...
        yield ServerSentEvent(event="done", data="")
    except Exception as exc:
        yield ServerSentEvent(event="error", data=str(exc))
    # context copy is discarded automatically when the Task exits — no reset() needed
```

### 4. `AgentRunner[X]` as the cross-module runner DI token

Every `AgentModule.for_root()` auto-generates a runner and registers
`AgentRunner[agent_cls]` aliases for every agent in `agents=`.  Controllers in
other modules use the parameterized form — **no named subclass boilerplate needed**:

```python
from lauren_ai import AgentRunner

class BankingChatController:
    def __init__(
        self,
        unauth_runner:   AgentRunner[UnauthenticatedCRMAgent],
        auth_runner:     AgentRunner[AuthenticatedCRMAgent],
        transfer_runner: AgentRunner[BankTransferAgent],
        disputes_runner: AgentRunner[DisputesAgent],
    ) -> None: ...
```

`AgentRunner[X]` is validated at construction time: passing a non-`@agent`
class raises `TypeError` immediately.  DI raises `MissingProviderError` at
startup if no module declares `X`.

For delegation tools, inject `runner: AgentRunner[TargetAgent]`:

```python
@tool()
class DelegateToBankingTransfer:
    def __init__(
        self,
        transfer_agent: BankTransferAgent,
        runner: AgentRunner[BankTransferAgent],   # ← parameterized token
    ) -> None:
        self._agent = transfer_agent
        self._runner = runner

    async def run(self, ctx: ToolContext, task: str) -> dict:
        response = await self._runner.run(
            self._agent, task,
            execution_context=ctx.execution_context,
        )
        return {"result": response.content}
```

### 5. `@use_knowledge_sources` — per-agent KB opt-in

`AgentModule.for_root(knowledge=[KnowledgeSource(...)])` declares sources at the
module level but does NOT auto-attach them to every agent.  KB access is
**opt-in** — each agent must declare `@use_knowledge_sources(source, ...)`.
Without the decorator the agent has **no** KB tools (zero tools from the
module's `knowledge=` list).

```python
from lauren_ai import use_knowledge_sources
from .knowledge_sources import PUBLIC_KB_SOURCE

@use_knowledge_sources(PUBLIC_KB_SOURCE)   # only this agent sees the KB tool
@agent(name="UnauthCRM", model="...")
class UnauthenticatedCRMAgent: ...
```

`@use_knowledge_sources` validation runs at module-build time:

- Source not in `for_root(knowledge=[...])` → `DecoratorUsageError`
- Inherited without redeclaring (subclass) → `MetadataInheritanceError`
- No args → `DecoratorUsageError`

### 6. Delegation tool lives in the calling module's `tools=`

The delegation tool belongs in the **calling module's `tools=`**, NOT in the
target module. The calling module imports the target module so the target
runner token is visible to DI:

```python
_TransferModule = AgentModule.for_root(
    agents=[BankTransferAgent],
    tools=[TransferFundsTool],
    imports=[LLMProvider, BankingModule],
)

_CRMModule = AgentModule.for_root(
    agents=[AuthenticatedCRMAgent],
    tools=[DelegateToBankingTransfer],      # ← delegation tool lives HERE
    imports=[LLMProvider, _TransferModule, BankingModule],  # makes AgentRunner[BankTransferAgent] visible
)
```

### 7. Register observability singletons in `providers=[]`

Any class that subscribes to `SignalBus` events or performs post-construction
wiring in `__init__` must appear in the `providers` list of the enclosing `@module`
so the Lauren lifecycle scheduler instantiates it eagerly at startup:

```python
@module(
    providers=[
        BankDatabase,
        EventForwarder,    # ← subscribes signal handlers in __init__; must be eager
    ],
    controllers=[BankingChatController],
)
class BankingModule: ...
```

Without appearing in `providers`, the singleton is never constructed and the
signal subscriptions never run.

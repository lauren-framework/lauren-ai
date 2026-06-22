# AGENTS.md — AI Agent Usage Guide for `lauren-ai`

This file is the primary reference for AI coding agents using `lauren-ai` to
build agentic applications.  Read this file before writing any agent, tool,
guardrail, team, or memory code.

---

## Quick start

```python
from lauren_ai import LLMConfig, LLMModule, LLMService, agent, tool, use_tools

# 1. Define a tool. Future annotations are supported as long as every
# referenced type resolves when @tool() runs.
@tool()
async def get_weather(city: str) -> dict:
    """Get weather for a city.
    Args:
        city: City name (e.g. 'London').
    """
    return {"city": city, "temp_c": 18, "condition": "cloudy"}

# 2. Define an agent — @agent() outermost, @use_tools() below it
@agent(system="You are a weather assistant.")
@use_tools(get_weather)
class WeatherAgent: ...

# 3. Wire via AgentModule (production) or AgentRunnerBase (testing/scripting)
from lauren_ai import AgentRunnerBase

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-...")
LLMProvider = LLMModule.for_root(cfg)
transport = LLMProvider.transport_instance
runner = AgentRunnerBase(transport=transport)

# 4. Run
result = await runner.run(WeatherAgent(), "What's the weather in Paris?")
print(result.content)
```

---

## All decorator APIs

### `@tool()`

Marks an async function as a tool.  **Must use parentheses.**

```python
# CORRECT — with parentheses
@tool()
async def my_tool(param: str) -> dict:
    """Short description.
    Args:
        param: Description of param.
    """
    return {"result": param}

# WRONG — no parentheses
@tool
async def bad_tool(...): ...
```

**Rules:**
- `from __future__ import annotations` works with `@tool()`, but every annotated
  type must be importable when schema generation runs. Avoid unresolved forward
  references or circular imports in function-form tool files.
- Use Google-style docstrings (`Args:` section) for parameter descriptions.
- The `ctx: ToolContext | None = None` parameter (if present) is injected
  internally and never included in the JSON schema.
- Return type should be `dict`, `list`, `str`, `int`, or `float`.
- Async is required (use `async def`).

**`@tool()` optional kwargs for state and dependencies:**

```python
@tool(
    label="Read File",                              # display name (TUI, logs)
    initial_state=lambda: {"start_ts": None},       # seeded into ctx.state each call
    initial_tool_state=lambda: {"seen": set()},     # seeded into ctx.tool_state once per run
    dependency_factory=lambda: {"client": httpx.AsyncClient()},  # ctx.dependencies, once per run
)
async def read_file(path: str, ctx: ToolContext) -> dict:
    ...
```

**`ToolContext` fields available to tools that declare `ctx: ToolContext`:**

| Field | Scope | Mutable? | Purpose |
|---|---|---|---|
| `ctx.state` | Per-call | Yes | Scratch pad; seeded by `initial_state()` each call |
| `ctx.tool_state` | Per-run, per-tool | Yes | Memory across multiple calls; seeded by `initial_tool_state()` at run start |
| `ctx.dependencies` | Per-run, per-tool | Convention: no | Singleton deps from `dependency_factory()`; same object all calls |
| `ctx.extras` | Per-call | Convention: no | Runner-injected per-call context |
| `ctx.metadata` | Per-call | No | Static `@set_metadata()` key-value pairs |
| `ctx.agent_context` | Per-call | No | Full `AgentContext`; use `.request`, `.execution_context` here |

> **Removed:** `ctx.request` and `ctx.execution_context` no longer exist on `ToolContext`.
> Use `ctx.agent_context.request` and `ctx.agent_context.execution_context` instead.

**Class-form tools** (for stateful / DI-injectable tools):

```python
@tool()
class DatabaseTool:
    """Query a database.
    Args:
        query: SQL query string.
    """
    def __init__(self, connection_string: str) -> None:
        self._conn = connection_string

    async def run(self, query: str) -> dict:
        # execute query — @tool() looks for run(), not __call__()
        return {"rows": []}
```

---

### `@agent()`

Marks a class as an AI agent.  **Must use parentheses.**

```python
@agent(
    model="claude-opus-4-6",        # LLM model identifier
    system="You are helpful.",       # System prompt (falls back to class docstring)
    max_turns=10,                    # Agentic loop limit
    temperature=0.7,                 # Sampling temperature
    max_cost_usd=0.50,              # Hard cost budget in USD (None = unlimited)
    memory=ShortTermMemory(max_tokens=60_000),      # Optional — reused across run() calls
    conversation_store=InMemoryConversationStore(), # Optional — auto-created if None
    config=AgentConfig(parallel_tool_calls=True),  # Full AgentConfig override
)
class MyAgent: ...
```

**Per-agent state parameters:**

| Parameter | Default | Purpose |
|---|---|---|
| `memory` | `None` | `ShortTermMemory` instance reused across **every** `run()` call (agentic persistent memory). When `None`, a fresh one is built per turn. |
| `conversation_store` | `None` | Per-agent conversation store. When `None`, `AgentModule.for_root()` auto-creates an `InMemoryConversationStore` and writes it back to AgentMeta. Two agents in the same module always get **distinct** stores. |

`AgentModule.for_root()` **does not accept `memory=` or `conversation_store=`** — these are per-agent; place them in `@agent()`.  Both can be overridden per-call via `runner.run(agent, ..., conversation_store=..., memory=...)`.

**Lifecycle hooks** (all optional, all async or sync):

```python
@agent(model="claude-opus-4-6")
class MyAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        """Called once before the first LLM turn."""

    async def on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None:
        """Called after each LLM turn."""

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        """Called after each tool execution.  Return a modified ToolResult or None."""

    async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
        """Called after the loop terminates."""
```

---

### `@use_tools(*tools)`

Attaches tool functions or classes to an agent.  Must be stacked **below** `@agent()`.

```python
@agent(model="claude-opus-4-6")
@use_tools(tool_a, tool_b, ToolClassC)
class MyAgent: ...
```

Multiple `@use_tools` can be stacked; tools are merged:

```python
@agent(model="claude-opus-4-6")
@use_tools(extra_tool)
@use_tools(base_tool)
class MyAgent: ...
# result: [base_tool, extra_tool]
```

---

### `@use_knowledge_sources(*sources: KnowledgeSource)`

Restricts an agent's knowledge-base tool visibility to the listed sources.
**KB access is opt-in** — an agent without this decorator sees **zero** KB
tools even when its module declares `knowledge=[...]`.

```python
from lauren_ai import use_knowledge_sources
from .knowledge_sources import PUBLIC_KB, INTERNAL_KB

@use_knowledge_sources(PUBLIC_KB)     # only this KB is visible to this agent
@agent(name="UnauthCRM", model="claude-opus-4-6")
class UnauthenticatedCRMAgent: ...

@agent(name="AuthCRM", model="claude-opus-4-6")  # no decorator → no KB tools
class AuthenticatedCRMAgent: ...
```

**Rules:**

- Must be declared on the agent class itself — not inherited (strict-inheritance; raises `MetadataInheritanceError` if subclass inherits without redeclaring).
- All referenced sources must be listed in the enclosing `AgentModule.for_root(knowledge=[...])` — if not, raises `DecoratorUsageError` at module-build time.
- Must be stacked **below** `@agent()` (executes first so `@agent()` can read it).
- Calling with no arguments (`@use_knowledge_sources()`) raises `DecoratorUsageError`.

Typical wiring — hoist the `KnowledgeSource` to a shared module to avoid circular imports:

```python
# knowledge_sources.py
from lauren_ai._knowledge import KnowledgeBase, KnowledgeSource, SentenceChunker, TextLoader
from lauren_ai._memory._vector import InMemoryVectorStore

PUBLIC_KB = KnowledgeSource(
    kb=KnowledgeBase(store=InMemoryVectorStore(), chunker=SentenceChunker()),
    tool_name="search_public_info",
    top_k=3,
    loaders=[TextLoader("docs/public.md")],   # loaded at app startup via @post_construct
)

# agent.py
from .knowledge_sources import PUBLIC_KB

@use_knowledge_sources(PUBLIC_KB)
@agent(name="UnauthCRM", ...)
class UnauthenticatedCRMAgent: ...

# module.py
from .knowledge_sources import PUBLIC_KB

AgentModule = AgentModule.for_root(
    agents=[UnauthenticatedCRMAgent, AuthenticatedCRMAgent],
    knowledge=[PUBLIC_KB],   # module declares it; only UnauthCRM opts in
    imports=[LLMProvider],
)
```

---

### `@use_guardrails(input=[...], output=[...])`

Attaches guardrail instances to an agent.  **Must use parentheses.**  Stack between
`@agent()` and `@use_tools()`.

```python
from lauren_ai import use_guardrails, PromptInjectionFilter, PIIRedactor, LengthFilter

@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[PromptInjectionFilter(), PIIRedactor()],
    output=[LengthFilter(max_chars=8000)],
)
@use_tools(my_tool)
class SafeAgent: ...
```

`None` entries are silently dropped (conditional guardrails):

```python
@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[PromptInjectionFilter(), TopicFilter(allowed_topics=topics) if topics else None],
)
class DynamicAgent: ...
```

**Built-in guardrails:**

| Class | Direction | Purpose |
|-------|-----------|---------|
| `PromptInjectionFilter()` | input | Blocks jailbreak / prompt-override attempts |
| `PIIRedactor(entities=[...])` | input | Redacts emails, phone numbers, SSNs, credit cards |
| `LengthFilter(max_chars=N)` | output | Caps response length |
| `TopicFilter(allowed_topics=[...])` | input/output | Allows only specified topics |
| `LLMGuardrail(llm, prompt, block_if)` | input/output | LLM-powered content evaluation |

**Custom guardrail class** (implement the protocol and inject via DI):

```python
from lauren_ai import guardrail, GuardrailDecision, GuardrailContext

@guardrail(kind="input")   # registers as DI-injectable singleton
class ProfanityFilter:
    async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
        if "badword" in message.lower():
            return GuardrailDecision(
                action="block",
                violation="Profanity detected.",
                guardrail_name="ProfanityFilter",
            )
        return GuardrailDecision(action="pass", guardrail_name="ProfanityFilter")
```

Note: `@guardrail(kind="input"|"output"|"any")` is for making DI-injectable guardrail
classes.  `@use_guardrails(input=[...], output=[...])` attaches instances to an agent.

---

### `@remember(store=..., extract=..., inject=..., top_k=...)`

Opts an agent into automatic user memory.  **Must use parentheses.**  Stack
between `@agent()` and `@guardrail()`.

```python
from lauren_ai import InMemoryUserMemoryStore

store = InMemoryUserMemoryStore()

@agent(model="claude-opus-4-6")
@remember(store=store, extract=True, inject=True, top_k=5)
@use_guardrails(input=[PromptInjectionFilter()])
@use_tools(my_tool)
class PersonalAssistant: ...
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `store` | `None` | DI token name for `UserMemoryStore`; `None` = use `InMemoryUserMemoryStore` |
| `extract` | `True` | Extract new facts from each conversation turn |
| `inject` | `True` | Inject relevant memories into system prompt |
| `top_k` | `5` | Number of memories to inject |
| `extraction_model` | `None` | Model for fact extraction (defaults to agent model) |

---

### `@team(name=..., mode=..., model=..., max_rounds=...)`

Marks a class as a multi-agent team.  **Must use parentheses.**

```python
@team(
    name="research-team",
    mode="coordinator",      # "coordinator" | "collaborate"
    model="claude-opus-4-6",
    max_rounds=4,
    coordinator_prompt=MY_PROMPT,  # Optional override
)
class ResearchTeam:
    def __init__(self, researcher: ResearchAgent, writer: WriterAgent) -> None:
        self.researcher = researcher
        self.writer = writer
```

**Modes:**
- `coordinator` — LLM coordinator routes sub-tasks one at a time until it declares `DONE:`.
- `collaborate` — all workers run sequentially, then the coordinator synthesises a final answer.

**`coordinator_prompt` template variables:** `{worker_descriptions}`, `{task}`, `{prior_outputs}`.

---

### `@traced(name=..., kind=...)`

Wraps a function in an observability span.  **Must use parentheses.**

```python
from lauren_ai import traced, SpanKind

@traced(name="my_operation", kind=SpanKind.AGENT)
async def my_operation(input: str) -> str:
    return "result"
```

---

## `AgentRunner` / `AgentRunnerBase` methods

`AgentRunner` is a `@runtime_checkable Protocol`. Use `AgentRunnerBase` for direct
construction. In production, `AgentModule.for_root()` auto-generates the runner and
wires everything; inject it via `runner: AgentRunner` when one agent module is in
scope, or `runner: AgentRunner[MyAgent]` when you need a specific module's runner.

```python
from lauren_ai import AgentRunnerBase

runner = AgentRunnerBase(
    transport=...,
    signals=...,           # Optional SignalBus
    cache_backend=...,     # Optional tool-result cache
    # No conversation_store — it lives on each @agent() now.
    # Pass it per-call via runner.run(..., conversation_store=...) to override.
)

# Blocking run — returns AgentResponse
response: AgentResponse = await runner.run(
    agent_instance,            # @agent()-decorated instance
    "User message",
    conversation_id="sess-1",  # Optional — loads prior history from agent's store, saves after
    metadata={"user_id": "u1"},
    run_id="run-abc",          # Optional — random hex if omitted
    conversation_store=...,    # Optional per-request override (wins over agent's store)
    memory=...,                # Optional per-request override (wins over agent's memory)
)
response.content        # str: final assistant text
response.turns          # int: number of agentic loop iterations
response.total_usage    # TokenUsage: cumulative token counts
response.tool_calls_made  # list[ToolCall]
response.stop_reason    # "end_turn" | "max_turns" | "budget_exceeded" | "delegated"

# Streaming run — yields CompletionChunk
async for chunk in await runner.run_stream(agent_instance, "message"):
    print(chunk.delta, end="", flush=True)

# HITL approval / rejection
await runner.approve_tool(agent_run_id, tool_use_id)
await runner.reject_tool(agent_run_id, tool_use_id, reason="Not permitted")
```

---

## `TeamRunner` (run, run_stream, TeamResult, TeamEvents)

```python
from lauren_ai import TeamRunner, TeamResult

team_runner = TeamRunner(
    team_cls=ResearchTeam,
    llm=llm_service,        # raw LLMService instance
    agent_runner=runner,    # shared AgentRunner
)

# Blocking
result: TeamResult = await team_runner.run("Research quantum computing")
result.final_answer       # str
result.worker_outputs     # dict[str, str] — per-worker output
result.rounds             # int

# Streaming — yields TeamEvent instances
from lauren_ai import (
    TeamWorkerStarted, TeamWorkerFinished,
    TeamCoordinatorDecision, TeamFinalAnswer,
)

async for event in team_runner.run_stream("Research topic"):
    if isinstance(event, TeamWorkerStarted):
        print(f"→ {event.worker_name} starting round {event.round}")
    elif isinstance(event, TeamWorkerFinished):
        print(f"✓ {event.worker_name} done: {event.result_content[:100]}")
    elif isinstance(event, TeamCoordinatorDecision):
        print(f"Coordinator: {event.decision}")
    elif isinstance(event, TeamFinalAnswer):
        print(f"Final: {event.content}")
```

---

## Delegation pattern

Use a class-form `@tool()` that injects the target agent and its runner via
`AgentRunner[TargetAgent]`. The delegation tool lives in the **calling module's
`tools=`**; the calling module imports the target module so the parameterized
runner token is visible to DI.  No named `AgentRunnerBase` subclass needed.

```python
# delegation.py — future annotations are allowed, but keep tool types importable
from lauren_ai import AgentRunner, tool, ToolContext

@tool()
class DelegateToSpecialist:
    """Delegate a task to the SpecialistAgent.
    Args:
        task: Full task description.
    """
    def __init__(
        self,
        agent: SpecialistAgent,
        runner: AgentRunner[SpecialistAgent],   # ← parameterized token — no boilerplate subclass
    ) -> None:
        self._agent = agent
        self._runner = runner

    async def run(self, ctx: ToolContext, task: str) -> dict:
        response = await self._runner.run(self._agent, task,
                                          execution_context=ctx.execution_context)
        return {"result": response.content}

# Target module — AgentRunner[SpecialistAgent] is auto-registered
SpecialistMod = AgentModule.for_root(
    agents=[SpecialistAgent],
    tools=[SpecialistTool],
    imports=[LLMProvider],
)

# Calling module — owns the delegation tool; imports target module
OrchestratorMod = AgentModule.for_root(
    agents=[OrchestratorAgent],
    tools=[DelegateToSpecialist],          # ← delegation tool lives HERE
    imports=[LLMProvider, SpecialistMod],  # ← makes AgentRunner[SpecialistAgent] visible
)
```

---

## Memory cookbook

### Conversation history across requests

Declare `conversation_store` on the `@agent()` decorator.  Then supply a
`conversation_id` on each `run()` call — the runner loads prior messages before
the new turn and saves the updated history afterward:

```python
from lauren_ai import agent, InMemoryConversationStore

# Per-agent store — declared on the class, isolated from all other agents
@agent(model="claude-opus-4-6", conversation_store=InMemoryConversationStore())
class MyAgent: ...

# AgentModule.for_root() auto-creates InMemoryConversationStore for agents that omit it
AgentModule.for_root(agents=[MyAgent], imports=LLMProvider)

# In a controller — each request carries the same session ID:
resp1 = await runner.run(agent, "My name is Alice.", conversation_id="sess-1")
resp2 = await runner.run(agent, "What is my name?",  conversation_id="sess-1")
# resp2 sees the full prior exchange — the agent replies "Alice"
```

Override the store for a single call without mutating the agent class:

```python
await runner.run(agent, msg, conversation_id="s1", conversation_store=other_store)
```

Without a store the runner creates a fresh `ShortTermMemory` per call; the
`conversation_id` is accepted but ignored.  Different IDs are completely isolated.

### Long-term user facts

```python
from lauren_ai import InMemoryUserMemoryStore, MemoryFact

store = InMemoryUserMemoryStore()

# Store a fact
await store.add(user_id="u1", fact=MemoryFact(content="User prefers dark mode", topics=["preferences"]))

# Retrieve relevant facts
facts = await store.search(user_id="u1", query="user interface preferences", top_k=3)
for f in facts:
    print(f.content, f.confidence)

# Build memory context string for injection
from lauren_ai._memory._remember import build_memory_context
context = build_memory_context(facts)
```

---

## Guardrails cookbook

```python
from lauren_ai import use_guardrails, LengthFilter, PIIRedactor, PromptInjectionFilter, TopicFilter

# Compose multiple guardrails on an agent
@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[
        PromptInjectionFilter(),
        PIIRedactor(),
        TopicFilter(allowed_topics=["customer support", "billing", "account"]),
    ],
    output=[
        LengthFilter(max_chars=4000),
    ],
)
class SupportAgent: ...
```

LLM-powered guardrail (slower but flexible):

```python
from lauren_ai import LLMGuardrail

@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[LLMGuardrail(
        llm=llm_service,
        prompt="Does this text contain medical advice? Reply YES or NO.\n\n{content}",
        block_if="YES",
    )],
)
class SafeAgent: ...
```

---

## Signal bus cookbook

```python
from lauren_ai import SignalBus, ModelCallComplete, AgentRunComplete

bus = SignalBus()

@bus.on(ModelCallComplete)
async def track_cost(event: ModelCallComplete) -> None:
    print(f"Model: {event.model}, cost: ${event.cost_usd:.6f}")

@bus.on(AgentRunComplete)
async def log_run(event: AgentRunComplete) -> None:
    print(f"Agent {event.agent_class.__name__} finished in {event.turns} turns, "
          f"total cost: ${event.total_cost_usd:.6f}")
```

---

## Cost / rate tracking cookbook

```python
from lauren_ai import CostTracker, TokenBudget, RateLimiter, default_pricing_table

# Cost tracking (injectable singleton)
tracker = CostTracker(pricing=default_pricing_table())

async with tracker.session() as session:
    result = await agent_runner.run(agent_instance, "Hello!")
    # session.estimate available during the run

report = await tracker.report()
print(f"Total cost: ${report.total_usd:.4f}")

# Token budget — prevent runaway costs
budget = TokenBudget(max_tokens_per_conversation=50_000)
# Raises BudgetExceededError when exceeded (checked by AgentRunner per turn)

# Rate limiting
limiter = RateLimiter(requests_per_minute=60)
await limiter.acquire()  # Raises RateLimitExhaustedError when ceiling breached
```

---

## Structured output

```python
from lauren_ai import StructuredLLM, Message
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float

# Obtain via LLMService (injected by DI)
structured: StructuredLLM[Sentiment] = llm_service.with_structured_output(Sentiment)
result: Sentiment = await structured.complete(
    [Message.user("I love this product!")]
)
print(result.label, result.score)
```

---

## Multimodal inputs

```python
from lauren_ai import ImageContent, Message

msg = Message.from_multimodal("user", [
    ImageContent(url="https://example.com/chart.png"),
    "Describe this chart.",
])
response = await llm_service.complete([msg])
```

---

## Semantic router

```python
from lauren_ai import SemanticRouter, Route

async def embed_fn(texts: list[str]) -> list[list[float]]:
    embeddings = await embed_service.embed(texts)
    return [e.vector for e in embeddings]

router = SemanticRouter(
    routes=[
        Route(name="weather", examples=["What's the weather?", "Is it raining?"]),
        Route(name="code", examples=["Write a function", "Debug this Python"]),
    ],
    embed_fn=embed_fn,
    min_confidence=0.7,
)
await router.compile()   # must be awaited before route()

match = await router.route("What temperature is it outside?")
print(match.route)       # "weather"
print(match.confidence)  # float
print(match.matched)     # bool — False if below min_confidence
```

---

## Agent DI integration

`@agent()` automatically applies `@injectable(scope=Scope.SINGLETON)`.
`AgentModule.for_root()` registers each agent class as a DI provider **and**
exports it, so controllers in the parent module can inject it directly.

**Single module — bare `AgentRunner` Protocol:**

```python
from lauren_ai import AgentRunner  # @runtime_checkable Protocol

class ChatController:
    def __init__(self, runner: AgentRunner, agent: ChatAgent) -> None:
        self._runner = runner
        self._agent = agent   # DI-resolved singleton

    async def chat(self, message: str) -> str:
        response = await self._runner.run(self._agent, message)
        return response.content
```

**Multiple modules — `AgentRunner[X]` parameterized form:**

When a controller or tool can see runners from two or more `AgentModule`s, the
bare `runner: AgentRunner` Protocol scan would find multiple matches and raise
`ProtocolAmbiguityError`.  Instead, use the parameterized form:

```python
from lauren_ai import AgentRunner

class BankingChatController:
    def __init__(
        self,
        unauth_runner:   AgentRunner[UnauthenticatedCRMAgent],
        auth_runner:     AgentRunner[AuthenticatedCRMAgent],
        transfer_runner: AgentRunner[BankTransferAgent],
    ) -> None:
        ...
```

`AgentModule.for_root()` automatically registers `AgentRunner[agent_cls]` for
every agent in `agents=`.  **No named runner subclass boilerplate needed.**

Always pass the **instance** (from DI) — never the class — to `runner.run()`.
Passing the class breaks lifecycle hooks because `on_start` / `on_turn_complete`
/ `on_tool_result` / `on_finish` are unbound at that point.

---

## Production patterns from the banking chatbot

The following patterns are drawn from
`lauren-examples/lauren-ai-chatbot/backend/` and represent battle-tested
conventions for production agent deployments.

### Identity from ExecutionContext, not the LLM

`SignatureGuard` (or any `lauren-guards` auth guard) pins a verified identity
to `request.state` before the handler runs.  The controller wraps the live
`Request` in an `ExecutionContext` and passes it to `AgentRunner.run()`.
`ToolExecutor` forwards that context object to every tool call as
`ToolContext.execution_context`.  The tool reads the identity from there — the
LLM never sees it and cannot override it.

```python
# The guard sets the identity (runs before the controller)
class SignatureGuard:
    async def can_activate(self, ctx: ExecutionContext) -> bool:
        payload = await verify_hmac(ctx.request)        # cryptographic check
        ctx.request.state.user_id = payload["user_id"]  # pinned — immutable from here
        return True

# The controller wraps and passes the context
@post("/chat")
async def stream(self, body: Json[ChatRequest], exec_ctx: ExecutionContext) -> EventStream:
    user_id = exec_ctx.request.state.get("user_id")  # trust state, not the body field
    response = await self._runner.run(
        self._agent, message,
        execution_context=exec_ctx,   # ← security anchor
    )

# The tool reads the verified identity — never adds user_id to its JSON schema
async def run(self, ctx: ToolContext, to_user: str, amount: float) -> dict:
    auth_uid = ctx.execution_context.request.state.get("user_id")
    if not auth_uid:
        return {"error": "Security error: unauthenticated."}
    result = self._db.transfer(from_user=auth_uid, to_user=to_user, amount=amount)
    ...
```

If you find yourself adding `authenticated_user: str` or `from_user: str` to a
tool's parameter list, that is a security vulnerability.  Read from
`ctx.execution_context.request.state` instead.

### Streaming agent output as SSE

Set the ContextVar **before** awaiting `runner.run()` inside the `generate()`
async generator.  Do not call `.reset()` — each keep-alive task gets its own
context copy and discards it automatically on exit.

```python
from app.ws.context import current_user_id

@post("/chat")
async def stream(self, body: Json[ChatRequest], exec_ctx: ExecutionContext) -> EventStream:
    account = ...  # resolved from exec_ctx.request.state

    async def generate():
        # Pin ContextVar so signal handlers can route events to the right WebSocket
        current_user_id.set(account.user_id)
        try:
            response = await self._runner.run(
                self._crm_agent, full_prompt,
                conversation_id=body.conversation_id,
                execution_context=exec_ctx,
            )
            content = response.content or ""
            for i in range(0, len(content), 40):
                yield ServerSentEvent(event="token", data=content[i : i + 40])
            yield ServerSentEvent(event="done", data="")
        except Exception as exc:
            yield ServerSentEvent(event="error", data=str(exc))

    return EventStream(generate(), keep_alive=15.0)
```

### SignalBus + WebSocket for real-time agent events

`EventForwarder` is a singleton injectable that subscribes to `SignalBus`
events in its `__init__` and uses the `current_user_id` ContextVar to route
events to the right user's WebSocket.  `asyncio.gather` (used by
`SignalBus.emit`) copies ContextVar state into every spawned task, so the
handler can call `.get()` to find the target user.

```python
@injectable(scope=Scope.SINGLETON)
class EventForwarder:
    def __init__(self, db: BankDatabase) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()
        # Subscribe once at construction — must be listed in providers=[] so the
        # Lauren lifecycle scheduler instantiates it eagerly before requests arrive
        signal_bus.on(ModelCallComplete)(self._on_model_complete)
        signal_bus.on(ToolCallStarted)(self._on_tool_started)
        db.add_transfer_listener(self._on_transfer)

    async def _on_model_complete(self, event: ModelCallComplete) -> None:
        user_id = current_user_id.get()   # ContextVar copied by asyncio.gather
        if user_id:
            await self.send_to_user(user_id, {"type": "token_usage", ...})
```

### Multi-runner disambiguation with `AgentRunner[X]`

Every `AgentModule.for_root()` auto-registers `AgentRunner[agent_cls]` aliases
for every agent in `agents=`.  Controllers that see runners from multiple
modules inject by agent class — **no named subclass boilerplate needed**:

```python
from lauren_ai import AgentRunner

class BankingChatController:
    def __init__(
        self,
        unauth_runner:   AgentRunner[UnauthenticatedCRMAgent],
        auth_runner:     AgentRunner[AuthenticatedCRMAgent],
        transfer_runner: AgentRunner[BankTransferAgent],
        disputes_runner: AgentRunner[DisputesAgent],
    ) -> None:
        self._registry = {
            UNAUTH:   (unauth_agent,   unauth_runner),
            AUTH:     (auth_agent,     auth_runner),
            TRANSFER: (transfer_agent, transfer_runner),
            DISPUTES: (disputes_agent, disputes_runner),
        }
```

Delegation tools use the same pattern:

```python
@tool()
class DelegateToBankingTransfer:
    def __init__(
        self,
        transfer_agent: BankTransferAgent,
        runner: AgentRunner[BankTransferAgent],   # ← parameterized DI token
    ) -> None:
        self._agent = transfer_agent
        self._runner = runner

    async def run(self, ctx: ToolContext, task: str) -> dict:
        response = await self._runner.run(
            self._agent, task,
            execution_context=ctx.execution_context,
        )
        return {"result": response.content, "stop_reason": response.stop_reason}
```

---

## By-Task Quick Lookup

| I need to… | Read first | Copy-paste guide |
|---|---|---|
| Define a new agent | `src/lauren_ai/_agents/_agent.py` | `skills/building-agents/` |
| Write a tool (function or class form) | `src/lauren_ai/_tools/__init__.py` | `skills/building-tools/` |
| Build a multi-agent team | `src/lauren_ai/_teams/` | `skills/building-teams/` |
| Add / query conversation or user memory | `src/lauren_ai/_memory/` | `skills/managing-memory/` |
| Add input/output guardrails | `src/lauren_ai/_guardrails/` | `skills/adding-guardrails/` |
| Secure agent identity from tools | `src/lauren_ai/_agents/_runner.py` | `skills/securing-agents/` |
| Test an agent without live API calls | `tests/unit/` | `skills/testing-agents/` |
| Wire agents into a Lauren web app | `src/lauren_ai/_module.py` | `skills/integrating-with-lauren/` |
| Inspect raw LLM stream / debug chunks | `src/lauren_ai/_transport/` | `skills/inspecting-streams/` |
| Debug a startup or schema error | `src/lauren_ai/_tools/__init__.py` | **Common Errors** section below |
| Migrate from LangChain / OpenAI SDK | `llms-full.txt` | `skills/migrating-to-lauren-ai/` |
| Copy a complete working agent | `tests/integration/` | `skills/common-agent-patterns/` |

## Common Errors

| Error / Symptom | Cause | Fix |
|---|---|---|
| Tool schema is `{}` or missing parameters | A tool annotation could not be resolved at schema-build time | Import the referenced type in the tool module, or avoid unresolved forward refs / circular imports |
| `AgentRunner[X]` resolves to wrong runner | Two `AgentModule.for_root()` calls register the same agent | Use `shared_tools=` to deduplicate, or merge into one module |
| `ModuleExportViolation` on `AgentRunner[X]` | Runner injected across module boundary without export | Add the type to `exports=` in the owning `AgentModule` |
| `ProtocolAmbiguityError` on `AgentRunner` | Bare `AgentRunner` annotation where two runners are visible | Use `AgentRunner[AgentX]` (parameterized form) — see CLAUDE.md §4 |
| Guardrail fires but SSE content not replaced | `guardrail_override` chunk not yielded in the streaming controller | Yield `ServerSentEvent(event="guardrail_override", data=chunk.guardrail_override)` |
| Memory not persisted across turns | Different `conversation_id` passed on each call | Pass the same `conversation_id` on every `runner.run()` / `runner.run_stream()` call |
| Signal handler fires N times (duplicate events) | `EventForwarder` re-created on hot-reload without clearing old handlers | Call `signal_bus.clear(EventType)` before re-registering in `__init__` |
| `@use_guardrails` has no effect | Decorator applied **below** `@agent()` instead of above | Apply `@use_guardrails` **above** `@agent()` — bottom-up decorator application |
| `TypeError: on_start() missing positional arg` | Agent class (not instance) passed to `runner.run()` | Pass a DI-resolved **instance**: `runner.run(my_agent_instance, ...)` |
| `DecoratorUsageError` on `@use_knowledge_sources` | Knowledge source not listed in `AgentModule.for_root(knowledge=[...])` | Add the source to the module's `knowledge=` argument |

## Skills Quick Index

| Task | Skill directory |
|---|---|
| Define agents, lifecycle hooks, streaming | `skills/building-agents/` |
| Write `@tool` functions and classes, ToolContext DI | `skills/building-tools/` |
| Coordinator and collaborate multi-agent teams | `skills/building-teams/` |
| Conversation, user-fact, and vector-store memory | `skills/managing-memory/` |
| Input/output guardrails, PII, scope enforcement | `skills/adding-guardrails/` |
| Identity trust chain, ToolContext security | `skills/securing-agents/` |
| MockTransport, AgentTestClient, multi-turn tests | `skills/testing-agents/` |
| LLMModule, AgentModule, SSE, ExecutionContext | `skills/integrating-with-lauren/` |
| Raw stream debug, chunk/event inspection | `skills/inspecting-streams/` |
| LangChain / OpenAI SDK → lauren-ai equivalents | `skills/migrating-to-lauren-ai/` |
| Copy-paste complete agents | `skills/common-agent-patterns/` |

Full index: [`skills/README.md`](skills/README.md)

## Docs Map

| Concept | Guide |
|---|---|
| Agent definition and lifecycle hooks | `docs/guides/agents.md` |
| Tool building (function, class, built-ins) | `docs/guides/tools.md` |
| Multi-agent teams (coordinator, collaborate) | `docs/guides/agent-teams.md` |
| Memory (conversation, user facts, vector store) | `docs/guides/memory.md` |
| Guardrails (input/output, LLM-powered) | `docs/guides/guardrails.md` |
| Knowledge base / RAG | `docs/guides/knowledge-base.md` |
| Cost tracking and token budgets | `docs/guides/cost-tracking.md` |
| SSE streaming with Lauren framework | `docs/guides/lauren-integration.md` |
| Testing with MockTransport | `docs/guides/testing.md` |
| Output parsers (JSON, Pydantic, Regex, …) | `docs/guides/output-parsers.md` |
| Tracing / observability | `docs/guides/tracing.md` |
| Evaluation framework (AccuracyEval, AgentJudge) | `docs/guides/evaluation.md` |

## Anti-patterns to avoid

- **Do not** rely on unresolved forward references in function-form `@tool()` files. `from __future__ import annotations` is supported, but `@tool()` still needs every referenced type to resolve when schema generation runs.
- **Do not** swap decorator order — `@agent` must be outermost (topmost in code), `@use_tools` innermost.
- **Do not** use bare `@agent`, `@tool`, `@use_guardrails`, `@guardrail`, `@remember`, `@team` (always use parentheses).
- **Do not** use `@guardrail(input=[...], output=[...])` on agents — that form is for DI-injectable guardrail classes; use `@use_guardrails(input=[...], output=[...])` on agents instead.
- **Do not** define `__call__` in class-form tools — `@tool()` looks for `run()`.
- **Do not** register the same tool in `ToolRegistry` twice — it will override silently.
- **Do not** pass `conversation_id=None` when you want session persistence — always supply one, and ensure a `conversation_store` is set on the agent via `@agent(conversation_store=...)`.
- **Do not** share `ShortTermMemory` instances between agents unless you want history to bleed across agent boundaries — each agent should have its own.
- **Do not** pass agent **classes** to `runner.run()` — always pass a DI-resolved **instance**. Passing a class bypasses DI and breaks lifecycle hooks (raises `TypeError: on_start() missing 1 required positional argument: 'ctx'`).
- **Do not** use `runner: AgentRunner` (bare Protocol annotation) in a controller or tool that can see runners from two or more `AgentModule`s — the structural Protocol scan finds multiple matches and raises `ProtocolAmbiguityError`. Use `AgentRunner[AgentX]` (parameterized form) instead.
- **Do not** pass `memory=` or `conversation_store=` to `AgentModule.for_root()` — these are per-agent and must be declared on `@agent(memory=..., conversation_store=...)`.
- **Do not** use `@use_knowledge_sources(KS)` without listing `KS` in the module's `for_root(knowledge=[...])` — raises `DecoratorUsageError` at module-build time.

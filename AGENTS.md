# AGENTS.md — AI Agent Usage Guide for `lauren-ai`

This file is the primary reference for AI coding agents using `lauren-ai` to
build agentic applications.  Read this file before writing any agent, tool,
guardrail, team, or memory code.

---

## Quick start

```python
from lauren_ai import LLMConfig, LLMModule, LLMService, agent, tool, use_tools

# 1. Define a tool (NO from __future__ import annotations in function-form tool files!)
@tool()
async def get_weather(city: str) -> dict:
    """Get weather for a city.
    Args:
        city: City name (e.g. 'London').
    """
    return {"city": city, "temp_c": 18, "condition": "cloudy"}

# 2. Define an agent — @agent() outermost, @use_tools() below it
@agent(model="claude-opus-4-6", system="You are a weather assistant.")
@use_tools(get_weather)
class WeatherAgent: ...

# 3. Wire via LLMModule + AgentRunner
from lauren_ai._agents._runner import AgentRunner
from lauren_ai._tools._registry import ToolRegistry

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-...")
LLMProvider = LLMModule.for_root(cfg)
transport = LLMProvider.transport_instance
registry = ToolRegistry()
registry.register(get_weather)
runner = AgentRunner(transport=transport, registry=registry, config=cfg)

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
- No `from __future__ import annotations` in the file containing `@tool()`.
- Use Google-style docstrings (`Args:` section) for parameter descriptions.
- The `ctx: ToolContext | None = None` parameter (if present) is injected
  internally and never included in the JSON schema.
- Return type should be `dict`, `list`, `str`, `int`, or `float`.
- Async is required (use `async def`).

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
    config=AgentConfig(parallel_tool_calls=True),  # Full AgentConfig override
)
class MyAgent: ...
```

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

## `AgentRunner` methods

```python
runner = AgentRunner(
    transport=...,
    registry=...,
    config=...,
    signals=...,           # Optional SignalBus
    cache_backend=...,     # Optional tool-result cache
    conversation_store=...,# Optional ConversationStore — enables history persistence
)

# Blocking run — returns AgentResponse
response: AgentResponse = await runner.run(
    agent_instance,            # @agent()-decorated instance
    "User message",
    conversation_id="sess-1",  # Optional — loads prior history, saves after run
    metadata={"user_id": "u1"},
    run_id="run-abc",          # Optional — random hex if omitted
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

The recommended way to give one agent access to another is via module-level
delegation tools that close over shared singletons:

```python
# delegation_tools.py — NO from __future__ import annotations
from lauren_ai import tool

_runner = None
_specialist = None

def init(runner, specialist) -> None:
    global _runner, _specialist
    _runner = runner
    _specialist = specialist

@tool()
async def delegate_to_specialist(task: str) -> dict:
    """Delegate a task to the specialist agent.
    Args:
        task: Description of the task.
    """
    response = await _runner.run(_specialist, task)
    return {"content": response.content, "turns": response.turns}
```

Call `init()` once at application startup (after all agents are instantiated),
then attach `delegate_to_specialist` to the orchestrator agent via `@use_tools`.

**Anti-pattern — do not** pass an `AgentRunner` as a constructor argument to
an agent; agents are plain classes and DI injects through `__init__` only for
controllers and services.

---

## Memory cookbook

### Conversation history across requests

Pass `conversation_store` to `AgentModule.for_root()` (preferred) or directly
to `AgentRunner`.  Then supply a `conversation_id` on each `run()` call — the
runner loads prior messages before the new turn and saves the updated history
afterward:

```python
from lauren_ai import InMemoryConversationStore, AgentModule

store = InMemoryConversationStore()

AIModule = AgentModule.for_root(
    agents=[MyAgent],
    conversation_store=store,   # wired to AgentRunner automatically
    imports=LLMProvider,
)

# In a controller — each request carries the same session ID:
resp1 = await runner.run(agent, "My name is Alice.", conversation_id="sess-1")
resp2 = await runner.run(agent, "What is my name?",  conversation_id="sess-1")
# resp2 sees the full prior exchange — the agent replies "Alice"
```

Without `conversation_store` the runner creates a fresh `ShortTermMemory` per
call; the `conversation_id` is accepted but ignored.  Different IDs are
completely isolated.

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
exports it, so controllers in the parent module can inject it directly:

```python
from lauren_ai import AgentRunner

class ChatController:
    def __init__(self, runner: AgentRunner, agent: ChatAgent) -> None:
        self._runner = runner
        self._agent = agent   # DI-resolved singleton

    async def chat(self, message: str) -> str:
        response = await self._runner.run(self._agent, message)
        return response.content
```

Always pass the **instance** (from DI) — never the class — to `runner.run()`.
Passing the class breaks lifecycle hooks because `on_start` / `on_turn_complete`
/ `on_tool_result` / `on_finish` are unbound at that point.

---

## Anti-patterns to avoid

- **Do not** use `from __future__ import annotations` in **function-form** `@tool()` files — it breaks schema generation (see note in `@tool()` section; class-form tools may use it for DI cycle-breaking).
- **Do not** swap decorator order — `@agent` must be outermost (topmost in code), `@use_tools` innermost.
- **Do not** use bare `@agent`, `@tool`, `@use_guardrails`, `@guardrail`, `@remember`, `@team` (always use parentheses).
- **Do not** use `@guardrail(input=[...], output=[...])` on agents — that form is for DI-injectable guardrail classes; use `@use_guardrails(input=[...], output=[...])` on agents instead.
- **Do not** define `__call__` in class-form tools — `@tool()` looks for `run()`.
- **Do not** register the same tool in `ToolRegistry` twice — it will override silently.
- **Do not** pass `conversation_id=None` when you want session persistence — always supply one, and ensure `conversation_store` was passed to `AgentModule.for_root()` or `AgentRunner.__init__`.
- **Do not** share `ShortTermMemory` across runs — it is created fresh per `AgentRunner.run()` call (prior history is loaded from `ConversationStore` when available).
- **Do not** pass agent **classes** to `runner.run()` — always pass a DI-resolved **instance**. Passing a class bypasses DI and breaks lifecycle hooks (raises `TypeError: on_start() missing 1 required positional argument: 'ctx'`).

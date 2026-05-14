# lauren-ai IDEAS

50 feature ideas drawing inspiration from [Pydantic AI](https://pydantic.dev/docs/ai/overview/),
the existing `lauren-ai` architecture, and first-principles thinking about production LLM systems.

Each idea notes the Pydantic AI concept (if any) that inspired it and how it would
fit the decorator-first, DI-driven, module-scoped model that `lauren-ai` already uses.

Recent shipped maintenance work is now documented elsewhere:
- `AgentRunnerBase` no longer takes `config=...`
- `AgentModule.for_root()` now supplies the default model fallback from `LLMConfig`
- contributor workflows live under `docs/development/`

This file remains a forward-looking roadmap for ideas that are not already part of
the current implementation.

---

## 1. Typed agent dependencies — `@agent(deps_type=T)`

**Inspiration:** Pydantic AI's `RunContext[DepsType]` generic

Agents are currently generic over nothing; every run carries untyped `metadata` dict.
Add a `deps_type` parameter so the agent and its tools share a fully type-checked
dependency object resolved from DI:

```python
@dataclass
class BankingDeps:
    user_id: str
    db: BankDatabase
    audit_logger: AuditLogger

@agent(model="claude-opus-4-6", deps_type=BankingDeps)
@use_tools(get_balance)
class BankingAgent: ...

# Runner resolves deps from DI and passes to every tool/hook via AgentContext.deps
response = await runner.run(agent, "What is my balance?", deps=BankingDeps(...))
```

Tools declare the dep type as the first `ToolContext` field:

```python
@tool()
async def get_balance(ctx: ToolContext[BankingDeps]) -> float:
    return await ctx.deps.db.balance(ctx.deps.user_id)
```

---

## 2. Structured output with automatic reflection retry

**Inspiration:** Pydantic AI's schema validation + reflection loop

When `result_type` is a Pydantic model and the LLM returns malformed JSON, feed the
validation error back as a user message and retry automatically (up to `max_retries`):

```python
class TravelPlan(BaseModel):
    destination: str
    budget_usd: float
    activities: list[str]

@agent(model="claude-opus-4-6", result_type=TravelPlan, result_retries=3)
class TravelAgent: ...
```

`AgentRunner.run()` returns `AgentResponse[TravelPlan]` with `response.output: TravelPlan`.
On each retry the schema error is appended to the message history before the next turn.

---

## 3. Result validators — `@agent.result_validator`

**Inspiration:** Pydantic AI's `@agent.result_validator`

Attach post-processing validation to an agent that runs after each complete turn.
Validators can accept/reject/modify the result, or raise `ModelRetry` to trigger another turn:

```python
@agent(model="claude-opus-4-6", result_type=TransferResult)
@use_tools(transfer_funds)
class TransferAgent: ...

@TransferAgent.result_validator
async def check_fraud(ctx: AgentContext, result: TransferResult) -> TransferResult:
    if await fraud_service.is_suspicious(result):
        raise ModelRetry("Suspicious transaction detected — please clarify intent.")
    return result
```

---

## 4. `ModelRetry` exception from tools

**Inspiration:** Pydantic AI's explicit `ModelRetry` raise

Tools can raise `ModelRetry(message)` to send a custom correction message back to the
LLM without counting it as a tool error.  Useful for soft validation:

```python
@tool()
async def create_transfer(ctx: ToolContext, amount: float, to_user: str) -> str:
    if amount > 10_000:
        raise ModelRetry("Transfers over $10,000 require explicit user confirmation. Ask the user to confirm.")
    return await process_transfer(amount, to_user)
```

---

## 5. Usage limits per run — `UsageLimits`

**Inspiration:** Pydantic AI's `UsageLimits`

Cap token consumption, number of LLM requests, and tool calls at the run level,
independently of the global `max_cost_usd` budget:

```python
from lauren_ai import UsageLimits

response = await runner.run(
    agent,
    "Research this topic deeply.",
    usage_limits=UsageLimits(
        max_tokens=8_000,
        max_requests=5,
        max_tool_calls=10,
    ),
)
```

Raises `UsageLimitExceededError` (distinct from `AgentBudgetExceededError`) when
any limit is crossed.

---

## 6. Agent-as-tool

**Inspiration:** Pydantic AI's agent-as-tool pattern

Any `@agent`-decorated class can be registered as a `@tool()` for another agent.
`AgentModule` auto-wraps it with a `DelegateToAgent`-backed tool schema:

```python
ResearchTool = research_agent.as_tool(
    name="research",
    description="Research a topic in depth and return a summary.",
)

@agent(model="claude-opus-4-6")
@use_tools(ResearchTool, write_report)
class ReportAgent: ...
```

This is different from `@team()` — there is no coordinator LLM; the parent agent
decides when and whether to invoke the sub-agent.

---

## 7. Agent specs — declarative YAML/TOML configuration

**Inspiration:** Pydantic AI's `Agent.from_file()`

Load agent definitions from config files without writing Python.  Useful for
non-engineer stakeholders and for A/B testing prompt variants:

```yaml
# agents/crm_agent.yaml
name: CRMAgent
model: claude-opus-4-6
system: |
  You are a CRM assistant for SecureBank.
  Always address the customer by name.
max_turns: 6
tools:
  - lookup_customer
  - update_ticket
guardrails:
  input:
    - PromptInjectionFilter
  output:
    - PIIRedactor
```

```python
CRMAgent = Agent.from_yaml("agents/crm_agent.yaml", tool_registry=registry)
```

---

## 8. Capabilities bundles — `@capability`

**Inspiration:** Pydantic AI's Capabilities (reusable tool+hook+instruction bundles)

A `@capability` groups a set of tools, guardrails, and lifecycle hooks into a single
reusable unit that can be composed onto any agent:

```python
from lauren_ai import capability

@capability(name="web-research")
class WebResearchCapability:
    tools = [WebSearchTool, HttpFetchTool]
    input_guardrails = [PromptInjectionFilter()]
    system_addendum = "You can search the web using the provided tools."

@agent(model="claude-opus-4-6")
@use_capabilities(WebResearchCapability, CodeExecutionCapability)
class DataAnalystAgent: ...
```

`AgentModule` merges capability tools/guardrails into the agent's resolved metadata
at startup.

---

## 9. Step-by-step run iteration — `runner.iter()`

**Inspiration:** Pydantic AI's `agent.iter()` / `AgentRun`

Expose each turn of the agentic loop as an async iterator so callers can observe
(or intervene) between steps:

```python
async for step in runner.iter(agent, "Analyze this dataset"):
    if isinstance(step, ToolCallStep):
        print(f"Calling tool: {step.tool_name}({step.args})")
    elif isinstance(step, AssistantTurnStep):
        print(f"Turn {step.turn}: {step.content[:80]}...")
    elif isinstance(step, RunCompleteStep):
        print(f"Done: {step.response.content}")
```

`runner.run()` and `runner.run_stream()` become thin wrappers over `runner.iter()`.

---

## 10. `PydanticGraph`-style typed workflow engine

**Inspiration:** Pydantic AI's `pydantic-graph` with type-hint-based node definitions

Replace the current untyped `Workflow(Step(...))` with a fully type-safe graph where
nodes are dataclasses and edges are return type annotations:

```python
from lauren_ai.graph import BaseNode, End, Graph, GraphState

@dataclass
class ExtractIntent(BaseNode[str]):
    message: str
    async def run(self, ctx: GraphState) -> "RouteToAgent | End[str]":
        intent = await llm.classify(self.message)
        if intent == "transfer":
            return RouteToAgent(agent="transfer")
        return End(f"Handled: {intent}")

graph = Graph(nodes=[ExtractIntent, RouteToAgent], state_type=ConversationState)
result = await graph.run(ExtractIntent(message="Send $100 to Alice"))
```

Snapshots enable persistence across process restarts; branching/parallel nodes
supported via return type unions.

---

## 11. Durable agent execution — checkpoint/resume

**Inspiration:** Pydantic AI's durable execution (Temporal/DBOS integration)

Long-running agents (research, code generation) can checkpoint their `ShortTermMemory`
and tool call history to a durable store.  On restart, execution resumes from the
last completed tool call:

```python
@agent(model="claude-opus-4-6", durable=True, checkpoint_store=RedisCheckpointStore())
@use_tools(long_running_analysis)
class ResearchAgent: ...
```

`AgentRunner` serialises `(memory_snapshot, tool_calls_made, turn_number)` after
each tool call.  On `runner.run(..., run_id="existing-run")` it detects existing
state and resumes.

---

## 12. Model Context Protocol (MCP) server support

**Inspiration:** Pydantic AI's MCP integration

Mount any MCP-compatible tool server so its tools are automatically available to agents.
`AgentModule` connects to the MCP server at startup and registers all discovered tools:

```python
from lauren_ai.mcp import MCPServer

AIModule = AgentModule.for_root(
    agents=[MyAgent],
    mcp_servers=[
        MCPServer.stdio("uvx", ["mcp-server-git"]),
        MCPServer.http("https://tools.example.com/mcp"),
    ],
    imports=LLMProvider,
)
```

Tools from MCP servers appear in the `ToolRegistry` with `source="mcp"` metadata.

---

## 13. Agent2Agent (A2A) protocol support

**Inspiration:** Pydantic AI's A2A interoperability

Expose `@agent`-decorated classes as A2A-compatible HTTP endpoints so they can be
called by agents in other frameworks (Pydantic AI, LangGraph, etc.):

```python
AIModule = AgentModule.for_root(
    agents=[ResearchAgent],
    a2a=A2AConfig(prefix="/a2a", auth=BearerAuth(token=API_KEY)),
    imports=LLMProvider,
)
```

`lauren-ai` also ships an `A2AClient` for calling remote A2A agents as tools:

```python
remote = A2AClient("https://research.partner.com/a2a/ResearchAgent")

@agent(model="claude-opus-4-6")
@use_tools(remote.as_tool())
class OrchestratorAgent: ...
```

---

## 14. Inline snapshot testing for agent outputs

**Inspiration:** Pydantic AI's inline snapshot approach (from Pydantic's test ecosystem)

A `assert_agent_snapshot` helper (built on `syrupy` / `pytest-inline-snapshot`) that
auto-generates and updates baseline snapshots for agent responses, making regression
testing trivial:

```python
from lauren_ai.testing import assert_agent_snapshot

async def test_balance_query(mock_runner):
    mock_runner.queue_response("Your balance is $1,500.")
    resp = await mock_runner.run(agent, "What is my balance?")
    assert_agent_snapshot(resp.content)
    # First run: creates snapshot; subsequent runs: compares.
```

---

## 15. Structured streaming output — `run_stream_structured`

**Inspiration:** Pydantic AI's streamed structured output with immediate validation

Stream partial Pydantic model completions, validating each partial JSON fragment as
it arrives.  Callers receive `PartialModel[T]` with fields populated as they stream:

```python
async with runner.run_stream_structured(agent, "Generate a report", result_type=Report) as stream:
    async for partial in stream:
        # partial.title might be None if not yet streamed
        print(f"Title so far: {partial.title!r}")
    final: Report = await stream.get_output()
```

---

## 16. Provider fallback chains — `FallbackTransport`

**Inspiration:** Resilience patterns in production AI deployments

Wrap multiple transports with automatic fallback on error or rate limiting.  The
first transport to respond successfully wins:

```python
LLMProvider = LLMModule.for_root(
    LLMConfig(provider="fallback", transports=[
        LLMConfig.for_anthropic(model="claude-opus-4-6"),
        LLMConfig.for_openai(model="gpt-4o"),
        LLMConfig.for_ollama(model="llama3"),
    ]),
)
```

`FallbackTransport` retries on `TransientTransportError` and tracks which provider
was used via a `SignalBus` event.

---

## 17. Model settings three-tier precedence

**Inspiration:** Pydantic AI's three-tier model settings (model → agent → run)

Currently `AgentConfig` has global settings with no per-run override.  Add
`model_settings` overrides at each level, merged at call time:

```python
@agent(model="claude-opus-4-6", model_settings=ModelSettings(temperature=0.3))
class PreciseAgent: ...

# Override for this specific run only
response = await runner.run(
    agent,
    "Be creative!",
    model_settings=ModelSettings(temperature=0.9, max_tokens=2000),
)
```

`ModelSettings` covers `temperature`, `max_tokens`, `top_p`, `stop_sequences`,
`thinking` (Anthropic), `reasoning_effort` (OpenAI o-series).

---

## 18. LLM-as-judge evaluator — `@evaluator`

**Inspiration:** Pydantic AI's evals harness + LLM-powered scoring

A `@evaluator` decorator marks a function that scores agent outputs using another
LLM call.  Combine with the existing `AccuracyEval` for human-quality scoring:

```python
from lauren_ai.eval import evaluator, EvalScore

@evaluator(criteria="factual accuracy, helpfulness, tone", scale=(1, 5))
async def banking_evaluator(output: str, expected: str) -> EvalScore:
    """Use an LLM judge to score the response against the expected answer."""
    ...  # implementation generated automatically from criteria

suite = EvalSuite(agent=BankingCRMAgent, evaluator=banking_evaluator)
report = await suite.run(dataset)
```

---

## 19. Concurrency limits — `ConcurrencyLimiter`

**Inspiration:** Pydantic AI's `max_concurrency` parameter

Limit the number of parallel `AgentRunner.run()` calls — globally, per agent class,
or per user — with a semaphore-backed limiter wired via DI:

```python
AIModule = AgentModule.for_root(
    agents=[BankingCRMAgent],
    concurrency=ConcurrencyLimits(
        global_max=50,
        per_agent={BankingCRMAgent: 10},
    ),
    imports=LLMProvider,
)
```

Callers that exceed the limit receive `ConcurrencyLimitError` (HTTP 429 via a guard).

---

## 20. Dynamic tool filtering per run

**Inspiration:** Pydantic AI's tool preparation hooks

Provide a `tool_filter` hook that receives `AgentContext` and returns the subset of
tools to expose to the LLM for that specific run.  Enables capability-scoped access:

```python
@agent(model="claude-opus-4-6")
@use_tools(read_balance, transfer_funds, admin_override)
class BankingAgent:
    async def prepare_tools(self, tools: list[ToolMeta], ctx: AgentContext) -> list[ToolMeta]:
        user_role = ctx.metadata.get("role", "customer")
        if user_role != "admin":
            return [t for t in tools if t.name != "admin_override"]
        return tools
```

---

## 21. Context window summarisation — `@summarise_on_overflow`

When `ShortTermMemory` approaches its `max_tokens` budget, automatically invoke a
summarisation LLM call to compress older turns instead of silently dropping them:

```python
@agent(
    model="claude-opus-4-6",
    memory_window_tokens=40_000,
    on_overflow="summarise",          # or "drop" (current default)
    summarise_model="claude-haiku-4-5",  # cheaper model for compression
)
class LongContextAgent: ...
```

The summarised block is stored as a synthetic `system` message marked with
`role="memory_summary"` so it is always preserved.

---

## 22. Semantic tool result cache

Extend the current TTL-based `cache_backend` to support semantic deduplication:
if a tool is called with a semantically similar query (cosine similarity > threshold),
return the cached result without re-invoking:

```python
from lauren_ai.cache import SemanticCache

AIModule = AgentModule.for_root(
    agents=[ResearchAgent],
    tool_cache=SemanticCache(
        embed_fn=embed_service.embed,
        similarity_threshold=0.92,
        ttl_seconds=3600,
    ),
    imports=LLMProvider,
)
```

---

## 23. Agent hot-reload — zero-downtime prompt updates

Expose a `runner.reload_agent(agent_cls)` method that rebuilds `AgentMeta` from the
class without restarting the process.  Integrates with Lauren's `@post_construct`
lifecycle for safe state migration:

```python
# In a management controller:
@post("/admin/reload/{agent_name}")
async def reload(self, agent_name: str) -> dict:
    await self._runner.reload_agent(agent_name)
    return {"reloaded": agent_name}
```

Useful for updating system prompts, swapping tools, or adjusting guardrails in
production without a deploy.

---

## 24. Prompt caching hints — `cache_control`

Surface Anthropic's prompt caching API and OpenAI's equivalent so long system
prompts and static few-shot examples are cached at the provider:

```python
@agent(
    model="claude-opus-4-6",
    system=LONG_SYSTEM_PROMPT,
    cache_system_prompt=True,   # adds cache_control: {"type": "ephemeral"}
)
class KnowledgeAgent: ...
```

`LLMConfig` can also set `cache_tools=True` to cache the (often large) tool schema
block, saving significant input tokens on repeated calls.

---

## 25. Run replay — deterministic re-execution

Record a full run (inputs, tool results, model responses) and replay it with
modified inputs or a different model, enabling offline debugging and A/B testing:

```python
from lauren_ai.replay import RunRecord, RunReplayer

# During production
recorder = RunRecorder(store=RunRecordStore())
AIModule = AgentModule.for_root(..., run_recorder=recorder)

# In tests / debugging
record: RunRecord = await store.load("run-abc123")
replayer = RunReplayer(record)
result = await replayer.run(
    modified_message="What is my balance in EUR?",
    model_override="gpt-4o",
)
```

---

## 26. Multi-provider A/B testing — `ExperimentTransport`

Route a percentage of production traffic to an alternative model/provider and
compare outputs using a judge function or user feedback signals:

```python
LLMProvider = LLMModule.for_root(
    LLMConfig(
        provider="experiment",
        control=LLMConfig.for_anthropic(model="claude-opus-4-6"),
        treatment=LLMConfig.for_openai(model="gpt-4o"),
        split=0.10,   # 10 % to treatment
        judge=llm_judge_fn,
    )
)
```

`SignalBus` emits `ExperimentCallComplete` events for analysis.

---

## 27. Agent catalog — `AgentRegistry`

A central registry (DI-scoped singleton) that knows every agent in the application —
its name, capabilities, tools, model, cost-per-call estimate, and average latency.
Exposed as a REST endpoint and useful for the coordinator pattern:

```python
@get("/agents")
async def list_agents(self, registry: AgentRegistry) -> list[AgentCard]:
    return registry.list()
```

`AgentCard` includes: `name`, `description`, `tools`, `model`, `avg_latency_ms`,
`p95_cost_usd`.  Enables runtime routing decisions and monitoring dashboards.

---

## 28. Agent versioning — `@agent(version="1.2")`

Tag each agent deployment with a semantic version stored in `AgentMeta`.  The
`AgentRunner` stamps every `AgentRunComplete` signal and stored conversation entry
with the agent version, enabling:

- Rollback: keep the previous agent class registered alongside the new one
- Analytics: compare quality metrics across versions
- Audit: reproduce the exact behaviour of a historical run

```python
@agent(model="claude-opus-4-6", version="2.1", replaces="2.0")
class BankingCRMAgent: ...
```

---

## 29. Fine-tuning data collector — `@capture_examples`

Automatically save (input, output) pairs that meet a quality threshold to a dataset
store, ready for supervised fine-tuning:

```python
@agent(model="claude-opus-4-6")
@capture_examples(
    store=DatasetStore("./dataset/crm_examples.jsonl"),
    min_quality_score=0.85,
    evaluator=banking_evaluator,
)
class BankingCRMAgent: ...
```

`DatasetStore` accumulates examples in OpenAI/Anthropic fine-tuning JSONL format
and exposes a `/admin/dataset/export` endpoint.

---

## 30. Memory time-to-live — `ConversationStore(ttl_seconds=...)`

Add TTL support to `InMemoryConversationStore` and the `ConversationStore` protocol.
Expired entries are evicted lazily on next access, with an optional background
cleanup task:

```python
store = InMemoryConversationStore(ttl_seconds=3600 * 24)  # 24-hour sessions

# Protocol addition:
class ConversationStore(Protocol):
    async def load(self, conversation_id: str) -> list[Message]: ...
    async def save(self, conversation_id: str, messages: list[Message], ttl: int | None = None) -> None: ...
    async def delete(self, conversation_id: str) -> None: ...
    async def touch(self, conversation_id: str) -> None: ...   # reset TTL
```

---

## 31. Parallel guardrail execution

Currently guardrails run sequentially; for latency-sensitive paths with multiple
LLM-based guardrails, run them concurrently with `asyncio.gather()` and short-circuit
on the first block decision:

```python
@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[PromptInjectionFilter(), PIIRedactor(), TopicFilter(topics=["banking"])],
    output=[LengthFilter(max_chars=2000), PIIRedactor()],
    input_mode="parallel",    # default: "sequential"
    output_mode="sequential",
)
class SafeAgent: ...
```

---

## 32. Structured tool errors — `ToolError[T]`

Tools currently return strings on error.  Allow typed error returns that the LLM
can reason about:

```python
from lauren_ai import ToolError

class InsufficientFundsError(BaseModel):
    available: float
    requested: float
    message: str = "Insufficient funds"

@tool()
async def transfer(ctx: ToolContext, amount: float, to: str) -> str | ToolError[InsufficientFundsError]:
    if amount > ctx.balance:
        return ToolError(InsufficientFundsError(available=ctx.balance, requested=amount))
    return await do_transfer(amount, to)
```

`AgentRunner` serialises the error to JSON and feeds it back to the LLM as a
structured tool result, enabling richer error reasoning.

---

## 33. Agent conversation branching — `runner.fork()`

Fork a conversation at any point to explore multiple response strategies in parallel,
then merge or select the best branch:

```python
fork_a, fork_b = await runner.fork(
    agent, "Suggest an investment strategy",
    conversation_id="sess-1",
    branches=[
        {"model_settings": ModelSettings(temperature=0.2)},
        {"model_settings": ModelSettings(temperature=0.9)},
    ],
)
# Evaluate both branches
best = await evaluator.pick_best([fork_a.response, fork_b.response])
```

---

## 34. Per-user rate limiting in the conversation store

Track tokens consumed per user per time window directly in the memory subsystem,
enabling per-user fairness without a separate rate limiter:

```python
store = InMemoryConversationStore(
    rate_limit=UserRateLimit(tokens_per_hour=50_000, requests_per_minute=20)
)
```

When a user exceeds the limit, `store.load()` raises `UserRateLimitExceededError`
which the guard catches before the agent runs.

---

## 35. Agent introspection API — `agent.spec()`

Return a machine-readable description of an agent's full configuration at runtime:

```python
spec: AgentSpec = BankingCRMAgent.spec()
# AgentSpec fields:
# .name, .model, .system, .tools: list[ToolSpec],
# .guardrails: GuardrailSpec, .memory: MemorySpec,
# .version, .team: str | None
```

`AgentSpec` serialises to JSON/OpenAPI-compatible schemas.  Expose via a DI-wired
`AgentCatalogController` at `GET /agents/{name}/spec`.

---

## 36. Tool pre/post middleware — `@tool_middleware`

Wrap all tool calls through a middleware chain (similar to Lauren's HTTP middleware)
for cross-cutting concerns like audit logging, cost attribution, and
latency tracking:

```python
from lauren_ai import tool_middleware, ToolCall, ToolResult

@tool_middleware()
class AuditToolMiddleware:
    def __init__(self, audit_log: AuditLogger) -> None:
        self._audit = audit_log

    async def dispatch(self, call: ToolCall, call_next) -> ToolResult:
        self._audit.log_before(call)
        result = await call_next(call)
        self._audit.log_after(call, result)
        return result
```

Register at module level: `AgentModule.for_root(..., tool_middlewares=[AuditToolMiddleware])`.

---

## 37. Hallucination / groundedness checker — `GroundednessGuardrail`

A built-in output guardrail that verifies every factual claim in the response against
a provided knowledge source using an embedding + citation check:

```python
@agent(model="claude-opus-4-6")
@use_guardrails(
    output=[GroundednessGuardrail(
        knowledge_base=kb,
        threshold=0.75,
        on_fail="warn",   # or "block"
    )]
)
class FactualAgent: ...
```

Emits a `GuardrailTriggered(kind="groundedness", score=0.62, ...)` signal when
confidence drops below threshold.

---

## 38. OpenTelemetry exporter for `@traced`

Extend the `TraceStore` exporter interface with a first-class OTEL exporter so
every `@traced()` span propagates to Grafana, Datadog, Honeycomb, or any
OTEL-compatible backend with zero extra code:

```python
from lauren_ai.tracing import OTelExporter

AIModule = AgentModule.for_root(
    agents=[MyAgent],
    trace_exporter=OTelExporter(
        endpoint="http://localhost:4317",
        service_name="banking-ai",
        resource_attributes={"env": "prod"},
    ),
    imports=LLMProvider,
)
```

`ModelCallComplete` and `AgentRunComplete` signals map to OTEL spans automatically.

---

## 39. Dynamic system prompt composition — `@system_prompt_fragment`

Allow multiple independently-registered fragments to compose the final system prompt
at run time.  Each fragment can be static or a DI-resolved async function:

```python
@system_prompt_fragment(priority=10)
async def user_greeting(ctx: AgentContext) -> str:
    name = await ctx.deps.db.get_name(ctx.metadata["user_id"])
    return f"The user's name is {name}. Always address them by name."

@system_prompt_fragment(priority=5)
def company_policy() -> str:
    return "Never disclose account numbers in full. Show only the last 4 digits."
```

Fragments are sorted by priority and concatenated; individual fragments can be
disabled per-run via `model_settings.disabled_fragments`.

---

## 40. Conversation summarisation middleware — `SummarisationMiddleware`

A Lauren middleware that detects long conversations and automatically summarises
previous turns before they are sent to the model, keeping the context window
within budget without data loss:

```python
@module(global_middlewares=[SummarisationMiddleware(
    trigger_tokens=30_000,
    summary_model="claude-haiku-4-5",
    keep_last_turns=4,
)])
class AppModule: ...
```

The summary is stored in `ConversationStore` alongside the full history and
injected as a synthetic message.

---

## 41. Multi-modal output — agents that return images/audio

Extend `AgentResponse` to carry multi-modal content chunks alongside text, enabling
agents backed by image-generation or text-to-speech models:

```python
@agent(model="dall-e-3", output_modality="image")
class ImageGeneratorAgent: ...

response = await runner.run(agent, "A sunset over the ocean, photorealistic")
image_url: str = response.output_image.url
```

`AgentResponse` gains `output_image: ImageContent | None`, `output_audio: AudioContent | None`.

---

## 42. `capture_run_messages` — diagnostic message capture on errors

**Inspiration:** Pydantic AI's `capture_run_messages` context manager

A context manager that intercepts all `Message` objects exchanged during an agent run,
even when the run raises an exception, for post-mortem debugging:

```python
from lauren_ai.testing import capture_run_messages

async with capture_run_messages() as messages:
    try:
        response = await runner.run(agent, "Do something complex")
    except AgentMaxTurnsError:
        pass   # we still get the messages

for msg in messages:
    print(msg["role"], msg["content"][:100])
```

---

## 43. Prompt optimiser — DSPy-style automatic prompt improvement

A `PromptOptimiser` that runs an agent against a labelled dataset and iteratively
rewrites the system prompt to maximise the evaluation score:

```python
from lauren_ai.optimiser import PromptOptimiser

optimiser = PromptOptimiser(
    agent=BankingCRMAgent,
    dataset=eval_dataset,
    evaluator=banking_evaluator,
    max_iterations=20,
    optimise_field="system",
)
best_prompt, history = await optimiser.run()
```

Each iteration: evaluate current prompt → generate candidate variations → pick best.

---

## 44. Persistent knowledge base with chunking strategies

Move `InMemoryVectorStore` to a pluggable backend (SQLite, pgvector, Pinecone) with
an explicit `KnowledgeBaseStore` protocol and enhanced chunking options:

```python
from lauren_ai.knowledge import KnowledgeBase, SemanticChunker

kb = KnowledgeBase(
    store=PgVectorStore(dsn=DATABASE_URL),
    chunker=SemanticChunker(model="claude-haiku-4-5", target_tokens=512),
    embed_fn=embed_service.embed,
)
await kb.load(TextLoader("docs/product_manual.pdf"))
```

`SemanticChunker` (new) uses the LLM to find natural split points rather than fixed
character counts.

---

## 45. Agent graph visualiser — `runner.to_mermaid()`

Generate a Mermaid diagram of the agent's tool call graph from a recorded run,
useful for debugging complex multi-tool chains:

```python
diagram = runner.to_mermaid(run_id="run-abc123")
# Returns Mermaid flowchart string showing every tool call,
# decision point, and delegation in the recorded run.
print(diagram)
```

Also expose as `GET /admin/runs/{run_id}/diagram` returning `text/plain`.

---

## 46. Structured agent-to-agent contract — `AgentContract`

Define a typed contract between two agents specifying the exact input/output types.
The runtime validates that the delegating agent passes the correct type and the
delegate returns the expected schema:

```python
TransferContract = AgentContract(
    input_type=TransferRequest,
    output_type=TransferResult,
    provider=BankingTransferAgent,
    consumer=BankingCRMAgent,
)

AIModule = AgentModule.for_root(
    ...,
    contracts=[TransferContract],
)
```

`AgentRunner` raises `ContractViolationError` at startup if mismatches are found.

---

## 47. Progressive result streaming — `@stream_field`

For structured output agents, stream individual fields as soon as they are complete
rather than waiting for the full Pydantic model:

```python
class AnalysisReport(BaseModel):
    summary: str = Field(..., stream_first=True)
    recommendations: list[str]
    risk_score: float

async with runner.run_stream_structured(agent, "Analyse Q3", result_type=AnalysisReport) as stream:
    async for event in stream.field_events():
        if event.field == "summary":
            print("Summary:", event.value)
        elif event.field == "risk_score":
            print("Risk:", event.value)
```

---

## 48. Tool dependency injection via Lauren's DI container

Class-form tools currently receive only `ToolContext`.  Allow them to declare
additional constructor dependencies resolved by the DI container, removing the
need for manual `use_factory` wiring:

```python
@tool()
class SendEmailTool:
    def __init__(self, smtp: SMTPService, templates: TemplateEngine) -> None:
        # ← injected by the DI container at startup
        self._smtp = smtp
        self._templates = templates

    async def run(self, ctx: ToolContext, to: str, subject: str, body: str) -> str:
        """Send an email. Args: to: recipient, subject: email subject, body: email body."""
        await self._smtp.send(to=to, subject=subject, body=self._templates.render(body))
        return f"Email sent to {to}"
```

No `BankingDelegationWiring`-style manual wiring required.

---

## 49. Agent analytics dashboard — built-in Prometheus metrics

Export agent performance metrics (token usage, cost, latency, tool call rate,
error rate) as a Prometheus endpoint with no extra configuration:

```python
AIModule = AgentModule.for_root(
    agents=[BankingCRMAgent],
    metrics=PrometheusMetrics(prefix="banking_ai"),
    imports=LLMProvider,
)
```

Automatically emits:
- `banking_ai_token_total{agent, model, direction}` (counter)
- `banking_ai_run_duration_seconds{agent}` (histogram)
- `banking_ai_cost_usd_total{agent, model}` (counter)
- `banking_ai_tool_calls_total{agent, tool, status}` (counter)

---

## 50. First-class WebSocket / real-time agent sessions

**Inspiration:** Pydantic AI's real-time voice/UI streaming + AG-UI protocol

Expose agents over WebSocket using Lauren's `@ws_controller`, with a standardised
event protocol (compatible with Vercel AI SDK's stream format and AG-UI):

```python
from lauren_ai.realtime import AgentWebSocketHandler

@ws_controller("/ws/banking")
class BankingWSController:
    def __init__(self, handler: AgentWebSocketHandler[BankingCRMAgent]) -> None:
        self._handler = handler

    @on_message
    async def handle(self, frame: ChatFrame) -> None:
        # Streams tokens, tool events, and final response back over WS
        await self._handler.run(frame.message, conversation_id=frame.session_id)
```

`AgentWebSocketHandler` bridges `runner.run_stream()` to WebSocket frames, handles
reconnection (using `Last-Event-ID` equivalent), and emits structured events:
`token`, `tool_call`, `tool_result`, `done`, `error`.

---

*Ideas range from small API additions (ideas 4, 5, 17) to major new subsystems
(ideas 10, 11, 13, 49, 50). Priority should be driven by user demand, architectural
readiness, and uniqueness of value relative to other frameworks.*

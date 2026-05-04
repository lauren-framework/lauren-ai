# Guides

In-depth how-to guides for every feature of `lauren-ai`.

## Core

<div class="grid cards" markdown>

- :material-chat-processing: **[LLM Calls](llm-calls.md)**

    ---

    Use `LLMService` for direct completions, streaming, and embeddings without
    the full agentic loop.

- :material-wrench: **[Tools](tools.md)**

    ---

    Decorate functions and classes with `@tool()` to expose them to agents.
    Covers schema generation, `ToolContext`, DI injection, and caching.

- :material-robot: **[Agents](agents.md)**

    ---

    Build agents with `@agent()`, `@use_tools()`, lifecycle hooks, delegation,
    and full DI integration via `AgentRunner`.

- :material-database: **[Memory](memory.md)**

    ---

    Four-tier memory architecture: short-term window, conversation history,
    per-user long-term facts, and vector store for RAG.

</div>

## Multi-agent & orchestration

<div class="grid cards" markdown>

- :material-account: **[User Memory](user-memory.md)**

    ---

    Persist facts about individual users across conversations with
    `@remember()`, `UserMemoryStore`, and `MemoryFact`.

- :material-graph: **[Multi-Agent Systems](multi-agent.md)**

    ---

    Delegate between agents using `DelegateToAgent`, tool-based handoff,
    and `AgentRunner` recursive delegation.

- :material-account-group: **[Agent Teams](agent-teams.md)**

    ---

    Compose specialist agents with `@team()` in coordinator or collaborate
    mode. Stream `TeamEvent` instances as workers produce results.

- :material-bookshelf: **[Knowledge Base](knowledge-base.md)**

    ---

    Load documents, chunk them, embed them into `InMemoryVectorStore`, and
    inject relevant context into agents automatically.

</div>

## Input & output

<div class="grid cards" markdown>

- :material-text-box-outline: **[Prompt Templates](prompt-templates.md)**

    ---

    Build reusable, composable prompts with `PromptTemplate`,
    `ChatPromptTemplate`, and `FewShotPromptTemplate`.

- :material-code-json: **[Output Parsers](output-parsers.md)**

    ---

    Transform raw LLM text into typed Python objects with `StrOutputParser`,
    `JSONOutputParser`, and `PydanticOutputParser`.

- :material-format-list-checks: **[Structured Output](structured-output.md)**

    ---

    Guarantee every completion matches a Pydantic schema using `StructuredLLM`
    and `llm_service.with_structured_output(Model)`.

- :material-image-multiple: **[Multimodal Inputs](multimodal.md)**

    ---

    Send images, audio, and documents to the LLM using `ImageContent`,
    `AudioContent`, and `DocumentContent` alongside text messages.

- :material-routes: **[Semantic Router](semantic-router.md)**

    ---

    Route natural-language queries to named handlers using embedding-based
    similarity with `SemanticRouter` and `Route`.

- :material-transit-connection-variant: **[Streaming](streaming.md)**

    ---

    Stream tokens, tool results, and agent turns via `run_stream()` and
    `CompletionChunk` — with SSE controller integration.

</div>

## Production & quality

<div class="grid cards" markdown>

- :material-shield-check: **[Guardrails](guardrails.md)**

    ---

    Block prompt injection, redact PII, enforce topic and length constraints,
    and build custom LLM-evaluated guardrails.

- :material-sitemap: **[Workflows & Chains](workflows.md)**

    ---

    Compose templates, LLM calls, and parsers into typed pipelines with
    `Chain`, `RunnableLambda`, and `|` operator chaining.

- :material-currency-usd: **[Cost Tracking](cost-tracking.md)**

    ---

    Track token usage and USD cost per model and conversation with
    `CostTracker`, `TokenBudget`, and `RateLimiter`.

- :material-chart-timeline: **[Tracing & Observability](tracing.md)**

    ---

    Record structured spans with `@traced()`, export to OpenTelemetry or
    a custom `TraceStore`, and inspect the full agent execution tree.

- :material-test-tube: **[Testing](testing.md)**

    ---

    Write deterministic, zero-network tests with `MockTransport`,
    `AgentTestClient`, and queued mock responses.

- :material-clipboard-check: **[Evaluation](evaluation.md)**

    ---

    Score agent outputs against expected answers using built-in evaluators
    and the `EvalSuite` runner.

- :material-thought-bubble: **[Extended Thinking](extended-thinking.md)**

    ---

    Enable chain-of-thought reasoning in supporting models by passing
    `thinking=True` to the transport or `AgentConfig`.

</div>

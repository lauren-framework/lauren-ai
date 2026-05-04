# Reference

Complete API reference for every public class, decorator, and function in `lauren-ai`.

## Data model & transport

<div class="grid cards" markdown>

- :material-transit-connection: **[Transport](transport.md)**

    ---

    Provider-agnostic data model: `Message`, `Completion`, `CompletionChunk`,
    `TokenUsage`, `ToolCall`, `ToolSchema`, and multimodal content types.

- :material-cog: **[Config](config.md)**

    ---

    `LLMConfig` (provider, model, API key, base URL) and `AgentConfig`
    (max turns, cost budget, memory window). `LLMModule` and `AgentModule`
    factory signatures.

</div>

## Agents & tools

<div class="grid cards" markdown>

- :material-robot: **[Agents](agents.md)**

    ---

    `@agent()`, `@use_tools()`, `AgentMeta`, `AgentConfig`, `AgentContext`,
    `AgentResponse`, and `AgentRunner` constructor and method signatures.

- :material-wrench: **[Tools](tools.md)**

    ---

    `@tool()`, `ToolMeta`, `ToolContext`, `ToolResult`, `ToolRegistry`,
    `ToolExecutor`, and JSON-schema generation rules.

</div>

## Memory

<div class="grid cards" markdown>

- :material-database: **[Memory](memory.md)**

    ---

    `ShortTermMemory`, `ConversationStore` protocol, `InMemoryConversationStore`
    (with `AgentRunner` wiring), `UserMemoryStore`, `MemoryFact`, and
    `InMemoryVectorStore`.

</div>

## HTTP integration

<div class="grid cards" markdown>

- :material-shield: **[Guards](guards.md)**

    ---

    HTTP guard factories: `token_budget_guard`, `requires_capability`,
    `safety_guard`, and `signature_guard` for request-level access control.

- :material-middleware: **[Middleware](middleware.md)**

    ---

    `conversation_middleware` (session ID injection) and `ai_rate_limit`
    (per-user token rate limiting) factory functions.

- :material-filter: **[Interceptors](interceptors.md)**

    ---

    `ai_metrics_interceptor` (latency + cost headers) and
    `token_usage_response_interceptor` (embed usage in response body).

</div>

## Observability & errors

<div class="grid cards" markdown>

- :material-signal: **[Signals](signals.md)**

    ---

    `SignalBus` and all lifecycle events: `ModelCallStarted`,
    `ModelCallComplete`, `ToolCallStarted`, `ToolCallComplete`,
    `AgentRunComplete`.

- :material-alert-circle: **[Exceptions](exceptions.md)**

    ---

    Full exception hierarchy from `LaurenAIError` — transport errors,
    tool errors, agent budget errors, guardrail violations, and more.

- :material-clipboard-check: **[Evaluation](evaluation.md)**

    ---

    `EvalCase`, `EvalResult`, `EvalSuite`, and built-in scorer functions
    for automated agent output assessment.

</div>

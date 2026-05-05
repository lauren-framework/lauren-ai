# Config Reference

## `LLMConfig`

Immutable (frozen) configuration for an LLM provider connection. Use the provider-specific classmethods for the most convenient construction. Use `dataclasses.replace()` to create modified copies.

```python
from lauren_ai import LLMConfig

# Provider-specific factories
cfg = LLMConfig.for_anthropic(model="claude-opus-4-6")
cfg = LLMConfig.for_openai(model="gpt-4o")
cfg = LLMConfig.for_ollama(model="llama3.2")

# Direct construction
cfg = LLMConfig(
    provider="anthropic",
    model="claude-opus-4-6",
    api_key="sk-ant-...",
    max_tokens=4096,
    temperature=1.0,
    timeout=60.0,
    max_retries=3,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `Literal["anthropic", "openai", "ollama", "litellm"]` | required | The backend provider. |
| `model` | `str` | required | The model identifier, e.g. `"claude-opus-4-6"` or `"gpt-4o"`. |
| `api_key` | `str \| None` | `None` | Provider API key. When `None`, the provider-specific environment variable is used (e.g. `ANTHROPIC_API_KEY`). |
| `base_url` | `str \| None` | `None` | Override the provider's default base URL. Useful for proxies, self-hosted deployments, or Ollama. |
| `max_tokens` | `int` | `4096` | Maximum tokens to generate per completion call. |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0–2.0 for most providers). |
| `timeout` | `float` | `60.0` | HTTP request timeout in seconds. |
| `max_retries` | `int` | `3` | Maximum automatic retries on transient errors. |
| `cache_system_prompt` | `bool` | `False` | Enable Anthropic prompt caching for the system prompt. No-op on other providers. |
| `cache_tools` | `bool` | `False` | Enable Anthropic prompt caching for tool definitions. No-op on other providers. |
| `embed_model` | `str \| None` | `None` | Model to use for embedding calls. Defaults to `model` when `None`. |
| `embed_dimensions` | `int \| None` | `None` | Desired embedding dimensionality for providers that support truncated embeddings. |

### Classmethods

#### `LLMConfig.for_anthropic(model="claude-opus-4-6", *, api_key=None, **kwargs)`

Creates a config pre-wired for Anthropic. Reads `ANTHROPIC_API_KEY` from the environment when `api_key` is `None`.

#### `LLMConfig.for_openai(model="gpt-4o", *, api_key=None, **kwargs)`

Creates a config pre-wired for OpenAI. Reads `OPENAI_API_KEY` from the environment when `api_key` is `None`.

#### `LLMConfig.for_ollama(model="llama3.2", *, base_url="http://localhost:11434", **kwargs)`

Creates a config pre-wired for a local Ollama server. No API key required.

#### `LLMConfig.for_testing() -> tuple[LLMConfig, MockTransport]`

Creates a test config paired with a `MockTransport`. No network calls are ever made. Queue deterministic responses before running code under test.

```python
from lauren_ai import LLMConfig
from lauren_ai.testing import AgentTestClient
from lauren_ai._transport import Completion, TokenUsage

cfg, mock = LLMConfig.for_testing()
mock.queue_response(
    Completion(
        id="test-1",
        model="mock-model",
        content="Hello!",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
)

client = AgentTestClient(agent=MyAgent, config=cfg, mock_transport=mock)
result = await client.run("Say hello")
assert result.content == "Hello!"
```

---

## `AgentConfig`

Immutable (frozen) configuration for an agent's runtime behaviour. Fields can be set directly on `@agent()` via `config_kwargs` or passed to `AgentModule.for_root()`.

```python
from lauren_ai import AgentConfig

cfg = AgentConfig(
    max_turns=5,
    max_cost_usd=0.10,
    parallel_tool_calls=True,
    tool_error_policy="return_error",
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | `str` | `"You are a helpful assistant."` | System prompt sent to the LLM at the start of every turn. |
| `max_turns` | `int` | `10` | Maximum agentic loop iterations before `AgentMaxTurnsError`. |
| `max_tokens_per_turn` | `int` | `4096` | Maximum output tokens requested per turn. |
| `temperature` | `float` | `1.0` | Sampling temperature. Overrides `LLMConfig.temperature` for this agent. |
| `memory_window_tokens` | `int` | `40_000` | Sliding-window size in tokens for conversation history passed to the model. |
| `max_cost_usd` | `float \| None` | `None` | Hard cost budget in USD. The runner raises `AgentBudgetExceededError` after each turn that crosses this limit. `None` = unlimited. |
| `parallel_tool_calls` | `bool` | `False` | When `True`, all tool calls in a single model turn are executed concurrently. |
| `tool_error_policy` | `Literal["raise", "return_error", "skip"]` | `"return_error"` | How to handle tool execution errors (see below). |
| `thinking` | `bool` | `False` | Enable extended thinking (Anthropic only). |
| `thinking_budget_tokens` | `int` | `8_000` | Token budget for the thinking phase when `thinking=True`. |
| `reasoning_effort` | `Literal["low", "medium", "high"] \| None` | `None` | OpenAI reasoning effort for o1/o3 models. `None` = provider default. |
| `include_reasoning_in_response` | `bool` | `False` | When `True`, thinking blocks are included in the `Completion` response. |

### `tool_error_policy` values

| Value | Behaviour |
|-------|-----------|
| `"raise"` | Re-raise the exception immediately, halting the agent. |
| `"return_error"` | Send the error message back to the model as a tool result so it can decide how to proceed. |
| `"skip"` | Silently omit the failing tool result. |

---

## `LLMModule`

Factory that creates a `@module` providing `LLMService` and `EmbedService` to the Lauren DI container.

```python
from lauren_ai._module import LLMModule
from lauren_ai import LLMConfig

LLMProvider = LLMModule.for_root(
    LLMConfig.for_anthropic(model="claude-opus-4-6", api_key="sk-ant-...")
)

# In tests (zero network calls):
cfg, mock = LLMConfig.for_testing()
TestLLMModule = LLMModule.for_root(cfg, transport_override=mock)
```

### `LLMModule.for_root(config, *, transport_override=None) -> type`

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `LLMConfig` | The LLM configuration. |
| `transport_override` | `Any \| None` | Pre-built transport to use instead of the one derived from `config`. Pass a `MockTransport` here in tests. |

The returned `@module` provides and exports:
- `LLMService` — completion, streaming, embedding, and token counting.
- `EmbedService` — embedding-only facade.
- `LLMConfig` — the raw config object.
- `Transport` — the underlying transport instance.

### `LLMService`

High-level service wrapping a `Transport` with application-level defaults from `LLMConfig`. Inject into controllers or other services.

| Method | Description |
|--------|-------------|
| `async complete(messages, *, system, tools, tool_choice, model, max_tokens, temperature, stream)` | Run a completion with merged per-call overrides. Returns `Completion` or `AsyncIterator[CompletionChunk]`. |
| `async complete_stream(messages, **kwargs)` | Convenience alias for `complete(..., stream=True)`. |
| `async embed(inputs, *, model=None) -> list[Embedding]` | Compute embeddings. |
| `async count_tokens(messages) -> int` | Count tokens; falls back to `chars / 4` heuristic when the transport doesn't support it. |
| `with_structured_output(model_cls) -> StructuredLLM[T]` | Return a `StructuredLLM` that forces schema-valid Pydantic output via tool-calling. |

### `EmbedService`

Embedding-only facade backed by `LLMService`. Provided for consumers that only need embeddings.

| Method | Description |
|--------|-------------|
| `async embed(inputs, *, model=None) -> list[Embedding]` | Compute embeddings. |

---

## `AgentModule`

Factory that creates a `@module` providing a unique `AgentRunnerBase` subclass (the module's runner token), all registered agent instances, and their tools.

Must import the `LLMModule` result via the `imports` parameter so `Transport` and `LLMConfig` tokens are visible inside the generated module.

```python
from lauren_ai._module import AgentModule

AIAgentModule = AgentModule.for_root(
    agents=[ResearchAgent, SummarizerAgent],
    tools=[WebSearchTool, CodeExecutionTool],
    imports=LLMProvider,           # required — exposes Transport and LLMConfig
    config=AgentConfig(max_turns=5, max_cost_usd=0.50),
    signals=signal_bus,
)

@module(imports=[LLMProvider, AIAgentModule])
class AppModule: ...
```

### `AgentModule.for_root(*, agents, tools=None, imports=None, signals=None, memory=None, conversation_store=None, config=None, tool_cache=None, knowledge=None) -> type`

| Parameter | Type | Description |
|-----------|------|-------------|
| `agents` | `list[type]` | `@agent()`-decorated classes to register. |
| `tools` | `list[Any] \| None` | Shared tools available to all agents (in addition to per-agent `@use_tools()` registrations). |
| `imports` | `type \| list[type] \| None` | `@module`-decorated class(es) to import. Pass the `LLMModule` result here. |
| `signals` | `Any \| None` | Optional `SignalBus` wired into the module's runner. |
| `memory` | `Any \| None` | Long-term memory store instance. |
| `conversation_store` | `Any \| None` | Conversation history store instance. |
| `config` | `AgentConfig \| None` | Default `AgentConfig` for all agents in this module. |
| `tool_cache` | `Any \| None` | Cache backend for tool result caching. |
| `knowledge` | `list[Any] \| None` | Knowledge base instances to pre-load into long-term memory. |

The returned `@module` provides and exports:
- A unique `AgentRunnerBase` subclass as the module's runner token. Inject it via
  `runner: AgentRunner` (Protocol scan) in single-module scope, or via the explicit
  named subclass passed to `injects=[MyRunner]` in multi-module scope.
- All agent classes registered as injectable singletons.

The `injects` parameter accepts one optional `AgentRunnerBase` subclass. When
omitted, an anonymous subclass is auto-generated. Pass an explicit subclass when
a controller or service imports two or more AgentModules and needs to disambiguate.

# Modules & Services

DI-wiring helpers for integrating `lauren-ai` into a Lauren application.

### `LLMModule`

```python
class LLMModule
```

Factory that creates a `@module` providing LLM services.

The returned module provides and exports:

* `LLMService` — completion + embedding + streaming
* `EmbedService` — embedding-only convenience wrapper

When `lauren` is installed the factory also registers the raw
`Transport` so other modules in the graph can depend on it.

Usage::

    LLMProviderModule = LLMModule.for_root(
        LLMConfig.for_anthropic(model="claude-opus-4-6", api_key="sk-...")
    )

    # Tests (zero network calls):
    cfg, mock = LLMConfig.for_testing()
    TestLLMModule = LLMModule.for_root(cfg, transport_override=mock)

#### `LLMModule.for_root`

```python
def for_root(cls, config: LLMConfig, transport_override: Any | None = None) -> type
```

Create a `@module` that provides `LLMService` and
`EmbedService`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `config` | `LLMConfig` | The LLM configuration. |
| `transport_override` | `Any | None` | Pre-built transport to use instead of the
one derived from *config*.  Pass a
`MockTransport` here in tests. |

**Returns:** `type` — A `@module`-decorated class.

### `AgentModule`

```python
class AgentModule
```

Factory that creates a `@module` providing the `AgentRunner`,
`ToolRegistry`, and all registered agent
class instances.

The module wires `AgentRunner` via `use_factory`, injecting the
`Transport` and `LLMConfig` tokens from the Lauren DI container.  Those
tokens are provided by the `@module` returned by
`LLMModule.for_root()`.  Because Lauren enforces NestJS-style module
encapsulation, the generated agent module can only *see* tokens that are
**exported by a module it explicitly imports**.  Pass the `LLMModule`
result via the `imports` parameter so the `Transport` + `LLMConfig`
tokens are visible inside the generated module and the `use_factory`
resolves correctly::

    LLMProvider = LLMModule.for_root(LLMConfig.for_anthropic(model="claude-opus-4-6"))

    AIAgentModule = AgentModule.for_root(
        agents=[ResearchAgent, SummarizerAgent],
        tools=[WebSearchTool, CodeExecutionTool],
        imports=LLMProvider,        # ← required so Transport is visible
        config=AgentConfig(max_turns=5, max_cost_usd=0.50),
    )

    @module(imports=[LLMProvider, AIAgentModule])
    class AppModule: ...

Without `imports=LLMProvider` the generated module has an empty
`imports` list, so `Transport` and `LLMConfig` are not in its visible
set and the `use_factory` injection raises `MissingProviderError` at
startup.

#### `AgentModule.for_root`

```python
def for_root(cls, agents: list[type], tools: list[Any] | None = None, imports: Any | None = None, signals: Any | None = None, config: AgentConfig | None = None, tool_cache: Any | None = None, knowledge: list[Any] | None = None, runner: type | None = None, injects: list[type] | None = None, export_tools: list[type] | None = None, shared_tools: list[type] | None = None) -> type
```

Create a `@module` providing the agent runner and all agent instances.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `agents` | `list[type]` | `@agent()`-decorated classes to register. |
| `tools` | `list[Any] | None` | Shared tools available to all agents (supplementing
per-agent `@use_tools()` registrations). |
| `imports` | `Any | None` | A single `@module`-decorated class **or** a list of
them to import into the generated agent module.  Pass the result of
`LLMModule.for_root()` here so `Transport` and `LLMConfig`
are visible inside the generated module and the `use_factory` for
`AgentRunner` can inject them.
Without this the two modules are siblings in the application module
graph, and the generated agent module cannot see the LLM module's
exports. |
| `signals` | `Any | None` | Optional `SignalBus` to wire
into the `AgentRunner` so it emits
`ModelCallComplete` / `AgentRunComplete` events. |
| `config` | `AgentConfig | None` | Default `AgentConfig`. |
| `tool_cache` | `Any | None` | Cache backend for tool result caching. |
| `knowledge` | `list[Any] | None` | List of `KnowledgeSource`
instances declared at module scope.  Each is converted to a
`@tool()` via `KnowledgeBase.as_tool()` and registered as a
DI provider via `use_value(provide=type(ks), value=ks)`.

**Visibility is opt-in.**  Agents must declare
`@use_knowledge_sources(...)` to attach a source's tool to
their schema.  An agent without that decorator sees **no** KB
tools (its `meta.knowledge_source_filter` is `None`).

Bare `KnowledgeBase` instances are rejected with
`TypeError` — wrap in
`KnowledgeSource(kb=..., tool_name=...)`.  Two sources with
the same tool name raise
`DecoratorUsageError`. |
| `runner` | `type | None` | Optional named `AgentRunnerBase`
subclass to use as this module's runner DI token.

**Default (`None`):** a unique `AgentRunnerBase` subclass is
auto-generated per `for_root()` call.  Providers inside this module
can inject it with `runner: AgentRunner` — the DI container resolves
it via structural Protocol scan.

**Explicit subclass:** pass a named `AgentRunnerBase` subclass
(decorated with `@injectable(scope=Scope.SINGLETON)`) when this
module coexists with other `AgentModule` instances in the same
import scope **and** a controller, service, or delegation tool needs
to inject a specific module's runner by name.  The named class becomes
the unambiguous DI token (e.g. `runner: TransferAgentRunner`).

Every `AgentModule.for_root()` call MUST have its own dedicated
runner — either the auto-generated one (default) or this explicit
subclass.  Sharing a runner across modules is not supported. |
| `injects` | `list[type] | None` | Optional list of additional provider classes to register
inside this module.  Use this to make extra singletons available to
the agents and tools wired by this module — for example, a shared
cache, a domain service, or a custom configuration class.  These
classes are added as providers but not exported; export them
explicitly if parent modules need them. |
| `shared_tools` | `list[type] | None` | Tool classes that are owned and exported by an imported module
and must not be auto-registered as providers here.  Pass tool classes that appear
in `@use_tools()` on an agent but are already provided by a module in `imports`,
to prevent `ModuleExportViolation` when the same class would otherwise be declared
as a provider in multiple `AgentModule` instances.

The tools remain fully usable by agents in this module — the DI container resolves
them through the import chain.  Only the *declaration* step is skipped; ownership,
lifecycle, and scope all remain in the providing module. |

**Returns:** `type` — A `@module`-decorated class.

### `LLMService`

```python
class LLMService(transport: Any, config: LLMConfig)
```

High-level service wrapping a `Transport`
with application-level defaults from `LLMConfig`.

Registered as a singleton provider by `LLMModule`.  Inject it
directly into controllers or agents::

    class AIController:
        def __init__(self, llm: LLMService) -> None:
            self._llm = llm

        @get("/complete")
        async def complete(self) -> dict:
            result = await self._llm.complete(
                [Message.user("Say hello!")]
            )
            return {"content": result.content}

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `transport` | `Any` | The underlying LLM transport. |
| `config` | `LLMConfig` | The LLM configuration supplying defaults. |

#### `LLMService.complete`

```python
def complete(self, messages: list[Message], system: str | None = None, tools: list[Any] | None = None, tool_choice: Any | None = None, model: str | None = None, max_tokens: int | None = None, temperature: float | None = None, stream: bool = False) -> Completion | AsyncIterator[CompletionChunk]
```

Run a completion with merged per-call overrides and config defaults.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | Conversation messages. |
| `system` | `str | None` | Optional system prompt. |
| `tools` | `list[Any] | None` | Optional tool schema list. |
| `tool_choice` | `Any | None` | Optional tool choice specifier. |
| `model` | `str | None` | Model override.  Uses `config.model` when `None`. |
| `max_tokens` | `int | None` | Max tokens override.  Uses `config.max_tokens`
when `None`. |
| `temperature` | `float | None` | Temperature override.  Uses `config.temperature`
when `None`. |
| `stream` | `bool` | When `True` returns an async iterator of chunks. |

**Returns:** `Completion | AsyncIterator[CompletionChunk]` — A `Completion` or an async
iterator of `CompletionChunk`.

#### `LLMService.complete_stream`

```python
def complete_stream(self, messages: list[Message], kwargs: Any = {}) -> AsyncIterator[CompletionChunk]
```

Run a streaming completion (convenience alias for `complete(..., stream=True)`).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | Conversation messages. |
| `kwargs` | `Any` | Additional keyword arguments forwarded to `complete()`. |

**Returns:** `AsyncIterator[CompletionChunk]` — Async iterator of `CompletionChunk`.

#### `LLMService.embed`

```python
def embed(self, inputs: list[str], model: str | None = None) -> list[Embedding]
```

Compute embeddings for a list of input strings.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `inputs` | `list[str]` | Texts to embed. |
| `model` | `str | None` | Embedding model override.  Uses `config.embed_model`
(or `config.model`) when `None`. |

**Returns:** `list[Embedding]` — One `Embedding` per input.

#### `LLMService.count_tokens`

```python
def count_tokens(self, messages: list[Message]) -> int
```

Count the tokens in *messages* for the configured model.

Falls back to a heuristic (`total_chars / 4`) when the transport
does not support `count_tokens`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The messages to count. |

**Returns:** `int` — Estimated or exact token count.

#### `LLMService.with_structured_output`

```python
def with_structured_output(self, model_cls: type[T]) -> StructuredLLM[T]
```

Return a StructuredLLM that forces schema-valid output.

Uses native tool-calling to guarantee the model returns
a valid instance of *model_cls*.

Usage::

    structured = llm.with_structured_output(MyModel)
    result: MyModel = await structured.complete(messages)

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model_cls` | `type[T]` | A Pydantic `BaseModel` subclass whose schema
the model must satisfy. |

**Returns:** `StructuredLLM[T]` — A `StructuredLLM`
bound to this service.

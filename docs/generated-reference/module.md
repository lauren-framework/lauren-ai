# Modules & Services

DI-wiring helpers for integrating `lauren-ai` into a Lauren application.

### `LLMModule`

```python
class LLMModule
```

Factory that creates a ``@module`` providing LLM services.

The returned module provides and exports:

* :class:`LLMService` — completion + embedding + streaming
* :class:`EmbedService` — embedding-only convenience wrapper

When ``lauren`` is installed the factory also registers the raw
``Transport`` so other modules in the graph can depend on it.

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

Create a ``@module`` that provides :class:`LLMService` and
:class:`EmbedService`.

:param config: The LLM configuration.
:type config: LLMConfig
:param transport_override: Pre-built transport to use instead of the
    one derived from *config*.  Pass a
    :class:`~lauren_ai._transport._mock.MockTransport` here in tests.
:type transport_override: Any | None
:return: A ``@module``-decorated class.
:rtype: type

### `AgentModule`

```python
class AgentModule
```

Factory that creates a ``@module`` providing the :class:`~lauren_ai._agents._runner.AgentRunner`,
:class:`~lauren_ai._tools._registry.ToolRegistry`, and all registered agent
class instances.

The module wires :class:`AgentRunner` via ``use_factory``, injecting the
``Transport`` and ``LLMConfig`` tokens from the Lauren DI container.  Those
tokens are provided by the ``@module`` returned by
:meth:`LLMModule.for_root`.  Because Lauren enforces NestJS-style module
encapsulation, the generated agent module can only *see* tokens that are
**exported by a module it explicitly imports**.  Pass the ``LLMModule``
result via the ``imports`` parameter so the ``Transport`` + ``LLMConfig``
tokens are visible inside the generated module and the ``use_factory``
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

Without ``imports=LLMProvider`` the generated module has an empty
``imports`` list, so ``Transport`` and ``LLMConfig`` are not in its visible
set and the ``use_factory`` injection raises ``MissingProviderError`` at
startup.

#### `AgentModule.for_root`

```python
def for_root(cls, agents: list[type], tools: list[Any] | None = None, imports: Any | None = None, signals: Any | None = None, config: AgentConfig | None = None, tool_cache: Any | None = None, knowledge: list[Any] | None = None, runner: type | None = None, injects: list[type] | None = None, export_tools: list[type] | None = None, shared_tools: list[type] | None = None) -> type
```

Create a ``@module`` providing the agent runner and all agent instances.

:param agents: ``@agent()``-decorated classes to register.
:type agents: list[type]
:param tools: Shared tools available to all agents (supplementing
    per-agent ``@use_tools()`` registrations).
:type tools: list[Any] | None
:param imports: A single ``@module``-decorated class **or** a list of
    them to import into the generated agent module.  Pass the result of
    :meth:`LLMModule.for_root` here so ``Transport`` and ``LLMConfig``
    are visible inside the generated module and the ``use_factory`` for
    :class:`~lauren_ai._agents._runner.AgentRunner` can inject them.
    Without this the two modules are siblings in the application module
    graph, and the generated agent module cannot see the LLM module's
    exports.
:type imports: type | list[type] | None
:param signals: Optional :class:`~lauren_ai._signals.SignalBus` to wire
    into the :class:`~lauren_ai._agents._runner.AgentRunner` so it emits
    ``ModelCallComplete`` / ``AgentRunComplete`` events.
:type signals: Any | None
:param config: Default :class:`~lauren_ai._config.AgentConfig`.
:type config: AgentConfig | None
:param tool_cache: Cache backend for tool result caching.
:type tool_cache: Any | None
:param knowledge: List of :class:`~lauren_ai._knowledge.KnowledgeSource`
    instances declared at module scope.  Each is converted to a
    ``@tool()`` via ``KnowledgeBase.as_tool()`` and registered as a
    DI provider via ``use_value(provide=type(ks), value=ks)``.

    **Visibility is opt-in.**  Agents must declare
    ``@use_knowledge_sources(...)`` to attach a source's tool to
    their schema.  An agent without that decorator sees **no** KB
    tools (its ``meta.knowledge_source_filter`` is ``None``).

    Bare :class:`KnowledgeBase` instances are rejected with
    ``TypeError`` — wrap in
    ``KnowledgeSource(kb=..., tool_name=...)``.  Two sources with
    the same tool name raise
    :class:`~lauren_ai._exceptions.DecoratorUsageError`.
:type knowledge: list[KnowledgeSource] | None
:param runner: Optional named :class:`~lauren_ai._agents._runner.AgentRunnerBase`
    subclass to use as this module's runner DI token.

    **Default (``None``):** a unique ``AgentRunnerBase`` subclass is
    auto-generated per ``for_root()`` call.  Providers inside this module
    can inject it with ``runner: AgentRunner`` — the DI container resolves
    it via structural Protocol scan.

    **Explicit subclass:** pass a named ``AgentRunnerBase`` subclass
    (decorated with ``@injectable(scope=Scope.SINGLETON)``) when this
    module coexists with other ``AgentModule`` instances in the same
    import scope **and** a controller, service, or delegation tool needs
    to inject a specific module's runner by name.  The named class becomes
    the unambiguous DI token (e.g. ``runner: TransferAgentRunner``).

    Every ``AgentModule.for_root()`` call MUST have its own dedicated
    runner — either the auto-generated one (default) or this explicit
    subclass.  Sharing a runner across modules is not supported.
:type runner: type | None
:param injects: Optional list of additional provider classes to register
    inside this module.  Use this to make extra singletons available to
    the agents and tools wired by this module — for example, a shared
    cache, a domain service, or a custom configuration class.  These
    classes are added as providers but not exported; export them
    explicitly if parent modules need them.
:type injects: list[type] | None
:param shared_tools: Tool classes that are owned and exported by an imported module
    and must not be auto-registered as providers here.  Pass tool classes that appear
    in ``@use_tools()`` on an agent but are already provided by a module in ``imports``,
    to prevent ``ModuleExportViolation`` when the same class would otherwise be declared
    as a provider in multiple ``AgentModule`` instances.

    The tools remain fully usable by agents in this module — the DI container resolves
    them through the import chain.  Only the *declaration* step is skipped; ownership,
    lifecycle, and scope all remain in the providing module.
:type shared_tools: list[type] | None
:return: A ``@module``-decorated class.
:rtype: type

### `LLMService`

```python
class LLMService(transport: Any, config: LLMConfig)
```

High-level service wrapping a :class:`~lauren_ai._transport.Transport`
with application-level defaults from :class:`~lauren_ai._config.LLMConfig`.

Registered as a singleton provider by :class:`LLMModule`.  Inject it
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

:param transport: The underlying LLM transport.
:type transport: Any
:param config: The LLM configuration supplying defaults.
:type config: LLMConfig

#### `LLMService.complete`

```python
def complete(self, messages: list[Message], system: str | None = None, tools: list[Any] | None = None, tool_choice: Any | None = None, model: str | None = None, max_tokens: int | None = None, temperature: float | None = None, stream: bool = False) -> Completion | AsyncIterator[CompletionChunk]
```

Run a completion with merged per-call overrides and config defaults.

:param messages: Conversation messages.
:type messages: list[Message]
:param system: Optional system prompt.
:type system: str | None
:param tools: Optional tool schema list.
:type tools: list[Any] | None
:param tool_choice: Optional tool choice specifier.
:type tool_choice: Any | None
:param model: Model override.  Uses ``config.model`` when ``None``.
:type model: str | None
:param max_tokens: Max tokens override.  Uses ``config.max_tokens``
    when ``None``.
:type max_tokens: int | None
:param temperature: Temperature override.  Uses ``config.temperature``
    when ``None``.
:type temperature: float | None
:param stream: When ``True`` returns an async iterator of chunks.
:type stream: bool
:return: A :class:`~lauren_ai._transport.Completion` or an async
    iterator of :class:`~lauren_ai._transport.CompletionChunk`.
:rtype: Completion | AsyncIterator[CompletionChunk]

#### `LLMService.complete_stream`

```python
def complete_stream(self, messages: list[Message], kwargs: Any = {}) -> AsyncIterator[CompletionChunk]
```

Run a streaming completion (convenience alias for ``complete(..., stream=True)``).

:param messages: Conversation messages.
:type messages: list[Message]
:param kwargs: Additional keyword arguments forwarded to :meth:`complete`.
:return: Async iterator of :class:`~lauren_ai._transport.CompletionChunk`.
:rtype: AsyncIterator[CompletionChunk]

#### `LLMService.embed`

```python
def embed(self, inputs: list[str], model: str | None = None) -> list[Embedding]
```

Compute embeddings for a list of input strings.

:param inputs: Texts to embed.
:type inputs: list[str]
:param model: Embedding model override.  Uses ``config.embed_model``
    (or ``config.model``) when ``None``.
:type model: str | None
:return: One :class:`~lauren_ai._transport.Embedding` per input.
:rtype: list[Embedding]

#### `LLMService.count_tokens`

```python
def count_tokens(self, messages: list[Message]) -> int
```

Count the tokens in *messages* for the configured model.

Falls back to a heuristic (``total_chars / 4``) when the transport
does not support ``count_tokens``.

:param messages: The messages to count.
:type messages: list[Message]
:return: Estimated or exact token count.
:rtype: int

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

:param model_cls: A Pydantic ``BaseModel`` subclass whose schema
    the model must satisfy.
:type model_cls: type[T]
:return: A :class:`~lauren_ai._transport._structured.StructuredLLM`
    bound to this service.
:rtype: StructuredLLM[T]


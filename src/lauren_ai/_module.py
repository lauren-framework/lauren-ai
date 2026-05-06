"""Module factories for ``lauren-ai``.

:class:`LLMModule` and :class:`AgentModule` are factory classes whose
``for_root()`` classmethods return ``@module``-decorated classes that
integrate seamlessly with the ``lauren`` framework's DI and module system.

Typical usage::

    import os
    from lauren import module, LaurenFactory
    from lauren_ai import LLMConfig, AgentConfig
    from lauren_ai._module import LLMModule, AgentModule

    LLMProviderModule = LLMModule.for_root(
        LLMConfig.for_anthropic(
            model="claude-opus-4-6",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    )

    AIAgentModule = AgentModule.for_root(
        agents=[ResearchAgent, SummarizerAgent],
        tools=[WebSearchTool],
    )

    @module(
        controllers=[ChatController],
        imports=[LLMProviderModule, AIAgentModule],
    )
    class AppModule: ...

    app = LaurenFactory.create(AppModule)

Both module factories return a ``@module``-decorated class ready for use
with the ``lauren`` DI and module system.
"""

from __future__ import annotations

__all__ = [
    "LLMModule",
    "AgentModule",
    "LLMService",
    "EmbedService",
]

import logging
from collections.abc import AsyncIterator
from typing import Any, TypeVar, get_origin

T = TypeVar("T")

from lauren_ai._config import AgentConfig, LLMConfig
from lauren_ai._exceptions import AgentConfigError
from lauren_ai._transport import Completion, CompletionChunk, Embedding, Message

logger = logging.getLogger(__name__)

from lauren import Scope, module, use_class, use_factory, use_value

# ---------------------------------------------------------------------------
# Transport builder
# ---------------------------------------------------------------------------


def _build_transport(config: LLMConfig, override: Any = None) -> Any:
    """Build a transport instance from *config*, or return *override* directly.

    :param config: The LLM configuration.
    :type config: LLMConfig
    :param override: Pre-built transport to use instead (used in tests).
    :type override: Any | None
    :return: A :class:`~lauren_ai._transport.Transport`-compatible instance.
    :rtype: Any
    :raises AgentConfigError: When *config.provider* is not recognised or the
        required optional package is missing.
    """
    if override is not None:
        return override

    if config.provider == "anthropic":
        from lauren_ai._transport._anthropic import AnthropicTransport  # noqa: PLC0415

        return AnthropicTransport(config)

    if config.provider == "openai":
        try:
            from lauren_ai._transport._openai import OpenAITransport  # noqa: PLC0415

            return OpenAITransport(config)
        except ImportError as exc:
            raise AgentConfigError(
                "OpenAI transport requires the 'openai' package.  "
                "Install it with: pip install openai",
                cause=exc,
            ) from exc

    if config.provider == "ollama":
        try:
            from lauren_ai._transport._ollama import OllamaTransport  # noqa: PLC0415

            return OllamaTransport(config)
        except ImportError as exc:
            raise AgentConfigError(
                "Ollama transport requires the 'httpx' package.  "
                "Install it with: pip install httpx",
                cause=exc,
            ) from exc

    if config.provider == "litellm":
        try:
            from lauren_ai._transport._litellm import LiteLLMTransport  # noqa: PLC0415

            return LiteLLMTransport(config)
        except ImportError as exc:
            raise AgentConfigError(
                "LiteLLM transport requires the 'litellm' package.  "
                "Install it with: pip install litellm",
                cause=exc,
            ) from exc

    raise AgentConfigError(
        f"Unknown LLM provider: {config.provider!r}.  "
        "Supported providers: 'anthropic', 'openai', 'ollama', 'litellm'."
    )


# ---------------------------------------------------------------------------
# LLMService
# ---------------------------------------------------------------------------


class LLMService:
    """High-level service wrapping a :class:`~lauren_ai._transport.Transport`
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
    """

    def __init__(self, transport: Any, config: LLMConfig) -> None:
        self._transport = transport
        self._config = config

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> Completion | AsyncIterator[CompletionChunk]:
        """Run a completion with merged per-call overrides and config defaults.

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
        """
        kwargs: dict[str, Any] = dict(
            model=model or self._config.model,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            stream=stream,
        )
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        return await self._transport.complete(messages, **kwargs)

    async def complete_stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        """Run a streaming completion (convenience alias for ``complete(..., stream=True)``).

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param kwargs: Additional keyword arguments forwarded to :meth:`complete`.
        :return: Async iterator of :class:`~lauren_ai._transport.CompletionChunk`.
        :rtype: AsyncIterator[CompletionChunk]
        """
        result = await self.complete(messages, stream=True, **kwargs)
        return result  # type: ignore[return-value]

    async def embed(
        self,
        inputs: list[str],
        *,
        model: str | None = None,
    ) -> list[Embedding]:
        """Compute embeddings for a list of input strings.

        :param inputs: Texts to embed.
        :type inputs: list[str]
        :param model: Embedding model override.  Uses ``config.embed_model``
            (or ``config.model``) when ``None``.
        :type model: str | None
        :return: One :class:`~lauren_ai._transport.Embedding` per input.
        :rtype: list[Embedding]
        """
        embed_model = model or self._config.embed_model or self._config.model
        return await self._transport.embed(
            inputs,
            model=embed_model,
            dimensions=self._config.embed_dimensions,
        )

    async def count_tokens(self, messages: list[Message]) -> int:
        """Count the tokens in *messages* for the configured model.

        Falls back to a heuristic (``total_chars / 4``) when the transport
        does not support ``count_tokens``.

        :param messages: The messages to count.
        :type messages: list[Message]
        :return: Estimated or exact token count.
        :rtype: int
        """
        if hasattr(self._transport, "count_tokens"):
            return await self._transport.count_tokens(
                messages,
                model=self._config.model,
            )
        total_chars = 0
        for m in messages:
            if isinstance(m.content, str):
                total_chars += len(m.content)
            else:
                for block in m.content:
                    total_chars += len(str(block))
        return total_chars // 4

    def with_structured_output(self, model_cls: type[T]) -> StructuredLLM[T]:
        """Return a StructuredLLM that forces schema-valid output.

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
        """
        from lauren_ai._transport._structured import StructuredLLM  # noqa: PLC0415

        return StructuredLLM(self, model_cls)


# ---------------------------------------------------------------------------
# EmbedService
# ---------------------------------------------------------------------------


class EmbedService:
    """Embedding-only facade backed by :class:`LLMService`.

    Provided as a separate binding so consumers that only need embeddings
    can inject ``EmbedService`` rather than the full ``LLMService``.

    :param llm_service: The underlying :class:`LLMService`.
    :type llm_service: LLMService
    """

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    async def embed(
        self,
        inputs: list[str],
        *,
        model: str | None = None,
    ) -> list[Embedding]:
        """Compute embeddings.

        :param inputs: Texts to embed.
        :type inputs: list[str]
        :param model: Optional embedding model override.
        :type model: str | None
        :return: One :class:`~lauren_ai._transport.Embedding` per input.
        :rtype: list[Embedding]
        """
        return await self._llm.embed(inputs, model=model)


# ---------------------------------------------------------------------------
# LLMModule
# ---------------------------------------------------------------------------


class LLMModule:
    """Factory that creates a ``@module`` providing LLM services.

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
    """

    @classmethod
    def for_root(
        cls,
        config: LLMConfig,
        *,
        transport_override: Any | None = None,
    ) -> type:
        """Create a ``@module`` that provides :class:`LLMService` and
        :class:`EmbedService`.

        :param config: The LLM configuration.
        :type config: LLMConfig
        :param transport_override: Pre-built transport to use instead of the
            one derived from *config*.  Pass a
            :class:`~lauren_ai._transport._mock.MockTransport` here in tests.
        :type transport_override: Any | None
        :return: A ``@module``-decorated class.
        :rtype: type
        """
        transport = _build_transport(config, override=transport_override)
        llm_service = LLMService(transport=transport, config=config)
        embed_service = EmbedService(llm_service=llm_service)

        # Register pre-built singletons via use_value providers so the DI
        # container hands out the same instances on every resolve.
        _llm_service_provider = use_value(
            provide=LLMService, value=llm_service
        )
        _embed_service_provider = use_value(
            provide=EmbedService, value=embed_service
        )
        _config_provider = use_value(
            provide=LLMConfig, value=config
        )

        providers = [
            _llm_service_provider,
            _embed_service_provider,
            _config_provider,
        ]
        exports = [LLMService, EmbedService, LLMConfig]

        # Also try to export the transport under the Transport protocol token
        try:
            from lauren_ai._transport import Transport as _Transport  # noqa: PLC0415

            _transport_provider = use_value(
                provide=_Transport, value=transport
            )
            providers.insert(0, _transport_provider)
            exports.insert(0, _Transport)
        except ImportError:
            pass

        @module(providers=providers, exports=exports)
        class _LLMModule:
            """Auto-generated LLM provider module."""

            transport_instance: Any = transport
            llm_service_instance: LLMService = llm_service
            embed_service_instance: EmbedService = embed_service

        _LLMModule.__name__ = "LLMModule"
        _LLMModule.__qualname__ = "LLMModule"
        return _LLMModule


# ---------------------------------------------------------------------------
# AgentModule
# ---------------------------------------------------------------------------


class AgentModule:
    """Factory that creates a ``@module`` providing the :class:`~lauren_ai._agents._runner.AgentRunner`,
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
    """

    @classmethod
    def for_root(
        cls,
        *,
        agents: list[type],
        tools: list[Any] | None = None,
        imports: Any | None = None,
        signals: Any | None = None,
        memory: Any | None = None,
        conversation_store: Any | None = None,
        config: AgentConfig | None = None,
        tool_cache: Any | None = None,
        knowledge: list[Any] | None = None,
        runner: type | None = None,
        injects: list[type] | None = None,
        export_tools: list[type] | None = None,
    ) -> type:
        """Create a ``@module`` providing the agent runner and all agent instances.

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
        :param memory: Long-term memory store instance.
        :type memory: Any | None
        :param conversation_store: Conversation history store instance.
        :type conversation_store: Any | None
        :param config: Default :class:`~lauren_ai._config.AgentConfig`.
        :type config: AgentConfig | None
        :param tool_cache: Cache backend for tool result caching.
        :type tool_cache: Any | None
        :param knowledge: Knowledge base instances to pre-load into long-term
            memory.
        :type knowledge: list[Any] | None
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
        :return: A ``@module``-decorated class.
        :rtype: type
        """
        import inspect as _inspect  # noqa: PLC0415

        from lauren_ai._agents import USE_TOOLS_META  # noqa: PLC0415
        from lauren_ai._agents._runner import AgentRunnerBase  # noqa: PLC0415
        from lauren_ai._tools import TOOL_META, _add_to_tool_map  # noqa: PLC0415

        if runner is not None:
            # Advanced path: caller supplied an explicit named runner subclass.
            _runner_cls: type = runner
        else:
            # Default path: generate a unique subclass per for_root() call so
            # each AgentModule gets its own DI token.  ``runner: AgentRunner``
            # in a tool or service resolves to this class via the framework's
            # structural Protocol fallback (own-module scan).
            _runner_name = "".join(a.__name__ for a in agents) + "Runner"
            _runner_cls = type(_runner_name, (AgentRunnerBase,), {})

        _captured_tool_cache = tool_cache
        _captured_signals = signals
        _captured_conversation_store = conversation_store

        _fn_tools: list[Any] = []
        _class_tools: list[Any] = []
        _seen_tool_names: set[str] = set()

        def _categorize(tool_item: Any) -> None:
            if tool_item is None:
                return
            # Support generic aliases like HandoffBackTo[SomeAgent]:
            # get_origin() returns the bare class for subscripted generics.
            _origin = get_origin(tool_item)
            _lookup = _origin if (_origin is not None and _inspect.isclass(_origin)) else tool_item
            meta = getattr(_lookup, TOOL_META, None)
            if meta is None:
                logger.warning(
                    "lauren_ai.AgentModule: %r has no @tool() metadata — skipping",
                    tool_item,
                )
                return
            if meta.name in _seen_tool_names:
                return  # deduplicate across shared + per-agent lists
            _seen_tool_names.add(meta.name)
            if _inspect.isclass(_lookup):
                _class_tools.append(tool_item)  # keep alias as DI token
            else:
                _fn_tools.append(tool_item)

        for tool_item in (tools or []):
            _categorize(tool_item)

        for agent_cls in agents:
            for tool_item in getattr(agent_cls, USE_TOOLS_META, ()):
                _categorize(tool_item)

        # ── Initialise eager-tools sentinel and effective scope ───────────────

        _eager_tools: dict | None = None

        # Effective scope: propagate narrowest class-tool scope upward so the
        # AgentRunner is never longer-lived than the tools it holds.
        _effective_scope = Scope.SINGLETON
        for _ct in _class_tools:
            _actual_cls = get_origin(_ct) if not _inspect.isclass(_ct) else _ct
            _ct_meta = getattr(_actual_cls, "__lauren_injectable__", None)
            _ct_scope = getattr(_ct_meta, "scope", Scope.SINGLETON) if _ct_meta else Scope.SINGLETON
            if _ct_scope < _effective_scope:
                _effective_scope = _ct_scope

        # ── Build providers / exports lists ──────────────────────────────────

        providers: list[Any] = []
        exports: list[Any] = []

        # Add class-form tools as DI providers only — they are module-internal
        # and resolved by the AgentRunner factory.  Exporting them would cause
        # ModuleExportViolation when two AgentModule instances share an import
        # chain (e.g. a CRM module importing a Transfer module).
        #
        # Generic aliases (e.g. HandoffBackTo[BankingCRMAgent]) are registered
        # via use_class so each subscripted variant gets its own singleton bound
        # to the alias token without declaring the origin class as a separate
        # provider token (which would cause DuplicateBindingError when two
        # AgentModules both need the same base tool class).
        for cls_tool in _class_tools:
            _alias_origin = get_origin(cls_tool)
            if _alias_origin is not None and _inspect.isclass(_alias_origin):
                providers.append(use_class(provide=cls_tool, use=_alias_origin))
            else:
                providers.append(cls_tool)

        # Register all agent classes as providers and exports so parent modules
        # can inject the resolved agent singleton (e.g. for runner.run(agent, …)).
        for agent_cls in agents:
            providers.append(agent_cls)
            exports.append(agent_cls)

        # AgentRunner: resolve Transport and LLMConfig from the DI graph.
        #
        # Both tokens are provided (and exported) by the @module that
        # LLMModule.for_root() generates.  For them to be visible inside
        # *this* generated module the caller must pass that LLM module via
        # the ``imports`` parameter; without it the two modules are siblings
        # in the application graph, and this module's use_factory cannot see
        # the sibling's exports — a MissingProviderError at startup.
        try:
            from lauren_ai._transport import Transport as _Transport  # noqa: PLC0415

            if _class_tools:
                # Class-form tools need DI resolution.  The runner factory
                # receives the DI-resolved instances positionally:
                #   args = (transport, *class_instances, config)
                _num_class_tools = len(_class_tools)
                _captured_fn_tools = list(_fn_tools)
                _captured_signals_ref = _captured_signals
                _captured_tool_cache_ref = _captured_tool_cache
                _captured_conv_store_ref = _captured_conversation_store

                def _build_runner_with_classes(*args: Any) -> AgentRunner:
                    transport = args[0]
                    class_instances = args[1:1 + _num_class_tools]
                    cfg = args[1 + _num_class_tools]
                    tools: dict = {}
                    for fn_tool in _captured_fn_tools:
                        try:
                            _add_to_tool_map(tools, fn_tool)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "lauren_ai.AgentModule: could not add fn tool %r: %s",
                                fn_tool,
                                exc,
                            )
                    for instance in class_instances:
                        try:
                            _add_to_tool_map(tools, type(instance), instance=instance)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "lauren_ai.AgentModule: could not add class tool instance %r: %s",
                                instance,
                                exc,
                            )
                    return _runner_cls(
                        transport=transport,
                        tools=tools,
                        config=cfg,
                        signals=_captured_signals_ref,
                        cache_backend=_captured_tool_cache_ref,
                        conversation_store=_captured_conv_store_ref,
                    )

                _runner_provider = use_factory(
                    provide=_runner_cls,
                    factory=_build_runner_with_classes,
                    injects=[_Transport, *_class_tools, LLMConfig],
                    scope=_effective_scope,
                )
            else:
                # No class-form tools — build the tools dict eagerly.
                _eager_tools = {}
                for fn_tool in _fn_tools:
                    try:
                        _add_to_tool_map(_eager_tools, fn_tool)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "lauren_ai.AgentModule: could not add fn tool %r: %s",
                            fn_tool,
                            exc,
                        )
                _captured_eager_tools = _eager_tools
                _captured_signals_ref = _captured_signals
                _captured_tool_cache_ref = _captured_tool_cache
                _captured_conv_store_ref = _captured_conversation_store
                _captured_runner_cls = _runner_cls

                _runner_provider = use_factory(
                    provide=_runner_cls,
                    factory=lambda transport, cfg: _captured_runner_cls(
                        transport=transport,
                        tools=_captured_eager_tools,
                        config=cfg,
                        signals=_captured_signals_ref,
                        cache_backend=_captured_tool_cache_ref,
                        conversation_store=_captured_conv_store_ref,
                    ),
                    injects=[_Transport, LLMConfig],
                    scope=_effective_scope,
                )

            providers.append(_runner_provider)
            exports.append(_runner_cls)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "lauren_ai.AgentModule: could not build AgentRunner use_factory "
                "provider (Transport or LLMConfig token unavailable at definition "
                "time): %s.  AgentRunner will need to be wired manually.",
                exc,
            )

        # Normalise the imports argument: None → [], single class → [cls], list → list.
        if imports is None:
            _imports: list[Any] = []
        elif isinstance(imports, (list, tuple)):
            _imports = list(imports)
        else:
            _imports = [imports]

        # Register any additional provider classes from ``injects=``.  These are
        # not exported; they are purely internal to this module.
        for _inj_cls in (injects or []):
            if _inj_cls not in providers:
                providers.append(_inj_cls)

        # Register and export delegation tools.  These tools are providers of
        # THIS module (so they can receive this module's runner via DI) but are
        # NOT included in the runner's tool map — they are consumed by OTHER
        # modules' runners.  Caller must NOT include them in `tools=` as well;
        # doing so creates a circular dependency (runner → tool → runner).
        _captured_export_tools: list[type] = list(export_tools or [])
        for _et in _captured_export_tools:
            if _et not in providers:
                providers.append(_et)
            if _et not in exports:
                exports.append(_et)

        @module(imports=_imports, providers=providers, exports=exports)
        class _AgentModule:
            """Auto-generated agent provider module."""

            # Exposes the pre-built tools dict when only function-form tools
            # are used (eager path).  None when class-form tools are present
            # (the dict is built inside the AgentRunner factory).
            tools_instance: dict | None = _eager_tools
            agent_classes: list[type] = list(agents)
            # The concrete runner class for this module.  Useful for
            # introspection; DI injection should use ``runner: AgentRunner``
            # (single-module scope) or the named subclass (multi-module scope).
            runner_class: type = _runner_cls
            exported_tools: list[type] = _captured_export_tools

        _AgentModule.__name__ = "".join([agent.__name__ for agent in agents]) + "Module"
        _AgentModule.__qualname__ = "And".join([agent.__name__ for agent in agents]) + "Module"
        return _AgentModule

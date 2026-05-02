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

Both module factories degrade gracefully when ``lauren`` is not importable.
In that case the returned class carries the service instances as class
attributes, and callers wire dependencies manually.
"""

from __future__ import annotations

__all__ = [
    "LLMModule",
    "AgentModule",
    "LLMService",
    "EmbedService",
]

import logging
from typing import Any, AsyncIterator, TypeVar

T = TypeVar("T")

from lauren_ai._config import AgentConfig, LLMConfig
from lauren_ai._exceptions import AgentConfigError
from lauren_ai._transport import Completion, CompletionChunk, Embedding, Message, TokenUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional lauren imports
# ---------------------------------------------------------------------------

try:
    from lauren import Scope, injectable, module, use_value, use_factory  # type: ignore[import]

    _LAUREN_AVAILABLE = True
except ImportError:
    _LAUREN_AVAILABLE = False

    def injectable(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        def decorator(cls: Any) -> Any:
            return cls
        return decorator

    def module(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        def decorator(cls: Any) -> Any:
            return cls
        return decorator

    def use_value(**kwargs: Any) -> Any:  # type: ignore[misc]
        return kwargs

    def use_factory(**kwargs: Any) -> Any:  # type: ignore[misc]
        return kwargs

    class Scope:  # type: ignore[no-redef]
        SINGLETON = "singleton"


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

    def with_structured_output(self, model_cls: type[T]) -> "StructuredLLM[T]":
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

        if _LAUREN_AVAILABLE:
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

            @module(providers=providers, exports=exports)  # type: ignore[misc]
            class _LLMModule:
                """Auto-generated LLM provider module."""

                transport_instance: Any = transport
                llm_service_instance: LLMService = llm_service
                embed_service_instance: EmbedService = embed_service

            _LLMModule.__name__ = "LLMModule"
            _LLMModule.__qualname__ = "LLMModule"
            return _LLMModule

        else:
            # lauren not available — plain class with instances as attributes
            class _LLMModuleFallback:
                """LLM module (lauren not available; wire services manually)."""

                transport_instance: Any = transport
                llm_service_instance: LLMService = llm_service
                embed_service_instance: EmbedService = embed_service

            _LLMModuleFallback.__name__ = "LLMModule"
            return _LLMModuleFallback  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# AgentModule
# ---------------------------------------------------------------------------


class AgentModule:
    """Factory that creates a ``@module`` providing the :class:`~lauren_ai._agents._runner.AgentRunner`,
    :class:`~lauren_ai._tools._registry.ToolRegistry`, and all registered agent
    class instances.

    Usage::

        AIAgentModule = AgentModule.for_root(
            agents=[ResearchAgent, SummarizerAgent],
            tools=[WebSearchTool, CodeExecutionTool],
            config=AgentConfig(max_turns=5, max_cost_usd=0.50),
        )
    """

    @classmethod
    def for_root(
        cls,
        *,
        agents: list[type],
        tools: list[Any] | None = None,
        memory: Any | None = None,
        conversation_store: Any | None = None,
        config: AgentConfig | None = None,
        tool_cache: Any | None = None,
        knowledge: list[Any] | None = None,
    ) -> type:
        """Create a ``@module`` providing the agent runner and all agent instances.

        :param agents: ``@agent()``-decorated classes to register.
        :type agents: list[type]
        :param tools: Shared tools available to all agents (supplementing
            per-agent ``@use_tools()`` registrations).
        :type tools: list[Any] | None
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
        :return: A ``@module``-decorated class.
        :rtype: type
        """
        from lauren_ai._tools._registry import ToolRegistry  # noqa: PLC0415
        from lauren_ai._agents._runner import AgentRunner  # noqa: PLC0415
        from lauren_ai._agents import USE_TOOLS_META  # noqa: PLC0415

        # Build and populate the shared ToolRegistry
        registry = ToolRegistry()

        def _safe_register(tool_item: Any, label: str) -> None:
            if tool_item is None:
                return
            from lauren_ai._tools import TOOL_META  # noqa: PLC0415

            tool_meta = getattr(tool_item, TOOL_META, None)
            if tool_meta is None:
                logger.warning(
                    "lauren_ai.AgentModule: %r has no @tool() metadata — skipping",
                    tool_item,
                )
                return
            if tool_meta.name in registry:
                return  # Already registered; skip silently
            try:
                registry.register(tool_item)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "lauren_ai.AgentModule: could not register %s %r: %s",
                    label,
                    tool_item,
                    exc,
                )

        # Register shared tools
        for tool_item in (tools or []):
            _safe_register(tool_item, "shared tool")

        # Register per-agent tools (from @use_tools())
        for agent_cls in agents:
            for tool_item in getattr(agent_cls, USE_TOOLS_META, ()):
                _safe_register(tool_item, f"tool for {agent_cls.__name__!r}")

        _captured_tool_cache = tool_cache

        if _LAUREN_AVAILABLE:
            # Use use_value for the registry (already built)
            _registry_provider = use_value(provide=ToolRegistry, value=registry)

            providers: list[Any] = [_registry_provider]
            exports: list[Any] = [ToolRegistry]

            # Register all agent classes as providers (they are already
            # @injectable(scope=Scope.SINGLETON) from @agent())
            for agent_cls in agents:
                providers.append(agent_cls)
                exports.append(agent_cls)

            # AgentRunner: resolve Transport and LLMConfig from the DI graph.
            # LLMModule (imported by the app module) must provide both.
            try:
                from lauren_ai._transport import Transport as _Transport  # noqa: PLC0415

                _captured_registry = registry
                _captured_tc = _captured_tool_cache

                _runner_provider = use_factory(
                    provide=AgentRunner,
                    factory=lambda transport, reg, cfg: AgentRunner(
                        transport=transport,
                        registry=reg,
                        config=cfg,
                        signals=None,
                        cache_backend=_captured_tc,
                    ),
                    inject=[_Transport, ToolRegistry, LLMConfig],
                    scope=Scope.SINGLETON,
                )
                providers.append(_runner_provider)
                exports.append(AgentRunner)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "lauren_ai.AgentModule: could not build AgentRunner use_factory "
                    "provider (Transport or LLMConfig token unavailable at definition "
                    "time): %s.  AgentRunner will need to be wired manually.",
                    exc,
                )

            @module(providers=providers, exports=exports)  # type: ignore[misc]
            class _AgentModule:
                """Auto-generated agent provider module."""

                registry_instance: ToolRegistry = registry
                agent_classes: list[type] = list(agents)

            _AgentModule.__name__ = "AgentModule"
            _AgentModule.__qualname__ = "AgentModule"
            return _AgentModule

        else:
            # lauren not available — plain class
            class _AgentModuleFallback:
                """Agent module (lauren not available; wire manually)."""

                registry_instance: ToolRegistry = registry  # type: ignore[assignment]
                agent_classes: list[type] = list(agents)

            _AgentModuleFallback.__name__ = "AgentModule"
            return _AgentModuleFallback  # type: ignore[return-value]

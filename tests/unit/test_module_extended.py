"""Extended unit tests for _module.py — LLMService, EmbedService, LLMModule, AgentModule."""

from __future__ import annotations

import pytest

from lauren_ai._config import AgentConfig, LLMConfig
from lauren_ai._exceptions import AgentConfigError
from lauren_ai._module import AgentModule, EmbedService, LLMModule, LLMService, _build_transport
from lauren_ai._transport import Completion, Message, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# _build_transport tests
# ---------------------------------------------------------------------------


class TestBuildTransport:
    def test_override_returned_directly(self):
        mock = MockTransport()
        cfg, _ = LLMConfig.for_testing()  # for_testing returns (config, mock)
        result = _build_transport(cfg, override=mock)
        assert result is mock

    def test_unknown_provider_raises(self):
        cfg = LLMConfig(
            provider="unknown_provider",
            model="some-model",
            api_key="key",
        )
        with pytest.raises(AgentConfigError) as exc_info:
            _build_transport(cfg)
        assert "unknown_provider" in str(exc_info.value).lower()

    def test_anthropic_provider_builds_transport(self):
        cfg = LLMConfig.for_anthropic(model="claude-haiku-4-5", api_key="test-key")
        # Should not raise (just creates transport without connecting)
        transport = _build_transport(cfg)
        assert transport is not None


# ---------------------------------------------------------------------------
# LLMService tests
# ---------------------------------------------------------------------------


class TestLLMService:
    def _make_service(self) -> tuple[LLMService, MockTransport]:
        mock = MockTransport()
        cfg, _ = LLMConfig.for_testing()  # for_testing returns (config, mock)
        service = LLMService(transport=mock, config=cfg)
        return service, mock

    @pytest.mark.asyncio
    async def test_complete_delegates_to_transport(self):
        service, mock = self._make_service()
        completion = Completion(
            id="c1",
            model="mock",
            content="Hello!",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        mock.queue_response(completion)

        result = await service.complete([Message.user("Hi")])
        assert result.content == "Hello!"
        assert len(mock.calls) == 1

    @pytest.mark.asyncio
    async def test_complete_uses_config_defaults(self):
        mock = MockTransport()
        cfg = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key="key",
            max_tokens=1024,
            temperature=0.5,
        )
        service = LLMService(transport=mock, config=cfg)
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="Hi",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            )
        )
        await service.complete([Message.user("Hello")])
        call = mock.calls[0]
        assert call.model == "claude-haiku-4-5"
        assert call.max_tokens == 1024
        assert call.temperature == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_complete_with_overrides(self):
        mock = MockTransport()
        cfg = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key="key",
            max_tokens=512,
            temperature=1.0,
        )
        service = LLMService(transport=mock, config=cfg)
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="Hi",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            )
        )
        await service.complete(
            [Message.user("Hello")],
            model="claude-opus-4-6",
            max_tokens=2048,
            temperature=0.0,
        )
        call = mock.calls[0]
        assert call.model == "claude-opus-4-6"
        assert call.max_tokens == 2048
        assert call.temperature == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self):
        service, mock = self._make_service()
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="OK",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=2),
            )
        )
        await service.complete([Message.user("Hi")], system="You are helpful.")
        call = mock.calls[0]
        assert call.system == "You are helpful."

    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        service, mock = self._make_service()
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="OK",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=2),
            )
        )
        fake_tools = [{"name": "search", "description": "Search", "input_schema": {}}]
        await service.complete([Message.user("Hi")], tools=fake_tools)
        call = mock.calls[0]
        assert call.tools == fake_tools

    @pytest.mark.asyncio
    async def test_complete_stream_alias(self):
        service, mock = self._make_service()
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="Streamed",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=2),
            )
        )
        result = await service.complete_stream([Message.user("Hi")])
        # Should return an async iterator
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    async def test_embed_delegates_to_transport(self):
        service, mock = self._make_service()
        from lauren_ai._transport import Embedding

        mock.queue_embed([Embedding(index=0, vector=[0.1, 0.2, 0.3])])
        embeddings = await service.embed(["hello world"])
        assert len(embeddings) == 1
        assert embeddings[0].vector == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_uses_embed_model(self):
        mock = MockTransport()
        cfg = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key="key",
            embed_model="text-embedding-3-small",
        )
        service = LLMService(transport=mock, config=cfg)
        from lauren_ai._transport import Embedding

        mock.queue_embed([Embedding(index=0, vector=[0.0])])
        await service.embed(["test"])
        # embed_model should be used (not main model)
        # (MockTransport doesn't record embed calls, just checking no exception)

    @pytest.mark.asyncio
    async def test_embed_fallback_to_main_model(self):
        mock = MockTransport()
        cfg = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key="key",
            embed_model=None,
        )
        service = LLMService(transport=mock, config=cfg)
        # Should use config.model as embed model
        embeddings = await service.embed(["test"])
        assert len(embeddings) == 1

    @pytest.mark.asyncio
    async def test_count_tokens_via_transport(self):
        service, mock = self._make_service()
        tokens = await service.count_tokens([Message.user("Hello world")])
        assert isinstance(tokens, int)
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_count_tokens_fallback_heuristic(self):
        """Test the heuristic fallback when transport has no count_tokens method."""

        class MinimalTransport:
            async def complete(self, *args, **kwargs):
                pass

            async def embed(self, *args, **kwargs):
                return []

        cfg, _ = LLMConfig.for_testing()
        service = LLMService(transport=MinimalTransport(), config=cfg)
        tokens = await service.count_tokens([Message.user("Hello world this is a test")])
        # Heuristic: len("Hello world this is a test") // 4 = 6
        assert isinstance(tokens, int)
        assert tokens >= 1

    @pytest.mark.asyncio
    async def test_count_tokens_list_content(self):
        """Test token counting with list-content messages."""
        from lauren_ai._transport import ContentBlock

        class MinimalTransport:
            async def complete(self, *args, **kwargs):
                pass

            async def embed(self, *args, **kwargs):
                return []

        cfg, _ = LLMConfig.for_testing()
        service = LLMService(transport=MinimalTransport(), config=cfg)
        msgs = [Message(role="user", content=[ContentBlock(type="text", text="Hello there")])]
        tokens = await service.count_tokens(msgs)
        assert isinstance(tokens, int)


# ---------------------------------------------------------------------------
# EmbedService tests
# ---------------------------------------------------------------------------


class TestEmbedService:
    @pytest.mark.asyncio
    async def test_embed_delegates_to_llm_service(self):
        mock = MockTransport()
        cfg, _ = LLMConfig.for_testing()
        llm = LLMService(transport=mock, config=cfg)
        embed_svc = EmbedService(llm_service=llm)
        from lauren_ai._transport import Embedding

        mock.queue_embed([Embedding(index=0, vector=[1.0, 2.0])])
        result = await embed_svc.embed(["test"])
        assert len(result) == 1
        assert result[0].vector == [1.0, 2.0]


# ---------------------------------------------------------------------------
# LLMModule tests
# ---------------------------------------------------------------------------


class TestBuildTransportExtended:
    def test_openai_transport_builds_successfully(self):
        """Test that OpenAI provider builds successfully when openai is installed."""
        cfg = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="key",
        )
        try:
            transport = _build_transport(cfg)
            assert transport is not None
        except AgentConfigError as exc:
            # If openai is not installed, expect AgentConfigError
            assert "openai" in str(exc).lower()

    def test_ollama_import_error_raises_agent_config_error(self):
        """Test that missing httpx package raises AgentConfigError."""
        cfg = LLMConfig(
            provider="ollama",
            model="llama3",
            api_key="",
        )
        try:
            import httpx  # noqa: F401

            # httpx present — OllamaTransport can be built, test different path
            # or just skip if we can't trigger the ImportError
            pytest.skip("httpx is installed, can't test missing-package path")
        except ImportError:
            with pytest.raises(AgentConfigError) as exc_info:
                _build_transport(cfg)
            assert "httpx" in str(exc_info.value).lower()

    def test_litellm_import_error_raises_agent_config_error(self):
        """Test that missing litellm package raises AgentConfigError."""
        cfg = LLMConfig(
            provider="litellm",
            model="gpt-3.5-turbo",
            api_key="key",
        )
        try:
            import litellm  # noqa: F401

            pytest.skip("litellm is installed, can't test missing-package path")
        except ImportError:
            with pytest.raises(AgentConfigError) as exc_info:
                _build_transport(cfg)
            assert "litellm" in str(exc_info.value).lower()


class TestLLMServiceExtended:
    def _make_service(self) -> tuple[LLMService, MockTransport]:
        mock = MockTransport()
        cfg, _ = LLMConfig.for_testing()
        service = LLMService(transport=mock, config=cfg)
        return service, mock

    @pytest.mark.asyncio
    async def test_complete_with_tool_choice(self):
        service, mock = self._make_service()
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="OK",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=2),
            )
        )
        # Use a simple dict as tool_choice to trigger the tool_choice kwarg path
        from lauren_ai._transport import ToolChoice

        await service.complete(
            [Message.user("Hi")],
            tool_choice=ToolChoice(type="auto"),
        )
        call = mock.calls[0]
        assert call.tool_choice is not None


class TestLLMModule:
    def test_for_root_with_mock_returns_class(self):
        cfg, mock = LLMConfig.for_testing()
        cls = LLMModule.for_root(cfg, transport_override=mock)
        assert cls is not None
        assert hasattr(cls, "llm_service_instance")
        assert hasattr(cls, "embed_service_instance")
        assert isinstance(cls.llm_service_instance, LLMService)
        assert isinstance(cls.embed_service_instance, EmbedService)

    def test_for_root_module_name(self):
        cfg, mock = LLMConfig.for_testing()
        cls = LLMModule.for_root(cfg, transport_override=mock)
        assert cls.__name__ == "LLMModule"

    def test_for_root_transport_instance_set(self):
        cfg, mock = LLMConfig.for_testing()
        cls = LLMModule.for_root(cfg, transport_override=mock)
        assert cls.transport_instance is mock


# ---------------------------------------------------------------------------
# AgentModule tests
# ---------------------------------------------------------------------------


class TestAgentModule:
    def test_for_root_with_agents(self):
        from lauren_ai._agents import agent, use_tools
        from lauren_ai._tools import tool

        @tool()
        async def my_tool(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        @agent(model="claude-haiku-4-5")
        @use_tools(my_tool)
        class TestAgent:
            """A test agent."""

        cls = AgentModule.for_root(agents=[TestAgent])
        assert cls is not None
        assert cls.__name__ == "TestAgentModule"

    def test_for_root_has_registry(self):
        from lauren_ai._agents import agent

        @agent()
        class SimpleAgent:
            """Simple."""

        cls = AgentModule.for_root(agents=[SimpleAgent])
        assert isinstance(cls.tools_instance, dict)

    def test_for_root_agent_classes_stored(self):
        from lauren_ai._agents import agent

        @agent()
        class AgentA:
            """A."""

        @agent()
        class AgentB:
            """B."""

        cls = AgentModule.for_root(agents=[AgentA, AgentB])
        assert AgentA in cls.agent_classes
        assert AgentB in cls.agent_classes

    def test_for_root_with_shared_tools(self):
        from lauren_ai._agents import agent
        from lauren_ai._tools import tool

        @tool()
        async def shared_tool(x: str) -> str:
            """Shared. Args: x: Input."""
            return x

        @agent()
        class TestAgent2:
            """Agent 2."""

        cls = AgentModule.for_root(agents=[TestAgent2], tools=[shared_tool])
        assert cls is not None

    def test_for_root_skips_none_tools(self):
        from lauren_ai._agents import agent

        @agent()
        class TestAgent3:
            """Agent 3."""

        # Passing None in tools should not raise
        cls = AgentModule.for_root(agents=[TestAgent3], tools=[None])
        assert cls is not None

    def test_for_root_warns_on_non_tool_item(self, caplog):
        from lauren_ai._agents import agent

        @agent()
        class TestAgent4:
            """Agent 4."""

        def not_a_tool():
            pass

        # Should log a warning but not raise
        cls = AgentModule.for_root(agents=[TestAgent4], tools=[not_a_tool])
        assert cls is not None

    def test_for_root_config_provided(self):
        from lauren_ai._agents import agent

        @agent()
        class AgentWithConfig:
            """Agent."""

        config = AgentConfig(max_turns=3, max_cost_usd=1.0)
        cls = AgentModule.for_root(agents=[AgentWithConfig], config=config)
        assert cls is not None

    def test_for_root_duplicate_tool_skipped(self):
        """A tool registered twice should not raise — second registration is silently skipped."""
        from lauren_ai._agents import agent, use_tools
        from lauren_ai._tools import tool

        @tool()
        async def shared_tool(x: str) -> str:
            """A shared tool. Args: x: Input."""
            return x

        @agent()
        @use_tools(shared_tool)
        class Agent1:
            """Agent 1."""

        # Pass the same tool in both shared tools AND per-agent tools
        cls = AgentModule.for_root(agents=[Agent1], tools=[shared_tool])
        assert cls is not None

    def test_for_root_tool_register_exception_logs(self, caplog):
        """If _add_to_tool_map raises (e.g. name collision), it should log a warning and continue."""
        import logging

        from lauren_ai._agents import agent
        from lauren_ai._tools import tool

        @tool()
        async def collision_tool(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        @tool(name="collision_tool")
        async def collision_tool2(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        @agent()
        class AgentX:
            """A."""

        with caplog.at_level(logging.WARNING, logger="lauren_ai._module"):
            cls = AgentModule.for_root(agents=[AgentX], tools=[collision_tool, collision_tool2])
        assert cls is not None

    def test_for_root_with_memory_and_conversation(self):
        from lauren_ai._agents import agent

        @agent()
        class AgentY:
            """Y."""

        class FakeMemory:
            pass

        class FakeConvStore:
            pass

        cls = AgentModule.for_root(
            agents=[AgentY],
            memory=FakeMemory(),
            conversation_store=FakeConvStore(),
        )
        assert cls is not None

    def test_for_root_with_tool_cache(self):
        from lauren_ai._agents import agent
        from lauren_ai._tools._executor import InMemoryCacheBackend

        @agent()
        class AgentZ:
            """Z."""

        cache = InMemoryCacheBackend()
        cls = AgentModule.for_root(agents=[AgentZ], tool_cache=cache)
        assert cls is not None

    def test_for_root_runner_class_exported_under_subclass_token(self):
        """runner=MyCustomRunner causes the module to export the subclass."""
        from lauren_ai._agents import agent
        from lauren_ai._agents._runner import AgentRunner, AgentRunnerBase

        class MyCustomRunner(AgentRunnerBase):
            """Marker subclass used as a distinct DI token."""

        @agent()
        class AgentForCustomRunner:
            """Agent."""

        cls = AgentModule.for_root(
            agents=[AgentForCustomRunner],
            runner=MyCustomRunner,
        )
        # The module's exports should contain MyCustomRunner, not the Protocol.
        assert MyCustomRunner in cls.__lauren_module__.exports
        assert AgentRunner not in cls.__lauren_module__.exports

    def test_for_root_runner_class_default_exports_dynamic_subclass(self):
        """Without runner= the module exports a dynamic AgentRunnerBase subclass."""
        from lauren_ai._agents import agent
        from lauren_ai._agents._runner import AgentRunner, AgentRunnerBase

        @agent()
        class AgentForDefaultRunner:
            """Agent."""

        cls = AgentModule.for_root(agents=[AgentForDefaultRunner])
        # A dynamic subclass of AgentRunnerBase is exported, not the Protocol itself.
        exported = cls.__lauren_module__.exports
        assert AgentRunner not in exported
        assert any(
            isinstance(e, type) and issubclass(e, AgentRunnerBase) and e is not AgentRunnerBase
            for e in exported
        )

    def test_runner_param_exported_under_subclass_token(self):
        """runner=SubClass exports that subclass, not the AgentRunner Protocol."""
        from lauren_ai._agents import agent
        from lauren_ai._agents._runner import AgentRunner, AgentRunnerBase

        class MyRunner(AgentRunnerBase):
            """Marker subclass."""

        @agent()
        class A:
            """A."""

        cls = AgentModule.for_root(agents=[A], runner=MyRunner)
        assert MyRunner in cls.__lauren_module__.exports
        assert AgentRunner not in cls.__lauren_module__.exports

    def test_runner_none_exports_dynamic_subclass(self):
        """runner=None (default) auto-generates a dynamic AgentRunnerBase subclass."""
        from lauren_ai._agents import agent
        from lauren_ai._agents._runner import AgentRunner, AgentRunnerBase

        @agent()
        class C:
            """C."""

        cls = AgentModule.for_root(agents=[C])
        exported = cls.__lauren_module__.exports
        assert AgentRunner not in exported
        assert any(
            isinstance(e, type) and issubclass(e, AgentRunnerBase) and e is not AgentRunnerBase
            for e in exported
        )

    def test_injects_adds_extra_providers(self):
        """injects=[Extra] registers the class as an additional provider (not exported)."""
        from lauren_ai._agents import agent

        class ExtraService:
            """An extra singleton for tools to use."""

        @agent()
        class D:
            """D."""

        cls = AgentModule.for_root(agents=[D], injects=[ExtraService])
        # injects= classes are providers, not exports
        provider_tokens = [
            getattr(p, "provide", p) if hasattr(p, "provide") else p
            for p in cls.__lauren_module__.providers
        ]
        assert ExtraService in provider_tokens or ExtraService in cls.__lauren_module__.providers
        assert ExtraService not in cls.__lauren_module__.exports


# ---------------------------------------------------------------------------
# Generic-alias tool support
# ---------------------------------------------------------------------------


class TestAgentModuleGenericAliasTools:
    """AgentModule.for_root(tools=[SomeTool[X]]) treats each alias as a distinct token."""

    def test_generic_alias_tool_not_skipped(self):
        """A @tool() Generic[T] subclass passed as MyTool[SomeClass] must appear in providers."""
        from typing import Generic
        from typing import TypeVar as _TypeVar

        from lauren._di.custom import CustomProvider

        from lauren_ai._agents import agent
        from lauren_ai._tools import tool

        _T = _TypeVar("_T")

        @tool()
        class DirectionTool(Generic[_T]):
            """Return a fixed direction.

            Args:
                summary: The summary.
            """

            async def run(self, summary: str) -> dict:
                return {"ok": True}

        @agent()
        class _Agent:
            """Agent."""

        class TargetA:
            """A."""

        alias = DirectionTool[TargetA]
        mod_cls = AgentModule.for_root(agents=[_Agent], tools=[alias])

        providers = mod_cls.__lauren_module__.providers
        found = any(isinstance(p, CustomProvider) and p.provide == alias for p in providers)
        assert found, f"Alias {alias!r} not found in module providers: {providers!r}"

    def test_generic_alias_two_aliases_are_distinct_tokens(self):
        """MyTool[ClassA] and MyTool[ClassB] in separate modules produce distinct DI tokens."""
        from typing import Generic
        from typing import TypeVar as _TypeVar

        from lauren._di.custom import CustomProvider

        from lauren_ai._agents import agent
        from lauren_ai._tools import tool

        _T = _TypeVar("_T")

        @tool()
        class FlexTool(Generic[_T]):
            """A flexible tool.

            Args:
                msg: The message.
            """

            async def run(self, msg: str) -> dict:
                return {"msg": msg}

        @agent()
        class _AgentA:
            """Agent A."""

        @agent()
        class _AgentB:
            """Agent B."""

        class MarkerA:
            """A."""

        class MarkerB:
            """B."""

        mod_a = AgentModule.for_root(agents=[_AgentA], tools=[FlexTool[MarkerA]])
        mod_b = AgentModule.for_root(agents=[_AgentB], tools=[FlexTool[MarkerB]])

        def _alias_tokens(mod):
            return {
                p.provide for p in mod.__lauren_module__.providers if isinstance(p, CustomProvider)
            }

        tokens_a = _alias_tokens(mod_a)
        tokens_b = _alias_tokens(mod_b)
        assert FlexTool[MarkerA] in tokens_a
        assert FlexTool[MarkerB] in tokens_b
        assert FlexTool[MarkerA] != FlexTool[MarkerB]
        assert tokens_a.isdisjoint(tokens_b)

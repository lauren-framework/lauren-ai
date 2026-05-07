"""Integration tests for the ``@use_knowledge_sources(...)`` decorator.

KB visibility is **opt-in**: an agent without the decorator has
``meta.knowledge_source_filter is None`` and sees **no** KB tools, even
when its module declares ``knowledge=[…]``.  An agent with the decorator
sees only the listed sources.

Coverage:
    * Filter attaches only listed sources to the schema.
    * No decorator ⇒ no KB tools (the new opt-in semantics).
    * Validation: source name not declared in the module's ``knowledge=``
      raises ``DecoratorUsageError`` at module-build time.
    * Strict-inheritance: a subclass that inherits the metadata without
      redeclaring raises ``MetadataInheritanceError``.
    * KnowledgeSource is registered as a DI provider per-source via
      ``use_value`` — consumers can ``Inject(KnowledgeSource)`` (or
      a subclass) and get the configured instance.
"""

from __future__ import annotations

import pytest

from lauren import LaurenFactory, module
from lauren.exceptions import MetadataInheritanceError
from lauren_ai import AgentModule, LLMModule, agent, use_knowledge_sources
from lauren_ai._agents import AGENT_META
from lauren_ai._config import LLMConfig
from lauren_ai._exceptions import DecoratorUsageError
from lauren_ai._knowledge import KnowledgeBase, KnowledgeSource, TextLoader
from lauren_ai._memory._vector import InMemoryVectorStore


async def _populated_kb(text: str = "Hello world.") -> KnowledgeBase:
    kb = KnowledgeBase(store=InMemoryVectorStore())
    await kb.load(TextLoader(text, is_file=False))
    return kb


async def _resolve(app, token):
    return await app._container.resolve(token)


class TestKnowledgeSourceFilter:
    @pytest.mark.asyncio
    async def test_no_decorator_means_no_kb_tools(self):
        """Without ``@use_knowledge_sources``, the agent sees zero KB tools."""
        kb = await _populated_kb()
        ks = KnowledgeSource(kb=kb, tool_name="search_x")

        @agent(model=None)
        class NoOptInAgent: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(
            agents=[NoOptInAgent],
            imports=[LLMProv],
            knowledge=[ks],
        )

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        app = LaurenFactory.create(AppMod)
        runner = await _resolve(app, AIMod.runner_class)
        meta = getattr(NoOptInAgent, AGENT_META)
        assert meta.knowledge_source_filter is None

        names = [s["name"] for s in runner._get_tool_schemas(meta)]
        assert "search_x" not in names

    @pytest.mark.asyncio
    async def test_two_agents_same_module_have_independent_kb_visibility(self):
        """One agent opts in, its sibling does not — each sees only its own set.

        This is the canonical real-world scenario: two agents share a module
        (and therefore a runner), but only one of them needs the KB tool.
        The runner must consult each agent's ``knowledge_source_filter``
        independently rather than applying a global setting.
        """
        kb_x = await _populated_kb("Topic X.")
        ks_x = KnowledgeSource(kb=kb_x, tool_name="search_x")

        @use_knowledge_sources(ks_x)
        @agent(model=None)
        class OptInAgent: ...

        @agent(model=None)
        class OptOutAgent: ...  # no @use_knowledge_sources

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(
            agents=[OptInAgent, OptOutAgent],
            imports=[LLMProv],
            knowledge=[ks_x],
        )

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        app = LaurenFactory.create(AppMod)
        runner = await _resolve(app, AIMod.runner_class)

        optin_meta = getattr(OptInAgent, AGENT_META)
        optout_meta = getattr(OptOutAgent, AGENT_META)

        optin_names = [s["name"] for s in runner._get_tool_schemas(optin_meta)]
        optout_names = [s["name"] for s in runner._get_tool_schemas(optout_meta)]

        assert "search_x" in optin_names, "OptInAgent must see search_x"
        assert "search_x" not in optout_names, "OptOutAgent must NOT see search_x"

    @pytest.mark.asyncio
    async def test_decorator_attaches_only_listed_sources(self):
        kb_x = await _populated_kb("Topic X.")
        kb_y = await _populated_kb("Topic Y.")
        ks_x = KnowledgeSource(kb=kb_x, tool_name="search_x")
        ks_y = KnowledgeSource(kb=kb_y, tool_name="search_y")

        @use_knowledge_sources(ks_x)  # only X
        @agent(model=None)
        class OnlyXAgent: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(
            agents=[OnlyXAgent],
            imports=[LLMProv],
            knowledge=[ks_x, ks_y],  # both at module level
        )

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        app = LaurenFactory.create(AppMod)
        runner = await _resolve(app, AIMod.runner_class)
        meta = getattr(OnlyXAgent, AGENT_META)
        names = [s["name"] for s in runner._get_tool_schemas(meta)]
        assert "search_x" in names
        assert "search_y" not in names


class TestKnowledgeSourceValidation:
    def test_unknown_source_name_raises(self):
        """``@use_knowledge_sources(KS)`` referencing a source NOT in the
        module's ``knowledge=`` raises at module-build time."""
        kb_in = KnowledgeBase(store=InMemoryVectorStore())
        kb_out = KnowledgeBase(store=InMemoryVectorStore())
        ks_in = KnowledgeSource(kb=kb_in, tool_name="search_in")
        ks_out = KnowledgeSource(kb=kb_out, tool_name="search_out")

        @use_knowledge_sources(ks_out)  # not declared at module level
        @agent(model=None)
        class StrayAgent: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)

        with pytest.raises(DecoratorUsageError, match="search_out"):
            AgentModule.for_root(
                agents=[StrayAgent],
                imports=[LLMProv],
                knowledge=[ks_in],
            )

    def test_empty_decorator_call_raises(self):
        with pytest.raises(DecoratorUsageError, match="at least one"):
            use_knowledge_sources()  # type: ignore[call-arg]

    def test_inherited_kb_sources_without_redeclare_raises(self):
        """Strict-inheritance — mirrors lauren-framework's golden rule #3."""
        kb = KnowledgeBase(store=InMemoryVectorStore())
        ks = KnowledgeSource(kb=kb, tool_name="search_inherit")

        @use_knowledge_sources(ks)
        @agent(model=None)
        class ParentAgent:
            """."""

        @agent(model=None)
        class ChildAgent(ParentAgent):
            """Inherits @use_knowledge_sources without redeclaring — forbidden."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)

        with pytest.raises(MetadataInheritanceError, match="inherits @use_knowledge_sources"):
            AgentModule.for_root(
                agents=[ChildAgent],
                imports=[LLMProv],
                knowledge=[ks],
            )


class TestKnowledgeSourceDIRegistration:
    @pytest.mark.asyncio
    async def test_knowledge_source_resolvable_from_di(self):
        """``KnowledgeSource`` is ``@injectable(SINGLETON)`` and ``for_root``
        registers each instance via ``use_value`` so consumers can inject it.
        """
        kb = await _populated_kb()
        ks = KnowledgeSource(kb=kb, tool_name="search_di")

        @use_knowledge_sources(ks)
        @agent(model=None)
        class DIAgent: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(
            agents=[DIAgent],
            imports=[LLMProv],
            knowledge=[ks],
        )

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        app = LaurenFactory.create(AppMod)
        resolved = await _resolve(app, KnowledgeSource)
        assert resolved is ks

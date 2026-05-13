"""Integration tests for the ``AgentRunner[X]`` generic-DI mechanism.

``AgentRunner[X]`` is a real cached subclass of an internal marker class
(:class:`_AgentRunnerParam`) — *not* a Protocol subclass.  This decouples
parameterized tokens from the structural-Protocol scan that resolves bare
``runner: AgentRunner`` injections.

Coverage:
    * Subscript returns a cached real class — ``AgentRunner[X] is AgentRunner[X]``.
    * Subscript validates that ``X`` is ``@agent``-decorated; otherwise raises
      ``TypeError`` at construction time.
    * ``await container.resolve(AgentRunner[Agent])`` returns the module's
      runner instance.
    * Two agents in the same module share the runner instance — correct
      semantics, since the runner consults each agent's AgentMeta on every
      ``run()`` call.
    * Bare ``runner: AgentRunner`` still works for in-module DI without
      ``ProtocolAmbiguityError``.
"""

from __future__ import annotations

import pytest
from lauren import LaurenFactory, module

from lauren_ai import AgentModule, LLMModule, agent
from lauren_ai._agents._runner import (
    AgentRunner,
    AgentRunnerBase,
    _AgentRunnerParam,
)
from lauren_ai._config import LLMConfig


async def _resolve(app, token):
    return await app._container.resolve(token)


class TestAgentRunnerSubscript:
    def test_subscript_returns_cached_subclass(self):
        @agent(model="mock")
        class A:
            """."""

        cls1 = AgentRunner[A]
        cls2 = AgentRunner[A]
        assert cls1 is cls2
        assert isinstance(cls1, type)

    def test_subscript_returns_param_marker_subclass(self):
        @agent(model="mock")
        class A:
            """."""

        cls = AgentRunner[A]
        # Parameterized form is a subclass of the marker — NOT of AgentRunner
        # (so the structural-Protocol scan doesn't pick it up alongside the
        # real runner).
        assert issubclass(cls, _AgentRunnerParam)
        assert not issubclass(cls, AgentRunner)
        assert cls._agent_param is A

    def test_subscript_rejects_non_class(self):
        with pytest.raises(TypeError, match="requires X to be a class"):
            AgentRunner["not-a-class"]  # type: ignore[index]

    def test_subscript_rejects_undecorated_class(self):
        class Plain:
            pass

        with pytest.raises(TypeError, match="not @agent-decorated"):
            AgentRunner[Plain]


class TestAgentRunnerDIResolution:
    @pytest.mark.asyncio
    async def test_resolves_to_module_runner(self):
        @agent(model="mock-model")
        class SoloAgent:
            """."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(agents=[SoloAgent], imports=[LLMProv])

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        app = LaurenFactory.create(AppMod)

        runner = await _resolve(app, AgentRunner[SoloAgent])
        assert isinstance(runner, AgentRunnerBase)

    @pytest.mark.asyncio
    async def test_two_agents_same_module_share_runner_instance(self):
        @agent(model="mock-model")
        class TwinA:
            """."""

        @agent(model="mock-model")
        class TwinB:
            """."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(agents=[TwinA, TwinB], imports=[LLMProv])

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        app = LaurenFactory.create(AppMod)

        runner_a = await _resolve(app, AgentRunner[TwinA])
        runner_b = await _resolve(app, AgentRunner[TwinB])
        # Same instance — they share a runner.  The runner consults each
        # agent's AgentMeta on every call, so distinct stores stay distinct.
        assert runner_a is runner_b

    @pytest.mark.asyncio
    async def test_bare_agent_runner_protocol_still_works_in_module(self):
        """A consumer in the same module can still inject ``runner: AgentRunner``."""
        from lauren import injectable

        @agent(model="mock-model")
        class HostAgent:
            """."""

        @injectable()
        class _RunnerConsumer:
            def __init__(self, runner: AgentRunner) -> None:
                self.runner = runner

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(
            agents=[HostAgent],
            imports=[LLMProv],
            injects=[_RunnerConsumer],
        )

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        app = LaurenFactory.create(AppMod)
        consumer = await _resolve(app, _RunnerConsumer)
        assert isinstance(consumer.runner, AgentRunnerBase)

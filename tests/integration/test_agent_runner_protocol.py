"""Integration tests — AgentRunner Protocol design.

Verifies that:
1. Single-module: ``runner: AgentRunner`` resolves to the dynamic subclass instance.
2. Multi-module: two ``AgentModule.for_root()`` calls produce no ``ProtocolAmbiguityError``.
3. Delegation tool resolved in target-module scope receives the target module's runner.
4. ``isinstance(runner, AgentRunner)`` is True (Protocol is ``@runtime_checkable``).
5. ``isinstance(runner, AgentRunnerBase)`` is True.
6. ``injects=[SubClass]`` still works; the subclass is exported, not the Protocol.
"""

from __future__ import annotations

import asyncio

import pytest

from lauren import LaurenFactory, Scope, controller, injectable, module, post
from lauren.testing import TestClient

from lauren_ai import AgentModule, AgentRunner, AgentRunnerBase, LLMConfig, LLMModule, agent, tool
from lauren_ai._transport import Completion, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(content: str) -> Completion:
    return Completion(
        id="c1",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TestSingleModuleProtocolResolution:
    """runner: AgentRunner in a single-module app resolves correctly."""

    def test_runner_resolves_as_agent_runner_protocol(self):
        """runner: AgentRunner resolves to the dynamic subclass instance."""

        @agent(model=None)
        class SimpleAgent:
            """Agent."""

        @controller("/test")
        class TestController:
            def __init__(self, runner: AgentRunner) -> None:
                self._runner = runner

            @post("/ok")
            async def ok(self) -> dict:
                return {"ok": True}

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response("hi")
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AgentMod = AgentModule.for_root(agents=[SimpleAgent], imports=[LLMProv])

        @module(imports=[LLMProv, AgentMod], controllers=[TestController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/test/ok")
        assert r.status_code == 200

    def test_isinstance_checks_pass(self):
        """isinstance(runner, AgentRunner) and isinstance(runner, AgentRunnerBase) both True."""

        @agent(model=None)
        class CheckAgent:
            """Agent."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AgentMod = AgentModule.for_root(agents=[CheckAgent], imports=[LLMProv])

        app = LaurenFactory.create(AgentMod)
        loop = asyncio.new_event_loop()
        try:
            runner = loop.run_until_complete(
                app.container.resolve(AgentMod.runner_class)
            )
        finally:
            loop.close()

        assert isinstance(runner, AgentRunner)
        assert isinstance(runner, AgentRunnerBase)

    def test_runner_class_is_dynamic_subclass_of_agent_runner_base(self):
        """for_root() with no injects= generates a subclass of AgentRunnerBase."""

        @agent(model=None)
        class DynAgent:
            """Agent."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AgentMod = AgentModule.for_root(agents=[DynAgent], imports=[LLMProv])

        assert issubclass(AgentMod.runner_class, AgentRunnerBase)
        assert AgentMod.runner_class is not AgentRunnerBase


class TestMultiModuleNoAmbiguity:
    """Two AgentModule.for_root() calls coexist without ProtocolAmbiguityError."""

    def test_two_agent_modules_no_conflict(self):
        """Two independent AgentModules start up without any ambiguity error."""

        @agent(model=None)
        class AlphaAgent:
            """Alpha agent."""

        @agent(model=None)
        class BetaAgent:
            """Beta agent."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AlphaMod = AgentModule.for_root(agents=[AlphaAgent], imports=[LLMProv])
        BetaMod = AgentModule.for_root(agents=[BetaAgent], imports=[LLMProv])

        @module(imports=[LLMProv, AlphaMod, BetaMod])
        class AppModule: ...

        # Should not raise ProtocolAmbiguityError.
        app = LaurenFactory.create(AppModule)
        assert app is not None

    def test_each_module_runner_class_is_distinct(self):
        """Two for_root() calls produce distinct runner classes."""

        @agent(model=None)
        class GammaAgent:
            """Gamma."""

        @agent(model=None)
        class DeltaAgent:
            """Delta."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        GammaMod = AgentModule.for_root(agents=[GammaAgent], imports=[LLMProv])
        DeltaMod = AgentModule.for_root(agents=[DeltaAgent], imports=[LLMProv])

        assert GammaMod.runner_class is not DeltaMod.runner_class
        assert issubclass(GammaMod.runner_class, AgentRunnerBase)
        assert issubclass(DeltaMod.runner_class, AgentRunnerBase)


class TestDelegationToolResolvesTargetRunner:
    """A delegation tool in the target module gets the target module's runner."""

    def test_delegation_tool_runner_matches_target_module(self):
        """DelegateTool with runner: AgentRunner resolves to the target runner."""

        @agent(model=None)
        class TargetAgent:
            """Target."""

        @tool()
        class DelegateTool:
            """Delegate task to the target agent.

            Args:
                task: Task to delegate.
            """

            def __init__(self, runner: AgentRunner) -> None:
                self._runner = runner

            async def run(self, task: str) -> dict:
                return {"delegated": True}

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        # DelegateTool goes in export_tools (NOT tools=): it is a provider of
        # TargetMod (receives the target runner via DI) but is not part of the
        # TargetAgent's own tool map.  Putting it in tools= would be circular.
        TargetMod = AgentModule.for_root(
            agents=[TargetAgent],
            imports=[LLMProv],
            export_tools=[DelegateTool],
        )

        @module(imports=[LLMProv, TargetMod])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        loop = asyncio.new_event_loop()
        try:
            tool_instance = loop.run_until_complete(
                app.container.resolve(DelegateTool)
            )
        finally:
            loop.close()

        assert isinstance(tool_instance._runner, AgentRunner)
        assert isinstance(tool_instance._runner, AgentRunnerBase)
        # The runner resolved into DelegateTool is the TargetMod runner.
        assert type(tool_instance._runner) is TargetMod.runner_class


class TestInjectsBackwardsCompatibility:
    """injects=[SubClass] still works; SubClass is exported, not the Protocol."""

    def test_injects_subclass_exported_not_protocol(self):
        """injects=[SubClass] exports the concrete subclass; AgentRunner Protocol not in exports."""

        @injectable(scope=Scope.SINGLETON)
        class MySpecialRunner(AgentRunnerBase):
            """Custom runner subclass."""

        @agent(model=None)
        class InjectsAgent:
            """Agent."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AgentMod = AgentModule.for_root(
            agents=[InjectsAgent],
            imports=[LLMProv],
            injects=[MySpecialRunner],
        )

        exports = AgentMod.__lauren_module__.exports
        assert MySpecialRunner in exports
        assert AgentRunner not in exports

    def test_injects_subclass_resolves_as_agent_runner(self):
        """An instance of injects=[SubClass] satisfies isinstance(…, AgentRunner)."""

        @injectable(scope=Scope.SINGLETON)
        class MyRunner2(AgentRunnerBase):
            """Custom runner 2."""

        @agent(model=None)
        class InjectsAgent2:
            """Agent."""

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AgentMod = AgentModule.for_root(
            agents=[InjectsAgent2],
            imports=[LLMProv],
            injects=[MyRunner2],
        )

        app = LaurenFactory.create(AgentMod)
        loop = asyncio.new_event_loop()
        try:
            runner = loop.run_until_complete(app.container.resolve(MyRunner2))
        finally:
            loop.close()

        assert isinstance(runner, AgentRunner)
        assert isinstance(runner, AgentRunnerBase)
        assert isinstance(runner, MyRunner2)

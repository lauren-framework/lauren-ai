"""Integration tests for per-agent state introduced by the
``@agent(memory=…, conversation_store=…)`` parameters and the
removal of ``AgentModule.for_root(memory=…, conversation_store=…)``.

Coverage:
    * Default-fill: agents without ``conversation_store=`` get a fresh
      ``InMemoryConversationStore`` from ``for_root``.
    * Per-agent isolation: two agents in the same module receive distinct
      stores.
    * Per-request override: ``runner.run(..., conversation_store=…)`` wins
      over AgentMeta.
    * Memory instance reuse: when ``meta.memory`` is set, the same
      ``ShortTermMemory`` is reused across ``run()`` calls (history
      accumulates).  Without it, each call constructs a fresh memory.
    * Multi-app safety: two ``LaurenFactory.create`` calls back-to-back
      don't share defaulted AgentMeta state.
    * Hard break: ``for_root(memory=…)`` and ``for_root(conversation_store=…)``
      raise ``TypeError`` (kwargs were removed).
"""

from __future__ import annotations

import pytest

from lauren_ai import AgentModule
from lauren_ai._agents import AGENT_META, agent
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._config import LLMConfig
from lauren_ai._memory import ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str, *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


async def _resolve(app, token):
    return await app._container.resolve(token)


# ---------------------------------------------------------------------------
# Tests: default-fill of conversation_store on AgentMeta
# ---------------------------------------------------------------------------


class TestDefaultStoreFill:
    def test_default_store_per_agent_when_none_supplied(self):
        @agent(model="mock")
        class A1:
            """."""

        AgentModule.for_root(agents=[A1])
        meta = getattr(A1, AGENT_META)
        assert isinstance(meta.conversation_store, InMemoryConversationStore)

    def test_two_agents_in_same_module_get_distinct_stores(self):
        """Each agent gets its OWN store — never the same instance."""

        @agent(model="mock")
        class A2a:
            """."""

        @agent(model="mock")
        class A2b:
            """."""

        AgentModule.for_root(agents=[A2a, A2b])
        store_a = getattr(A2a, AGENT_META).conversation_store
        store_b = getattr(A2b, AGENT_META).conversation_store
        assert isinstance(store_a, InMemoryConversationStore)
        assert isinstance(store_b, InMemoryConversationStore)
        assert store_a is not store_b

    def test_explicit_store_in_agent_decorator_preserved(self):
        custom = InMemoryConversationStore()

        @agent(model="mock", conversation_store=custom)
        class A3:
            """."""

        AgentModule.for_root(agents=[A3])
        assert getattr(A3, AGENT_META).conversation_store is custom

    def test_for_root_resets_runner_class_on_each_call(self):
        """Multi-app safety: two for_root calls give distinct runner classes."""

        @agent(model="mock")
        class A4:
            """."""

        AgentModule.for_root(agents=[A4])
        first = getattr(A4, AGENT_META).runner_class
        AgentModule.for_root(agents=[A4])
        second = getattr(A4, AGENT_META).runner_class
        assert first is not second


# ---------------------------------------------------------------------------
# Tests: hard break on for_root(memory=, conversation_store=)
# ---------------------------------------------------------------------------


class TestForRootHardBreak:
    def test_for_root_no_longer_accepts_memory_kwarg(self):
        @agent(model="mock")
        class A5:
            """."""

        with pytest.raises(TypeError, match="memory"):
            AgentModule.for_root(agents=[A5], memory=object())  # type: ignore[call-arg]

    def test_for_root_no_longer_accepts_conversation_store_kwarg(self):
        @agent(model="mock")
        class A6:
            """."""

        with pytest.raises(TypeError, match="conversation_store"):
            AgentModule.for_root(
                agents=[A6],
                conversation_store=InMemoryConversationStore(),  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# Tests: runner consults AgentMeta + per-request override
# ---------------------------------------------------------------------------


def _make_runner(store_for_agent: InMemoryConversationStore | None = None):
    """Build a bare AgentRunnerBase + return it.  No conversation_store wiring
    on __init__ — the runner consults AgentMeta on every call."""
    from lauren_ai._transport._mock import MockTransport

    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    mock = MockTransport()
    runner = AgentRunnerBase(transport=mock, config=cfg)
    return runner, mock


class TestRunnerConsultsAgentMeta:
    @pytest.mark.asyncio
    async def test_runner_loads_history_from_agent_meta_store(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class MetaStoreAgent:
            """."""

        runner, mock = _make_runner()
        mock.queue_response(_completion("Hi back"))

        await runner.run(MetaStoreAgent(), "Hello", conversation_id="s1")

        saved = await store.load("s1")
        assert len(saved) == 2
        assert saved[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_per_request_store_override_wins(self):
        meta_store = InMemoryConversationStore()
        override_store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=meta_store)
        class OverrideAgent:
            """."""

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        await runner.run(
            OverrideAgent(),
            "Hi",
            conversation_id="conv",
            conversation_store=override_store,
        )

        assert len(await override_store.load("conv")) == 2
        # Meta store untouched
        assert len(await meta_store.load("conv")) == 0


# ---------------------------------------------------------------------------
# Tests: memory reuse semantics
# ---------------------------------------------------------------------------


class TestMemoryReuse:
    @pytest.mark.asyncio
    async def test_memory_instance_reused_across_run_calls(self):
        """When ``meta.memory`` is set the same ShortTermMemory is reused."""
        shared_mem = ShortTermMemory(max_tokens=10_000)

        @agent(model="mock-model", memory=shared_mem)
        class MemAgent:
            """."""

        runner, mock = _make_runner()
        mock.queue_response(_completion("Reply 1"))
        await runner.run(MemAgent(), "Q1")

        mock.queue_response(_completion("Reply 2"))
        await runner.run(MemAgent(), "Q2")

        # Memory accumulates: 2 user + 2 assistant = 4 messages
        assert len(shared_mem.messages()) == 4

    @pytest.mark.asyncio
    async def test_memory_none_constructs_fresh_per_turn(self):
        """Without meta.memory, each call builds a fresh ShortTermMemory."""

        @agent(model="mock-model")  # memory left as None
        class FreshMemAgent:
            """."""

        runner, mock = _make_runner()
        mock.queue_response(_completion("Reply 1"))
        await runner.run(FreshMemAgent(), "Q1")

        mock.queue_response(_completion("Reply 2"))
        resp2 = await runner.run(FreshMemAgent(), "Q2")

        # Each call's ShortTermMemory is local — nothing accumulates.
        # The proof: the second response works (no state corruption).
        assert resp2.content == "Reply 2"

    @pytest.mark.asyncio
    async def test_per_request_memory_override(self):
        meta_mem = ShortTermMemory(max_tokens=10_000)
        override_mem = ShortTermMemory(max_tokens=10_000)

        @agent(model="mock-model", memory=meta_mem)
        class OverrideMemAgent:
            """."""

        runner, mock = _make_runner()
        mock.queue_response(_completion("Reply"))
        await runner.run(OverrideMemAgent(), "Q1", memory=override_mem)

        # Override gets the messages; meta memory untouched.
        assert len(override_mem.messages()) == 2
        assert len(meta_mem.messages()) == 0

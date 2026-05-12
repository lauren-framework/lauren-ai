"""Integration tests for Skill 49: Agent Session Isolation per User / Tenant.

Tests cover:
- Two tenants run agents → each has isolated conversation store
- Same user_id in different tenants → different history
- Same tenant, different user_ids → different history
- Conversation count correct per tenant
- Agent responses are independent across tenants

NOTE: from __future__ import annotations is safe here.
"""

from __future__ import annotations

import asyncio

from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model=None, system="You are a helpful assistant.")
class IsolatedAgent: ...


# ---------------------------------------------------------------------------
# TenantIsolatedAgentRunner implementation (inline)
# ---------------------------------------------------------------------------


class TenantIsolatedAgentRunner:
    """Wraps AgentRunner to namespace conversation IDs per tenant."""

    def __init__(self, runner: AgentRunner, agent_instance):
        self._runner = runner
        self._agent = agent_instance
        self._stores: dict[str, InMemoryConversationStore] = {}

    def _get_store(self, tenant_id: str) -> InMemoryConversationStore:
        if tenant_id not in self._stores:
            self._stores[tenant_id] = InMemoryConversationStore()
        return self._stores[tenant_id]

    async def run(self, tenant_id: str, user_id: str, prompt: str):
        conversation_id = f"{tenant_id}:{user_id}"
        store = self._get_store(tenant_id)
        return await self._runner.run(
            self._agent,
            prompt,
            conversation_id=conversation_id,
            conversation_store=store,
        )

    def get_conversation_count(self, tenant_id: str) -> int:
        store = self._stores.get(tenant_id)
        if store is None:
            return 0
        return len(store)

    def has_tenant(self, tenant_id: str) -> bool:
        return tenant_id in self._stores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, config=cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_two_tenants_get_separate_stores(self):
        mock = MockTransport()
        mock.queue_response(_completion("Hello tenant A"))
        mock.queue_response(_completion("Hello tenant B"))
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())

        asyncio.run(isolated.run("tenant_a", "user1", "Hello"))
        asyncio.run(isolated.run("tenant_b", "user1", "Hello"))

        assert isolated.has_tenant("tenant_a") is True
        assert isolated.has_tenant("tenant_b") is True

    def test_same_user_different_tenants_have_separate_histories(self):
        mock = MockTransport()
        mock.queue_response(_completion("Response for tenant A user1"))
        mock.queue_response(_completion("Response for tenant B user1"))
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())

        asyncio.run(isolated.run("tenant_a", "user1", "My message"))
        asyncio.run(isolated.run("tenant_b", "user1", "My message"))

        assert isolated.get_conversation_count("tenant_a") == 1
        assert isolated.get_conversation_count("tenant_b") == 1

    def test_responses_are_independent(self):
        mock = MockTransport()
        mock.queue_response(_completion("Tenant A gets this"))
        mock.queue_response(_completion("Tenant B gets that"))
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())

        r_a = asyncio.run(isolated.run("tenant_a", "u1", "Hi"))
        r_b = asyncio.run(isolated.run("tenant_b", "u1", "Hi"))

        assert r_a.content == "Tenant A gets this"
        assert r_b.content == "Tenant B gets that"

    async def test_conversation_id_is_namespaced(self):
        """Verify conversation IDs are prefixed with tenant_id."""
        mock = MockTransport()
        mock.queue_response(_completion("OK"))
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())

        await isolated.run("mycompany", "alice", "Hello")
        store = isolated._stores["mycompany"]
        conversations = await store.list_conversations()
        assert "mycompany:alice" in conversations

    def test_same_tenant_different_users_isolated(self):
        mock = MockTransport()
        mock.queue_response(_completion("Reply for alice"))
        mock.queue_response(_completion("Reply for bob"))
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())

        asyncio.run(isolated.run("company", "alice", "Alice's question"))
        asyncio.run(isolated.run("company", "bob", "Bob's question"))

        assert isolated.get_conversation_count("company") == 2


class TestConversationCount:
    def test_zero_conversations_for_new_tenant(self):
        mock = MockTransport()
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        assert isolated.get_conversation_count("nonexistent") == 0

    def test_one_conversation_after_single_run(self):
        mock = MockTransport()
        mock.queue_response(_completion("Hi"))
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())

        asyncio.run(isolated.run("acme", "user1", "Hello"))
        assert isolated.get_conversation_count("acme") == 1

    def test_two_conversations_for_two_users(self):
        mock = MockTransport()
        mock.queue_response(_completion("R1"))
        mock.queue_response(_completion("R2"))
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())

        asyncio.run(isolated.run("acme", "user1", "Q1"))
        asyncio.run(isolated.run("acme", "user2", "Q2"))
        assert isolated.get_conversation_count("acme") == 2


class TestInMemoryConversationStore:
    async def test_store_saves_and_loads_history(self):
        store = InMemoryConversationStore()
        messages = [{"role": "user", "content": "Hello"}]
        await store.save("conv1", messages)
        loaded = await store.load("conv1")
        assert loaded[0]["content"] == "Hello"

    async def test_different_conversation_ids_are_isolated(self):
        store = InMemoryConversationStore()
        await store.save("conv1", [{"role": "user", "content": "Msg1"}])
        await store.save("conv2", [{"role": "user", "content": "Msg2"}])
        loaded1 = await store.load("conv1")
        loaded2 = await store.load("conv2")
        assert loaded1[0]["content"] == "Msg1"
        assert loaded2[0]["content"] == "Msg2"


class TestTenantIsolatedRunnerDirect:
    def test_get_conversation_count_zero_without_run(self):
        mock = MockTransport()
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        assert isolated.get_conversation_count("any_tenant") == 0

    def test_has_tenant_false_without_run(self):
        mock = MockTransport()
        runner = _make_runner(mock)
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        assert isolated.has_tenant("any_tenant") is False

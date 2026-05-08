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

import pytest

from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._agents import agent


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
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTenantIsolation:
    async def test_two_tenants_get_separate_stores(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Hello tenant A"))
        mock.queue_response(_completion("Hello tenant B"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        await isolated.run("tenant_a", "user1", "Hello")
        await isolated.run("tenant_b", "user1", "Hello")

        assert isolated.has_tenant("tenant_a")
        assert isolated.has_tenant("tenant_b")

    async def test_tenant_stores_are_different_objects(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("A"))
        mock.queue_response(_completion("B"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        await isolated.run("t1", "u1", "Hi")
        await isolated.run("t2", "u1", "Hi")

        store_t1 = isolated._stores["t1"]
        store_t2 = isolated._stores["t2"]
        assert store_t1 is not store_t2

    async def test_same_user_different_tenants_have_separate_histories(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Response for tenant A user1"))
        mock.queue_response(_completion("Response for tenant B user1"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        await isolated.run("tenant_a", "user1", "My message")
        await isolated.run("tenant_b", "user1", "My message")

        # Each tenant's store should have 1 conversation (for user1)
        assert isolated.get_conversation_count("tenant_a") == 1
        assert isolated.get_conversation_count("tenant_b") == 1

    async def test_conversation_id_is_namespaced(self):
        """Verify conversation IDs are prefixed with tenant_id."""
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        await isolated.run("mycompany", "alice", "Hello")

        store = isolated._stores["mycompany"]
        conversations = await store.list_conversations()
        assert "mycompany:alice" in conversations

    async def test_same_tenant_different_users_isolated(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Reply for alice"))
        mock.queue_response(_completion("Reply for bob"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        await isolated.run("company", "alice", "Alice's question")
        await isolated.run("company", "bob", "Bob's question")

        store = isolated._stores["company"]
        conversations = await store.list_conversations()
        assert "company:alice" in conversations
        assert "company:bob" in conversations
        # They should be separate conversations
        assert len(conversations) == 2

    async def test_responses_are_independent(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Tenant A gets this"))
        mock.queue_response(_completion("Tenant B gets that"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        r_a = await isolated.run("tenant_a", "u1", "Hi")
        r_b = await isolated.run("tenant_b", "u1", "Hi")

        assert r_a.content == "Tenant A gets this"
        assert r_b.content == "Tenant B gets that"


class TestConversationCount:
    async def test_zero_conversations_for_new_tenant(self):
        runner, _ = _make_runner()
        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        assert isolated.get_conversation_count("nonexistent") == 0

    async def test_one_conversation_after_single_run(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Hi"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        await isolated.run("acme", "user1", "Hello")
        assert isolated.get_conversation_count("acme") == 1

    async def test_two_conversations_for_two_users(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("R1"))
        mock.queue_response(_completion("R2"))

        isolated = TenantIsolatedAgentRunner(runner, IsolatedAgent())
        await isolated.run("acme", "user1", "Q1")
        await isolated.run("acme", "user2", "Q2")
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

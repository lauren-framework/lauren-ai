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

from pydantic import BaseModel

import pytest

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json, Path
from lauren.testing import TestClient
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
# Module-level mock and runner state
# ---------------------------------------------------------------------------

_MOCK = MockTransport()
_runner_state: dict = {}


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools={}, config=cfg)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _TenantRunRequest(BaseModel):
    tenant_id: str
    user_id: str
    prompt: str


@controller("/tenant")
class TenantController:
    def __init__(self, mock: MockTransport) -> None:
        self._mock = mock

    @post("/run")
    async def run(self, body: Json[_TenantRunRequest]) -> dict:
        isolated = _runner_state["isolated"]
        resp = await isolated.run(body.tenant_id, body.user_id, body.prompt)
        return {"content": resp.content}

    @get("/conversations/{tenant_id}")
    async def conversations(self, tenant_id: str) -> dict:
        isolated = _runner_state["isolated"]
        return {"count": isolated.get_conversation_count(tenant_id)}

    @get("/has-tenant/{tenant_id}")
    async def has_tenant(self, tenant_id: str) -> dict:
        isolated = _runner_state["isolated"]
        return {"has_tenant": isolated.has_tenant(tenant_id)}


@module(
    controllers=[TenantController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class TenantModule: ...


# ---------------------------------------------------------------------------
# Build app helper
# ---------------------------------------------------------------------------


def build_app(*responses: str) -> TestClient:
    _MOCK.reset()
    for c in responses:
        _MOCK.queue_response(_completion(c))
    runner = _make_runner(_MOCK)
    _runner_state["isolated"] = TenantIsolatedAgentRunner(runner, IsolatedAgent())
    return TestClient(LaurenFactory.create(TenantModule))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_two_tenants_get_separate_stores(self):
        client = build_app("Hello tenant A", "Hello tenant B")
        client.post("/tenant/run", json={"tenant_id": "tenant_a", "user_id": "user1", "prompt": "Hello"})
        client.post("/tenant/run", json={"tenant_id": "tenant_b", "user_id": "user1", "prompt": "Hello"})
        r_a = client.get("/tenant/has-tenant/tenant_a")
        r_b = client.get("/tenant/has-tenant/tenant_b")
        assert r_a.json()["has_tenant"] is True
        assert r_b.json()["has_tenant"] is True

    def test_same_user_different_tenants_have_separate_histories(self):
        client = build_app("Response for tenant A user1", "Response for tenant B user1")
        client.post("/tenant/run", json={"tenant_id": "tenant_a", "user_id": "user1", "prompt": "My message"})
        client.post("/tenant/run", json={"tenant_id": "tenant_b", "user_id": "user1", "prompt": "My message"})
        r_a = client.get("/tenant/conversations/tenant_a")
        r_b = client.get("/tenant/conversations/tenant_b")
        assert r_a.json()["count"] == 1
        assert r_b.json()["count"] == 1

    def test_responses_are_independent(self):
        client = build_app("Tenant A gets this", "Tenant B gets that")
        r_a = client.post("/tenant/run", json={"tenant_id": "tenant_a", "user_id": "u1", "prompt": "Hi"})
        r_b = client.post("/tenant/run", json={"tenant_id": "tenant_b", "user_id": "u1", "prompt": "Hi"})
        assert r_a.json()["content"] == "Tenant A gets this"
        assert r_b.json()["content"] == "Tenant B gets that"

    async def test_conversation_id_is_namespaced(self):
        """Verify conversation IDs are prefixed with tenant_id."""
        client = build_app("OK")
        client.post("/tenant/run", json={"tenant_id": "mycompany", "user_id": "alice", "prompt": "Hello"})
        isolated = _runner_state["isolated"]
        store = isolated._stores["mycompany"]
        conversations = await store.list_conversations()
        assert "mycompany:alice" in conversations

    def test_same_tenant_different_users_isolated(self):
        client = build_app("Reply for alice", "Reply for bob")
        client.post("/tenant/run", json={"tenant_id": "company", "user_id": "alice", "prompt": "Alice's question"})
        client.post("/tenant/run", json={"tenant_id": "company", "user_id": "bob", "prompt": "Bob's question"})
        r = client.get("/tenant/conversations/company")
        assert r.json()["count"] == 2


class TestConversationCount:
    def test_zero_conversations_for_new_tenant(self):
        client = build_app()
        r = client.get("/tenant/conversations/nonexistent")
        assert r.json()["count"] == 0

    def test_one_conversation_after_single_run(self):
        client = build_app("Hi")
        client.post("/tenant/run", json={"tenant_id": "acme", "user_id": "user1", "prompt": "Hello"})
        r = client.get("/tenant/conversations/acme")
        assert r.json()["count"] == 1

    def test_two_conversations_for_two_users(self):
        client = build_app("R1", "R2")
        client.post("/tenant/run", json={"tenant_id": "acme", "user_id": "user1", "prompt": "Q1"})
        client.post("/tenant/run", json={"tenant_id": "acme", "user_id": "user2", "prompt": "Q2"})
        r = client.get("/tenant/conversations/acme")
        assert r.json()["count"] == 2


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

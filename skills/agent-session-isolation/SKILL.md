---
name: agent-session-isolation
description: Implements per-tenant and per-user session isolation for agent conversations using namespaced conversation IDs. Use when building multi-tenant applications where different users or tenants must have fully isolated conversation histories that cannot bleed into each other.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Agent Session Isolation per User / Tenant

## Overview

`TenantIsolatedAgentRunner` wraps `AgentRunnerBase` and prefixes every
`conversation_id` with `"{tenant_id}:{user_id}"`.  Each tenant gets its own
`InMemoryConversationStore` so histories cannot cross tenant boundaries.

---

## Implementation

```python
# isolation/tenant_runner.py
from __future__ import annotations
from lauren_ai import agent, ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore

class TenantIsolatedAgentRunner:
    """Wraps AgentRunner to namespace conversation IDs per tenant."""

    def __init__(self, runner, agent_instance):
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
```

---

## Why this is safe

- `conversation_id = f"{tenant_id}:{user_id}"` ensures that `user123` in
  `tenant_a` and `user123` in `tenant_b` have different keys.
- Each tenant has its own `InMemoryConversationStore` dict — there is no
  shared mutable state.
- Switching to a Redis-backed store in production only requires replacing
  `InMemoryConversationStore` with a `RedisConversationStore`.

---

## Per-user isolation without multi-tenancy

When you only need user isolation (not tenant isolation), use the user ID
directly as the `conversation_id`:

```python
response = await runner.run(
    agent_instance,
    prompt,
    conversation_id=user_id,
    conversation_store=shared_store,
)
```

Each user's messages are stored under their own key in the same store.

---

## Attaching to a Lauren controller

```python
@controller("/chat")
class ChatController:
    def __init__(self, runner: AgentRunner) -> None:
        self._isolated = TenantIsolatedAgentRunner(runner, MyAgent())

    @post("/message")
    async def message(self, body: Json[ChatRequest], ctx: ExecutionContext) -> dict:
        tenant_id = ctx.request.state.tenant_id
        user_id   = ctx.request.state.user_id
        response = await self._isolated.run(tenant_id, user_id, body.text)
        return {"reply": response.content}
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_memory/_stores.py` | `InMemoryConversationStore` |
| `src/lauren_ai/_agents/_runner.py` | `run(conversation_id=, conversation_store=)` |

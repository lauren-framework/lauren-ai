---
name: conversation-memory
description: Adds persistent multi-turn conversation history to agents using InMemoryConversationStore and ShortTermMemory sliding window. Use when an agent must remember prior turns within or across sessions, implementing sliding-window trimming to fit token budgets, or storing conversation history keyed by session/user ID.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Conversation Memory

## Memory tiers at a glance

| Tier | Class | Scope | Notes |
|------|-------|-------|-------|
| In-run buffer | `ShortTermMemory` | Single `runner.run()` call | Automatic, no config needed |
| Cross-run history | `InMemoryConversationStore` | Session (`conversation_id`) | Persists across calls |

---

## Short-term memory (sliding window)

Pass `memory=ShortTermMemory(max_tokens=N)` to `@agent()` to share a buffer
across runs or shrink the rolling window.  Without this, each `run()` starts
fresh.

```python
from lauren_ai import agent
from lauren_ai._memory import ShortTermMemory

shared_mem = ShortTermMemory(max_tokens=8_000)

@agent(model="claude-opus-4-6", system="You are a helpful assistant.", memory=shared_mem)
class MemAgent: ...
```

The runner trims the oldest non-system messages automatically when
`token_estimate` exceeds `max_tokens`.

---

## Conversation store (cross-session history)

`InMemoryConversationStore` persists the full message exchange keyed by a
`conversation_id` string.  Pass it to `@agent(conversation_store=...)` and
supply `conversation_id` on each `runner.run()` call.

```python
from lauren_ai import agent
from lauren_ai._memory._stores import InMemoryConversationStore

STORE = InMemoryConversationStore()

@agent(
    model="claude-opus-4-6",
    system="You are a helpful assistant with memory of past turns.",
    conversation_store=STORE,
)
class MemoryAgent: ...
```

### Running with a conversation ID

```python
# Turn 1 — agent learns the user's name
resp1 = await runner.run(MemoryAgent(), "My name is Alice.", conversation_id="sess-1")

# Turn 2 — agent recalls the name from stored history
resp2 = await runner.run(MemoryAgent(), "What is my name?", conversation_id="sess-1")
# resp2.content → "Your name is Alice."
```

Without `conversation_id`, the store is not consulted and history is not saved.

---

## Per-request overrides

Pass `memory=` or `conversation_store=` directly to `runner.run()` to override
the agent-level configuration for a single run:

```python
override_store = InMemoryConversationStore()
await runner.run(MemoryAgent(), "hi", conversation_id="s", conversation_store=override_store)
```

---

## Implementing a custom persistent store

Implement the `ConversationStore` protocol (two async methods):

```python
from lauren_ai._memory import ConversationStore

class RedisConversationStore:
    async def load(self, conversation_id: str) -> list:
        raw = await redis.get(f"conv:{conversation_id}")
        return json.loads(raw) if raw else []

    async def save(self, conversation_id: str, messages: list) -> None:
        await redis.set(f"conv:{conversation_id}", json.dumps(messages))

    async def delete(self, conversation_id: str) -> None:
        await redis.delete(f"conv:{conversation_id}")
```

---

## Testing

Use `AgentTestClient` for integration testing.  Queue two responses and verify
the second call's messages include prior turns.

```python
from lauren_ai.testing import AgentTestClient
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()
mock.queue_response(completion("I'll remember that!"))
mock.queue_response(completion("Your name is Alice."))

store = InMemoryConversationStore()

@agent(model=None, conversation_store=store)
class MemAgent: ...

client = AgentTestClient(MemAgent(), mock)
await client.run_async("My name is Alice.", conversation_id="s")
await client.run_async("What is my name?", conversation_id="s")

history = await store.load("s")
assert len(history) == 4  # 2 user + 2 assistant
```

---

## Reference

- `lauren_ai._memory`: `ShortTermMemory`, `ConversationStore`, `InMemoryConversationStore`
- `lauren_ai._memory._stores`: `InMemoryConversationStore`
- Skills: `managing-memory`, `long-term-vector-memory`, `conversation-state-db`

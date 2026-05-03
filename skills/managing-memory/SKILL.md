---
name: managing-memory
description: Manages memory tiers in lauren-ai agents — short-term rolling window, conversation history, user memory facts with @remember, and vector store for RAG. Use when adding persistent memory to an agent, implementing @remember() for long-term facts, using ConversationStore across sessions, or building RAG with InMemoryVectorStore.
---

# Managing Memory in Agents

## Memory tier overview

| Tier | Class | Scope | Persistence | Managed by |
|------|-------|-------|-------------|------------|
| 1. Short-term | `ShortTermMemory` | Single agent run | None — resets each run | `AgentRunner` (automatic) |
| 2. Conversation | `ConversationStore` / `InMemoryConversationStore` | Session (conversation ID) | In-memory or custom backend | Application |
| 3. User memory | `UserMemoryStore` / `InMemoryUserMemoryStore` + `@remember` | Per user (user ID) | In-memory or custom backend | `@remember` decorator |
| 4. Vector | `InMemoryVectorStore` | Application-wide | In-memory | Application |

---

## Quick start: `@remember` for long-term user facts

```python
from lauren_ai import (
    agent, remember, use_guardrails, use_tools,
    InMemoryUserMemoryStore, PromptInjectionFilter,
)

_memory = InMemoryUserMemoryStore()

@agent(model="claude-opus-4-6", system="You are a personal assistant.")
@remember(store=_memory, extract=True, inject=True, top_k=5)
@use_guardrails(input=[PromptInjectionFilter()])
@use_tools(my_tool)
class PersonalAssistant: ...
```

**How it works:**

1. **Inject** (`inject=True`): Before each LLM call, `top_k` relevant facts are
   retrieved from the store and prepended to the system prompt as:
   ```
   ## What I remember about you:
   - User prefers dark mode (confidence: high)
   - User works on Python projects (confidence: medium)
   ```

2. **Extract** (`extract=True`): After each conversation turn, the LLM extracts
   new facts from the exchange and stores them.

---

## Decorator order (mandatory)

`@remember()` must sit between `@agent()` and `@use_guardrails()`:

```python
@agent(model="claude-opus-4-6")   # outermost — applied last
@remember(store=_memory, ...)      # reads/sets REMEMBER_META
@use_guardrails(input=[...])       # optional
@use_tools(my_tool)                # innermost — applied first
class MyAgent: ...
```

Python applies decorators bottom-up.  Wrong ordering causes the decorator that
reads metadata to run before the decorator that writes it, silently dropping
tools, guardrails, or memory configuration.

---

## Tier 2: Conversation history across runs

Use `ConversationStore` to persist full message history across multiple agent
runs within a session:

```python
from lauren_ai import InMemoryConversationStore, AgentRunner, LLMConfig

store = InMemoryConversationStore()
cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
runner = AgentRunner(transport=..., config=cfg, conversation_store=store)

# Each call with the same conversation_id shares history
result1 = await runner.run(MyAgent(), "My name is Alice.", conversation_id="sess-1")
result2 = await runner.run(MyAgent(), "What is my name?", conversation_id="sess-1")
```

---

## Tier 4: Vector store for RAG

```python
from lauren_ai import InMemoryVectorStore

store = InMemoryVectorStore()

# Index documents (requires EmbedService)
await store.add(
    texts=["Python is a programming language.", "FastAPI is an async web framework."],
    embed_service=embed_service,
)

# Query
results = await store.search(
    query="web development with Python",
    embed_service=embed_service,
    top_k=3,
)
for doc, score in results:
    print(f"{score:.3f} — {doc}")
```

---

## Reference

See [memory.md](memory.md) for:

- Full `ShortTermMemory` API (including `ctx.memory` access from lifecycle hooks)
- `ConversationStore` protocol and implementing a custom persistent backend
- `@remember` parameters (`store`, `extract`, `inject`, `top_k`, `store_token`)
- Manual `store.add()` / `store.search()` usage with `MemoryFact`
- Implementing a custom `UserMemoryStore` (e.g. Postgres-backed)
- `InMemoryVectorStore` full API
- `TeamMemory` shared state for multi-agent teams

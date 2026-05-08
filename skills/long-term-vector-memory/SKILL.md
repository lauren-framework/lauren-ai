---
name: long-term-vector-memory
description: Adds long-term user-specific memory to agents using the @remember decorator with InMemoryUserMemoryStore. Use when an agent needs to remember facts about individual users across sessions, extracting preferences/facts from conversations and injecting them into future system prompts, or building personalized assistants.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Long-Term Memory with Vector Store Persistence

## Overview

Long-term user memory operates at a level above conversation history.  Instead
of replaying raw messages, the `@remember` decorator:

1. **Injects** (`inject=True`) — retrieves the top-K relevant facts for the
   current query and prepends them to the system prompt before each LLM call.
2. **Extracts** (`extract=True`) — after each turn the LLM extracts new facts
   from the exchange and persists them to the store.

```
System prompt
 ┌─────────────────────────────────────────────────────┐
 │ ## What I remember about you:                       │
 │ - User prefers dark mode (confidence: high)         │
 │ - User works on Python projects (confidence: medium)│
 └─────────────────────────────────────────────────────┘
```

---

## Quick start

```python
from lauren_ai import agent, remember
from lauren_ai._memory._in_memory_user import InMemoryUserMemoryStore

_memory_store = InMemoryUserMemoryStore()

@agent(model="claude-opus-4-6", system="You are a personalized assistant.")
@remember(store=_memory_store, extract=True, inject=True, top_k=5)
class PersonalizedAgent: ...
```

### Decorator order (mandatory)

```python
@agent(model="claude-opus-4-6")   # outermost — applied last
@remember(store=_memory, ...)      # between @agent and @use_tools
@use_tools(my_tool)                # innermost — applied first
class MyAgent: ...
```

Python applies decorators bottom-up. `@remember` must sit directly below
`@agent` so the runner can find its metadata.

---

## `@remember` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `store` | required | `UserMemoryStore` or DI token string |
| `extract` | `True` | Extract new facts after each turn |
| `inject` | `True` | Prepend facts to system prompt |
| `top_k` | `5` | Maximum facts to inject per call |
| `extraction_model` | `None` | Override model for fact extraction |

---

## Accessing per-user memory

Pass `user_id` on each `runner.run()` call so the memory store can segment
facts by user:

```python
resp = await runner.run(PersonalizedAgent(), "I prefer Python", user_id="user-42")
```

Without `user_id`, facts are stored under a global key (suitable for single-user
applications).

---

## InMemoryUserMemoryStore

The built-in in-memory store is suitable for development and testing.  In
production, replace it with a database-backed implementation that satisfies
the `UserMemoryStore` protocol:

```python
class UserMemoryStore(Protocol):
    async def add(self, user_id: str, fact: MemoryFact) -> None: ...
    async def search(self, user_id: str, query: str, *, top_k: int = 5) -> list[MemoryFact]: ...
    async def delete(self, user_id: str, fact_id: str) -> None: ...
    async def clear(self, user_id: str) -> None: ...
```

---

## Testing

Because extraction depends on the LLM parsing the conversation, unit tests
should verify that `@remember` attaches the correct metadata and that the
agent can run end-to-end with `MockTransport`:

```python
from lauren_ai._memory import REMEMBER_META

meta = getattr(PersonalizedAgent, REMEMBER_META)
assert meta.inject is True
assert meta.extract is True
assert meta.top_k == 5
```

For a full end-to-end test, queue two completions: one for the main answer and
one for the extraction sub-call (if `extract=True`).

---

## Reference

- `lauren_ai._memory`: `REMEMBER_META`, `RememberMeta`, `remember`
- `lauren_ai._memory._in_memory_user`: `InMemoryUserMemoryStore`
- `lauren_ai._memory._user`: `MemoryFact`, `UserMemoryStore`
- Skills: `managing-memory`, `conversation-memory`, `rag-pipeline`

# User Memory

The user memory system lets agents remember facts about specific users across
conversations. Facts are stored in a `UserMemoryStore`, optionally injected
into the system prompt before each LLM call, and optionally extracted from
each conversation turn using a secondary LLM call.

## Core types

### MemoryFact

```python
from lauren_ai._memory import MemoryFact

fact = MemoryFact(
    memory_id="m1",
    user_id="user-abc",
    content="User prefers Python over JavaScript",
    topics=["language", "preferences"],
    confidence=1.0,
)
```

Fields:

| Field | Type | Description |
|---|---|---|
| `memory_id` | `str` | Unique identifier for the fact. |
| `user_id` | `str` | The user this fact belongs to. |
| `content` | `str` | The fact as a human-readable sentence. |
| `topics` | `list[str]` | Topic tags used for filtering and search. |
| `confidence` | `float` | Score in `[0.0, 1.0]`; higher means more certain. |
| `created_at` | `datetime` | When the fact was first recorded. |
| `last_seen_at` | `datetime` | When the fact was last reinforced. |
| `source_conversation_id` | `str \| None` | Originating conversation, if tracked. |

**Lifecycle helpers:**

```python
fact.reinforce()        # boosts confidence by 0.1 (capped at 1.0), updates last_seen_at
fact.decay(factor=0.8)  # multiplies confidence by factor
```

### UserMemoryStore protocol

Any class that satisfies `UserMemoryStore` can back the memory system:

```python
from lauren_ai._memory import UserMemoryStore

class MyPgStore:
    async def add(self, fact: MemoryFact) -> None: ...
    async def get(self, user_id: str, memory_id: str) -> MemoryFact | None: ...
    async def search(self, user_id: str, query: str, top_k: int = 10) -> list[MemoryFact]: ...
    async def list(self, user_id: str, topic: str | None = None) -> list[MemoryFact]: ...
    async def update(self, memory_id: str, *, content: str | None = None, confidence: float | None = None) -> None: ...
    async def delete(self, memory_id: str) -> None: ...
    async def clear(self, user_id: str) -> None: ...
```

All methods are async. The `search` method receives a plain-text query and
returns facts ranked by relevance (implementation-specific).

## InMemoryUserMemoryStore

Use the in-process store for tests and local development. It uses simple
substring matching for `search`:

```python
from lauren_ai._memory import InMemoryUserMemoryStore, MemoryFact

store = InMemoryUserMemoryStore()

await store.add(MemoryFact("m1", "alice", "User loves hiking", topics=["hobbies"]))
await store.add(MemoryFact("m2", "alice", "User works in Berlin", topics=["location"]))

# Retrieve by ID
fact = await store.get("alice", "m1")

# Full-text search within a user's facts
results = await store.search("alice", "hiking")   # returns [fact for m1]

# List all facts for a user
all_facts = await store.list("alice")

# Filter by topic
location_facts = await store.list("alice", topic="location")

# Update fields
await store.update("m1", content="User loves hiking and cycling")

# Delete
await store.delete("m1")

# Remove all facts for a user
await store.clear("alice")
```

## @remember() decorator

`@remember()` opts a `@agent()` class into automatic user memory integration.
It must be applied **below** `@agent()`:

```python
from lauren_ai import agent
from lauren_ai._memory import remember

@agent(model="claude-haiku-4-5", system="You are a personal assistant.")
@remember(store="user_memory", extract=True, inject=True, top_k=5)
class PersonalAssistant:
    async def on_start(self, ctx): ...
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `store` | `None` | DI token name for the `UserMemoryStore`. `None` means inject `UserMemoryStore` directly. |
| `extract` | `True` | Extract new facts from each conversation turn after the LLM responds. |
| `inject` | `True` | Inject relevant memories into the system prompt before each LLM call. |
| `top_k` | `5` | Number of memories to inject per turn. |
| `extraction_model` | `None` | Model to use for fact extraction. Defaults to the agent's configured model. |

`@remember` must always be called with parentheses. Using it bare (`@remember`
without parentheses) raises `MemoryConfigError` immediately.

### Memory injection

When `inject=True`, the runner fetches the top `top_k` facts matching the
current query and prepends a block like:

```
## What I remember about you:
- User loves hiking (confidence: high)
- User works in Berlin (confidence: high)
- User prefers dark mode (confidence: medium)
```

This block is added to the system prompt before the LLM call so the model can
personalise its response naturally.

### Fact extraction

When `extract=True`, after the assistant responds the runner sends the
conversation turn to an LLM with a structured prompt asking for facts in JSON:

```
[{"content": "User prefers short answers", "topics": ["communication"]}]
```

Each extracted fact is stored in the `UserMemoryStore` with `confidence=1.0`.

## Helper functions

### extract_facts

```python
from lauren_ai._memory._remember import extract_facts

facts = await extract_facts(
    user_id="alice",
    conversation_turn="User: I hate long explanations. Assistant: Got it.",
    llm=llm_service,
    model=None,   # uses llm_service's default model
)
# [{"content": "User prefers short explanations", "topics": ["communication"]}]
```

### build_memory_context

```python
from lauren_ai._memory._remember import build_memory_context

block = build_memory_context(facts)
# "## What I remember about you:\n- User prefers short explanations (confidence: high)"
```

## Wiring with DI

Provide the store as a singleton in your module:

```python
from lauren import module, use_value
from lauren_ai._memory import InMemoryUserMemoryStore, UserMemoryStore

store = InMemoryUserMemoryStore()

@module(
    providers=[use_value(provide=UserMemoryStore, value=store)],
    exports=[UserMemoryStore],
)
class MemoryModule: ...
```

Then import `MemoryModule` alongside `LLMModule` and `AgentModule`:

```python
from lauren import module, LaurenFactory
from lauren_ai._module import LLMModule, AgentModule

LLMProviderModule = LLMModule.for_root(LLMConfig.for_anthropic(...))
AIAgentModule = AgentModule.for_root(agents=[PersonalAssistant])

@module(imports=[LLMProviderModule, AIAgentModule, MemoryModule])
class AppModule: ...

app = LaurenFactory.create(AppModule)
```

## Confidence lifecycle

Facts start at `confidence=1.0`. You can schedule periodic decay jobs to
remove stale knowledge:

```python
async def decay_old_facts(store: InMemoryUserMemoryStore, user_id: str) -> None:
    facts = await store.list(user_id)
    for fact in facts:
        fact.decay(factor=0.9)
        if fact.confidence < 0.1:
            await store.delete(fact.memory_id)
        else:
            await store.update(fact.memory_id, confidence=fact.confidence)
```

When a fact is mentioned again in conversation, call `fact.reinforce()` before
calling `store.update()` to bump its confidence back towards 1.0.

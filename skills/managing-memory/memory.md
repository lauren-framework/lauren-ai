# Memory

`lauren-ai` provides four memory tiers with different scopes and persistence
characteristics.

---

## Tier 1 — Short-term memory (`ShortTermMemory`)

Automatically managed by `AgentRunner`.  Holds the rolling conversation
window for a single agent run.

```python
from lauren_ai import ShortTermMemory

mem = ShortTermMemory(max_tokens=40_000)
mem.add_user("Hello, who are you?")
mem.add_assistant(completion)          # Completion or str
mem.add_tool_result(tool_result)       # ToolResult

messages = mem.messages()             # list[Message] for the LLM
print(mem.token_estimate)             # rough token count
```

You never need to create `ShortTermMemory` manually — `AgentRunner.run()` creates
a fresh one per call.  To access it from lifecycle hooks, use `ctx.memory`:

```python
from lauren_ai import agent, AgentContext

@agent(model="claude-opus-4-6")
class MyAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        print(f"Memory has {ctx.memory.token_estimate} tokens so far")
```

---

## Tier 2 — Conversation store (`ConversationStore`)

Persists full conversation history across multiple agent runs within a session.
Declare the store **per-agent** via `@agent(conversation_store=...)`:

```python
from lauren_ai import agent, InMemoryConversationStore

@agent(model="claude-opus-4-6", conversation_store=InMemoryConversationStore())
class MyAgent: ...

# AgentModule.for_root() also auto-creates InMemoryConversationStore for agents
# that omit it, so every agent always has an isolated store.
AgentModule.for_root(agents=[MyAgent], imports=LLMProvider)
```

When `runner.run()` receives a `conversation_id`, the runner automatically
loads prior messages from the agent's store before the new turn and saves the
updated history after:

```python
# Turn 1 — no prior history
resp1 = await runner.run(agent, "My name is Alice.", conversation_id="sess-1")

# Turn 2 — full prior exchange is injected into ShortTermMemory
resp2 = await runner.run(agent, "What is my name?", conversation_id="sess-1")
# resp2.content → "Your name is Alice."
```

Per-request override (leaves the agent's own store untouched):

```python
await runner.run(agent, msg, conversation_id="s1", conversation_store=other_store)
```

`AgentModule.for_root()` **no longer accepts `conversation_store=`** — it was
removed in favour of the per-agent pattern above.

Direct store manipulation uses `save` / `load` / `delete`:

```python
await store.save("sess-1", messages)        # overwrite full history
history = await store.load("sess-1")        # [] if not found
await store.delete("sess-1")               # remove conversation
```

### Custom persistent store

Implement the `ConversationStore` protocol:

```python
class RedisConversationStore(ConversationStore):
    async def load(self, conversation_id: str) -> list[Message]: ...
    async def save(self, conversation_id: str, messages: list[Message]) -> None: ...
    async def delete(self, conversation_id: str) -> None: ...
```

---

## Tier 3 — User memory with `@remember()`

Long-term facts extracted from conversations and injected as context in future
turns.

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

**Decorator order:** `@agent` / `@remember` / `@use_guardrails` / `@use_tools`

**How it works:**

1. **Inject** (`inject=True`): Before each LLM call, `top_k` relevant facts are
   retrieved from the store and prepended to the system prompt as:
   ```
   ## What I remember about you:
   - User prefers dark mode (confidence: high)
   - User works on Python projects (confidence: medium)
   ```

2. **Extract** (`extract=True`): After each conversation turn, the LLM extracts
   new facts from the exchange and stores them automatically.

### `@remember` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `UserMemoryStore` or `str` | required | Store instance, or DI token string |
| `extract` | `bool` | `True` | Extract new facts after each turn |
| `inject` | `bool` | `True` | Inject remembered facts into system prompt |
| `top_k` | `int` | `5` | Number of most relevant facts to inject |

### Using a DI token for the store

```python
@agent(model="claude-opus-4-6")
@remember(store="UserMemoryStore", extract=True, inject=True)
class PersonalAssistant: ...
```

The string `"UserMemoryStore"` is resolved from the DI container at runtime.

### Manual store access

```python
from lauren_ai import InMemoryUserMemoryStore, MemoryFact

store = InMemoryUserMemoryStore()

# Add a fact
await store.add(
    user_id="user-123",
    fact=MemoryFact(
        content="User prefers dark mode",
        topics=["preferences", "ui"],
        confidence=0.9,
    ),
)

# Search relevant facts
facts = await store.search(
    user_id="user-123",
    query="what are the user's UI preferences?",
    top_k=3,
)

# Build context string
from lauren_ai._memory._remember import build_memory_context
context_block = build_memory_context(facts)
```

### Custom `UserMemoryStore`

Implement the protocol:

```python
from lauren_ai import UserMemoryStore, MemoryFact

class PostgresUserMemoryStore(UserMemoryStore):
    async def add(self, user_id: str, fact: MemoryFact) -> None: ...
    async def search(self, user_id: str, query: str, top_k: int = 5) -> list[MemoryFact]: ...
    async def get_all(self, user_id: str) -> list[MemoryFact]: ...
    async def delete(self, user_id: str, fact_id: str) -> None: ...
```

---

## Tier 4 — Vector store (`InMemoryVectorStore`)

Semantic retrieval for RAG (retrieval-augmented generation) patterns.

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

### Using `EmbedService` in a controller

```python
from lauren_ai import Embed  # DI extractor

@controller("/api/search")
class SearchController:
    def __init__(self, embed: Embed) -> None:
        self._embed = embed
        self._store = InMemoryVectorStore()

    @post("/index")
    async def index(self, body: Json[IndexRequest]) -> dict:
        await self._store.add(texts=body.texts, embed_service=self._embed)
        return {"indexed": len(body.texts)}

    @post("/query")
    async def query(self, body: Json[QueryRequest]) -> list[dict]:
        results = await self._store.search(
            query=body.query,
            embed_service=self._embed,
            top_k=body.top_k,
        )
        return [{"text": doc, "score": score} for doc, score in results]
```

---

## Team memory (`TeamMemory`)

Shared key-value state across workers in a team run.  Populated automatically
by `TeamRunner`.

```python
from lauren_ai._teams._memory import TeamMemory

mem = TeamMemory()
await mem.set("researcher", "Quantum computing is ...")
output = await mem.get("researcher")
all_outputs = await mem.get_all()  # dict[str, str]
```

`TeamMemory` is created and managed by `TeamRunner` — you do not need to
instantiate it directly in normal usage.  It is passed to each worker via
`AgentContext.metadata["team_memory"]` if you need to read prior worker
outputs from within an agent lifecycle hook.

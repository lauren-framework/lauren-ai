# Memory

`lauren-ai` provides three memory tiers with different scopes and persistence
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
a fresh one per call.  If you need access to it, use the `on_start` hook via
`ctx.memory`.

---

## Tier 2 — Conversation store (`ConversationStore`)

Persists full conversation history across multiple agent runs within a session.

```python
from lauren_ai import InMemoryConversationStore, ConversationStore

# In-memory (non-persistent)
store: ConversationStore = InMemoryConversationStore()

await store.append(conversation_id="sess-1", messages=[
    Message(role="user", content="Hi"),
    Message(role="assistant", content="Hello!"),
])

history = await store.get(conversation_id="sess-1")
# list[Message]
```

**Custom persistent store:** Implement the `ConversationStore` protocol:

```python
class RedisConversationStore(ConversationStore):
    async def get(self, conversation_id: str) -> list[Message]: ...
    async def append(self, conversation_id: str, messages: list[Message]) -> None: ...
    async def clear(self, conversation_id: str) -> None: ...
```

---

## Tier 3 — User memory with `@remember()`

Long-term facts extracted from conversations and injected as context in future
turns.

```python
from lauren_ai import InMemoryUserMemoryStore, remember, agent, guardrail, use_tools

_memory = InMemoryUserMemoryStore()

@agent(model="openai/gpt-4o-mini", system="You are a personal assistant.")
@remember(store=None, extract=True, inject=True, top_k=5)
@guardrail(input=[PromptInjectionFilter()])
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

**Manual store access:**

```python
from lauren_ai import MemoryFact

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

**Custom `UserMemoryStore`:** Implement the protocol:

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

# Memory Reference

`lauren-ai` has a four-tier memory architecture:

| Tier | Class | Scope | Purpose |
|------|-------|-------|---------|
| Short-term | `ShortTermMemory` | Per run | Rolling message window passed to the LLM each turn |
| Conversation | `ConversationStore` / `InMemoryConversationStore` | Per session | Persists message history across runs |
| User | `UserMemoryStore` / `InMemoryUserMemoryStore` | Per user | Long-term facts extracted by `@remember()` |
| Vector | `InMemoryVectorStore` | Application | Semantic retrieval (TF-IDF) for RAG patterns |

---

## `ShortTermMemory`

Sliding-window conversation buffer for a single agent run. Automatically trims to fit within a token budget. Uses the heuristic `chars / 4 ≈ tokens` when no token-counting transport is available.

```python
from lauren_ai import ShortTermMemory

memory = ShortTermMemory(max_tokens=8000)
memory.add_user("Hello!")
memory.add_assistant(completion)
msgs = memory.messages()  # trimmed list ready to pass to the transport
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int` | `40_000` | Maximum tokens to retain in the sliding window. |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_user` | `(content: str \| list[Any]) -> None` | Append a user message. |
| `add_assistant` | `(completion: Any) -> None` | Append a `Completion` object or `{"role": "assistant", "content": "..."}` dict. Tool-use blocks are included automatically. |
| `add_tool_result` | `(result: Any) -> None` | Append a `ToolResult` object or dict as a `tool_result` block. |
| `messages` | `() -> list[Any]` | Return the message list trimmed to the token budget. Does **not** mutate the internal buffer. |
| `trim_to_fit` | `(max_tokens: int) -> None` | Drop oldest non-system messages in-place until within budget. |
| `clear` | `() -> None` | Remove all messages from the buffer. |
| `snapshot` | `() -> list[Any]` | Return a deep copy of the current message list. |
| `restore` | `(messages: list[Any]) -> None` | Replace the internal buffer with the given message list. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `token_estimate` | `int` | Heuristic token count: `total_chars // 4`. |

Supports `len()` and `repr()`.

---

## `ConversationStore` protocol

Protocol for persisting and retrieving full conversation histories, keyed by an arbitrary `conversation_id` string (typically a session or user identifier).

```python
from lauren_ai import ConversationStore
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `async (conversation_id: str) -> list[Message]` | Load the message history. Returns `[]` when not found. |
| `save` | `async (conversation_id: str, messages: list[Message]) -> None` | Persist the message history. Overwrites any existing history. |
| `delete` | `async (conversation_id: str) -> None` | Delete the history for the given ID. |

### `InMemoryConversationStore`

In-process implementation of `ConversationStore`. Suitable for development and testing. State is lost on restart.

#### Wiring to `AgentRunner`

Declare the store on the agent, or pass it per call as an override:

```python
from lauren_ai import agent, InMemoryConversationStore

store = InMemoryConversationStore()

@agent(model="claude-opus-4-6", conversation_store=store)
class MyAgent: ...
```

Then pass `conversation_id` to `runner.run()` to activate history persistence:

```python
# Turn 1
await runner.run(agent, "My name is Alice.", conversation_id="sess-1")
# Turn 2 — agent sees the full prior exchange
resp = await runner.run(agent, "What is my name?", conversation_id="sess-1")
```

Without `conversation_store` configured on the agent or passed to `runner.run()`,
`conversation_id` is accepted but has no effect — each run starts with an empty
`ShortTermMemory`.

#### Additional methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `list_conversations` | `async () -> list[str]` | Return sorted list of all stored conversation IDs. |
| `clear` | `async () -> None` | Remove all stored conversations. |

Supports `len()` (`__len__`) and `in` (`__contains__`).

---

## `UserMemoryStore` protocol

Protocol for user-level persistent memory that spans conversations. Used by `@remember()`.

```python
from lauren_ai._memory import UserMemoryStore, MemoryFact
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `async (fact: MemoryFact) -> None` | Persist a new memory fact. |
| `get` | `async (user_id, memory_id) -> MemoryFact \| None` | Retrieve a specific fact by ID. |
| `search` | `async (user_id, query, top_k=10) -> list[MemoryFact]` | Semantic search over facts for a user. |
| `list` | `async (user_id, topic=None) -> list[MemoryFact]` | List all facts for a user, optionally filtered by topic. |
| `update` | `async (memory_id, *, content=None, confidence=None) -> None` | Update a fact's content or confidence score. |
| `delete` | `async (memory_id) -> None` | Delete a specific fact. |
| `clear` | `async (user_id) -> None` | Delete all facts for a user. |

### `InMemoryUserMemoryStore`

In-process implementation of `UserMemoryStore`. Suitable for development and testing.

---

## `MemoryFact`

A single persisted fact about a user. Facts carry a confidence score in `[0.0, 1.0]` that can decay over time.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `memory_id` | `str` | Unique identifier for this fact. |
| `user_id` | `str` | The user this fact belongs to. |
| `content` | `str` | The fact text (third person, e.g. `"User prefers dark mode"`). |
| `topics` | `list[str]` | Topic tags extracted from the fact. |
| `confidence` | `float` | Confidence score in `[0.0, 1.0]`. Defaults to `1.0`. |
| `created_at` | `datetime` | UTC timestamp when the fact was created. |
| `last_seen_at` | `datetime` | UTC timestamp when the fact was last reinforced. |
| `source_conversation_id` | `str \| None` | The conversation that generated this fact, if known. |

### Methods

| Method | Description |
|--------|-------------|
| `reinforce()` | Update `last_seen_at` to now and boost `confidence` by `+0.1` (capped at `1.0`). |
| `decay(factor=0.8)` | Reduce `confidence` by multiplying by `factor`. |

---

## `@remember()` decorator

Opts an `@agent()`-decorated class into automatic user memory extraction and injection. Must be applied below `@agent()`:

```python
@agent(model="claude-haiku-4-5")
@remember(store="user_memory", extract=True, inject=True, top_k=5)
class PersonalAssistant: ...
```

When `inject=True`, relevant memories are prepended to the system prompt before each LLM call.

When `extract=True`, new facts are extracted from each conversation turn and stored in the `UserMemoryStore`.

Must be called **with parentheses** — bare `@remember` raises `MemoryConfigError`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `str \| None` | `None` | DI token name for the `UserMemoryStore`. `None` auto-injects `UserMemoryStore` directly. |
| `extract` | `bool` | `True` | Extract new facts from each conversation turn. |
| `inject` | `bool` | `True` | Inject relevant memories into the system prompt before each turn. |
| `top_k` | `int` | `5` | Number of memories to inject. |
| `extraction_model` | `str \| None` | `None` | Model used for fact extraction. Defaults to the agent's model. |

### `RememberMeta`

Metadata stored on the decorated class under the `REMEMBER_META` attribute (`"__lauren_ai_remember__"`).

| Field | Type | Description |
|-------|------|-------------|
| `store_token` | `str \| None` | DI token name for the store, or `None`. |
| `extract` | `bool` | Whether to extract facts. |
| `inject` | `bool` | Whether to inject memories. |
| `top_k` | `int` | Number of memories to inject. |
| `extraction_model` | `str \| None` | Extraction model override. |

---

## `InMemoryVectorStore`

In-process vector store implementing the `MemoryStore` protocol using TF-IDF cosine similarity. No external dependencies required. Uses `numpy` for performance when available, falling back to pure Python.

```python
from lauren_ai._memory._vector import InMemoryVectorStore

store = InMemoryVectorStore()
doc_id = await store.upsert("The quick brown fox", metadata={"tag": "test"})
results = await store.search("quick fox", k=3)
```

### `MemoryStore` protocol methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `upsert` | `async (content, *, id=None, metadata=None, embedding=None) -> str` | Insert or update a document. Returns the document ID. When `embedding` is provided it is used directly (after L2-normalisation); otherwise TF-IDF is computed. |
| `search` | `async (query, *, k=5, filter=None) -> list[MemoryResult]` | Semantic search. Returns up to `k` results ordered by descending cosine similarity. `filter` applies exact-match metadata filtering. |
| `get` | `async (id) -> MemoryResult \| None` | Retrieve a document by ID. |
| `delete` | `async (ids: list[str]) -> None` | Delete documents. Non-existent IDs are silently ignored. |
| `clear` | `async () -> None` | Remove all documents. |

### `MemoryResult`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier of the stored document. |
| `content` | `str` | Original text content. |
| `score` | `float` | Similarity score in `[0.0, 1.0]`; higher is more similar. |
| `metadata` | `dict[str, Any]` | Key/value metadata attached at upsert time. |

Supports `len()` and `repr()`.

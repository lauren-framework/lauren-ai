# Memory

Short-term memory, conversation stores, vector stores, and user memory.

## Short-term memory

### `ShortTermMemory`

```python
class ShortTermMemory(max_tokens: int = 40000)
```

Sliding-window conversation buffer for a single agent run.

Stores the ordered message history and automatically trims to fit within a
token budget when requested.  Uses the heuristic `chars / 4 ≈ tokens`
when no token-counting transport is available.

:Example:

.. code-block:: python

    memory = ShortTermMemory(max_tokens=8000)
    memory.add_user("Hello, how are you?")
    memory.add_assistant(completion)
    msgs = memory.messages()  # trimmed to budget

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `max_tokens` | `int` | Maximum number of tokens to retain in the window.
Defaults to 40 000. |

#### `ShortTermMemory.add_user`

```python
def add_user(self, content: str | list[Any]) -> None
```

Append a user message to the buffer.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `content` | `str | list[Any]` | Plain text string or list of content blocks. |

#### `ShortTermMemory.add_assistant`

```python
def add_assistant(self, completion: Any) -> None
```

Append an assistant completion to the buffer.

Accepts a `Completion` dataclass (with `.content` and
`.tool_calls` attributes) or a plain dict.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `completion` | `Any` | A `Completion` object or `{"role": "assistant",
"content": "..."}` dict. |

#### `ShortTermMemory.add_tool_result`

```python
def add_tool_result(self, result: Any) -> None
```

Append a tool result message to the buffer.

Accepts a `ToolResult` dataclass or a plain dict.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `result` | `Any` | A `ToolResult` object or dict. |

#### `ShortTermMemory.messages`

```python
def messages(self) -> list[Any]
```

Return the current message list, trimmed to fit the token window.

The trim is applied in-place on a copy; the internal buffer is NOT
modified.  Call `trim_to_fit()` explicitly to mutate the buffer.

**Returns:** `list[Message]` — Ordered list of messages within the token budget.

#### `ShortTermMemory.trim_to_fit`

```python
def trim_to_fit(self, max_tokens: int) -> None
```

Drop oldest non-system messages until the token estimate fits.

Unlike `messages()` this *mutates* the internal buffer.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `max_tokens` | `int` | Target token budget. |

#### `ShortTermMemory.clear`

```python
def clear(self) -> None
```

Clear all messages from the buffer.

**Returns:** `None` — None

#### `ShortTermMemory.snapshot`

```python
def snapshot(self) -> list[Any]
```

Return a deep copy of the current message list.

The returned list is independent of the internal buffer; mutations to
it do not affect the memory.

**Returns:** `list[Message]` — Immutable snapshot of the conversation history.

#### `ShortTermMemory.restore`

```python
def restore(self, messages: list[Any]) -> None
```

Restore the message buffer from a snapshot.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `messages` | `list[Any]` | Ordered list of `Message` objects (typically
produced by `snapshot()`). |

## Conversation store

### `ConversationStore`

```python
class ConversationStore
```

Protocol for persisting and retrieving full conversation histories.

Keyed by an arbitrary string `conversation_id` (typically a session or
user identifier).

#### `ConversationStore.load`

```python
def load(self, conversation_id: str) -> list[Any]
```

Load the message history for *conversation_id*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation / session identifier. |

**Returns:** `list[Message]` — Ordered list of `Message` objects (empty list when not
found).

#### `ConversationStore.save`

```python
def save(self, conversation_id: str, messages: list[Any]) -> None
```

Persist the message history for *conversation_id*.

Overwrites any existing history for that ID.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation / session identifier. |
| `messages` | `list[Any]` | Ordered list of `Message` objects to persist. |

#### `ConversationStore.delete`

```python
def delete(self, conversation_id: str) -> None
```

Delete the history for *conversation_id*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation / session identifier. |

### `InMemoryConversationStore`

```python
class InMemoryConversationStore()
```

In-memory store for full conversation histories.

Implements the `ConversationStore` protocol.  Each conversation is keyed
by an arbitrary string identifier (typically a user ID or session UUID).
Deep copies are used on both `load` and `save` so that the caller
cannot inadvertently mutate stored data.

:Example:

.. code-block:: python

    store = InMemoryConversationStore()
    await store.save("session-abc", messages)
    loaded = await store.load("session-abc")

#### `InMemoryConversationStore.load`

```python
def load(self, conversation_id: str) -> list[Any]
```

Load the message history for *conversation_id*.

Returns an empty list when the conversation does not exist.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation identifier. |

**Returns:** `list[Message]` — A deep copy of the stored message list (empty list when not
found).

#### `InMemoryConversationStore.save`

```python
def save(self, conversation_id: str, messages: list[Any]) -> None
```

Persist the message history for *conversation_id*.

Overwrites any existing history for that identifier.  A deep copy of
*messages* is stored to prevent the caller from mutating the stored
data.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation identifier. |
| `messages` | `list[Any]` | Ordered list of `Message` objects to persist. |

#### `InMemoryConversationStore.delete`

```python
def delete(self, conversation_id: str) -> None
```

Delete the history for *conversation_id*.

Silently does nothing when the conversation does not exist.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation identifier. |

#### `InMemoryConversationStore.list_conversations`

```python
def list_conversations(self) -> list[str]
```

Return a sorted list of all stored conversation identifiers.

**Returns:** `list[str]` — Sorted list of conversation IDs.

#### `InMemoryConversationStore.clear`

```python
def clear(self) -> None
```

Remove all stored conversation histories.

**Returns:** `None` — None

## Vector store

### `InMemoryVectorStore`

```python
class InMemoryVectorStore()
```

In-memory vector store using TF-IDF cosine similarity.

Implements the `MemoryStore` protocol.  Suitable for development and
testing; no external dependencies required.

:Example:

.. code-block:: python

    store = InMemoryVectorStore()
    doc_id = await store.upsert("The quick brown fox", metadata={"tag": "test"})
    results = await store.search("quick fox", k=3)

#### `InMemoryVectorStore.upsert`

```python
def upsert(self, content: str, id: str | None = None, metadata: dict[str, Any] | None = None, embedding: list[float] | None = None) -> str
```

Insert or update a document.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `content` | `str` | The text content to store. |
| `id` | `str | None` | Optional stable identifier.  A UUID4 is generated when
`None`. |
| `metadata` | `dict[str, Any] | None` | Optional key/value metadata dict. |
| `embedding` | `list[float] | None` | Pre-computed embedding vector as a list of floats.
When provided it is used directly (after L2-normalisation); the
TF-IDF computation is skipped.  Must be dense and compatible with
the cosine-similarity computation. |

**Returns:** `str` — The document's identifier.

#### `InMemoryVectorStore.search`

```python
def search(self, query: str, k: int = 5, filter: dict[str, Any] | None = None) -> list[MemoryResult]
```

Search for documents semantically similar to *query*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `query` | `str` | Natural-language query string. |
| `k` | `int` | Maximum number of results to return. |
| `filter` | `dict[str, Any] | None` | Optional metadata filter.  Only documents whose
metadata contains **all** specified key/value pairs are returned. |

**Returns:** `list[MemoryResult]` — Up to *k* results ordered by descending cosine similarity.

#### `InMemoryVectorStore.get`

```python
def get(self, id: str) -> MemoryResult | None
```

Retrieve a document by its identifier.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `id` | `str` | Document identifier. |

**Returns:** `MemoryResult | None` — The `MemoryResult`, or `None` when not found.

#### `InMemoryVectorStore.delete`

```python
def delete(self, ids: list[str]) -> None
```

Delete documents by their identifiers.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `ids` | `list[str]` | List of document identifiers to remove. |

#### `InMemoryVectorStore.clear`

```python
def clear(self) -> None
```

Remove all documents from the store.

**Returns:** `None` — None

## User memory

### `MemoryFact`

A single persisted fact about a user.

Facts are stored with a confidence score [0.0–1.0] that
can decay over time if not reinforced.

### `UserMemoryStore`

Protocol for user-level persistent memory stores.

### `InMemoryUserMemoryStore`

In-process UserMemoryStore for testing and development.

Uses simple substring matching for search (no vector similarity).

## `@remember` decorator

### `remember`

Opt a @agent() class into automatic user memory extraction/injection.

Must be applied BELOW @agent()::

    @agent(model="claude-haiku-4-5")
    @remember(store="user_memory", extract=True, inject=True, top_k=5)
    class PersonalAssistant: ...

When inject=True, relevant memories are prepended to the system prompt
before each LLM call.

When extract=True, new facts are extracted from each conversation turn
and stored in the UserMemoryStore.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `store` | `str | None` | DI token name for UserMemoryStore (None = auto-inject). |
| `extract` | `bool` | Extract new facts after each turn. |
| `inject` | `bool` | Inject relevant memories before each turn. |
| `top_k` | `int` | Number of memories to inject. |
| `extraction_model` | `str | None` | Model for fact extraction (defaults to agent model). |

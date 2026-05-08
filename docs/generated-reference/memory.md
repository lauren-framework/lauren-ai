# Memory

Short-term memory, conversation stores, vector stores, and user memory.

## Short-term memory

### `ShortTermMemory`

```python
class ShortTermMemory(max_tokens: int = 40000)
```

Sliding-window conversation buffer for a single agent run.

Stores the ordered message history and automatically trims to fit within a
token budget when requested.  Uses the heuristic ``chars / 4 ≈ tokens``
when no token-counting transport is available.

:param max_tokens: Maximum number of tokens to retain in the window.
    Defaults to 40 000.
:type max_tokens: int

:Example:

.. code-block:: python

    memory = ShortTermMemory(max_tokens=8000)
    memory.add_user("Hello, how are you?")
    memory.add_assistant(completion)
    msgs = memory.messages()  # trimmed to budget

#### `ShortTermMemory.add_user`

```python
def add_user(self, content: str | list[Any]) -> None
```

Append a user message to the buffer.

:param content: Plain text string or list of content blocks.
:type content: str | list[Any]

#### `ShortTermMemory.add_assistant`

```python
def add_assistant(self, completion: Any) -> None
```

Append an assistant completion to the buffer.

Accepts a ``Completion`` dataclass (with ``.content`` and
``.tool_calls`` attributes) or a plain dict.

:param completion: A ``Completion`` object or ``{"role": "assistant",
    "content": "..."}`` dict.
:type completion: Any

#### `ShortTermMemory.add_tool_result`

```python
def add_tool_result(self, result: Any) -> None
```

Append a tool result message to the buffer.

Accepts a ``ToolResult`` dataclass or a plain dict.

:param result: A ``ToolResult`` object or dict.
:type result: Any

#### `ShortTermMemory.messages`

```python
def messages(self) -> list[Any]
```

Return the current message list, trimmed to fit the token window.

The trim is applied in-place on a copy; the internal buffer is NOT
modified.  Call ``trim_to_fit()`` explicitly to mutate the buffer.

:return: Ordered list of messages within the token budget.
:rtype: list[Message]

#### `ShortTermMemory.trim_to_fit`

```python
def trim_to_fit(self, max_tokens: int) -> None
```

Drop oldest non-system messages until the token estimate fits.

Unlike ``messages()`` this *mutates* the internal buffer.

:param max_tokens: Target token budget.
:type max_tokens: int

#### `ShortTermMemory.clear`

```python
def clear(self) -> None
```

Clear all messages from the buffer.

:return: None
:rtype: None

#### `ShortTermMemory.snapshot`

```python
def snapshot(self) -> list[Any]
```

Return a deep copy of the current message list.

The returned list is independent of the internal buffer; mutations to
it do not affect the memory.

:return: Immutable snapshot of the conversation history.
:rtype: list[Message]

#### `ShortTermMemory.restore`

```python
def restore(self, messages: list[Any]) -> None
```

Restore the message buffer from a snapshot.

:param messages: Ordered list of ``Message`` objects (typically
    produced by ``snapshot()``).
:type messages: list[Message]

## Conversation store

### `ConversationStore`

```python
class ConversationStore
```

Protocol for persisting and retrieving full conversation histories.

Keyed by an arbitrary string ``conversation_id`` (typically a session or
user identifier).

#### `ConversationStore.load`

```python
def load(self, conversation_id: str) -> list[Any]
```

Load the message history for *conversation_id*.

:param conversation_id: Unique conversation / session identifier.
:type conversation_id: str
:return: Ordered list of ``Message`` objects (empty list when not
    found).
:rtype: list[Message]

#### `ConversationStore.save`

```python
def save(self, conversation_id: str, messages: list[Any]) -> None
```

Persist the message history for *conversation_id*.

Overwrites any existing history for that ID.

:param conversation_id: Unique conversation / session identifier.
:type conversation_id: str
:param messages: Ordered list of ``Message`` objects to persist.
:type messages: list[Message]

#### `ConversationStore.delete`

```python
def delete(self, conversation_id: str) -> None
```

Delete the history for *conversation_id*.

:param conversation_id: Unique conversation / session identifier.
:type conversation_id: str

### `InMemoryConversationStore`

```python
class InMemoryConversationStore()
```

In-memory store for full conversation histories.

Implements the ``ConversationStore`` protocol.  Each conversation is keyed
by an arbitrary string identifier (typically a user ID or session UUID).
Deep copies are used on both ``load`` and ``save`` so that the caller
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

:param conversation_id: Unique conversation identifier.
:type conversation_id: str
:return: A deep copy of the stored message list (empty list when not
    found).
:rtype: list[Message]

#### `InMemoryConversationStore.save`

```python
def save(self, conversation_id: str, messages: list[Any]) -> None
```

Persist the message history for *conversation_id*.

Overwrites any existing history for that identifier.  A deep copy of
*messages* is stored to prevent the caller from mutating the stored
data.

:param conversation_id: Unique conversation identifier.
:type conversation_id: str
:param messages: Ordered list of ``Message`` objects to persist.
:type messages: list[Message]

#### `InMemoryConversationStore.delete`

```python
def delete(self, conversation_id: str) -> None
```

Delete the history for *conversation_id*.

Silently does nothing when the conversation does not exist.

:param conversation_id: Unique conversation identifier.
:type conversation_id: str

#### `InMemoryConversationStore.list_conversations`

```python
def list_conversations(self) -> list[str]
```

Return a sorted list of all stored conversation identifiers.

:return: Sorted list of conversation IDs.
:rtype: list[str]

#### `InMemoryConversationStore.clear`

```python
def clear(self) -> None
```

Remove all stored conversation histories.

:return: None
:rtype: None

## Vector store

### `InMemoryVectorStore`

```python
class InMemoryVectorStore()
```

In-memory vector store using TF-IDF cosine similarity.

Implements the ``MemoryStore`` protocol.  Suitable for development and
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

:param content: The text content to store.
:type content: str
:param id: Optional stable identifier.  A UUID4 is generated when
    ``None``.
:type id: str | None
:param metadata: Optional key/value metadata dict.
:type metadata: dict[str, Any] | None
:param embedding: Pre-computed embedding vector as a list of floats.
    When provided it is used directly (after L2-normalisation); the
    TF-IDF computation is skipped.  Must be dense and compatible with
    the cosine-similarity computation.
:type embedding: list[float] | None
:return: The document's identifier.
:rtype: str

#### `InMemoryVectorStore.search`

```python
def search(self, query: str, k: int = 5, filter: dict[str, Any] | None = None) -> list[MemoryResult]
```

Search for documents semantically similar to *query*.

:param query: Natural-language query string.
:type query: str
:param k: Maximum number of results to return.
:type k: int
:param filter: Optional metadata filter.  Only documents whose
    metadata contains **all** specified key/value pairs are returned.
:type filter: dict[str, Any] | None
:return: Up to *k* results ordered by descending cosine similarity.
:rtype: list[MemoryResult]

#### `InMemoryVectorStore.get`

```python
def get(self, id: str) -> MemoryResult | None
```

Retrieve a document by its identifier.

:param id: Document identifier.
:type id: str
:return: The ``MemoryResult``, or ``None`` when not found.
:rtype: MemoryResult | None

#### `InMemoryVectorStore.delete`

```python
def delete(self, ids: list[str]) -> None
```

Delete documents by their identifiers.

:param ids: List of document identifiers to remove.
:type ids: list[str]

#### `InMemoryVectorStore.clear`

```python
def clear(self) -> None
```

Remove all documents from the store.

:return: None
:rtype: None

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

:param store: DI token name for UserMemoryStore (None = auto-inject).
:param extract: Extract new facts after each turn.
:param inject: Inject relevant memories before each turn.
:param top_k: Number of memories to inject.
:param extraction_model: Model for fact extraction (defaults to agent model).


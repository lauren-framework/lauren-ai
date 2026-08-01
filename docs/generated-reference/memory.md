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

#### `ShortTermMemory.add_tool_results`

```python
def add_tool_results(self, results: list[Any]) -> None
```

Append multiple tool results as a **single** consolidated message.

Anthropic requires all `tool_result` blocks for a given assistant
turn to appear in the *same* immediately-following user message.
Calling `add_tool_result` in a loop creates N separate messages,
which causes Anthropic to report that only the first ID is answered
and the rest are missing (400 error).

This method consolidates all results into one `role:"user"` message
with multiple content blocks, satisfying Anthropic's constraint while
remaining compatible with OpenAI (the transport converts each block to
a separate `role:"tool"` message as needed).

For a single result, delegates to `add_tool_result` unchanged.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `results` | `list[Any]` | List of `ToolResult` objects or dicts to add. |

#### `ShortTermMemory.set_summary`

```python
def set_summary(self, text: str) -> None
```

Store *text* as the conversation summary.

Called by the runner after a summarisation LLM call completes.
The summary is persisted via `snapshot()` / `restore()` so
resumed sessions carry it forward.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `text` | `str` | Compressed summary of older conversation turns. |

#### `ShortTermMemory.messages_to_summarize`

```python
def messages_to_summarize(self, keep_recent: int = 6) -> list[Any]
```

Return the slice of messages that should be compressed.

Returns the oldest non-system messages outside the (boundary-safe)
recent window.  System messages are excluded because they are already
managed separately (they are never dropped by `messages()` either).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `keep_recent` | `int` | Number of most-recent non-system messages to
preserve verbatim.  Defaults to 6 (≈ 3 user/assistant pairs).  The
boundary is snapped so a `tool_use`/`tool_result` pair is never
split (see `_safe_keep_recent()`). |

**Returns:** `list[Any]` — List of messages to feed to the summarisation LLM call.

#### `ShortTermMemory.trim_to_recent`

```python
def trim_to_recent(self, keep_recent: int = 6) -> None
```

Drop all but the most-recent *keep_recent* non-system messages.

Called by the runner after the summarisation call so the buffer
only holds recent turns while the older context lives in
`self._summary`.  The boundary is snapped so a `tool_use`/
`tool_result` pair is never split (see `_safe_keep_recent()`).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `keep_recent` | `int` | Number of most-recent non-system messages to
keep.  Defaults to 6. |

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

#### `ShortTermMemory.ensure_valid`

```python
def ensure_valid(self) -> None
```

Heal dangling tool_calls in-place before making an API request.

Unlike `messages()` (which has a `has_moved_on` guard to avoid
healing mid-flight tool calls) this method heals *unconditionally*.
It should be called once immediately before each `run_stream()` /
`run()` invocation to handle cases where a previous agent turn was
interrupted while a tool was suspended — e.g. when the user cancels
a plan-approval overlay after the LLM has already called the approval
tool for a second time.

The method is idempotent — calling it multiple times is safe.

#### `ShortTermMemory.clear`

```python
def clear(self) -> None
```

Clear all messages from the buffer.

**Returns:** `None` — None

#### `ShortTermMemory.snapshot`

```python
def snapshot(self) -> Any
```

Return a deep copy of the current memory state.

The returned object includes both the message list and the
conversation summary (if any).  It is independent of the internal
buffer — mutations do not affect the memory.

The format is a `dict` with `"messages"` and `"summary"` keys
so that resumed sessions carry the summary forward.  Old snapshots
that are plain `list` objects are still accepted by `restore()`
for backward compatibility.

**Returns:** `dict[str, Any]` — Snapshot dict `{"messages": [...], "summary": str | None}`.

#### `ShortTermMemory.restore`

```python
def restore(self, data: Any) -> None
```

Restore the memory buffer from a snapshot.

Accepts both the new `dict` snapshot format (`{"messages": [...],
"summary": ...}`) and the legacy plain `list` format produced by
older versions of `snapshot()`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `data` | `Any` | Snapshot produced by `snapshot()`, or a plain list of
message objects for backward compatibility. |

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
def load(self, conversation_id: str) -> Any
```

Load the conversation snapshot for *conversation_id*.

Returns an empty list when the conversation does not exist (backward
compat — callers that check `if prior:` still work on empty lists).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation identifier. |

**Returns:** `dict[str, Any] | list[Any]` — A deep copy of the stored snapshot.  When the snapshot was
created by `ShortTermMemory.snapshot()` this is a
`{"messages": [...], "summary": ...}` dict; for legacy plain
lists the raw list is returned.

#### `InMemoryConversationStore.save`

```python
def save(self, conversation_id: str, snapshot: Any) -> None
```

Persist the conversation snapshot for *conversation_id*.

Overwrites any existing entry for that identifier.  A deep copy is
stored to prevent the caller from mutating the stored data.

Accepts both the new dict snapshot format
(`{"messages": [...], "summary": ...}`) produced by
`ShortTermMemory.snapshot()` and the legacy plain `list[Message]`
format so that code written against the old API continues to work.
Plain lists are automatically normalised to the dict format so that
`load()` always returns a consistent shape.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `conversation_id` | `str` | Unique conversation identifier. |
| `snapshot` | `Any` | Snapshot dict or message list to persist. |

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

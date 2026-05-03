# Transport Reference

The transport layer defines the provider-agnostic data model shared by all LLM backend implementations (Anthropic, OpenAI, Ollama, MockTransport).

---

## `Message`

A single message in a conversation.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `role` | `Literal["user", "assistant"]` | The sender role. |
| `content` | `str \| list[ContentBlock]` | Message content. Either a plain string or a list of `ContentBlock` instances for multi-modal / tool-use messages. |

### Classmethods

| Method | Signature | Description |
|--------|-----------|-------------|
| `Message.user` | `(content) -> Message` | Convenience factory for a user message. |
| `Message.assistant` | `(content) -> Message` | Convenience factory for an assistant message. |
| `Message.from_multimodal` | `(role, parts: list[Any]) -> Message` | Create a message from a mixed list of text strings and multimodal content objects (`ImageContent`, `AudioContent`, `DocumentContent`). |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `text` | `() -> str` | Extract all text blocks from `content` and return them as a single concatenated string. |

---

## `ContentBlock`

A single typed block within a `Message.content` list when the content is structured.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["text", "tool_use", "tool_result", "image"]` | Block type discriminator. |
| `text` | `str \| None` | Text content (for `type="text"`). |
| `tool_use_id` | `str \| None` | Provider-assigned tool call identifier. |
| `name` | `str \| None` | Tool name (for `type="tool_use"`). |
| `input` | `dict[str, Any] \| None` | Tool input arguments (for `type="tool_use"`). |
| `content` | `str \| list[Any] \| None` | Tool result payload (for `type="tool_result"`). |
| `source` | `dict[str, Any] \| None` | Image source descriptor (for `type="image"`), e.g. `{"type": "base64", "media_type": "image/png", "data": "..."}`. |

### Classmethods

| Method | Description |
|--------|-------------|
| `ContentBlock.text_block(text)` | Create a plain text block. |
| `ContentBlock.tool_use_block(name, tool_input, tool_use_id=None)` | Create a tool-use block. Generates a random `tool_use_id` when `None`. |
| `ContentBlock.tool_result_block(tool_use_id, content)` | Create a tool-result block. |

---

## `Completion`

A finished (non-streaming) model completion.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Provider-assigned completion identifier. |
| `model` | `str` | The model that produced this completion. |
| `content` | `str` | The primary text response from the model. |
| `tool_calls` | `list[ToolCall]` | All tool calls requested by the model in this turn. |
| `stop_reason` | `Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]` | Why the model stopped generating. |
| `usage` | `TokenUsage` | Token usage statistics for this completion. |
| `thinking_blocks` | `list[ThinkingBlock \| RedactedThinkingBlock]` | Extended-thinking blocks (Anthropic only). Defaults to `[]`. |

### `stop_reason` values

| Value | Meaning |
|-------|---------|
| `"end_turn"` | Natural end of response. |
| `"tool_use"` | One or more tool calls were requested. |
| `"max_tokens"` | The configured token limit was reached. |
| `"stop_sequence"` | A stop sequence was hit. |

---

## `CompletionChunk`

A single chunk from a streaming model completion. Only one of `delta`, `thinking_delta`, or `tool_call_delta` is typically populated per chunk.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `delta` | `str` | Text content delta for this chunk. |
| `thinking_delta` | `str \| None` | Reasoning / thinking text delta (extended thinking). |
| `tool_call_delta` | `ToolCallDelta \| None` | Partial tool call update for this chunk. |
| `stop_reason` | `str \| None` | Stop reason; populated only in the final chunk. |
| `usage` | `TokenUsage \| None` | Token usage; populated only in the final chunk. |
| `pending_approval` | `PendingApproval \| None` | Human-in-the-loop signal present when a tool call requires confirmation. |

---

## `TokenUsage`

Token accounting for a single model call.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Number of tokens in the prompt / input messages. |
| `output_tokens` | `int` | Number of tokens in the completion. |
| `cache_read_tokens` | `int` | Tokens read from the prompt cache (Anthropic). Defaults to `0`. |
| `cache_write_tokens` | `int` | Tokens written to the prompt cache (Anthropic). Defaults to `0`. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_tokens` | `int` | Sum of `input_tokens + output_tokens`. |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `cost_usd` | `(model: str) -> float` | Estimate cost in USD using a built-in price table with prefix-matching. Falls back to `$1/$3` per million tokens for unknown models. |
| `__add__` | `(other: TokenUsage) -> TokenUsage` | Add two `TokenUsage` instances by summing all fields. |

---

## `ToolCall`

A completed tool call extracted from a model response.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_use_id` | `str` | Provider-assigned identifier for this tool call. |
| `name` | `str` | The registered tool name. |
| `input` | `dict[str, Any]` | The parsed JSON input arguments. |

---

## `ToolSchema`

JSON Schema descriptor for a tool exposed to the model.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | The tool's registered name (snake_case). |
| `description` | `str` | Human-readable description used in the model's context. |
| `input_schema` | `dict[str, Any]` | JSON Schema `object` describing the tool's input parameters. |

---

## `ToolChoice`

Constraint on which tool(s) the model may call.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["auto", "any", "tool"]` | `"auto"` — model decides; `"any"` — must call at least one tool; `"tool"` — must call the specific named tool. |
| `name` | `str \| None` | Required when `type="tool"`. |

### Classmethods

| Method | Description |
|--------|-------------|
| `ToolChoice.auto()` | Model decides whether to call a tool. |
| `ToolChoice.required()` | Model must call at least one tool (maps to `type="any"`). |
| `ToolChoice.specific(name)` | Model must call exactly the named tool. Raises `ValueError` for empty `name`. |

---

## `ToolCallDelta`

A partial tool call update in a streaming response.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_use_id` | `str` | Provider-assigned identifier for this tool call. |
| `name` | `str \| None` | Tool name; populated only in the first delta chunk. |
| `input_delta` | `str` | Partial JSON string fragment for the input arguments. |

---

## `ThinkingBlock` / `RedactedThinkingBlock`

Extended-thinking blocks from an Anthropic model (Anthropic only).

### `ThinkingBlock` fields

| Field | Type | Description |
|-------|------|-------------|
| `thinking` | `str` | The model's reasoning text. |
| `signature` | `str` | Cryptographic signature from Anthropic confirming authenticity. |
| `type` | `Literal["thinking"]` | Always `"thinking"`. |

### `RedactedThinkingBlock` fields

| Field | Type | Description |
|-------|------|-------------|
| `data` | `str` | Opaque base64-encoded redacted thinking data. |
| `type` | `Literal["redacted_thinking"]` | Always `"redacted_thinking"`. |

---

## `Transport` protocol

Provider-agnostic interface that all transport implementations must satisfy.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `complete` | `async (messages, *, model, system=None, tools=None, tool_choice=None, max_tokens=4096, temperature=1.0, stop_sequences=None, stream=False, thinking=False, thinking_budget_tokens=8000) -> Completion \| AsyncIterator[CompletionChunk]` | Send messages to the model. Returns `Completion` when `stream=False`; returns `AsyncIterator[CompletionChunk]` when `stream=True`. |
| `embed` | `async (inputs, *, model, dimensions=None) -> list[Embedding]` | Generate embeddings for a list of text strings. |
| `count_tokens` | `async (messages, *, model, system=None, tools=None) -> int` | Estimate token count without generating a completion. |

### `Embedding`

| Field | Type | Description |
|-------|------|-------------|
| `index` | `int` | Zero-based index of this embedding in the batch. |
| `vector` | `list[float]` | The floating-point embedding vector. |

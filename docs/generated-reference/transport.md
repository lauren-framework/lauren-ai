# Transport & Multimodal

Core message types exchanged with LLM providers.

## Messages & completions

### `Message`

```python
class Message(role: Literal['user', 'assistant'], content: str | list[ContentBlock])
```

A single message in a conversation.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `role` | `Literal['user', 'assistant']` | The sender role.  Either `"user"` or `"assistant"`. |
| `content` | `str | list[ContentBlock]` | Message content.  Either a plain string or a list of
`ContentBlock` instances for multi-modal / tool-use messages. |

#### `Message.user`

```python
def user(cls, content: str | list[ContentBlock]) -> Message
```

Convenience factory for a user message.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `content` | `str | list[ContentBlock]` | Message content. |

**Returns:** `Message` — A new user `Message`.

#### `Message.assistant`

```python
def assistant(cls, content: str | list[ContentBlock]) -> Message
```

Convenience factory for an assistant message.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `content` | `str | list[ContentBlock]` | Message content. |

**Returns:** `Message` — A new assistant `Message`.

#### `Message.from_multimodal`

```python
def from_multimodal(cls, role: str, parts: list[Any]) -> Message
```

Create a Message from a list of text strings and content objects.

Accepts any mix of plain strings, `ImageContent`,
`AudioContent`, and
`DocumentContent` instances.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `role` | `str` | The sender role — `"user"` or `"assistant"`. |
| `parts` | `list[Any]` | Mixed list of text strings and multimodal content objects. |

**Returns:** `Message` — A new `Message` whose `content` is the *parts* list.

#### `Message.text`

```python
def text(self) -> str
```

Extract all text from this message as a single concatenated string.

**Returns:** `str` — All text content concatenated together.

### `Completion`

```python
class Completion(id: str, model: str, content: str, tool_calls: list[ToolCall], stop_reason: Literal['end_turn', 'tool_use', 'max_tokens', 'stop_sequence'], usage: TokenUsage, thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] = list(), provider: str | None = None, request_id: str | None = None, provider_metadata: dict[str, Any] = dict(), raw_response: Any = None)
```

A finished (non-streaming) model completion.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `id` | `str` | Provider-assigned completion identifier. |
| `model` | `str` | The model that produced this completion. |
| `content` | `str` | The primary text response from the model. |
| `tool_calls` | `list[ToolCall]` | All tool calls requested by the model in this turn. |
| `stop_reason` | `Literal['end_turn', 'tool_use', 'max_tokens', 'stop_sequence']` | Why the model stopped generating:

* `"end_turn"` — natural end of response.
* `"tool_use"` — one or more tool calls were requested.
* `"max_tokens"` — the configured token limit was reached.
* `"stop_sequence"` — a stop sequence was hit. |
| `usage` | `TokenUsage` | Token usage statistics for this completion. |
| `thinking_blocks` | `list[ThinkingBlock | RedactedThinkingBlock]` | Extended-thinking blocks (Anthropic only). |

### `CompletionChunk`

```python
class CompletionChunk(delta: str = '', thinking_delta: str | None = None, tool_call_delta: ToolCallDelta | None = None, stop_reason: str | None = None, usage: TokenUsage | None = None, pending_approval: PendingApproval | None = None, guardrail_override: str | None = None, system_notice: str | None = None, thinking_signature: str | None = None, redacted_thinking_data: str | None = None, provider_metadata: dict[str, Any] | None = None, raw_event: Any = None)
```

A single chunk from a streaming model completion.

Only one of *delta*, *thinking_delta*, or *tool_call_delta* is populated
per chunk (though they are not mutually exclusive in the schema).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `delta` | `str` | Text content delta for this chunk. |
| `thinking_delta` | `str | None` | Reasoning / thinking text delta (extended thinking). |
| `tool_call_delta` | `ToolCallDelta | None` | Partial tool call update for this chunk. |
| `stop_reason` | `str | None` | Stop reason, populated only in the final chunk. |
| `usage` | `TokenUsage | None` | Token usage, populated only in the final chunk. |
| `pending_approval` | `PendingApproval | None` | Human-in-the-loop pending-approval signal.
Present when a tool call requires confirmation before execution. |
| `guardrail_override` | `str | None` | When set, an output guardrail fired and this
string is the replacement content.  The runner emits one sentinel
chunk with only this field populated (`delta=""` etc.) after all
normal chunks have been yielded.  Callers should replace the
accumulated streaming text with this value. |
| `system_notice` | `str | None` | An out-of-band, user-facing status line (e.g.
`"⎋ Compacting conversation…"`).  Emitted as a sentinel chunk with no
other field populated; callers surface it to the user but must NOT add
it to the assistant's accumulated content. |
| `thinking_signature` | `str | None` | The cryptographic signature for the thinking block
that just completed (Anthropic extended thinking).  Emitted when the
`signature_delta` arrives; finalises the accumulated `thinking_delta`
text into a complete thinking block that must be round-tripped verbatim. |
| `redacted_thinking_data` | `str | None` | Opaque base64 payload of a
`redacted_thinking` block, delivered whole.  Must be preserved and sent
back unchanged. |

## Usage & calls

### `TokenUsage`

```python
class TokenUsage(input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0, audio_input_tokens: int = 0, audio_output_tokens: int = 0, provider_metadata: dict[str, Any] = dict())
```

Token accounting for a single model call.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `input_tokens` | `int` | Number of tokens in the prompt / input messages. |
| `output_tokens` | `int` | Number of tokens in the completion. |
| `cache_read_tokens` | `int` | Tokens read from the prompt cache (Anthropic). |
| `cache_write_tokens` | `int` | Tokens written to the prompt cache (Anthropic). |

#### `TokenUsage.cost_usd`

```python
def cost_usd(self, model: str) -> float
```

Estimate the cost in USD for this usage against *model*.

Uses a bundled price table with prefix-matching.  Falls back to
`$1/$3` per million tokens when the model is unknown.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `str` | Model identifier used for the completion. |

**Returns:** `float` — Estimated cost in USD.

### `ToolCall`

```python
class ToolCall(tool_use_id: str, name: str, input: dict[str, Any])
```

A completed tool call extracted from a model response.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `tool_use_id` | `str` | Provider-assigned identifier for this tool call. |
| `name` | `str` | The registered tool name. |
| `input` | `dict[str, Any]` | The parsed JSON input arguments. |

### `ToolSchema`

```python
class ToolSchema(name: str, description: str, input_schema: dict[str, Any], kind: Literal['function', 'hosted', 'mcp', 'computer_use', 'tool_search'] = 'function', provider_options: dict[str, Any] = dict())
```

JSON Schema descriptor for a tool exposed to the model.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | The tool's registered name (snake_case). |
| `description` | `str` | Human-readable description used in the model's system
context. |
| `input_schema` | `dict[str, Any]` | JSON Schema `object` describing the tool's input
parameters. |

### `Embedding`

```python
class Embedding(index: int, vector: list[float])
```

A single embedding vector.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `index` | `int` | Zero-based index of this embedding in the batch. |
| `vector` | `list[float]` | The floating-point embedding vector. |

## Structured output

### `StructuredLLM`

```python
class StructuredLLM(llm: Any, model_cls: type[T])
```

Typed wrapper over LLMService that forces structured output.

Created via `llm.with_structured_output(MyModel)`.

Usage::

    structured = llm.with_structured_output(SentimentResult)
    result: SentimentResult = await structured.complete([...])

#### `StructuredLLM.complete`

```python
def complete(self, messages: list[Any]) -> T
```

Complete *messages* and return a validated model instance.

Uses tool-calling to force the model to emit JSON that matches the
schema, then constructs and returns a `model_cls` instance.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `messages` | `list[Any]` | Conversation messages. |

**Returns:** `T` — A validated instance of *model_cls*.

**Raises:**

| Exception | Description |
|---|---|
| `OutputParserError` | When the model's response cannot be parsed
or validated against the schema. |

## Multimodal content

### `ImageContent`

```python
class ImageContent(_data: bytes | None = None, _url: str | None = None, mime_type: str = 'image/png')
```

An image content block for multimodal messages.

Usage::

    img = ImageContent.from_file("/tmp/chart.png")
    img = ImageContent.from_url("https://example.com/photo.jpg")
    img = ImageContent.from_bytes(b"...", mime_type="image/jpeg")

#### `ImageContent.from_file`

```python
def from_file(cls, path: str | Path) -> ImageContent
```

Load image bytes from *path* and detect MIME type from extension.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `path` | `str | Path` | Path to the image file. |

**Returns:** `ImageContent` — A new `ImageContent` with bytes loaded.

#### `ImageContent.from_url`

```python
def from_url(cls, url: str, mime_type: str = 'image/jpeg') -> ImageContent
```

Create an image referencing a remote URL.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `url` | `str` | The URL of the image. |
| `mime_type` | `str` | MIME type hint.  Defaults to `"image/jpeg"`. |

**Returns:** `ImageContent` — A new `ImageContent` referencing the URL.

#### `ImageContent.from_bytes`

```python
def from_bytes(cls, data: bytes, mime_type: str) -> ImageContent
```

Create an image from raw bytes.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `data` | `bytes` | Raw image bytes. |
| `mime_type` | `str` | MIME type of the image, e.g. `"image/png"`. |

**Returns:** `ImageContent` — A new `ImageContent` wrapping the bytes.

#### `ImageContent.from_base64`

```python
def from_base64(cls, b64: str, mime_type: str) -> ImageContent
```

Create an image from a base64-encoded string.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `b64` | `str` | Base64-encoded image data. |
| `mime_type` | `str` | MIME type of the image. |

**Returns:** `ImageContent` — A new `ImageContent` decoded from *b64*.

#### `ImageContent.to_anthropic_block`

```python
def to_anthropic_block(self) -> dict[str, Any]
```

Serialize to Anthropic API image-block format.

**Returns:** `dict[str, Any]` — Dictionary suitable for the Anthropic messages API.

#### `ImageContent.to_openai_block`

```python
def to_openai_block(self) -> dict[str, Any]
```

Serialize to OpenAI API image-block format.

**Returns:** `dict[str, Any]` — Dictionary suitable for the OpenAI messages API.

### `AudioContent`

```python
class AudioContent(_data: bytes, mime_type: str)
```

An audio content block.

Note: Only supported by OpenAI (input_audio).
Anthropic and Ollama raise `UnsupportedContentError`.

#### `AudioContent.from_file`

```python
def from_file(cls, path: str | Path) -> AudioContent
```

Load audio bytes from *path* and detect MIME type from extension.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `path` | `str | Path` | Path to the audio file. |

**Returns:** `AudioContent` — A new `AudioContent` with bytes loaded.

#### `AudioContent.from_bytes`

```python
def from_bytes(cls, data: bytes, mime_type: str) -> AudioContent
```

Create audio from raw bytes.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `data` | `bytes` | Raw audio bytes. |
| `mime_type` | `str` | MIME type, e.g. `"audio/mpeg"` or `"audio/wav"`. |

**Returns:** `AudioContent` — A new `AudioContent` wrapping the bytes.

#### `AudioContent.to_openai_block`

```python
def to_openai_block(self) -> dict[str, Any]
```

Serialize to OpenAI API input_audio block format.

**Returns:** `dict[str, Any]` — Dictionary suitable for the OpenAI messages API.

### `DocumentContent`

```python
class DocumentContent(_data: bytes | None = None, _url: str | None = None, mime_type: str = 'application/pdf')
```

A document (PDF) content block.

Anthropic supports native PDF documents.
Other providers raise `UnsupportedContentError`.

#### `DocumentContent.from_file`

```python
def from_file(cls, path: str | Path) -> DocumentContent
```

Load document bytes from *path*.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `path` | `str | Path` | Path to the PDF file. |

**Returns:** `DocumentContent` — A new `DocumentContent` with bytes loaded.

#### `DocumentContent.from_url`

```python
def from_url(cls, url: str) -> DocumentContent
```

Create a document referencing a remote URL.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `url` | `str` | The URL of the document. |

**Returns:** `DocumentContent` — A new `DocumentContent` referencing the URL.

#### `DocumentContent.from_bytes`

```python
def from_bytes(cls, data: bytes, mime_type: str = 'application/pdf') -> DocumentContent
```

Create a document from raw bytes.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `data` | `bytes` | Raw document bytes. |
| `mime_type` | `str` | MIME type.  Defaults to `"application/pdf"`. |

**Returns:** `DocumentContent` — A new `DocumentContent` wrapping the bytes.

#### `DocumentContent.to_anthropic_block`

```python
def to_anthropic_block(self) -> dict[str, Any]
```

Serialize to Anthropic API document-block format.

**Returns:** `dict[str, Any]` — Dictionary suitable for the Anthropic messages API.

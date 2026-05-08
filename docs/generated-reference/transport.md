# Transport & Multimodal

Core message types exchanged with LLM providers.

## Messages & completions

### `Message`

```python
class Message(role: Literal['user', 'assistant'], content: str | list[ContentBlock])
```

A single message in a conversation.

:param role: The sender role.  Either ``"user"`` or ``"assistant"``.
:type role: Literal["user", "assistant"]
:param content: Message content.  Either a plain string or a list of
    :class:`ContentBlock` instances for multi-modal / tool-use messages.
:type content: str | list[ContentBlock]

#### `Message.user`

```python
def user(cls, content: str | list[ContentBlock]) -> Message
```

Convenience factory for a user message.

:param content: Message content.
:type content: str | list[ContentBlock]
:return: A new user :class:`Message`.
:rtype: Message

#### `Message.assistant`

```python
def assistant(cls, content: str | list[ContentBlock]) -> Message
```

Convenience factory for an assistant message.

:param content: Message content.
:type content: str | list[ContentBlock]
:return: A new assistant :class:`Message`.
:rtype: Message

#### `Message.from_multimodal`

```python
def from_multimodal(cls, role: str, parts: list[Any]) -> Message
```

Create a Message from a list of text strings and content objects.

Accepts any mix of plain strings, :class:`~lauren_ai._transport._multimodal.ImageContent`,
:class:`~lauren_ai._transport._multimodal.AudioContent`, and
:class:`~lauren_ai._transport._multimodal.DocumentContent` instances.

:param role: The sender role — ``"user"`` or ``"assistant"``.
:type role: str
:param parts: Mixed list of text strings and multimodal content objects.
:type parts: list[Any]
:return: A new :class:`Message` whose ``content`` is the *parts* list.
:rtype: Message

#### `Message.text`

```python
def text(self) -> str
```

Extract all text from this message as a single concatenated string.

:return: All text content concatenated together.
:rtype: str

### `Completion`

```python
class Completion(id: str, model: str, content: str, tool_calls: list[ToolCall], stop_reason: Literal['end_turn', 'tool_use', 'max_tokens', 'stop_sequence'], usage: TokenUsage, thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] = list())
```

A finished (non-streaming) model completion.

:param id: Provider-assigned completion identifier.
:type id: str
:param model: The model that produced this completion.
:type model: str
:param content: The primary text response from the model.
:type content: str
:param tool_calls: All tool calls requested by the model in this turn.
:type tool_calls: list[ToolCall]
:param stop_reason: Why the model stopped generating:

    * ``"end_turn"`` — natural end of response.
    * ``"tool_use"`` — one or more tool calls were requested.
    * ``"max_tokens"`` — the configured token limit was reached.
    * ``"stop_sequence"`` — a stop sequence was hit.
:type stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
:param usage: Token usage statistics for this completion.
:type usage: TokenUsage
:param thinking_blocks: Extended-thinking blocks (Anthropic only).
:type thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock]

### `CompletionChunk`

```python
class CompletionChunk(delta: str = '', thinking_delta: str | None = None, tool_call_delta: ToolCallDelta | None = None, stop_reason: str | None = None, usage: TokenUsage | None = None, pending_approval: PendingApproval | None = None, guardrail_override: str | None = None)
```

A single chunk from a streaming model completion.

Only one of *delta*, *thinking_delta*, or *tool_call_delta* is populated
per chunk (though they are not mutually exclusive in the schema).

:param delta: Text content delta for this chunk.
:type delta: str
:param thinking_delta: Reasoning / thinking text delta (extended thinking).
:type thinking_delta: str | None
:param tool_call_delta: Partial tool call update for this chunk.
:type tool_call_delta: ToolCallDelta | None
:param stop_reason: Stop reason, populated only in the final chunk.
:type stop_reason: str | None
:param usage: Token usage, populated only in the final chunk.
:type usage: TokenUsage | None
:param pending_approval: Human-in-the-loop pending-approval signal.
    Present when a tool call requires confirmation before execution.
:type pending_approval: PendingApproval | None
:param guardrail_override: When set, an output guardrail fired and this
    string is the replacement content.  The runner emits one sentinel
    chunk with only this field populated (``delta=""`` etc.) after all
    normal chunks have been yielded.  Callers should replace the
    accumulated streaming text with this value.
:type guardrail_override: str | None

## Usage & calls

### `TokenUsage`

```python
class TokenUsage(input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0)
```

Token accounting for a single model call.

:param input_tokens: Number of tokens in the prompt / input messages.
:type input_tokens: int
:param output_tokens: Number of tokens in the completion.
:type output_tokens: int
:param cache_read_tokens: Tokens read from the prompt cache (Anthropic).
:type cache_read_tokens: int
:param cache_write_tokens: Tokens written to the prompt cache (Anthropic).
:type cache_write_tokens: int

#### `TokenUsage.cost_usd`

```python
def cost_usd(self, model: str) -> float
```

Estimate the cost in USD for this usage against *model*.

Uses a bundled price table with prefix-matching.  Falls back to
``$1/$3`` per million tokens when the model is unknown.

:param model: Model identifier used for the completion.
:type model: str
:return: Estimated cost in USD.
:rtype: float

### `ToolCall`

```python
class ToolCall(tool_use_id: str, name: str, input: dict[str, Any])
```

A completed tool call extracted from a model response.

:param tool_use_id: Provider-assigned identifier for this tool call.
:type tool_use_id: str
:param name: The registered tool name.
:type name: str
:param input: The parsed JSON input arguments.
:type input: dict[str, Any]

### `ToolSchema`

```python
class ToolSchema(name: str, description: str, input_schema: dict[str, Any])
```

JSON Schema descriptor for a tool exposed to the model.

:param name: The tool's registered name (snake_case).
:type name: str
:param description: Human-readable description used in the model's system
    context.
:type description: str
:param input_schema: JSON Schema ``object`` describing the tool's input
    parameters.
:type input_schema: dict[str, Any]

### `Embedding`

```python
class Embedding(index: int, vector: list[float])
```

A single embedding vector.

:param index: Zero-based index of this embedding in the batch.
:type index: int
:param vector: The floating-point embedding vector.
:type vector: list[float]

## Structured output

### `StructuredLLM`

```python
class StructuredLLM(llm: Any, model_cls: type[T])
```

Typed wrapper over LLMService that forces structured output.

Created via ``llm.with_structured_output(MyModel)``.

Usage::

    structured = llm.with_structured_output(SentimentResult)
    result: SentimentResult = await structured.complete([...])

#### `StructuredLLM.complete`

```python
def complete(self, messages: list[Any]) -> T
```

Complete *messages* and return a validated model instance.

Uses tool-calling to force the model to emit JSON that matches the
schema, then constructs and returns a ``model_cls`` instance.

:param messages: Conversation messages.
:type messages: list[Any]
:return: A validated instance of *model_cls*.
:rtype: T
:raises OutputParserError: When the model's response cannot be parsed
    or validated against the schema.

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

:param path: Path to the image file.
:type path: str | Path
:return: A new :class:`ImageContent` with bytes loaded.
:rtype: ImageContent

#### `ImageContent.from_url`

```python
def from_url(cls, url: str, mime_type: str = 'image/jpeg') -> ImageContent
```

Create an image referencing a remote URL.

:param url: The URL of the image.
:type url: str
:param mime_type: MIME type hint.  Defaults to ``"image/jpeg"``.
:type mime_type: str
:return: A new :class:`ImageContent` referencing the URL.
:rtype: ImageContent

#### `ImageContent.from_bytes`

```python
def from_bytes(cls, data: bytes, mime_type: str) -> ImageContent
```

Create an image from raw bytes.

:param data: Raw image bytes.
:type data: bytes
:param mime_type: MIME type of the image, e.g. ``"image/png"``.
:type mime_type: str
:return: A new :class:`ImageContent` wrapping the bytes.
:rtype: ImageContent

#### `ImageContent.from_base64`

```python
def from_base64(cls, b64: str, mime_type: str) -> ImageContent
```

Create an image from a base64-encoded string.

:param b64: Base64-encoded image data.
:type b64: str
:param mime_type: MIME type of the image.
:type mime_type: str
:return: A new :class:`ImageContent` decoded from *b64*.
:rtype: ImageContent

#### `ImageContent.to_anthropic_block`

```python
def to_anthropic_block(self) -> dict[str, Any]
```

Serialize to Anthropic API image-block format.

:return: Dictionary suitable for the Anthropic messages API.
:rtype: dict[str, Any]

#### `ImageContent.to_openai_block`

```python
def to_openai_block(self) -> dict[str, Any]
```

Serialize to OpenAI API image-block format.

:return: Dictionary suitable for the OpenAI messages API.
:rtype: dict[str, Any]

### `AudioContent`

```python
class AudioContent(_data: bytes, mime_type: str)
```

An audio content block.

Note: Only supported by OpenAI (input_audio).
Anthropic and Ollama raise :class:`UnsupportedContentError`.

#### `AudioContent.from_file`

```python
def from_file(cls, path: str | Path) -> AudioContent
```

Load audio bytes from *path* and detect MIME type from extension.

:param path: Path to the audio file.
:type path: str | Path
:return: A new :class:`AudioContent` with bytes loaded.
:rtype: AudioContent

#### `AudioContent.from_bytes`

```python
def from_bytes(cls, data: bytes, mime_type: str) -> AudioContent
```

Create audio from raw bytes.

:param data: Raw audio bytes.
:type data: bytes
:param mime_type: MIME type, e.g. ``"audio/mpeg"`` or ``"audio/wav"``.
:type mime_type: str
:return: A new :class:`AudioContent` wrapping the bytes.
:rtype: AudioContent

#### `AudioContent.to_openai_block`

```python
def to_openai_block(self) -> dict[str, Any]
```

Serialize to OpenAI API input_audio block format.

:return: Dictionary suitable for the OpenAI messages API.
:rtype: dict[str, Any]

### `DocumentContent`

```python
class DocumentContent(_data: bytes | None = None, _url: str | None = None, mime_type: str = 'application/pdf')
```

A document (PDF) content block.

Anthropic supports native PDF documents.
Other providers raise :class:`UnsupportedContentError`.

#### `DocumentContent.from_file`

```python
def from_file(cls, path: str | Path) -> DocumentContent
```

Load document bytes from *path*.

:param path: Path to the PDF file.
:type path: str | Path
:return: A new :class:`DocumentContent` with bytes loaded.
:rtype: DocumentContent

#### `DocumentContent.from_url`

```python
def from_url(cls, url: str) -> DocumentContent
```

Create a document referencing a remote URL.

:param url: The URL of the document.
:type url: str
:return: A new :class:`DocumentContent` referencing the URL.
:rtype: DocumentContent

#### `DocumentContent.from_bytes`

```python
def from_bytes(cls, data: bytes, mime_type: str = 'application/pdf') -> DocumentContent
```

Create a document from raw bytes.

:param data: Raw document bytes.
:type data: bytes
:param mime_type: MIME type.  Defaults to ``"application/pdf"``.
:type mime_type: str
:return: A new :class:`DocumentContent` wrapping the bytes.
:rtype: DocumentContent

#### `DocumentContent.to_anthropic_block`

```python
def to_anthropic_block(self) -> dict[str, Any]
```

Serialize to Anthropic API document-block format.

:return: Dictionary suitable for the Anthropic messages API.
:rtype: dict[str, Any]


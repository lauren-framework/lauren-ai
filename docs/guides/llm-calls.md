# Direct LLM Calls

Use `LLMService` for direct completions without the agentic loop.  This is
useful in controllers, chains, or anywhere you want a single-shot LLM call
rather than a multi-turn agent.

---

## LLMConfig — provider configuration

`LLMConfig` is a frozen dataclass that holds provider connection details and
per-call defaults.

### Factory classmethods

```python
from lauren_ai import LLMConfig

# Anthropic (reads ANTHROPIC_API_KEY from env when api_key=None)
cfg = LLMConfig.for_anthropic(
    model="claude-opus-4-6",
    api_key="sk-ant-...",   # or omit to use ANTHROPIC_API_KEY
    temperature=0.7,
    max_tokens=2048,
)

# OpenAI (reads OPENAI_API_KEY from env when api_key=None)
cfg = LLMConfig.for_openai(model="gpt-4o")

# Ollama (local, no API key required)
cfg = LLMConfig.for_ollama(
    model="llama3.2",
    base_url="http://localhost:11434",
)

# Tests — zero network calls
cfg, mock = LLMConfig.for_testing()
```

### Direct constructor

```python
from lauren_ai import LLMConfig

cfg = LLMConfig(
    provider="anthropic",       # "anthropic" | "openai" | "ollama" | "litellm"
    model="claude-opus-4-6",
    api_key="sk-ant-...",
    max_tokens=4096,            # max output tokens per call
    temperature=1.0,            # sampling temperature (0.0–2.0)
    timeout=60.0,               # HTTP timeout in seconds
    max_retries=3,              # automatic retries on transient errors
    embed_model=None,           # embedding model (defaults to model)
    embed_dimensions=None,      # embedding dimensionality
    cache_system_prompt=False,  # Anthropic prompt caching (system)
    cache_tools=False,          # Anthropic prompt caching (tools)
)
```

---

## LLMModule — wiring into the DI container

`LLMModule.for_root()` creates a `@module` that provides `LLMService` and
`EmbedService` as singletons:

```python
import os
from lauren import module, LaurenFactory
from lauren_ai import LLMConfig
from lauren_ai._module import LLMModule

LLMProvider = LLMModule.for_root(
    LLMConfig.for_anthropic(
        model="claude-opus-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
)

@module(
    controllers=[MyController],
    imports=[LLMProvider],
)
class AppModule: ...

app = LaurenFactory.create(AppModule)
```

The generated module exports `LLMService`, `EmbedService`, and `LLMConfig`.
Inject any of them by type in controllers, agents, or other injectables.

---

## LLMService.complete()

`LLMService` is injectable wherever you need a direct completion:

```python
from lauren_ai._module import LLMService
from lauren_ai._transport import Message

class SummaryController:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    @post("/summarise")
    async def summarise(self, body: SummariseBody) -> dict:
        result = await self._llm.complete(
            [Message.user(f"Summarise this text:\n\n{body.text}")],
            system="You are a concise summariser.",
            max_tokens=512,
            temperature=0.5,
        )
        return {"summary": result.content}
```

`complete()` parameters (all keyword-only except `messages`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | required | Conversation messages |
| `system` | `str \| None` | `None` | System prompt |
| `model` | `str \| None` | `LLMConfig.model` | Model override |
| `max_tokens` | `int \| None` | `LLMConfig.max_tokens` | Max output tokens |
| `temperature` | `float \| None` | `LLMConfig.temperature` | Sampling temperature |
| `tools` | `list[ToolSchema] \| None` | `None` | Tool schemas |
| `tool_choice` | `ToolChoice \| None` | `None` | Tool selection constraint |
| `stream` | `bool` | `False` | Return async iterator of chunks |

Returns a `Completion` (when `stream=False`) or an `AsyncIterator[CompletionChunk]`.

### Completion fields

```python
result = await llm.complete([Message.user("Hello")])

result.content          # str — the model's text response
result.tool_calls       # list[ToolCall] — tool calls requested by the model
result.stop_reason      # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
result.usage            # TokenUsage(input_tokens, output_tokens)
result.usage.cost_usd("claude-opus-4-6")  # estimated cost in USD
result.thinking_blocks  # list[ThinkingBlock | RedactedThinkingBlock] — Anthropic only
```

### Completion with extended thinking

Pass `thinking=True` to let the model reason internally before responding.
`thinking_budget_tokens` controls the token ceiling for the reasoning phase:

```python
from lauren_ai._transport import ThinkingBlock, RedactedThinkingBlock

result = await llm.complete(
    [Message.user("Explain the trade-offs between SQL and NoSQL.")],
    thinking=True,
    thinking_budget_tokens=8_000,
)

for block in result.thinking_blocks:
    if isinstance(block, ThinkingBlock):
        print("THINKING:", block.thinking)
    elif isinstance(block, RedactedThinkingBlock):
        print("REDACTED:", block.data[:40])

print("ANSWER:", result.content)
```

> **Temperature is incompatible with extended thinking.**  When `thinking=True`
> is passed, the temperature parameter is omitted from the API call automatically.

See the [extended thinking guide](extended-thinking.md) for the full reference.

---

## LLMService.complete_stream()

Convenience alias for `complete(..., stream=True)`:

```python
stream = await llm.complete_stream(
    [Message.user("Write a poem about the sea.")],
    system="You are a poet.",
)
async for chunk in stream:
    print(chunk.delta, end="", flush=True)
```

See the [streaming guide](streaming.md) for SSE integration.

---

## LLMService.embed()

Compute embedding vectors for a list of texts:

```python
from lauren_ai._module import LLMService

class EmbedController:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    @post("/embed")
    async def embed(self, body: EmbedBody) -> dict:
        embeddings = await self._llm.embed(body.texts)
        return {"vectors": [e.vector for e in embeddings]}
```

Each `Embedding` in the returned list has `.index` (position in the input
list) and `.vector` (list of floats).

Use a dedicated `EmbedService` if you only need embeddings and want to avoid
depending on the full `LLMService`:

```python
from lauren_ai._module import EmbedService

class VectorController:
    def __init__(self, embed: EmbedService) -> None:
        self._embed = embed
```

---

## LLMService.with_structured_output()

Force the model to return a valid Pydantic model instance using native
tool-calling:

```python
from pydantic import BaseModel
from lauren_ai._module import LLMService
from lauren_ai._transport import Message

class RecipeSummary(BaseModel):
    title: str
    ingredients: list[str]
    cook_time_minutes: int

class RecipeController:
    def __init__(self, llm: LLMService) -> None:
        self._structured = llm.with_structured_output(RecipeSummary)

    @post("/parse-recipe")
    async def parse(self, body: RecipeBody) -> dict:
        result: RecipeSummary = await self._structured.complete(
            [Message.user(body.recipe_text)]
        )
        return result.model_dump()
```

---

## Message construction

```python
from lauren_ai._transport import Message

# Simple text messages
msgs = [
    Message.user("What is the capital of France?"),
    Message.assistant("Paris."),
    Message.user("And Germany?"),
]

# Build from role string
msg = Message(role="user", content="Hello")

# Extract all text from a message (handles multi-block content)
text = msg.text()
```

---

## Token counting

```python
count = await llm.count_tokens([Message.user("Hello, world!")])
print(f"Estimated tokens: {count}")
```

Falls back to a `chars / 4` heuristic when the transport does not support
native token counting.

# Streaming

`lauren-ai` supports streaming at two levels: the agent runner
(`AgentRunner.run_stream()`) and the raw LLM service (`LLMService.complete_stream()`).
Both yield `CompletionChunk` objects as the model produces tokens.

---

## CompletionChunk

Every streaming call yields `CompletionChunk` instances:

| Field | Type | Description |
|---|---|---|
| `delta` | `str` | Text content for this chunk (empty string when no text in this chunk) |
| `thinking_delta` | `str \| None` | Reasoning text delta (Anthropic extended thinking only) |
| `tool_call_delta` | `ToolCallDelta \| None` | Partial tool call update |
| `stop_reason` | `str \| None` | Set only in the final chunk (`"end_turn"`, `"tool_use"`, etc.) |
| `usage` | `TokenUsage \| None` | Token usage, set only in the final chunk |
| `pending_approval` | `PendingApproval \| None` | HITL approval required for a tool call |

The `delta` field is the most commonly used — append it to a string buffer to
reconstruct the full response:

```python
full_text = ""
async for chunk in stream:
    full_text += chunk.delta
```

---

## AgentRunner.run_stream()

`run_stream()` drives the full agentic loop with streaming output.  Tool calls
execute silently between turns; only the model's text output is yielded.

```python
from lauren_ai import AgentRunner

runner: AgentRunner = ...  # injected

async def stream_agent_response(agent, message: str):
    full_response = ""
    async for chunk in await runner.run_stream(agent, message):
        if chunk.delta:
            full_response += chunk.delta
            yield chunk.delta  # forward to client
```

`run_stream()` accepts the same keyword arguments as `run()`:

```python
async for chunk in await runner.run_stream(
    agent,
    "Explain quantum computing",
    conversation_id="session-42",
    metadata={"user_id": "usr-1"},
):
    print(chunk.delta, end="", flush=True)
```

---

## LLMService streaming

For direct LLM calls without the agentic loop, use `complete_stream()` or
pass `stream=True` to `complete()`:

```python
from lauren_ai._module import LLMService
from lauren_ai._transport import Message

llm: LLMService = ...  # injected

# Option 1: complete_stream() convenience method
stream = await llm.complete_stream([Message.user("Tell me a joke.")])

# Option 2: complete() with stream=True
stream = await llm.complete(
    [Message.user("Tell me a joke.")],
    stream=True,
)

async for chunk in stream:
    print(chunk.delta, end="", flush=True)
```

---

## Streaming in web controllers (SSE)

Pair streaming with Server-Sent Events to push tokens to a browser client in
real time.

### Using AgentRunner.run_stream()

```python
from lauren_ai import AgentRunner
from collections.abc import AsyncIterator

class ChatController:
    def __init__(self, runner: AgentRunner, agent: AssistantAgent) -> None:
        self._runner = runner
        self._agent = agent

    @get("/chat/stream")
    async def stream_chat(self, request: Request) -> AsyncIterator[str]:
        message = request.query_params["message"]

        async def generate() -> AsyncIterator[str]:
            async for chunk in await self._runner.run_stream(
                self._agent, message
            ):
                if chunk.delta:
                    yield f"data: {chunk.delta}\n\n"
            yield "data: [DONE]\n\n"

        return EventStream(generate())
```

### Using LLMService directly

```python
from lauren_ai._module import LLMService
from lauren_ai._transport import Message
from collections.abc import AsyncIterator

class CompletionController:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    @post("/complete/stream")
    async def stream(self, body: CompletionBody) -> AsyncIterator[str]:
        stream = await self._llm.complete_stream(
            [Message.user(body.prompt)],
            system=body.system,
            max_tokens=body.max_tokens,
        )

        async def generate() -> AsyncIterator[str]:
            async for chunk in stream:
                if chunk.delta:
                    yield f"data: {chunk.delta}\n\n"
            yield "data: [DONE]\n\n"

        return EventStream(generate())
```

---

## Chain streaming

`Chain.stream()` renders the template steps and then streams from the first
`LLMService` step in the chain.  Output parser steps are not applied during
streaming — apply them to the aggregated result from `Chain.invoke()` instead.

```python
from lauren_ai import chain
from lauren_ai._prompts import PromptTemplate
from lauren_ai._module import LLMService

template = PromptTemplate("Write a short story about {topic}.")
llm: LLMService = ...

pipeline = chain(template, llm)

stream = await pipeline.stream(topic="a robot who learns to paint")
async for chunk in stream:
    print(chunk.delta, end="", flush=True)
```

---

## Checking stop reason and usage in streaming

The final chunk carries `stop_reason` and `usage`:

```python
stop_reason = None
total_input_tokens = 0
total_output_tokens = 0

async for chunk in stream:
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    if chunk.stop_reason is not None:
        stop_reason = chunk.stop_reason
    if chunk.usage is not None:
        total_input_tokens = chunk.usage.input_tokens
        total_output_tokens = chunk.usage.output_tokens

print(f"\nStop: {stop_reason}, tokens: {total_input_tokens}+{total_output_tokens}")
```

---

## Extended thinking deltas (Anthropic only)

When `thinking=True` is set in `AgentConfig`, the model emits thinking text
before its response.  Thinking arrives in `chunk.thinking_delta`:

```python
thinking_text = ""
response_text = ""

async for chunk in stream:
    if chunk.thinking_delta:
        thinking_text += chunk.thinking_delta
    if chunk.delta:
        response_text += chunk.delta
```

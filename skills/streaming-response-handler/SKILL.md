---
name: streaming-response-handler
description: Streams LLM responses chunk-by-chunk from an agent via runner.run_stream(). Use when building real-time UIs, SSE endpoints, WebSocket streaming, or any feature that needs to display tokens as they arrive rather than waiting for the full response.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Streaming Response Handler

## Quick start

```python
# from __future__ import annotations is safe here (no @tool definitions)
from __future__ import annotations

from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport._mock import MockTransport

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-ant-...")
runner = AgentRunner(transport=transport)

async for chunk in await runner.run_stream(my_agent_instance, "Tell me a joke"):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    if chunk.stop_reason:
        print()  # newline at end
```

---

## CompletionChunk fields

| Field | Type | Description |
|-------|------|-------------|
| `delta` | `str` | Text fragment from this chunk (may be empty string) |
| `stop_reason` | `str \| None` | Set on the final chunk: `"end_turn"`, `"tool_use"`, etc. |
| `usage` | `TokenUsage \| None` | Set on the final chunk with token counts |
| `thinking_delta` | `str \| None` | Anthropic extended thinking fragment |
| `tool_call_delta` | `ToolCallDelta \| None` | Streaming tool-call fragment |
| `guardrail_override` | `str \| None` | Set if a guardrail replaced the output |

---

## Accumulate full text

```python
async def stream_and_collect(runner, agent_instance, prompt):
    full_text = ""
    async for chunk in await runner.run_stream(agent_instance, prompt):
        if chunk.delta:
            full_text += chunk.delta
    return full_text
```

---

## Lauren SSE integration

```python
from lauren import sse, EventStream
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner

@get("/chat/stream")
async def chat_stream(
    prompt: Query[str],
    agent: MyAgent,
    runner: AgentRunner,
) -> EventStream:
    async def _generate():
        async for chunk in await runner.run_stream(agent, prompt):
            if chunk.delta:
                yield chunk.delta

    return EventStream(_generate())
```

---

## Testing pattern — queue stream chunks

```python
from lauren_ai._transport import CompletionChunk, TokenUsage
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()
mock.queue_stream([
    CompletionChunk(delta="Hello"),
    CompletionChunk(delta=" world"),
    CompletionChunk(
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=2),
    ),
])

chunks = []
async for chunk in await runner.run_stream(my_agent, "hi"):
    chunks.append(chunk)
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_agents/_runner.py` | `run_stream()` implementation |
| `src/lauren_ai/_transport/__init__.py` | `CompletionChunk`, `ToolCallDelta` |
| `src/lauren_ai/_transport/_mock.py` | `queue_stream()` helper |

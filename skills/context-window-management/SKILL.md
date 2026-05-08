---
name: context-window-management
description: Shows how to manage an agent's context window using ShortTermMemory sliding-window trimming and manual message truncation helpers. Use when limiting token usage, implementing conversation history pruning, or estimating prompt sizes before sending to the LLM.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Agent Context Window Management & Trimming

## Overview

`ShortTermMemory` automatically trims the oldest non-system messages when the
conversation grows beyond `max_tokens`.  Configure it per-agent via
`@agent(memory=ShortTermMemory(max_tokens=N))`.

---

## Configuring a sliding window

```python
# agents.py — from __future__ import annotations is safe here
from __future__ import annotations
from lauren_ai import agent, ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore

@agent(
    model="claude-opus-4-6",
    system="You are a helpful assistant.",
    memory=ShortTermMemory(max_tokens=4000),
    conversation_store=InMemoryConversationStore(),
)
class ContextManagedAgent: ...
```

`ShortTermMemory.messages()` returns a trimmed snapshot without mutating the
internal buffer.  The oldest non-system messages are dropped first.

---

## Manual trim helper

```python
def trim_messages(messages: list[dict], max_turns: int = 10) -> list[dict]:
    """Keep system messages + the last *max_turns* user/assistant pairs."""
    system = [m for m in messages if m.get("role") == "system"]
    turns  = [m for m in messages if m.get("role") != "system"]
    if len(turns) > max_turns * 2:   # *2 for user + assistant
        turns = turns[-(max_turns * 2):]
    return system + turns
```

---

## Token estimation

```python
def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate — 4 chars ≈ 1 token."""
    return sum(len(m.get("content", "")) // 4 for m in messages)
```

`ShortTermMemory.token_estimate` uses the same heuristic internally.

---

## When to use each approach

| Approach | When to use |
|----------|-------------|
| `ShortTermMemory(max_tokens=N)` | Automatic per-run sliding window |
| `trim_messages(msgs, max_turns=N)` | Manual preprocessing before injecting history |
| `memory.trim_to_fit(max_tokens)` | Mutate the buffer in-place (use sparingly) |

---

## Full ShortTermMemory API

```python
mem = ShortTermMemory(max_tokens=8000)
mem.add_user("Hello")          # add a user message
mem.add_assistant(completion)  # add a Completion or dict
mem.add_tool_result(result)    # add a ToolResult
msgs = mem.messages()          # trimmed snapshot
snap = mem.snapshot()          # deep copy, no trimming
mem.restore(snap)              # load a snapshot
mem.clear()                    # wipe the buffer
tok  = mem.token_estimate      # heuristic token count
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_memory/__init__.py` | `ShortTermMemory` implementation |
| `src/lauren_ai/_memory/_stores.py` | `InMemoryConversationStore` |
| `src/lauren_ai/_agents/_runner.py` | Memory creation and restore in `run()` |

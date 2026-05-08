---
name: multi-tool-orchestration
description: Attaches multiple tools to a single agent and lets the LLM decide which to call. Use when an agent needs access to several capabilities (search, database, calculation, API calls) and the model orchestrates which tools to invoke in sequence or in parallel.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Multi-Tool Orchestration & Selection

## Quick start

```python
# agent.py — from __future__ import annotations IS safe here (no @tool)
from __future__ import annotations

from lauren_ai import agent, use_tools
from .tools import search_web, query_database, calculate

@agent(model="claude-opus-4-6", system="Use available tools to answer questions.")
@use_tools(search_web, query_database, calculate)
class ResearchAgent: ...
```

---

## Parallel tool calls

Enable `parallel_tool_calls=True` to run all tool calls in a single LLM turn
concurrently rather than sequentially:

```python
@agent(
    model="claude-opus-4-6",
    system="Use tools efficiently.",
    parallel_tool_calls=True,
)
@use_tools(search_web, query_database, calculate)
class ParallelAgent: ...
```

---

## Tool error policy

| Policy | Behaviour |
|--------|-----------|
| `"return_error"` (default) | Send error text back to LLM as a tool result |
| `"raise"` | Re-raise as `ToolExecutionError` immediately |
| `"skip"` | Omit the failing tool result; LLM continues without it |

```python
@agent(model="claude-opus-4-6", tool_error_policy="return_error")
@use_tools(search_web, calculate)
class RobustAgent: ...
```

---

## Testing sequential tool calls

```python
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()

# LLM calls search_web first
mock.queue_tool_use("search_web", {"query": "latest AI news"})
# Then calls calculate
mock.queue_tool_use("calculate", {"expression": "42 * 2"})
# Final answer
mock.queue_response(Completion(
    id="c3", model="mock-model",
    content="Based on my research: the answer is 84.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=30, output_tokens=20),
))
```

---

## Inspecting tool calls made

```python
resp = await runner.run(ResearchAgent(), "Find recent AI news and compute 42*2")
assert len(resp.tool_calls_made) == 2
assert resp.tool_calls_made[0].name == "search_web"
assert resp.tool_calls_made[1].name == "calculate"
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_agents/__init__.py` | `@use_tools()`, `AgentMeta.tool_classes` |
| `src/lauren_ai/_tools/_executor.py` | `ToolExecutor` — parallel dispatch |
| `src/lauren_ai/_agents/_runner.py` | `AgentRunnerBase` — multi-turn loop |

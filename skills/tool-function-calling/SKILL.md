---
name: tool-function-calling
description: Defines and registers tools for lauren-ai agents using @tool() on async functions or classes. Use when adding LLM-callable tools, injecting DI services into tools via ToolContext, attaching tools to agents with @use_tools(), or testing tool dispatch with MockTransport.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Tool / Function Calling Definition & Registration

## Critical rule — no PEP 563 in tool files

**Never add `from __future__ import annotations` to any file that defines `@tool()`.**

`@tool()` calls `inspect.signature()` at decoration time to build the JSON
schema. PEP 563 lazy evaluation converts all annotations to strings, silently
breaking schema generation.

---

## Function-form tool (most common)

```python
# tools.py — NO from __future__ import annotations

from lauren_ai import tool

@tool()
async def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name (e.g. 'London').
        units: Temperature units — 'celsius' or 'fahrenheit'.
    """
    return {"city": city, "temp": 18, "units": units, "condition": "cloudy"}
```

---

## Class-form tool (for DI injection)

```python
# tools.py — NO from __future__ import annotations

from lauren_ai import tool, ToolContext
from lauren import injectable, Scope

@tool()
@injectable(scope=Scope.SINGLETON)
class DatabaseQueryTool:
    """Query the application database.

    Args:
        query: SQL SELECT query string.
    """

    def __init__(self, db: DatabaseService) -> None:
        self._db = db

    async def run(self, ctx: ToolContext, query: str) -> list[dict]:
        return await self._db.execute(query)
```

---

## Attach tools to an agent

```python
# agent.py — from __future__ import annotations IS safe here

from __future__ import annotations

from lauren_ai import agent, use_tools
from .tools import get_weather, DatabaseQueryTool

@agent(model="claude-opus-4-6", system="You are a helpful assistant.")
@use_tools(get_weather, DatabaseQueryTool)
class AssistantAgent: ...
```

Decorator order: `@agent` outermost, `@use_tools` innermost (applied first).

---

## Testing tool dispatch with MockTransport

```python
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._tools import _add_to_tool_map
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()

# Queue a tool-use completion then the final answer
mock.queue_tool_use("get_weather", {"city": "London"})
mock.queue_response(Completion(
    id="c2", model="mock-model",
    content="It is 18°C in London.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=10, output_tokens=5),
))

tools = {}
_add_to_tool_map(tools, get_weather)
cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
runner = AgentRunner(transport=mock, tools=tools, config=cfg)

resp = await runner.run(MyAgent(), "What's the weather in London?")
assert "18°C" in resp.content
```

---

## ToolMeta structure

`ToolMeta` is stored on the decorated object under `TOOL_META`:

```python
from lauren_ai._tools import TOOL_META

meta = getattr(get_weather, TOOL_META)
meta.name         # "get_weather"
meta.description  # Extracted from docstring
meta.parameters   # dict — {"name": ..., "description": ..., "input_schema": {...}}
meta.is_async     # True
meta.reads_context  # True if a param is annotated ToolContext
```

`meta.parameters["input_schema"]` contains the JSON Schema properties object.

---

## ToolContext

When a tool is a class, `run(self, ctx: ToolContext, ...)` receives:

| Attribute | Type | Description |
|-----------|------|-------------|
| `ctx.tool_use_id` | `str` | Provider tool call identifier |
| `ctx.execution_context` | `Any` | Lauren `ExecutionContext` if injected |
| `ctx.run_id` | `str` | Unique agent run identifier |

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `@tool()`, `ToolContext`, `_add_to_tool_map` |
| `src/lauren_ai/_tools/_executor.py` | `ToolExecutor` — dispatch, caching, HITL |
| `src/lauren_ai/_agents/__init__.py` | `@use_tools()`, `AgentMeta.tool_classes` |

# Writing Tools

Tools are async Python functions (or classes) decorated with `@tool()`.  The
decorator builds a JSON schema from the function's type annotations and
Google-style docstring, which is sent to the LLM so it knows how to call your
tool.

---

## Critical rule

**Never add `from __future__ import annotations` to a file that defines `@tool()`.**

The `@tool()` decorator calls `inspect.signature()` at class-decoration time.
PEP 563 lazy evaluation converts all annotations to strings, breaking schema
generation.  Add this comment at the top of every tool file:

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
```

---

## Function-form tool

```python
# tools.py — NO from __future__ import annotations
from lauren_ai import tool

@tool()
async def search_database(query: str, limit: int = 10) -> list:
    """Search the product database for matching items.

    Args:
        query: Full-text search query.
        limit: Maximum number of results to return (1-100).

    Returns a list of product dicts with 'id', 'name', and 'price' keys.
    """
    # ... implementation ...
    return [{"id": 1, "name": "Widget", "price": 9.99}]
```

**Schema rules:**
- `str`, `int`, `float`, `bool` → primitive JSON types.
- `dict` → object (no nested schema).
- `list` → array (no item schema).
- `Optional[T]` / `T | None` → type T, not required.
- `ctx: ToolContext | None = None` → **excluded** from schema, injected internally.

---

## Class-form tool (stateful)

Use a class when the tool needs injected dependencies or state.
**Class-form tools must define a `run()` method** — `@tool()` looks for `run`;
a `__call__` is silently ignored and the decorator will raise `ValueError`.

```python
from lauren_ai import tool, ToolContext

@tool()
class EmailTool:
    """Send an email to a recipient.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
    """

    def __init__(self, smtp_host: str, smtp_port: int = 587) -> None:
        self._host = smtp_host
        self._port = smtp_port

    async def run(
        self,
        to: str,
        subject: str,
        body: str,
        ctx: ToolContext | None = None,
    ) -> dict:
        # ... send email ...
        return {"sent": True, "to": to}
```

---

## Human-in-the-loop (HITL) tool

Tools can pause for human approval before executing:

```python
from lauren_ai import tool, ToolContext

@tool()
async def delete_record(record_id: str, ctx: ToolContext | None = None) -> dict:
    """Permanently delete a record.

    Args:
        record_id: The ID of the record to delete.
    """
    # Signal that this tool needs approval
    if ctx is not None:
        runner = ctx.agent_context.metadata.get("runner")
        if runner is not None:
            # Caller must call runner.approve_tool(run_id, tool_use_id)
            # before this tool proceeds — wire your UI to do that.
            pass
    # ... delete ...
    return {"deleted": record_id}
```

Approve / reject from your controller:

```python
await runner.approve_tool(agent_run_id="abc", tool_use_id="toolu_1")
await runner.reject_tool(agent_run_id="abc", tool_use_id="toolu_1", reason="Not permitted")
```

---

## Tool result caching

Pass a `CacheBackend` to `AgentRunner` to cache repeated tool calls:

```python
from lauren_ai._tools._executor import CacheBackend

class InMemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._store[key] = value

runner = AgentRunner(
    transport=transport,
    registry=registry,
    config=config,
    cache_backend=InMemoryCache(),
)
```

---

## Tool lifecycle hooks on the agent

```python
@agent(model="openai/gpt-4o-mini")
class MyAgent:
    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        """Inspect or modify a tool result before it's sent back to the LLM."""
        if result.is_error:
            # Log the error, return modified result or None to keep original
            print(f"Tool error: {result.content}")
        return None  # Return None to keep original result unchanged
```

---

## Built-in skills

Import from `lauren_ai._skills`:

```python
from lauren_ai._skills import WebSearchTool, HttpFetchTool, CodeExecutionTool, DelegateToAgentTool
```

| Skill | What it does |
|-------|-------------|
| `WebSearchTool` | Stub web search (override with real API) |
| `HttpFetchTool` | HTTP GET/POST, returns status + body (truncated at 8 KB) |
| `CodeExecutionTool` | Runs Python in a subprocess sandbox (10s timeout) |
| `DelegateToAgentTool` | Low-level delegation primitive |

Register them in `ToolRegistry` before creating `AgentRunner`:

```python
from lauren_ai._skills import HttpFetchTool, CodeExecutionTool
from lauren_ai._tools._registry import ToolRegistry

registry = ToolRegistry()
registry.register(HttpFetchTool)
registry.register(CodeExecutionTool)
```

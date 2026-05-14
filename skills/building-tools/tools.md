# Writing Tools

Tools are async Python functions (or classes) decorated with `@tool()`.  The
decorator builds a JSON schema from the function's type annotations and
Google-style docstring, which is sent to the LLM so it knows how to call your
tool.

---

## Contents

- [Critical rule](#critical-rule)
- [Function-form tool](#function-form-tool)
- [Class-form tool](#class-form-tool-stateful)
- [ToolContext — DI injection](#toolcontext--di-injection)
- [Human-in-the-loop (HITL) approval](#human-in-the-loop-hitl-approval)
- [Tool result caching](#tool-result-caching)
- [Tool lifecycle hook on the agent](#tool-lifecycle-hook-on-the-agent)
- [Built-in skills](#built-in-skills)

---

## Critical rule

**`from __future__ import annotations` is supported, but every type used by
`@tool()` must resolve when schema generation runs.**

`@tool()` resolves annotations when it builds the JSON schema. Future
annotations are supported, but unresolved forward references and circular
imports in function-form tool files still break schema generation. A safe
reminder comment is:

```python

# @tool() resolves parameter annotations when this module is imported.
# Keep tool signature types importable and avoid unresolved forward refs.
```

---

## Function-form tool

The simplest form: an `async def` decorated with `@tool()`.

```python

from lauren_ai import tool

@tool()
async def search_database(query: str, limit: int = 10) -> list:
    """Search the product database for matching items.

    Args:
        query: Full-text search query string.
        limit: Maximum number of results to return (1–100).
    """
    # ... implementation ...
    return [{"id": 1, "name": "Widget", "price": 9.99}]
```

**Schema rules:**

- `str`, `int`, `float`, `bool` map to their JSON primitive equivalents.
- `dict` → `object` (no nested property schema).
- `list` → `array` (no item schema).
- `Optional[T]` / `T | None` → type T, marked as not required.
- Parameters with a default value are not required.
- `ctx: ToolContext` / `ctx: ToolContext | None = None` → **excluded from schema**, injected internally.

The Google-style `Args:` section of the docstring supplies descriptions for each
parameter.  Always document every non-`ctx` parameter.

Generated schema for the example above:

```json
{
  "name": "search_database",
  "description": "Search the product database for matching items.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Full-text search query string."},
      "limit": {"type": "number", "description": "Maximum number of results to return (1–100)."}
    },
    "required": ["query"]
  }
}
```

---

## Class-form tool (stateful)

Use a class when the tool needs injected dependencies or persistent state.

**Class-form tools must define a `run()` method — not `__call__()`.**
`@tool()` inspects the class for a method named `run`.  If you define
`__call__` instead, it is silently ignored and the decorator raises `ValueError`
because no callable entry point is found.

```python

from lauren_ai import tool, ToolContext

@tool()
class EmailTool:
    """Send an email to a recipient.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body (plain text).
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
        # ... send email via SMTP ...
        return {"sent": True, "to": to}
```

The schema is derived from `run()`'s parameters (excluding `self` and `ctx`).

### Class-form with DI injection

When used with the lauren DI container, the `__init__` parameters are injected
automatically.  Register the tool class as a provider and let the container
instantiate it:

```python

from lauren import injectable, Scope
from lauren_ai import tool, ToolContext
from app.db import Database

@tool()
@injectable(scope=Scope.SINGLETON)
class LookupAccountTool:
    """Look up account information for a user.

    Args:
        account_id: The account identifier to retrieve.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def run(self, account_id: str, ctx: ToolContext | None = None) -> dict:
        account = await self._db.get_account(account_id)
        if account is None:
            return {"error": f"Account {account_id!r} not found"}
        return {"id": account.id, "balance": account.balance, "status": account.status}
```

---

## ToolContext — DI injection

`ToolContext` carries metadata injected by `ToolExecutor` — it is never part
of the JSON schema and the LLM cannot supply or override it.

| Attribute | Type | Description |
|-----------|------|-------------|
| `ctx.agent_context` | `AgentContext` | The running agent's context (memory, config, metadata) |
| `ctx.execution_context` | `ExecutionContext \| None` | The HTTP request context forwarded from the controller |
| `ctx.tool_use_id` | `str` | Unique ID for this tool invocation |
| `ctx.turn` | `int` | Current turn index in the agentic loop |

The most important use of `ToolContext` is reading the **authenticated user
identity** set by a guard before the LLM ever ran:

```python
def _auth_uid(ctx: ToolContext) -> str:
    exec_ctx = ctx.execution_context
    if exec_ctx is None:
        return ""
    request = getattr(exec_ctx, "request", None)
    if request is None:
        return ""
    state = getattr(request, "state", None)
    if state is None:
        return ""
    return (state.get("user_id") or "").lower()
```

See [security.md](../securing-agents/security.md) for the full authentication
pattern.

---

## Human-in-the-loop (HITL) approval

Tools that perform irreversible or high-risk actions can pause for human review
before executing.  The pattern uses a `runner.approve_tool()` / `runner.reject_tool()`
call from your controller or UI layer:

```python

from lauren_ai import tool, ToolContext

@tool()
async def delete_record(record_id: str, ctx: ToolContext | None = None) -> dict:
    """Permanently delete a record from the database.

    Args:
        record_id: The unique ID of the record to delete.
    """
    # The runner pauses here until approve_tool() or reject_tool() is called.
    # Wire your UI to call the appropriate endpoint when the tool is pending.
    return {"deleted": record_id}
```

Approve or reject from a controller endpoint:

```python
# approve — tool proceeds
await runner.approve_tool(agent_run_id="run-abc", tool_use_id="toolu_1")

# reject — tool receives a rejection error; LLM sees the refusal
await runner.reject_tool(
    agent_run_id="run-abc",
    tool_use_id="toolu_1",
    reason="Manager did not approve this deletion",
)
```

The agent run blocks until `approve_tool` or `reject_tool` is called.  Use
server-sent events or a webhook to notify the UI that approval is needed.

---

## Tool result caching

Pass a `CacheBackend` to `AgentRunner` to avoid re-running identical tool calls
within a session.  The cache key is derived from the tool name and its input
parameters.

```python
from lauren_ai._tools._executor import CacheBackend
from lauren_ai import AgentRunnerBase

class InMemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._store[key] = value


runner = AgentRunnerBase(
    transport=transport,
    tools={},
    config=config,
    cache_backend=InMemoryCache(),
)
```

Implement a Redis-backed cache for distributed deployments:

```python
import json
import redis.asyncio as redis
from lauren_ai._tools._executor import CacheBackend

class RedisCacheBackend(CacheBackend):
    def __init__(self, client: redis.Redis) -> None:
        self._redis = client

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(key)
        return value.decode() if value else None

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        await self._redis.setex(key, ttl, value)
```

---

## Tool lifecycle hook on the agent

The agent class can intercept every tool result via `on_tool_result`.  Return
a modified `ToolResult` to replace the original, or `None` to keep it unchanged.

```python
from lauren_ai import agent, use_tools, AgentContext, ToolResult

@agent(model="claude-opus-4-6", system="You are helpful.")
@use_tools(search_database)
class MyAgent:
    async def on_tool_result(
        self, result: ToolResult, ctx: AgentContext
    ) -> ToolResult | None:
        """Inspect or rewrite a tool result before it goes back to the LLM."""
        if result.is_error:
            print(f"[warn] Tool {result.tool_use_id} errored: {result.content}")
            # Optionally return a friendlier error message
            from lauren_ai._tools import ToolResult as TR
            return TR(
                tool_use_id=result.tool_use_id,
                content="The tool encountered a temporary error. Please try again.",
                is_error=True,
            )
        return None  # keep original result unchanged
```

---

## Built-in skills

Import from `lauren_ai._skills`:

```python
from lauren_ai._skills import (
    WebSearchTool,
    HttpFetchTool,
    CodeExecutionTool,
    DelegateToAgentTool,
)
```

| Skill | Description |
|-------|-------------|
| `WebSearchTool` | Stub web search — override with a real search API for production |
| `HttpFetchTool` | HTTP GET/POST; returns status code + body (truncated at 8 KB) |
| `CodeExecutionTool` | Runs Python in a subprocess sandbox with a 10-second timeout |
| `DelegateToAgentTool` | Low-level delegation primitive used to forward tasks to sub-agents |

Register skills via `AgentModule.for_root()` (production) or directly for scripting:

```python
from lauren_ai._skills import HttpFetchTool, CodeExecutionTool
from lauren_ai import AgentModule, LLMModule, LLMConfig

LLMProvider = LLMModule.for_root(LLMConfig.for_anthropic())
AgentProvider = AgentModule.for_root(
    agents=[MyAgent],
    tools=[HttpFetchTool, CodeExecutionTool],
    imports=[LLMProvider],
)
```

Use built-in skills in an agent:

```python
from lauren_ai import agent, use_tools
from lauren_ai._skills import WebSearchTool, HttpFetchTool

@agent(
    model="claude-opus-4-6",
    system="You are a research assistant. Search the web and fetch pages as needed.",
)
@use_tools(WebSearchTool, HttpFetchTool)
class ResearchAgent: ...
```

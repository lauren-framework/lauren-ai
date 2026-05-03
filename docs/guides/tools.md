# Tools

Tools are functions or classes decorated with `@tool()` that agents can invoke
during the agentic loop.  The `@tool()` decorator inspects the function
signature at decoration time to generate the JSON schema that is sent to the
LLM.

> **Important:** Do not add `from __future__ import annotations` to any file
> that defines `@tool()`-decorated functions.  PEP 563 lazy evaluation turns
> all annotations into strings, which breaks schema generation silently.  Add
> this comment to the top of tool files as a reminder:
>
> ```python
> # NOTE: Do NOT add `from __future__ import annotations` to this file.
> ```

---

## Function-form tools

The simplest form — a plain async function decorated with `@tool()`:

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
from lauren_ai import tool

@tool()
async def get_weather(city: str, unit: str = "celsius") -> dict:
    """Return current weather for a city.

    Args:
        city: The city name to look up.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'.
    """
    # ... call a real weather API
    return {"city": city, "temperature": 22, "unit": unit}
```

`@tool()` must be called **with parentheses**.  Bare `@tool` raises
`DecoratorUsageError`.

---

## Class-form tools

Use the class form when the tool needs constructor-injected dependencies (a
database client, an HTTP client, etc.):

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
from lauren_ai import tool
from lauren import Scope, injectable

@tool()
@injectable(scope=Scope.SINGLETON)
class SearchTool:
    """Search the product catalogue."""

    def __init__(self, db: DatabaseService) -> None:
        self._db = db

    async def run(self, query: str, max_results: int = 10) -> list[dict]:
        """Search for products matching query.

        Args:
            query: The search query.
            max_results: Maximum number of results (1-50).
        """
        return await self._db.search(query, limit=max_results)
```

Key rules for class-form tools:

- The entry point must be an `async def run(self, ...)` method.
- `@tool()` automatically applies `@injectable(scope=Scope.SINGLETON)` if not
  already present.
- The schema is generated from `run()`'s parameters, not `__init__`.

---

## ToolContext — injected context parameter

Declare a `ctx: ToolContext` parameter to receive the agent context at
execution time.  `ToolContext` is **never** included in the JSON schema sent
to the LLM — it is injected internally by `ToolExecutor`.

```python
# NOTE: Do NOT add `from __future__ import annotations` to this file.
from lauren_ai import tool
from lauren_ai._tools import ToolContext

@tool()
async def log_action(action: str, ctx: ToolContext) -> str:
    """Log an action taken by the agent.

    Args:
        action: Description of the action.
    """
    turn = ctx.turn
    run_id = ctx.agent_context.agent_run_id
    print(f"[{run_id}] turn={turn}: {action}")
    return "logged"
```

The parameter may be named anything (e.g. `ctx`, `agent_ctx`); `@tool()`
detects it by its type annotation.

`ToolContext` fields:

| Field | Type | Description |
|---|---|---|
| `agent_context` | `AgentContext` | The owning agent's context |
| `tool_use_id` | `str` | Provider-assigned identifier for this call |
| `turn` | `int` | Agentic loop iteration that triggered the call |
| `request` | `Any \| None` | Originating HTTP request, if any |
| `state` | `dict[str, Any]` | Mutable per-call state bag |

---

## Schema generation rules

The JSON schema is built from the function/`run()` signature:

| Python type | JSON Schema type |
|---|---|
| `str` | `"string"` |
| `int`, `float` | `"number"` |
| `bool` | `"boolean"` |
| `dict` | `"object"` |
| `list` | `"array"` |
| `Optional[X]` / `X \| None` | `X` type, not required |

- Parameter descriptions are extracted from the Google-style `Args:` section
  of the docstring.
- Parameters with a default value are marked as not required in the schema.
- `ctx: ToolContext` parameters are excluded entirely.

Override the inferred name or description:

```python
@tool(name="web_search", description="Search the web for current information.")
async def search(query: str) -> list[dict]: ...
```

---

## @tool() parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str \| None` | function/class name | Tool name sent to the LLM |
| `description` | `str \| None` | first docstring paragraph | Description sent to the LLM |
| `requires_confirmation` | `bool` | `False` | Pause for HITL approval before executing |
| `pre_hook` | `Callable \| None` | `None` | Called before the tool runs |
| `post_hook` | `Callable \| None` | `None` | Called after a successful run |
| `error_hook` | `Callable \| None` | `None` | Called when the tool raises |
| `cache_ttl` | `int \| None` | `None` | Cache successful results for N seconds |
| `cache_key_fn` | `Callable \| None` | `None` | Custom cache key factory |

---

## Human-in-the-loop (HITL) approval

Set `requires_confirmation=True` to pause execution before the tool runs.
The runner stores a pending future; resume it via `AgentRunner.approve_tool()`
or `AgentRunner.reject_tool()`:

```python
@tool(requires_confirmation=True)
async def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
    """
    ...

# In a separate handler (e.g. a webhook from your UI):
await runner.approve_tool(agent_run_id="...", tool_use_id="toolu_abc123")
# or:
await runner.reject_tool(agent_run_id="...", tool_use_id="toolu_abc123", reason="Not approved")
```

---

## Tool result caching

Cache repeated calls with the same inputs using `cache_ttl`:

```python
@tool(cache_ttl=300)   # cache for 5 minutes
async def fetch_exchange_rate(base: str, quote: str) -> float:
    """Fetch the current exchange rate between two currencies.

    Args:
        base: Base currency code (e.g. USD).
        quote: Quote currency code (e.g. EUR).
    """
    ...
```

Caching requires a `CacheBackend` to be passed to `AgentModule.for_root()` via
the `tool_cache=` argument.  Without a backend, results are not cached and no
error is raised.

Provide a custom cache key to control how inputs are normalised:

```python
@tool(cache_ttl=60, cache_key_fn=lambda inp: inp["city"].lower())
async def get_weather(city: str) -> dict: ...
```

---

## ToolResult

Tools return plain Python values.  `ToolExecutor` wraps them in a `ToolResult`
automatically.  You can also return a `ToolResult` directly for fine-grained
control:

```python
from lauren_ai._tools import ToolResult

@tool()
async def risky_operation(param: str) -> ToolResult:
    """Attempt a risky operation."""
    try:
        result = do_thing(param)
        return ToolResult.ok(result, tool_use_id="")
    except Exception as exc:
        return ToolResult.error(str(exc), tool_use_id="")
```

`tool_use_id` is filled in by the executor; you can leave it as `""` when
returning from a tool function — the executor will patch it.

---

## Built-in skills

`lauren_ai.skills` ships four ready-to-use tools:

```python
from lauren_ai.skills import WebSearchTool, HttpFetchTool, CodeExecutionTool, DelegateToAgentTool
```

| Tool | Description |
|---|---|
| `WebSearchTool` | Web search (stub — wire up a real API in production) |
| `HttpFetchTool` | Fetch a URL via HTTP (requires `httpx`) |
| `CodeExecutionTool` | Execute Python in a subprocess sandbox |
| `DelegateToAgentTool` | Hand off to another registered agent by class name |

Attach them to agents via `@use_tools()`:

```python
from lauren_ai import agent, use_tools
from lauren_ai.skills import WebSearchTool, HttpFetchTool

@agent(model="claude-opus-4-6", system="You are a research assistant.")
@use_tools(WebSearchTool, HttpFetchTool)
class ResearchAgent: ...
```

# Custom Tool Decorators

Tool decorators let you express composable cross-cutting concerns — authentication,
permission checking, rate limiting, auditing — as reusable Python decorators that
wrap any `@tool()` function or class-form `run()` method.

The key utility that makes this work is
`get_tool_context_from_func_args(*args, **kwargs)`, which extracts the
`ToolContext` from a tool call's argument list without requiring the decorator
to know the exact parameter name used in the tool's signature.

## Why tool decorators?

Guards in Lauren protect HTTP request boundaries.  Tool decorators protect the
*agent tool boundary* — ensuring that a tool's business logic only runs when
the caller (an LLM acting on behalf of a user) satisfies a requirement.
Decorators are:

- **Composable**: stack `@require_auth`, `@require_scope("billing")`, and
  `@audit_log` on the same tool just like HTTP guards on a controller.
- **Reusable**: write once, apply to any function-form or class-form tool.
- **Context-aware**: `ToolContext.agent_context.execution_context` carries the same
  `ExecutionContext` that guards and controllers use, giving tool decorators
  access to `request.state`, headers, and the full Lauren DI graph.

## `get_tool_context_from_func_args`

```python
from lauren_ai import get_tool_context_from_func_args, ToolContext

ctx = get_tool_context_from_func_args(*args, **kwargs)
# → ToolContext instance, or None if not present in this call
```

The tool executor injects `ToolContext` as a **keyword argument** using
whatever parameter name was annotated `ToolContext` in the tool's signature
(`ctx`, `tool_ctx`, `context`, or any name).  When writing a decorator, you
receive the raw `*args` / `**kwargs` and need to find the context without
knowing that name — this helper handles both positional and keyword injection.

## `ToolContext` and `ExecutionContext`

`ToolContext.agent_context.execution_context` is the Lauren `ExecutionContext` that flowed
from the HTTP handler through the `AgentRunner.run(..., execution_context=ec)`
call.  It exposes:

| Attribute | Type | Description |
|---|---|---|
| `request` | `lauren.types.Request` | The originating HTTP request |
| `request.state` | `lauren.types.State` | Mutable state bag set by guards |
| `request.headers` | `lauren.types.Headers` | HTTP headers |
| `get_handler_metadata(key)` | `Any` | Route-level metadata set with `set_metadata` |

```python
async def run(self, ctx: ToolContext, amount: float) -> dict:
    # Identity set by SignatureGuard before any LLM code ran
    user_id = ctx.agent_context.execution_context.request.state.get("user_id")
```

If the agent was invoked outside a web context (scripts, tests), `execution_context`
is `None`.  Always guard with `if ctx.agent_context.execution_context is not None:`.

## Writing a decorator

### Function-form tool

```python
import functools
from lauren_ai import tool, ToolContext, get_tool_context_from_func_args


def require_auth(fn):
    """Reject the tool call when no authenticated user is present."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        ctx = get_tool_context_from_func_args(*args, **kwargs)
        if ctx is None or ctx.agent_context.execution_context is None:
            return {"error": "no_context", "message": "Tool called outside a web request"}
        user_id = ctx.agent_context.execution_context.request.state.get("user_id")
        if not user_id:
            return {"error": "unauthenticated", "message": "Authentication required"}
        return await fn(*args, **kwargs)
    return wrapper


# Apply above the tool function
@tool()
@require_auth
async def get_account_balance(ctx: ToolContext | None = None) -> dict:
    """Return the authenticated user's account balance.

    Args: (none — identity comes from ctx.agent_context.execution_context.request.state)
    """
    user_id = ctx.agent_context.execution_context.request.state.get("user_id")
    return {"user_id": user_id, "balance_usd": 1234.56}
```

### Class-form tool

Decorators work identically on class-form `run()` methods.  The `self`
argument is positional and is skipped by `get_tool_context_from_func_args`
because it is not a `ToolContext` instance.

```python
from lauren import injectable, Scope
from lauren_ai import tool, ToolContext, get_tool_context_from_func_args


def require_auth(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        ctx = get_tool_context_from_func_args(*args, **kwargs)
        if ctx is None or ctx.agent_context.execution_context is None:
            return {"error": "unauthenticated"}
        user_id = ctx.agent_context.execution_context.request.state.get("user_id")
        if not user_id:
            return {"error": "unauthenticated"}
        return await fn(*args, **kwargs)
    return wrapper


@tool()
@injectable(scope=Scope.SINGLETON)
class TransferFundsTool:
    """Transfer funds between accounts.

    Args:
        to_user: Recipient user ID.
        amount: Transfer amount in USD.
    """
    def __init__(self, db: BankDatabase) -> None:
        self._db = db

    @require_auth
    async def run(self, ctx: ToolContext, to_user: str, amount: float) -> dict:
        from_user = ctx.agent_context.execution_context.request.state.get("user_id")
        result = await self._db.transfer(from_user=from_user, to_user=to_user, amount=amount)
        return {"tx_id": result.tx_id, "amount": amount}
```

## Composable decorators

Stack multiple decorators to express layered requirements:

```python
from enum import Enum


class Scope(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


# Scope hierarchy: ADMIN > WRITE > READ
_SCOPE_LEVEL = {Scope.READ: 0, Scope.WRITE: 1, Scope.ADMIN: 2}


def require_scope(required: Scope):
    """Reject when the caller's scope is below *required*."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            ctx = get_tool_context_from_func_args(*args, **kwargs)
            if ctx is None or ctx.agent_context.execution_context is None:
                return {"error": f"scope_required:{required.value}"}
            granted_str = ctx.agent_context.execution_context.request.state.get("scope", "read")
            granted = Scope(granted_str)
            if _SCOPE_LEVEL[granted] < _SCOPE_LEVEL[required]:
                return {"error": "forbidden", "required": required.value, "granted": granted.value}
            return await fn(*args, **kwargs)
        return wrapper
    return decorator


def audit_log(action: str):
    """Append an audit entry after the tool call completes."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            ctx = get_tool_context_from_func_args(*args, **kwargs)
            result = await fn(*args, **kwargs)
            user_id = (
                ctx.agent_context.execution_context.request.state.get("user_id")
                if ctx and ctx.agent_context.execution_context
                else "anonymous"
            )
            print(f"[AUDIT] action={action} user={user_id} result={result}")
            return result
        return wrapper
    return decorator


@tool()
class DeleteRecordTool:
    """Permanently delete a record.

    Args:
        record_id: ID of the record to delete.
    """
    @require_auth
    @require_scope(Scope.ADMIN)
    @audit_log("delete_record")
    async def run(self, ctx: ToolContext, record_id: str) -> dict:
        # Only reached when: authenticated + admin scope
        return {"deleted": record_id}
```

Decorators execute from **innermost to outermost** (bottom-up) on the way in,
so the call stack is: `require_auth` → `require_scope` → `audit_log` → `run`.

## Retry decorator

```python
import asyncio


def retry(max_attempts: int = 3, delay: float = 0.5):
    """Retry a tool call on transient errors."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
            return {"error": "max_retries_exceeded", "detail": str(last_exc)}
        return wrapper
    return decorator


@tool()
@retry(max_attempts=3, delay=0.2)
async def fetch_external_data(url: str, ctx: ToolContext | None = None) -> dict:
    """Fetch JSON from an external API with automatic retry.

    Args:
        url: The URL to fetch.
    """
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
```

## Accessing agent metadata

`ToolContext.agent_context` exposes the running agent's metadata:

```python
def require_agent_metadata(key: str, expected: str):
    """Reject unless agent metadata contains a specific key=value pair."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            ctx = get_tool_context_from_func_args(*args, **kwargs)
            if ctx is None:
                return {"error": "no_context"}
            value = ctx.get_metadata(key)
            if value != expected:
                return {"error": f"missing_metadata:{key}={expected}", "got": value}
            return await fn(*args, **kwargs)
        return wrapper
    return decorator


@tool()
@require_agent_metadata("transfer_approved", "true")
async def execute_transfer(to_user: str, amount: float, ctx: ToolContext | None = None) -> dict:
    """Execute a pre-approved transfer.  Requires 'transfer_approved' in agent metadata.

    Args:
        to_user: Recipient user ID.
        amount: Transfer amount in USD.
    """
    ...
```

## Testing tool decorators

Since `ToolContext` is a plain dataclass you can construct it directly in
tests.  Combine with `AgentTestClient` or the `MockTransport` pattern:

```python
import pytest
from lauren_ai import ToolContext


def _ctx(user_id: str | None = None, scope: str = "read") -> ToolContext:
    """Build a minimal ToolContext for decorator tests."""
    from unittest.mock import MagicMock

    request = MagicMock()
    request.state.get = lambda key, default=None: (
        {"user_id": user_id, "scope": scope}.get(key, default)
    )
    exec_ctx = MagicMock()
    exec_ctx.request = request

    return ToolContext(
        agent_context=None,
        tool_use_id="t1",
        turn=0,
        execution_context=exec_ctx if user_id else None,
    )


class TestRequireAuth:
    async def test_authenticated_call_passes(self):
        ctx = _ctx(user_id="alice")
        result = await get_account_balance(ctx=ctx)
        assert "error" not in result

    async def test_unauthenticated_call_blocked(self):
        ctx = _ctx(user_id=None)
        result = await get_account_balance(ctx=ctx)
        assert result["error"] == "unauthenticated"

    async def test_no_context_blocked(self):
        result = await get_account_balance()  # ctx defaults to None
        assert "error" in result
```

## Summary

| Building block | Purpose |
|---|---|
| `get_tool_context_from_func_args(*args, **kwargs)` | Extract `ToolContext` from any decorator's argument list |
| `ctx.agent_context.execution_context.request.state` | Access guard-verified identity and other HTTP state |
| `ctx.agent_context.execution_context.request.headers` | Read request headers inside a tool |
| `ctx.get_metadata(key)` | Read agent-level metadata set by other tools or the runner |
| `ctx.state` | Per-run mutable state bag for carrying data between tool calls |
| `@functools.wraps(fn)` | Preserve the original function's signature (required — the tool executor inspects it) |

!!! warning "Always preserve the signature"
    Tool decorators **must** use `@functools.wraps(fn)` to preserve the
    original function's annotations.  The tool executor uses
    `inspect.signature` and `typing.get_type_hints` at decoration time to
    build the JSON schema; if the wrapper hides the original signature, the
    schema will be empty and the LLM cannot call the tool correctly.

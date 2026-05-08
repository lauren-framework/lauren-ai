---
name: agent-permission-scoping
description: Scope tool execution behind permission levels (read_only, mutate, admin). Use when different users or agent roles should have different tool access rights, injecting the permission level via ToolContext metadata.
---

> Use `codemap find "ToolContext"` to check the metadata injection API.

# Agent Permission Scoping

Define an ordered `Permission` enum and wrap tools with a `require_permission`
guard. The granted permission is injected via `ToolContext.get_metadata`.

## Permission model

```python
from enum import Enum

class Permission(str, Enum):
    READ_ONLY = "read_only"
    MUTATE    = "mutate"
    ADMIN     = "admin"

PERMISSION_ORDER = {
    Permission.READ_ONLY: 0,
    Permission.MUTATE:    1,
    Permission.ADMIN:     2,
}
```

## Tool guard decorator

```python
from lauren_ai._tools import ToolContext

def require_permission(required: Permission):
    def decorator(cls):
        original_run = cls.run
        async def guarded_run(self, ctx: ToolContext, *args, **kwargs):
            granted_str = ctx.get_metadata("permission") or Permission.READ_ONLY
            granted = Permission(granted_str)
            if PERMISSION_ORDER[granted] < PERMISSION_ORDER[required]:
                return {"error": f"Permission denied: requires {required.value}, got {granted.value}"}
            return await original_run(self, ctx, *args, **kwargs)
        cls.run = guarded_run
        return cls
    return decorator
```

## Protected tool

```python
from lauren_ai._tools import tool

@tool()
@require_permission(Permission.MUTATE)
class UpdateRecordTool:
    """Update a database record.

    Args:
        record_id: The record to update.
        data: The new data.
    """

    async def run(self, ctx: ToolContext, record_id: str, data: str) -> dict:
        return {"updated": record_id, "data": data}
```

## Injecting permission at call time

Pass the permission level as metadata when invoking the runner:

```python
response = await runner.run(
    agent_instance,
    "Update record 42 with new data",
    metadata={"permission": Permission.MUTATE.value},
)
```

`ToolContext.get_metadata("permission")` delegates to
`AgentContext.get_metadata`, which reads from the `metadata` dict supplied
to `runner.run`.

## Checking permissions in a tool

```python
@tool()
class AdminOnlyTool:
    """Perform an admin action.

    Args:
        action: The action to perform.
    """

    async def run(self, ctx: ToolContext, action: str) -> dict:
        permission_str = ctx.get_metadata("permission") or "read_only"
        if permission_str != Permission.ADMIN.value:
            return {"error": "Admin permission required"}
        return {"performed": action}
```

## Notes

- Permission strings are passed as plain values in `metadata` — they travel
  through `AgentContext` and are available in every tool via `ctx.get_metadata`.
- For request-level permission injection in a Lauren web app, extract the
  permission from `request.state` in the controller and forward it as
  `metadata={"permission": user.role}`.
- Extend the pattern by loading permissions from a database inside the guard.

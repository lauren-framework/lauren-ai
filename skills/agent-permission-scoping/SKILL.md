---
name: agent-permission-scoping
description: Scope tool execution behind permission levels (read_only, mutate, admin). Use when different users or agent roles should have different tool access rights, injecting the permission level via ToolContext metadata and declaring requirements via @set_metadata.
---

> Use `codemap find "ToolContext"` to check the metadata injection API.

# Agent Permission Scoping

Define an ordered `Permission` enum and check permissions inline in the tool's
`run()` method. Declare the minimum permission level with `@set_metadata` for
introspection and documentation. Inject the caller's permission via
`runner.run(metadata={"caller_permission": ...})`.

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

def _check_permission(ctx: ToolContext, required: Permission) -> dict | None:
    """Return error dict if caller lacks required permission, else None."""
    granted_str = ctx.get_metadata("caller_permission") or Permission.READ_ONLY.value
    try:
        granted = Permission(granted_str)
    except ValueError:
        granted = Permission.READ_ONLY
    if PERMISSION_ORDER[granted] < PERMISSION_ORDER[required]:
        return {"error": f"Permission denied: requires {required.value}, got {granted.value}"}
    return None
```

## Protected tool

Apply `@set_metadata` first (outermost), then `@tool()`. The `@set_metadata`
decorator stores the declared minimum permission on the tool class for
documentation and introspection. Permission enforcement happens in `run()`.

```python
from lauren_ai._tools import ToolContext, set_metadata, tool

@set_metadata("min_permission", "mutate")
@tool()
class UpdateRecordTool:
    """Update a database record.

    Args:
        record_id: The record to update.
        data: The new data.
    """

    async def run(self, ctx: ToolContext, record_id: str, data: str) -> dict:
        err = _check_permission(ctx, Permission.MUTATE)
        if err:
            return err
        return {"updated": record_id, "data": data}
```

## Injecting permission at call time

Pass the permission level as metadata when invoking the runner or TestClient:

```python
# Production runner
response = await runner.run(
    agent_instance,
    "Update record 42 with new data",
    metadata={"caller_permission": "mutate"},
)

# In tests via TestClient
result = client.run("Update record 42", metadata={"caller_permission": "mutate"})
```

`ToolContext.get_metadata("caller_permission")` checks tool-level static
metadata first, then delegates to `AgentContext.get_metadata`, which reads from
the `metadata` dict supplied to `runner.run`.

## AgentRunner test pattern

```python
import json
from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, set_metadata, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


@agent(model=None, system="Permission test agent")
@use_tools(UpdateRecordTool, DeleteRecordTool, ReadRecordTool)
class PermissionTestAgent(_Capture):
    def __init__(self):
        _Capture.__init__(self)


def _c(text):
    return Completion(id="c1", model="mock", content=text, tool_calls=[],
                      stop_reason="end_turn", usage=TokenUsage(10, 5))


def test_mutate_permission_allows_update():
    agent_inst = PermissionTestAgent()
    client = TestClient(agent_inst)
    client.mock.queue_tool_use("update_record_tool", {"record_id": "1", "data": "x"})
    client.mock.queue_response(_c("Updated."))
    client.run("Update record", metadata={"caller_permission": "mutate"})
    out = json.loads(agent_inst.captured[0].content)
    assert "error" not in out
    assert out["updated"] == "1"


def test_read_only_permission_blocks_update():
    agent_inst = PermissionTestAgent()
    client = TestClient(agent_inst)
    client.mock.queue_tool_use("update_record_tool", {"record_id": "1", "data": "x"})
    client.mock.queue_response(_c("Denied."))
    client.run("Update record", metadata={"caller_permission": "read_only"})
    out = json.loads(agent_inst.captured[0].content)
    assert "error" in out
```

## Reading declared minimum permission in the tool

```python
required = ctx.get_metadata("min_permission")   # → "mutate" (from @set_metadata)
granted  = ctx.get_metadata("caller_permission") # → from runner.run(metadata=...)
```

`ctx.get_metadata` checks tool-level static metadata (`TOOL_METADATA` attribute)
first, then agent-level runtime metadata (`AgentContext.metadata`).

## Notes

- Permission strings are passed as plain values in `metadata`.
- `@set_metadata` is read-only at call time — it's for declaration and
  introspection, not enforcement. Always enforce in `run()`.
- Decorator ordering matters: `@set_metadata` must be outermost (applied last),
  `@tool()` must be innermost (applied first) so that `@tool()` inspects the
  original `run()` signature with `ctx: ToolContext`.
- For request-level permission injection in a Lauren web app, extract the
  permission from `request.state` in the controller and forward it as
  `metadata={"caller_permission": user.role}`.

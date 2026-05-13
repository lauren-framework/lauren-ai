"""Integration tests for the agent permission scoping pattern (Skill 36).

Tests cover:
- Tool with READ_ONLY permission denied when MUTATE is required
- Tool with MUTATE permission succeeds when MUTATE is required
- Tool with ADMIN permission succeeds when MUTATE is required (higher rank)
- Tool with READ_ONLY denied when ADMIN is required
- Tool with ADMIN succeeds when ADMIN is required
- Permission metadata propagated via AgentContext

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import json
from enum import StrEnum

from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, set_metadata, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Permission model
# ---------------------------------------------------------------------------


class Permission(StrEnum):
    READ_ONLY = "read_only"
    MUTATE = "mutate"
    ADMIN = "admin"


PERMISSION_ORDER = {
    Permission.READ_ONLY: 0,
    Permission.MUTATE: 1,
    Permission.ADMIN: 2,
}


def _check_permission(ctx: ToolContext, required: Permission) -> dict | None:
    """Return an error dict if the caller lacks the required permission, else None."""
    granted_str = ctx.get_metadata("caller_permission") or Permission.READ_ONLY.value
    try:
        granted = Permission(granted_str)
    except ValueError:
        granted = Permission.READ_ONLY
    if PERMISSION_ORDER[granted] < PERMISSION_ORDER[required]:
        return {"error": f"Permission denied: requires {required.value}, got {granted.value}"}
    return None


# ---------------------------------------------------------------------------
# Tool definitions — permission check inline, @tool() sees original signatures
# ---------------------------------------------------------------------------


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


@set_metadata("min_permission", "admin")
@tool()
class DeleteRecordTool:
    """Delete a database record.

    Args:
        record_id: The record to delete.
    """

    async def run(self, ctx: ToolContext, record_id: str) -> dict:
        err = _check_permission(ctx, Permission.ADMIN)
        if err:
            return err
        return {"deleted": record_id}


@tool()
class ReadRecordTool:
    """Read a database record.

    Args:
        record_id: The record to read.
    """

    async def run(self, ctx: ToolContext, record_id: str) -> dict:
        return {"id": record_id, "value": "data"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text, *, n=1, stop="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock",
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(10, 5),
    )


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="Permission test agent")
@use_tools(UpdateRecordTool, DeleteRecordTool, ReadRecordTool)
class PermissionTestAgent(_Capture):
    def __init__(self):
        _Capture.__init__(self)


# ---------------------------------------------------------------------------
# Tests: UpdateRecordTool (requires MUTATE)
# ---------------------------------------------------------------------------


class TestUpdateRecordToolPermissions:
    def test_read_only_denied(self):
        """READ_ONLY permission is denied when MUTATE is required."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("update_record_tool", {"record_id": "42", "data": "new data"})
        client.mock.queue_response(_c("Cannot update."))
        client.run("Update record 42", metadata={"caller_permission": "read_only"})
        assert len(agent_inst.captured) == 1
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data
        assert "Permission denied" in data["error"]
        assert "mutate" in data["error"]

    def test_mutate_succeeds(self):
        """MUTATE permission succeeds when MUTATE is required."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("update_record_tool", {"record_id": "42", "data": "new data"})
        client.mock.queue_response(_c("Updated."))
        client.run("Update record 42", metadata={"caller_permission": "mutate"})
        assert len(agent_inst.captured) == 1
        data = json.loads(agent_inst.captured[0].content)
        assert "error" not in data
        assert data["updated"] == "42"
        assert data["data"] == "new data"

    def test_admin_succeeds_for_mutate_tool(self):
        """ADMIN permission (higher rank) succeeds when MUTATE is required."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "update_record_tool", {"record_id": "99", "data": "admin update"}
        )
        client.mock.queue_response(_c("Updated."))
        client.run("Update record 99", metadata={"caller_permission": "admin"})
        data = json.loads(agent_inst.captured[0].content)
        assert "error" not in data
        assert data["updated"] == "99"

    def test_no_permission_defaults_to_read_only_and_denied(self):
        """When no permission is set, defaults to READ_ONLY and is denied."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("update_record_tool", {"record_id": "1", "data": "x"})
        client.mock.queue_response(_c("Denied."))
        client.run("Update record 1")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# Tests: DeleteRecordTool (requires ADMIN)
# ---------------------------------------------------------------------------


class TestDeleteRecordToolPermissions:
    def test_read_only_denied_for_admin_tool(self):
        """READ_ONLY is denied when ADMIN is required."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("delete_record_tool", {"record_id": "5"})
        client.mock.queue_response(_c("Denied."))
        client.run("Delete record 5", metadata={"caller_permission": "read_only"})
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data

    def test_mutate_denied_for_admin_tool(self):
        """MUTATE is denied when ADMIN is required."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("delete_record_tool", {"record_id": "5"})
        client.mock.queue_response(_c("Denied."))
        client.run("Delete record 5", metadata={"caller_permission": "mutate"})
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data
        assert "admin" in data["error"]

    def test_admin_succeeds_for_admin_tool(self):
        """ADMIN permission succeeds when ADMIN is required."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("delete_record_tool", {"record_id": "5"})
        client.mock.queue_response(_c("Deleted."))
        client.run("Delete record 5", metadata={"caller_permission": "admin"})
        data = json.loads(agent_inst.captured[0].content)
        assert "error" not in data
        assert data["deleted"] == "5"


# ---------------------------------------------------------------------------
# Tests: ReadRecordTool (no guard — always accessible)
# ---------------------------------------------------------------------------


class TestReadRecordToolNoGuard:
    def test_read_tool_accessible_without_permission(self):
        """An unguarded tool is accessible regardless of permission level."""
        agent_inst = PermissionTestAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("read_record_tool", {"record_id": "7"})
        client.mock.queue_response(_c("Here is the record."))
        client.run("Read record 7")
        data = json.loads(agent_inst.captured[0].content)
        assert data["id"] == "7"
        assert data["value"] == "data"


# ---------------------------------------------------------------------------
# Tests: set_metadata propagation
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_update_tool_min_permission_metadata(self):
        """@set_metadata on UpdateRecordTool stores 'min_permission' in tool metadata."""
        from lauren_ai._tools import TOOL_METADATA

        meta = getattr(UpdateRecordTool, TOOL_METADATA, {})
        assert meta.get("min_permission") == "mutate"

    def test_delete_tool_min_permission_metadata(self):
        """@set_metadata on DeleteRecordTool stores 'min_permission' in tool metadata."""
        from lauren_ai._tools import TOOL_METADATA

        meta = getattr(DeleteRecordTool, TOOL_METADATA, {})
        assert meta.get("min_permission") == "admin"

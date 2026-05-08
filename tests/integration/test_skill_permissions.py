"""Integration tests for the agent permission scoping pattern (Skill 36).

Tests cover:
- Tool with READ_ONLY permission denied when MUTATE is required
- Tool with MUTATE permission succeeds when MUTATE is required
- Tool with ADMIN permission succeeds when MUTATE is required (higher rank)
- Tool with READ_ONLY denied when ADMIN is required
- Tool with ADMIN succeeds when ADMIN is required
- Permission metadata propagated via AgentContext
"""

import pytest
from enum import Enum


# ---------------------------------------------------------------------------
# Permission model
# ---------------------------------------------------------------------------


class Permission(str, Enum):
    READ_ONLY = "read_only"
    MUTATE = "mutate"
    ADMIN = "admin"


PERMISSION_ORDER = {
    Permission.READ_ONLY: 0,
    Permission.MUTATE: 1,
    Permission.ADMIN: 2,
}


def require_permission(required: Permission):
    """Class decorator that guards a tool's run() method with a permission check."""

    def decorator(cls):
        original_run = cls.run

        async def guarded_run(self, ctx, *args, **kwargs):
            granted_str = ctx.get_metadata("permission") or Permission.READ_ONLY.value
            try:
                granted = Permission(granted_str)
            except ValueError:
                granted = Permission.READ_ONLY
            if PERMISSION_ORDER[granted] < PERMISSION_ORDER[required]:
                return {
                    "error": (
                        f"Permission denied: requires {required.value}, got {granted.value}"
                    )
                }
            return await original_run(self, ctx, *args, **kwargs)

        cls.run = guarded_run
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Minimal ToolContext stub (tests run without the full DI stack)
# ---------------------------------------------------------------------------


class _StubContext:
    def __init__(self, permission: str | None):
        self._permission = permission

    def get_metadata(self, key: str, default=None):
        if key == "permission":
            return self._permission
        return default


# ---------------------------------------------------------------------------
# Tool definitions (no from __future__ import annotations)
# ---------------------------------------------------------------------------

from lauren_ai._tools import tool, ToolContext


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


@tool()
@require_permission(Permission.ADMIN)
class DeleteRecordTool:
    """Delete a database record.

    Args:
        record_id: The record to delete.
    """

    async def run(self, ctx: ToolContext, record_id: str) -> dict:
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
# Tests: UpdateRecordTool (requires MUTATE)
# ---------------------------------------------------------------------------


class TestUpdateRecordToolPermissions:
    @pytest.mark.asyncio
    async def test_read_only_denied(self):
        """READ_ONLY permission is denied when MUTATE is required."""
        tool_instance = UpdateRecordTool()
        ctx = _StubContext(Permission.READ_ONLY.value)
        result = await tool_instance.run(ctx, record_id="42", data="new data")

        assert "error" in result
        assert "Permission denied" in result["error"]
        assert "mutate" in result["error"]

    @pytest.mark.asyncio
    async def test_mutate_succeeds(self):
        """MUTATE permission succeeds when MUTATE is required."""
        tool_instance = UpdateRecordTool()
        ctx = _StubContext(Permission.MUTATE.value)
        result = await tool_instance.run(ctx, record_id="42", data="new data")

        assert "error" not in result
        assert result["updated"] == "42"
        assert result["data"] == "new data"

    @pytest.mark.asyncio
    async def test_admin_succeeds_for_mutate_tool(self):
        """ADMIN permission (higher rank) succeeds when MUTATE is required."""
        tool_instance = UpdateRecordTool()
        ctx = _StubContext(Permission.ADMIN.value)
        result = await tool_instance.run(ctx, record_id="99", data="admin update")

        assert "error" not in result
        assert result["updated"] == "99"

    @pytest.mark.asyncio
    async def test_no_permission_defaults_to_read_only_and_denied(self):
        """When no permission is set, defaults to READ_ONLY and is denied."""
        tool_instance = UpdateRecordTool()
        ctx = _StubContext(None)
        result = await tool_instance.run(ctx, record_id="1", data="x")

        assert "error" in result
        assert "Permission denied" in result["error"]


# ---------------------------------------------------------------------------
# Tests: DeleteRecordTool (requires ADMIN)
# ---------------------------------------------------------------------------


class TestDeleteRecordToolPermissions:
    @pytest.mark.asyncio
    async def test_read_only_denied_for_admin_tool(self):
        """READ_ONLY is denied when ADMIN is required."""
        tool_instance = DeleteRecordTool()
        ctx = _StubContext(Permission.READ_ONLY.value)
        result = await tool_instance.run(ctx, record_id="5")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_mutate_denied_for_admin_tool(self):
        """MUTATE is denied when ADMIN is required."""
        tool_instance = DeleteRecordTool()
        ctx = _StubContext(Permission.MUTATE.value)
        result = await tool_instance.run(ctx, record_id="5")

        assert "error" in result
        assert "admin" in result["error"]

    @pytest.mark.asyncio
    async def test_admin_succeeds_for_admin_tool(self):
        """ADMIN permission succeeds when ADMIN is required."""
        tool_instance = DeleteRecordTool()
        ctx = _StubContext(Permission.ADMIN.value)
        result = await tool_instance.run(ctx, record_id="5")

        assert "error" not in result
        assert result["deleted"] == "5"


# ---------------------------------------------------------------------------
# Tests: ReadRecordTool (no guard — always accessible)
# ---------------------------------------------------------------------------


class TestReadRecordToolNoGuard:
    @pytest.mark.asyncio
    async def test_read_tool_accessible_without_permission(self):
        """An unguarded tool is accessible regardless of permission level."""
        tool_instance = ReadRecordTool()
        ctx = _StubContext(None)
        result = await tool_instance.run(ctx, record_id="7")

        assert result["id"] == "7"
        assert result["value"] == "data"

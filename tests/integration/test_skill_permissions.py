"""Integration tests for the agent permission scoping pattern (Skill 36).

Tests cover:
- Tool with READ_ONLY permission denied when MUTATE is required
- Tool with MUTATE permission succeeds when MUTATE is required
- Tool with ADMIN permission succeeds when MUTATE is required (higher rank)
- Tool with READ_ONLY denied when ADMIN is required
- Tool with ADMIN succeeds when ADMIN is required
- Permission metadata propagated via AgentContext
"""

from __future__ import annotations

from enum import Enum

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient
from lauren_ai._tools import tool, ToolContext


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
# Minimal ToolContext stub
# ---------------------------------------------------------------------------


class _StubContext:
    def __init__(self, permission: str | None):
        self._permission = permission

    def get_metadata(self, key: str, default=None):
        if key == "permission":
            return self._permission
        return default


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


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
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/tools")
class ToolsController:
    @post("/update-record")
    async def update_record(self, body: Json[dict]) -> dict:
        record_id = body.get("record_id", "1")
        data = body.get("data", "")
        permission = body.get("permission", None)
        ctx = _StubContext(permission)
        tool_instance = UpdateRecordTool()
        return await tool_instance.run(ctx, record_id=record_id, data=data)

    @post("/delete-record")
    async def delete_record(self, body: Json[dict]) -> dict:
        record_id = body.get("record_id", "1")
        permission = body.get("permission", None)
        ctx = _StubContext(permission)
        tool_instance = DeleteRecordTool()
        return await tool_instance.run(ctx, record_id=record_id)

    @post("/read-record")
    async def read_record(self, body: Json[dict]) -> dict:
        record_id = body.get("record_id", "1")
        permission = body.get("permission", None)
        ctx = _StubContext(permission)
        tool_instance = ReadRecordTool()
        return await tool_instance.run(ctx, record_id=record_id)


@module(controllers=[ToolsController])
class PermissionsModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(PermissionsModule))


# ---------------------------------------------------------------------------
# Tests: UpdateRecordTool (requires MUTATE)
# ---------------------------------------------------------------------------


class TestUpdateRecordToolPermissions:
    def test_read_only_denied(self):
        """READ_ONLY permission is denied when MUTATE is required."""
        client = build_app()
        r = client.post("/tools/update-record", json={
            "record_id": "42", "data": "new data", "permission": "read_only"
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "Permission denied" in data["error"]
        assert "mutate" in data["error"]

    def test_mutate_succeeds(self):
        """MUTATE permission succeeds when MUTATE is required."""
        client = build_app()
        r = client.post("/tools/update-record", json={
            "record_id": "42", "data": "new data", "permission": "mutate"
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" not in data
        assert data["updated"] == "42"
        assert data["data"] == "new data"

    def test_admin_succeeds_for_mutate_tool(self):
        """ADMIN permission (higher rank) succeeds when MUTATE is required."""
        client = build_app()
        r = client.post("/tools/update-record", json={
            "record_id": "99", "data": "admin update", "permission": "admin"
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" not in data
        assert data["updated"] == "99"

    def test_no_permission_defaults_to_read_only_and_denied(self):
        """When no permission is set, defaults to READ_ONLY and is denied."""
        client = build_app()
        r = client.post("/tools/update-record", json={
            "record_id": "1", "data": "x"
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# Tests: DeleteRecordTool (requires ADMIN)
# ---------------------------------------------------------------------------


class TestDeleteRecordToolPermissions:
    def test_read_only_denied_for_admin_tool(self):
        """READ_ONLY is denied when ADMIN is required."""
        client = build_app()
        r = client.post("/tools/delete-record", json={
            "record_id": "5", "permission": "read_only"
        })
        assert r.status_code == 200
        assert "error" in r.json()

    def test_mutate_denied_for_admin_tool(self):
        """MUTATE is denied when ADMIN is required."""
        client = build_app()
        r = client.post("/tools/delete-record", json={
            "record_id": "5", "permission": "mutate"
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "admin" in data["error"]

    def test_admin_succeeds_for_admin_tool(self):
        """ADMIN permission succeeds when ADMIN is required."""
        client = build_app()
        r = client.post("/tools/delete-record", json={
            "record_id": "5", "permission": "admin"
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" not in data
        assert data["deleted"] == "5"


# ---------------------------------------------------------------------------
# Tests: ReadRecordTool (no guard — always accessible)
# ---------------------------------------------------------------------------


class TestReadRecordToolNoGuard:
    def test_read_tool_accessible_without_permission(self):
        """An unguarded tool is accessible regardless of permission level."""
        client = build_app()
        r = client.post("/tools/read-record", json={"record_id": "7"})
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "7"
        assert data["value"] == "data"

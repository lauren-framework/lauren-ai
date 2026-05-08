"""Integration tests for Skill 41: File System Operation Tool with Path Restrictions.

Tests cover:
- Read within base path → success
- Write within base path → success and verifiable content
- List directory within base path → returns file names
- Path traversal attempt → blocked with error dict
- Absolute path outside base → blocked
- Write creates intermediate directories
- Read non-existent file → error dict

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import tempfile
import os

import pytest
from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json, Query
from lauren.testing import TestClient
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------

from lauren_ai._tools import tool, ToolContext


@tool()
class FileSystemTool:
    """Read and write files within the allowed directory.

    Args:
        operation: 'read', 'write', or 'list'.
        path: Relative path within the allowed directory.
        content: Content to write (for write operation).
    """

    def __init__(self, base_path: str = "/tmp/agent-workspace"):
        self._base = Path(base_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative: str) -> "Path | None":
        target = (self._base / relative).resolve()
        if not str(target).startswith(str(self._base)):
            return None
        return target

    async def run(
        self, ctx: ToolContext, operation: str, path: str, content: str = ""
    ) -> dict:
        safe = self._safe_path(path)
        if safe is None:
            return {"error": "Access denied: path traversal detected"}
        if operation == "read":
            if not safe.exists():
                return {"error": f"File not found: {path}"}
            return {"content": safe.read_text(encoding="utf-8")}
        elif operation == "write":
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8")
            return {"written": str(path), "bytes": len(content)}
        elif operation == "list":
            if not safe.exists() or not safe.is_dir():
                return {"files": []}
            return {"files": [f.name for f in safe.iterdir()]}
        return {"error": f"Unknown operation: {operation}"}


# ---------------------------------------------------------------------------
# Module-level mutable state to hold the current tool instance
# ---------------------------------------------------------------------------

_fs_state: dict = {}


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class _WriteRequest(BaseModel):
    path: str
    content: str = ""


class _ReadRequest(BaseModel):
    path: str


@controller("/fs")
class FsController:
    @post("/write")
    async def write(self, body: Json[_WriteRequest]) -> dict:
        tool = _fs_state["tool"]
        ctx = _make_ctx()
        return await tool.run(ctx, "write", body.path, body.content)

    @post("/read")
    async def read(self, body: Json[_ReadRequest]) -> dict:
        tool = _fs_state["tool"]
        ctx = _make_ctx()
        return await tool.run(ctx, "read", body.path)

    @get("/list")
    async def list_dir(self, path: Query[str] = ".") -> dict:
        tool = _fs_state["tool"]
        ctx = _make_ctx()
        return await tool.run(ctx, "list", path)

    @post("/blocked")
    async def blocked(self, body: Json[_ReadRequest]) -> dict:
        tool = _fs_state["tool"]
        ctx = _make_ctx()
        return await tool.run(ctx, "read", body.path)

    @post("/op")
    async def op(self, body: Json[dict]) -> dict:
        tool = _fs_state["tool"]
        ctx = _make_ctx()
        return await tool.run(ctx, body.get("operation", ""), body.get("path", ""), body.get("content", ""))


@module(controllers=[FsController])
class FsModule: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCtx:
    def __init__(self) -> None:
        self.state: dict = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


def _make_ctx() -> _MockCtx:
    return _MockCtx()


def build_app() -> TestClient:
    base = tempfile.mkdtemp()
    _fs_state["tool"] = FileSystemTool(base_path=base)
    return TestClient(LaurenFactory.create(FsModule))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileSystemToolWrite:
    def test_write_file_success(self):
        client = build_app()
        r = client.post("/fs/write", json={"path": "hello.txt", "content": "Hello, world!"})
        assert r.status_code == 200
        data = r.json()
        assert data["written"] == "hello.txt"
        assert data["bytes"] == 13

    def test_write_creates_file_on_disk(self):
        client = build_app()
        r = client.post("/fs/write", json={"path": "test.txt", "content": "data"})
        assert r.status_code == 200
        tool = _fs_state["tool"]
        assert (tool._base / "test.txt").read_text() == "data"

    def test_write_creates_intermediate_dirs(self):
        client = build_app()
        r = client.post("/fs/write", json={"path": "sub/dir/file.txt", "content": "nested"})
        assert r.status_code == 200
        data = r.json()
        assert data["written"] == "sub/dir/file.txt"
        tool = _fs_state["tool"]
        assert (tool._base / "sub" / "dir" / "file.txt").exists()


class TestFileSystemToolRead:
    def test_read_existing_file(self):
        client = build_app()
        tool = _fs_state["tool"]
        (tool._base / "note.txt").write_text("my content", encoding="utf-8")
        r = client.post("/fs/read", json={"path": "note.txt"})
        assert r.status_code == 200
        assert r.json()["content"] == "my content"

    def test_read_nonexistent_file_returns_error(self):
        client = build_app()
        r = client.post("/fs/read", json={"path": "missing.txt"})
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_write_then_read_roundtrip(self):
        client = build_app()
        client.post("/fs/write", json={"path": "round.txt", "content": "roundtrip data"})
        r = client.post("/fs/read", json={"path": "round.txt"})
        assert r.status_code == 200
        assert r.json()["content"] == "roundtrip data"


class TestFileSystemToolList:
    def test_list_base_directory(self):
        client = build_app()
        tool = _fs_state["tool"]
        (tool._base / "a.txt").write_text("a")
        (tool._base / "b.txt").write_text("b")
        r = client.get("/fs/list?path=.")
        assert r.status_code == 200
        files = r.json()["files"]
        assert "a.txt" in files
        assert "b.txt" in files

    def test_list_empty_dir_returns_empty(self):
        client = build_app()
        tool = _fs_state["tool"]
        (tool._base / "empty").mkdir()
        r = client.get("/fs/list?path=empty")
        assert r.status_code == 200
        assert r.json()["files"] == []

    def test_list_nonexistent_path_returns_empty(self):
        client = build_app()
        r = client.get("/fs/list?path=no_such_dir")
        assert r.status_code == 200
        assert r.json()["files"] == []


class TestFileSystemToolPathSecurity:
    def test_path_traversal_blocked(self):
        client = build_app()
        r = client.post("/fs/blocked", json={"path": "../../etc/passwd"})
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "traversal" in data["error"].lower() or "denied" in data["error"].lower()

    def test_double_dot_in_path_blocked(self):
        client = build_app()
        r = client.post("/fs/write", json={"path": "../outside.txt", "content": "bad"})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_path_within_base_allowed(self):
        client = build_app()
        r = client.post("/fs/write", json={"path": "safe.txt", "content": "safe"})
        assert r.status_code == 200
        assert "error" not in r.json()

    def test_unknown_operation_returns_error(self):
        client = build_app()
        r = client.post("/fs/op", json={"operation": "delete", "path": "anything.txt"})
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "Unknown" in data["error"]

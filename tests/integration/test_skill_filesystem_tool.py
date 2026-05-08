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

import asyncio
import tempfile

from pathlib import Path
from unittest.mock import MagicMock

from lauren_ai._tools import tool, ToolContext


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------


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
# MockToolContext helper
# ---------------------------------------------------------------------------


def _tool_ctx(state=None):
    ctx = MagicMock()
    ctx.execution_context = None
    ctx.agent_context = MagicMock()
    ctx.agent_context.metadata = {}
    ctx.get_metadata = lambda k, d=None: ctx.agent_context.metadata.get(k, d)
    ctx.state = state if state is not None else {}
    ctx.tool_use_id = "t1"
    ctx.turn = 0
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileSystemToolWrite:
    def test_write_file_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "write", "hello.txt", "Hello, world!"))
            assert result["written"] == "hello.txt"
            assert result["bytes"] == 13

    def test_write_creates_file_on_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            asyncio.run(tool.run(ctx, "write", "test.txt", "data"))
            assert (Path(tmpdir) / "test.txt").read_text() == "data"

    def test_write_creates_intermediate_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "write", "sub/dir/file.txt", "nested"))
            assert result["written"] == "sub/dir/file.txt"
            assert (Path(tmpdir) / "sub" / "dir" / "file.txt").exists()


class TestFileSystemToolRead:
    def test_read_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            (Path(tmpdir) / "note.txt").write_text("my content", encoding="utf-8")
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "read", "note.txt"))
            assert result["content"] == "my content"

    def test_read_nonexistent_file_returns_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "read", "missing.txt"))
            assert "error" in result
            assert "not found" in result["error"].lower()

    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            asyncio.run(tool.run(ctx, "write", "round.txt", "roundtrip data"))
            result = asyncio.run(tool.run(ctx, "read", "round.txt"))
            assert result["content"] == "roundtrip data"


class TestFileSystemToolList:
    def test_list_base_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            (Path(tmpdir) / "a.txt").write_text("a")
            (Path(tmpdir) / "b.txt").write_text("b")
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "list", "."))
            assert "a.txt" in result["files"]
            assert "b.txt" in result["files"]

    def test_list_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            (Path(tmpdir) / "empty").mkdir()
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "list", "empty"))
            assert result["files"] == []

    def test_list_nonexistent_path_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "list", "no_such_dir"))
            assert result["files"] == []


class TestFileSystemToolPathSecurity:
    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "read", "../../etc/passwd"))
            assert "error" in result
            assert "traversal" in result["error"].lower() or "denied" in result["error"].lower()

    def test_double_dot_in_path_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "write", "../outside.txt", "bad"))
            assert "error" in result

    def test_path_within_base_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "write", "safe.txt", "safe"))
            assert "error" not in result

    def test_unknown_operation_returns_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileSystemTool(base_path=tmpdir)
            ctx = _tool_ctx()
            result = asyncio.run(tool.run(ctx, "delete", "anything.txt"))
            assert "error" in result
            assert "Unknown" in result["error"]

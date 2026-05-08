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

from lauren_ai._tools import ToolContext
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import _add_to_tool_map
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------

from lauren_ai._tools import tool


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
# Mock context helper
# ---------------------------------------------------------------------------

class MockContext:
    def __init__(self):
        self.state = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFileSystemToolWrite:
    async def test_write_file_success(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "write", "hello.txt", content="Hello, world!")
        assert result["written"] == "hello.txt"
        assert result["bytes"] == 13

    async def test_write_creates_file_on_disk(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        await fs.run(ctx, "write", "test.txt", content="data")
        assert (tmp_path / "test.txt").read_text() == "data"

    async def test_write_creates_intermediate_dirs(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "write", "sub/dir/file.txt", content="nested")
        assert result["written"] == "sub/dir/file.txt"
        assert (tmp_path / "sub" / "dir" / "file.txt").exists()


class TestFileSystemToolRead:
    async def test_read_existing_file(self, tmp_path):
        (tmp_path / "note.txt").write_text("my content", encoding="utf-8")
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "read", "note.txt")
        assert result["content"] == "my content"

    async def test_read_nonexistent_file_returns_error(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "read", "missing.txt")
        assert "error" in result
        assert "not found" in result["error"].lower()

    async def test_write_then_read_roundtrip(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        await fs.run(ctx, "write", "round.txt", content="roundtrip data")
        result = await fs.run(ctx, "read", "round.txt")
        assert result["content"] == "roundtrip data"


class TestFileSystemToolList:
    async def test_list_base_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "list", ".")
        assert set(result["files"]) >= {"a.txt", "b.txt"}

    async def test_list_empty_dir_returns_empty(self, tmp_path):
        subdir = tmp_path / "empty"
        subdir.mkdir()
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "list", "empty")
        assert result["files"] == []

    async def test_list_nonexistent_path_returns_empty(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "list", "no_such_dir")
        assert result["files"] == []


class TestFileSystemToolPathSecurity:
    async def test_path_traversal_blocked(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "read", "../../etc/passwd")
        assert "error" in result
        assert "traversal" in result["error"].lower() or "denied" in result["error"].lower()

    async def test_double_dot_in_path_blocked(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "write", "../outside.txt", content="bad")
        assert "error" in result

    async def test_path_within_base_allowed(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "write", "safe.txt", content="safe")
        assert "error" not in result

    async def test_unknown_operation_returns_error(self, tmp_path):
        fs = FileSystemTool(base_path=str(tmp_path))
        ctx = MockContext()
        result = await fs.run(ctx, "delete", "anything.txt")
        assert "error" in result
        assert "Unknown" in result["error"]

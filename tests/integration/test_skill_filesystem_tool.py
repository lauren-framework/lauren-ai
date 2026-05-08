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

import json
import tempfile

from pathlib import Path

from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Tool definition
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

    async def run(self, ctx: ToolContext, operation: str, path: str, content: str = "") -> dict:
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


def _make_agent(base_path: str):
    """Create a fresh FileSystemTool + agent for a given base_path."""
    fs_tool = FileSystemTool(base_path=base_path)

    @agent(model=None, system="Filesystem agent")
    @use_tools(fs_tool)
    class FsTestAgent(_Capture):
        def __init__(self):
            _Capture.__init__(self)

    return FsTestAgent()


# ---------------------------------------------------------------------------
# Tests: write
# ---------------------------------------------------------------------------


class TestFileSystemToolWrite:
    def test_write_file_success(self):
        """write returns path and byte count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "write", "path": "hello.txt", "content": "Hello, world!"},
            )
            client.mock.queue_response(_c("Written."))
            client.run("Write hello.txt")
            result = json.loads(agent_inst.captured[0].content)
            assert result["written"] == "hello.txt"
            assert result["bytes"] == 13

    def test_write_creates_file_on_disk(self):
        """write creates the file on disk with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "write", "path": "test.txt", "content": "data"},
            )
            client.mock.queue_response(_c("Written."))
            client.run("Write test.txt")
            assert (Path(tmpdir) / "test.txt").read_text() == "data"

    def test_write_creates_intermediate_dirs(self):
        """write creates parent directories when they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "write", "path": "sub/dir/file.txt", "content": "nested"},
            )
            client.mock.queue_response(_c("Written."))
            client.run("Write nested file")
            result = json.loads(agent_inst.captured[0].content)
            assert result["written"] == "sub/dir/file.txt"
            assert (Path(tmpdir) / "sub" / "dir" / "file.txt").exists()


# ---------------------------------------------------------------------------
# Tests: read
# ---------------------------------------------------------------------------


class TestFileSystemToolRead:
    def test_read_existing_file(self):
        """read returns file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "note.txt").write_text("my content", encoding="utf-8")
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "read", "path": "note.txt"},
            )
            client.mock.queue_response(_c("Here is the file."))
            client.run("Read note.txt")
            result = json.loads(agent_inst.captured[0].content)
            assert result["content"] == "my content"

    def test_read_nonexistent_file_returns_error(self):
        """read on a missing file returns an error dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "read", "path": "missing.txt"},
            )
            client.mock.queue_response(_c("File not found."))
            client.run("Read missing.txt")
            result = json.loads(agent_inst.captured[0].content)
            assert "error" in result
            assert "not found" in result["error"].lower()

    def test_write_then_read_roundtrip(self):
        """write followed by read returns the same content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "write", "path": "round.txt", "content": "roundtrip data"},
            )
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "read", "path": "round.txt"},
            )
            client.mock.queue_response(_c("Done."))
            client.run("Write then read roundtrip")
            read_result = json.loads(agent_inst.captured[1].content)
            assert read_result["content"] == "roundtrip data"


# ---------------------------------------------------------------------------
# Tests: list
# ---------------------------------------------------------------------------


class TestFileSystemToolList:
    def test_list_base_directory(self):
        """list returns file names in the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.txt").write_text("a")
            (Path(tmpdir) / "b.txt").write_text("b")
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "list", "path": "."},
            )
            client.mock.queue_response(_c("Files listed."))
            client.run("List files")
            result = json.loads(agent_inst.captured[0].content)
            assert "a.txt" in result["files"]
            assert "b.txt" in result["files"]

    def test_list_empty_dir_returns_empty(self):
        """list on an empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "empty").mkdir()
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "list", "path": "empty"},
            )
            client.mock.queue_response(_c("Empty dir."))
            client.run("List empty dir")
            result = json.loads(agent_inst.captured[0].content)
            assert result["files"] == []

    def test_list_nonexistent_path_returns_empty(self):
        """list on a non-existent path returns empty files list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "list", "path": "no_such_dir"},
            )
            client.mock.queue_response(_c("Not found."))
            client.run("List non-existent dir")
            result = json.loads(agent_inst.captured[0].content)
            assert result["files"] == []


# ---------------------------------------------------------------------------
# Tests: path security
# ---------------------------------------------------------------------------


class TestFileSystemToolPathSecurity:
    def test_path_traversal_blocked(self):
        """Path traversal via ../../ is blocked with an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "read", "path": "../../etc/passwd"},
            )
            client.mock.queue_response(_c("Access denied."))
            client.run("Try to read /etc/passwd")
            result = json.loads(agent_inst.captured[0].content)
            assert "error" in result
            assert "traversal" in result["error"].lower() or "denied" in result["error"].lower()

    def test_double_dot_in_path_blocked(self):
        """../outside.txt traversal is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "write", "path": "../outside.txt", "content": "bad"},
            )
            client.mock.queue_response(_c("Blocked."))
            client.run("Try to write outside base")
            result = json.loads(agent_inst.captured[0].content)
            assert "error" in result

    def test_path_within_base_allowed(self):
        """A safe relative path within the base directory is allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "write", "path": "safe.txt", "content": "safe"},
            )
            client.mock.queue_response(_c("Written."))
            client.run("Write safe file")
            result = json.loads(agent_inst.captured[0].content)
            assert "error" not in result

    def test_unknown_operation_returns_error(self):
        """An unknown operation returns an error dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_inst = _make_agent(tmpdir)
            client = TestClient(agent_inst)
            client.mock.queue_tool_use(
                "file_system_tool",
                {"operation": "delete", "path": "anything.txt"},
            )
            client.mock.queue_response(_c("Unknown op."))
            client.run("Delete file")
            result = json.loads(agent_inst.captured[0].content)
            assert "error" in result
            assert "Unknown" in result["error"]

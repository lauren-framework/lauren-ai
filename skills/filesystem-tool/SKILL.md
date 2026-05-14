---
name: filesystem-tool
description: Implements a sandboxed file system tool for agents using @tool() class-form. Use when building tools that read, write, or list files within a restricted base directory, with path traversal protection to prevent access outside the allowed workspace.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# File System Operation Tool with Path Restrictions

## Critical rule — tool annotations must resolve

**`from __future__ import annotations` is supported, but every type used by `@tool()` must resolve when schema generation runs.**

`@tool()` resolves annotations when it builds the JSON schema. Future
annotations are supported, but unresolved forward references and circular
imports in function-form tool files still break schema generation.

---

## Overview

The `FileSystemTool` is a class-form `@tool()` that reads, writes, and lists
files within a configurable base directory. It resolves all paths via
`Path.resolve()` and rejects any path that resolves outside the base — blocking
`../` traversal attacks.

---

## Implementation

```python
# tools/filesystem_tool.py — future annotations are allowed; keep tool types importable
from pathlib import Path
from lauren_ai import tool, ToolContext

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

    def _safe_path(self, relative: str) -> Path | None:
        target = (self._base / relative).resolve()
        if not str(target).startswith(str(self._base)):
            return None  # path traversal detected
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
```

---

## AgentRunner test pattern

Use `tempfile.TemporaryDirectory` for isolated test directories. Create a fresh
tool instance and agent factory per test to avoid shared state.

```python
import json
import tempfile
from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolResult
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


def _make_agent(base_path: str):
    fs_tool = FileSystemTool(base_path=base_path)

    @agent(model=None, system="Filesystem agent")
    @use_tools(fs_tool)
    class FsTestAgent(_Capture):
        def __init__(self):
            _Capture.__init__(self)

    return FsTestAgent()


def _c(text):
    return Completion(id="c1", model="mock", content=text, tool_calls=[],
                      stop_reason="end_turn", usage=TokenUsage(10, 5))


def test_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_inst = _make_agent(tmpdir)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "file_system_tool",
            {"operation": "write", "path": "test.txt", "content": "hello"},
        )
        client.mock.queue_response(_c("Done."))
        client.run("Write test.txt")
        out = json.loads(agent_inst.captured[0].content)
        assert out["written"] == "test.txt"


def test_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_inst = _make_agent(tmpdir)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "file_system_tool",
            {"operation": "read", "path": "../../etc/passwd"},
        )
        client.mock.queue_response(_c("Access denied."))
        client.run("Try to read /etc/passwd")
        out = json.loads(agent_inst.captured[0].content)
        assert "error" in out
        assert "traversal" in out["error"].lower() or "denied" in out["error"].lower()
```

---

## Path safety rules

| Input path | Result |
|-----------|--------|
| `"notes.txt"` | Allowed — stays in base |
| `"subdir/notes.txt"` | Allowed — stays in base |
| `"../../etc/passwd"` | Blocked — resolves outside base |
| `"/etc/passwd"` | Blocked — absolute path outside base |

The safety check uses `str(target).startswith(str(self._base))` after both
paths are fully resolved. This is robust against symlink tricks on most
platforms.

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `@tool()`, `ToolContext` |
| `src/lauren_ai/_tools/_executor.py` | `ToolExecutor` dispatch |
| `src/lauren_ai/_tools/_schema.py` | JSON schema generation |

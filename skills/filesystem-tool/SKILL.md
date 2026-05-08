---
name: filesystem-tool
description: Implements a sandboxed file system tool for agents using @tool() class-form. Use when building tools that read, write, or list files within a restricted base directory, with path traversal protection to prevent access outside the allowed workspace.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# File System Operation Tool with Path Restrictions

## Critical rule — no PEP 563 in tool files

**Never add `from __future__ import annotations` to any file that defines `@tool()`.**

`@tool()` calls `inspect.signature()` at decoration time to build the JSON
schema. PEP 563 lazy evaluation converts all annotations to strings, silently
breaking schema generation.

---

## Overview

The `FileSystemTool` is a class-form `@tool()` that reads, writes, and lists
files within a configurable base directory.  It resolves all paths via
`Path.resolve()` and rejects any path that resolves outside the base — blocking
`../` traversal attacks.

---

## Implementation

```python
# tools/filesystem_tool.py — NO from __future__ import annotations
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

## Attaching to an agent

```python
# agents.py — from __future__ import annotations is safe here
from __future__ import annotations
from lauren_ai import agent, use_tools
from .tools.filesystem_tool import FileSystemTool

@agent(model="claude-opus-4-6", system="You are a file management assistant.")
@use_tools(FileSystemTool)
class FileAgent: ...
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
paths are fully resolved.  This is robust against symlink tricks on most
platforms.

---

## Testing

Use `tempfile.mkdtemp()` as the `base_path` so each test gets an isolated
temporary directory that is cleaned up automatically.

```python
import tempfile, pytest
from .tools.filesystem_tool import FileSystemTool
from lauren_ai._tools import ToolContext

@pytest.fixture
def tool_instance(tmp_path):
    return FileSystemTool(base_path=str(tmp_path))
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `@tool()`, `ToolContext` |
| `src/lauren_ai/_tools/_executor.py` | `ToolExecutor` dispatch |
| `src/lauren_ai/_tools/_schema.py` | JSON schema generation |

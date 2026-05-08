---
name: code-execution-sandbox
description: Execute Python code in a restricted sandbox using exec() with limited builtins, stdout capture, and asyncio timeout. Use when an agent needs to run user-supplied or LLM-generated Python code safely without access to the filesystem or dangerous builtins.
---

> Use `codemap find "CodeExecutionTool"` after adding the pattern to your project.

# Code Execution Sandbox Tool

Run Python code snippets inside a restricted `exec()` environment. Blocks
dangerous builtins (`open`, `exec`, `eval`, `__import__`), captures stdout,
and enforces a configurable timeout.

## Pattern

```python
import asyncio
import io
import sys
from contextlib import redirect_stdout
from lauren_ai._tools import tool, ToolContext

# Build a safe builtins dict by excluding dangerous names
_BLOCKED = {"open", "exec", "eval", "__import__", "compile", "globals", "locals"}

def _safe_builtins() -> dict:
    raw = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    return {k: v for k, v in raw.items() if k not in _BLOCKED}

SAFE_BUILTINS = _safe_builtins()

@tool()
class CodeExecutionTool:
    """Execute Python code in a restricted sandbox.

    Args:
        code: The Python code to execute.
        timeout: Maximum execution time in seconds (capped by tool default).
    """

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout

    async def run(self, ctx: ToolContext, code: str, timeout: float = 5.0) -> dict:
        effective_timeout = min(timeout, self._timeout)
        stdout_capture = io.StringIO()
        local_vars: dict = {}

        # exec() is synchronous and CPU-bound — run in a thread so the event
        # loop can enforce the timeout even for tight loops that never yield.
        def _exec_sync():
            with redirect_stdout(stdout_capture):
                exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)

        try:
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, _exec_sync),
                timeout=effective_timeout,
            )
            return {
                "stdout": stdout_capture.getvalue(),
                "locals": {
                    k: repr(v)
                    for k, v in local_vars.items()
                    if not k.startswith("_")
                },
                "success": True,
            }
        except asyncio.TimeoutError:
            return {
                "error": f"Execution timed out after {effective_timeout}s",
                "success": False,
            }
        except Exception as e:
            return {"error": str(e), "success": False}
```

## What is blocked

| Builtin | Risk |
|---|---|
| `open` | Filesystem access |
| `exec` / `eval` | Dynamic code execution escape |
| `__import__` | Arbitrary module imports |
| `compile` | Bytecode manipulation |
| `globals` / `locals` | Scope escape |

Math, string operations, list comprehensions, `print`, `len`, `range`, `zip`,
`enumerate`, `sorted`, `sum`, etc. all remain available.

## Example safe code

```python
code = """
x = [i**2 for i in range(10)]
print(sum(x))
"""
result = await tool.run(ctx, code)
# result == {"stdout": "285\n", "locals": {"x": "[0, 1, 4, ...]"}, "success": True}
```

## Example blocked code

```python
result = await tool.run(ctx, "open('/etc/passwd').read()")
# result == {"error": "name 'open' is not defined", "success": False}
```

## Notes

- `asyncio.wait_for` wraps a coroutine; `exec` itself is synchronous but runs
  inside an async wrapper so the event loop can apply the timeout.
- For CPU-bound loops that do not yield, consider using
  `anyio.to_thread.run_sync` with a real thread timeout.
- Never allow network access inside the sandbox without explicit vetting.

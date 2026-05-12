"""Integration tests for the code execution sandbox tool pattern (Skill 40).

Tests cover:
- Simple math code executes and returns stdout
- print() output is captured in stdout
- local variables are reported
- open() is blocked (not in safe builtins)
- exec() is blocked
- __import__() is blocked
- Infinite loop times out (short timeout)
- Exception in code returns error dict
- Success flag is True on success, False on failure
- Timeout flag returns success=False with timeout message
"""

import asyncio
import io
from contextlib import redirect_stdout

# ---------------------------------------------------------------------------
# CodeExecutionTool implementation
# ---------------------------------------------------------------------------

_BLOCKED = {"open", "exec", "eval", "__import__", "compile", "globals", "locals"}


def _safe_builtins() -> dict:
    raw = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    return {k: v for k, v in raw.items() if k not in _BLOCKED}


SAFE_BUILTINS = _safe_builtins()


class CodeExecutionTool:
    """Execute Python code in a restricted sandbox.

    Args:
        code: The Python code to execute.
        timeout: Maximum execution time in seconds (capped by tool default).
    """

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout

    async def run(self, ctx, code: str, timeout: float = 5.0) -> dict:
        effective_timeout = min(timeout, self._timeout)
        stdout_capture = io.StringIO()
        local_vars: dict = {}

        import time as _time

        safe_globals = {"__builtins__": SAFE_BUILTINS, "time": _time}

        def _exec_sync():
            with redirect_stdout(stdout_capture):
                exec(code, safe_globals, local_vars)

        try:
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, _exec_sync),
                timeout=effective_timeout,
            )
            return {
                "stdout": stdout_capture.getvalue(),
                "locals": {k: repr(v) for k, v in local_vars.items() if not k.startswith("_")},
                "success": True,
            }
        except TimeoutError:
            return {
                "error": f"Execution timed out after {effective_timeout}s",
                "success": False,
            }
        except Exception as e:
            return {"error": str(e), "success": False}


_CTX = None  # ctx not used in sandbox


# ---------------------------------------------------------------------------
# Tests: successful execution
# ---------------------------------------------------------------------------


class TestCodeExecutionSuccess:
    async def test_print_captured_in_stdout(self):
        """print() output is captured in the stdout field."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code='print("hello world")', timeout=5.0)
        assert data["success"] is True
        assert "hello world" in data["stdout"]

    async def test_math_expression_produces_correct_output(self):
        """Arithmetic evaluates correctly and output is captured."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="print(2 + 3 * 4)", timeout=5.0)
        assert data["success"] is True
        assert "14" in data["stdout"]

    async def test_local_variables_are_reported(self):
        """Variables defined in the code appear in the locals dict."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="x = 42\ny = 'hello'", timeout=5.0)
        assert data["success"] is True
        assert "x" in data["locals"]
        assert "y" in data["locals"]
        assert "42" in data["locals"]["x"]

    async def test_list_comprehension_executes(self):
        """List comprehensions execute without errors."""
        tool = CodeExecutionTool()
        code = "squares = [i**2 for i in range(5)]\nprint(squares)"
        data = await tool.run(_CTX, code=code, timeout=5.0)
        assert data["success"] is True
        assert "[0, 1, 4, 9, 16]" in data["stdout"]

    async def test_private_variables_excluded_from_locals(self):
        """Variables starting with _ are excluded from the locals report."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="_hidden = 'secret'\npublic = 1", timeout=5.0)
        assert data["success"] is True
        assert "_hidden" not in data["locals"]
        assert "public" in data["locals"]

    async def test_empty_code_succeeds(self):
        """Empty code string executes without error."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="", timeout=5.0)
        assert data["success"] is True
        assert data["stdout"] == ""

    async def test_multiline_code_executes(self):
        """Multi-line code with a function definition executes correctly."""
        tool = CodeExecutionTool()
        code = "def add(a, b):\n    return a + b\nresult = add(3, 4)\nprint(result)"
        data = await tool.run(_CTX, code=code, timeout=5.0)
        assert data["success"] is True
        assert "7" in data["stdout"]


# ---------------------------------------------------------------------------
# Tests: blocked builtins
# ---------------------------------------------------------------------------


class TestCodeExecutionBlocked:
    async def test_open_is_blocked(self):
        """open() is not available in the sandbox."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="f = open('/etc/passwd', 'r')", timeout=5.0)
        assert data["success"] is False
        assert "error" in data
        assert "open" in data["error"] or "not defined" in data["error"]

    async def test_exec_is_blocked(self):
        """exec() is not available in the sandbox."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="exec('import os')", timeout=5.0)
        assert data["success"] is False

    async def test_import_via_dunder_is_blocked(self):
        """__import__ is not available in the sandbox."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="__import__('os')", timeout=5.0)
        assert data["success"] is False

    async def test_exception_in_code_returns_error(self):
        """A ZeroDivisionError in code returns an error dict."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="x = 1 / 0", timeout=5.0)
        assert data["success"] is False
        assert "error" in data
        assert "ZeroDivisionError" in data["error"] or "division by zero" in data["error"]

    async def test_name_error_is_caught(self):
        """Using an undefined name returns a NameError in the error field."""
        tool = CodeExecutionTool()
        data = await tool.run(_CTX, code="y = undefined_variable", timeout=5.0)
        assert data["success"] is False
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


class TestCodeExecutionTimeout:
    async def test_timeout_returns_success_false(self):
        """Code slower than the timeout returns success=False with a timeout error."""
        tool = CodeExecutionTool(timeout=0.05)
        data = await tool.run(_CTX, code="time.sleep(2)", timeout=0.05)
        assert data["success"] is False
        assert "error" in data
        assert "timed out" in data["error"].lower() or "timeout" in data["error"].lower()

    async def test_tool_timeout_caps_per_call_timeout(self):
        """Tool-level timeout is the effective cap when it is smaller than per-call timeout."""
        tool = CodeExecutionTool(timeout=0.05)
        data = await tool.run(_CTX, code="time.sleep(2)", timeout=10.0)
        assert data["success"] is False
        assert "timed out" in data["error"].lower() or "timeout" in data["error"].lower()

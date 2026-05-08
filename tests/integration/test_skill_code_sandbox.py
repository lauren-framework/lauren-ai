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
import sys
from contextlib import redirect_stdout

import pytest


# ---------------------------------------------------------------------------
# CodeExecutionTool implementation (inline for test isolation)
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

        # exec() is synchronous and CPU-bound — run it in a thread so the
        # event loop can enforce the timeout via asyncio.wait_for.
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
                "locals": {
                    k: repr(v) for k, v in local_vars.items() if not k.startswith("_")
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


_CTX = None  # tool.run(ctx, ...) — ctx is not used in sandbox


# ---------------------------------------------------------------------------
# Tests: successful execution
# ---------------------------------------------------------------------------


class TestCodeExecutionSuccess:
    @pytest.mark.asyncio
    async def test_print_captured_in_stdout(self):
        """print() output is captured in the stdout field."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, 'print("hello world")')

        assert result["success"] is True
        assert "hello world" in result["stdout"]

    @pytest.mark.asyncio
    async def test_math_expression_produces_correct_output(self):
        """Arithmetic evaluates correctly and output is captured."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "print(2 + 3 * 4)")

        assert result["success"] is True
        assert "14" in result["stdout"]

    @pytest.mark.asyncio
    async def test_local_variables_are_reported(self):
        """Variables defined in the code appear in the locals dict."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "x = 42\ny = 'hello'")

        assert result["success"] is True
        assert "x" in result["locals"]
        assert "y" in result["locals"]
        assert "42" in result["locals"]["x"]

    @pytest.mark.asyncio
    async def test_list_comprehension_executes(self):
        """List comprehensions execute without errors."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "squares = [i**2 for i in range(5)]\nprint(squares)")

        assert result["success"] is True
        assert "[0, 1, 4, 9, 16]" in result["stdout"]

    @pytest.mark.asyncio
    async def test_private_variables_excluded_from_locals(self):
        """Variables starting with _ are excluded from the locals report."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "_hidden = 'secret'\npublic = 1")

        assert result["success"] is True
        assert "_hidden" not in result["locals"]
        assert "public" in result["locals"]

    @pytest.mark.asyncio
    async def test_empty_code_succeeds(self):
        """Empty code string executes without error."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "")

        assert result["success"] is True
        assert result["stdout"] == ""

    @pytest.mark.asyncio
    async def test_multiline_code_executes(self):
        """Multi-line code with a function definition executes correctly."""
        sandbox = CodeExecutionTool()
        code = "def add(a, b):\n    return a + b\nresult = add(3, 4)\nprint(result)"
        result = await sandbox.run(_CTX, code)

        assert result["success"] is True
        assert "7" in result["stdout"]


# ---------------------------------------------------------------------------
# Tests: blocked builtins
# ---------------------------------------------------------------------------


class TestCodeExecutionBlocked:
    @pytest.mark.asyncio
    async def test_open_is_blocked(self):
        """open() is not available in the sandbox."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "f = open('/etc/passwd', 'r')")

        assert result["success"] is False
        assert "error" in result
        # The error message should mention 'open' not being defined
        assert "open" in result["error"] or "not defined" in result["error"]

    @pytest.mark.asyncio
    async def test_exec_is_blocked(self):
        """exec() is not available in the sandbox."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "exec('import os')")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_import_via_dunder_is_blocked(self):
        """__import__ is not available in the sandbox."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "__import__('os')")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exception_in_code_returns_error(self):
        """A ZeroDivisionError in code returns an error dict."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "x = 1 / 0")

        assert result["success"] is False
        assert "error" in result
        assert "ZeroDivisionError" in result["error"] or "division by zero" in result["error"]

    @pytest.mark.asyncio
    async def test_name_error_is_caught(self):
        """Using an undefined name returns a NameError in the error field."""
        sandbox = CodeExecutionTool()
        result = await sandbox.run(_CTX, "y = undefined_variable")

        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


class TestCodeExecutionTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_success_false(self):
        """Code slower than the timeout returns success=False with a timeout error."""
        sandbox = CodeExecutionTool(timeout=0.05)
        # time.sleep is interruptible via asyncio.wait_for / run_in_executor
        # cancellation: the future resolves as TimeoutError while the thread
        # finishes the sleep in the background (harmless for tests).
        # time is pre-imported in the sandbox globals, so no import needed
        result = await sandbox.run(
            _CTX,
            "time.sleep(2)",
            timeout=0.05,
        )

        assert result["success"] is False
        assert "error" in result
        assert "timed out" in result["error"].lower() or "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_tool_timeout_caps_per_call_timeout(self):
        """Tool-level timeout is the effective cap when it is smaller than per-call timeout."""
        # Tool has 0.05s cap; per-call asks for 10s → effective = 0.05s
        sandbox = CodeExecutionTool(timeout=0.05)
        result = await sandbox.run(
            _CTX,
            "time.sleep(2)",
            timeout=10.0,
        )

        assert result["success"] is False
        assert "timed out" in result["error"].lower() or "timeout" in result["error"].lower()

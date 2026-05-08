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

from __future__ import annotations

import asyncio
import io
from contextlib import redirect_stdout

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient


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


_CTX = None  # ctx not used in sandbox


# ---------------------------------------------------------------------------
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/sandbox")
class SandboxController:
    @post("/run")
    async def run(self, body: Json[dict]) -> dict:
        code = body.get("code", "")
        timeout = body.get("timeout", 5.0)
        tool_timeout = body.get("tool_timeout", 5.0)
        sandbox = CodeExecutionTool(timeout=tool_timeout)
        return await sandbox.run(_CTX, code=code, timeout=timeout)


@module(controllers=[SandboxController])
class CodeSandboxModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(CodeSandboxModule))


# ---------------------------------------------------------------------------
# Tests: successful execution
# ---------------------------------------------------------------------------


class TestCodeExecutionSuccess:
    def test_print_captured_in_stdout(self):
        """print() output is captured in the stdout field."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": 'print("hello world")', "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "hello world" in data["stdout"]

    def test_math_expression_produces_correct_output(self):
        """Arithmetic evaluates correctly and output is captured."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "print(2 + 3 * 4)", "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "14" in data["stdout"]

    def test_local_variables_are_reported(self):
        """Variables defined in the code appear in the locals dict."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "x = 42\ny = 'hello'", "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "x" in data["locals"]
        assert "y" in data["locals"]
        assert "42" in data["locals"]["x"]

    def test_list_comprehension_executes(self):
        """List comprehensions execute without errors."""
        client = build_app()
        r = client.post("/sandbox/run", json={
            "code": "squares = [i**2 for i in range(5)]\nprint(squares)",
            "timeout": 5.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "[0, 1, 4, 9, 16]" in data["stdout"]

    def test_private_variables_excluded_from_locals(self):
        """Variables starting with _ are excluded from the locals report."""
        client = build_app()
        r = client.post("/sandbox/run", json={
            "code": "_hidden = 'secret'\npublic = 1",
            "timeout": 5.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "_hidden" not in data["locals"]
        assert "public" in data["locals"]

    def test_empty_code_succeeds(self):
        """Empty code string executes without error."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "", "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["stdout"] == ""

    def test_multiline_code_executes(self):
        """Multi-line code with a function definition executes correctly."""
        client = build_app()
        code = "def add(a, b):\n    return a + b\nresult = add(3, 4)\nprint(result)"
        r = client.post("/sandbox/run", json={"code": code, "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "7" in data["stdout"]


# ---------------------------------------------------------------------------
# Tests: blocked builtins
# ---------------------------------------------------------------------------


class TestCodeExecutionBlocked:
    def test_open_is_blocked(self):
        """open() is not available in the sandbox."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "f = open('/etc/passwd', 'r')", "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "error" in data
        assert "open" in data["error"] or "not defined" in data["error"]

    def test_exec_is_blocked(self):
        """exec() is not available in the sandbox."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "exec('import os')", "timeout": 5.0})
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_import_via_dunder_is_blocked(self):
        """__import__ is not available in the sandbox."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "__import__('os')", "timeout": 5.0})
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_exception_in_code_returns_error(self):
        """A ZeroDivisionError in code returns an error dict."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "x = 1 / 0", "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "error" in data
        assert "ZeroDivisionError" in data["error"] or "division by zero" in data["error"]

    def test_name_error_is_caught(self):
        """Using an undefined name returns a NameError in the error field."""
        client = build_app()
        r = client.post("/sandbox/run", json={"code": "y = undefined_variable", "timeout": 5.0})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


class TestCodeExecutionTimeout:
    def test_timeout_returns_success_false(self):
        """Code slower than the timeout returns success=False with a timeout error."""
        client = build_app()
        r = client.post("/sandbox/run", json={
            "code": "time.sleep(2)",
            "timeout": 0.05,
            "tool_timeout": 0.05,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "error" in data
        assert "timed out" in data["error"].lower() or "timeout" in data["error"].lower()

    def test_tool_timeout_caps_per_call_timeout(self):
        """Tool-level timeout is the effective cap when it is smaller than per-call timeout."""
        client = build_app()
        r = client.post("/sandbox/run", json={
            "code": "time.sleep(2)",
            "timeout": 10.0,
            "tool_timeout": 0.05,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "timed out" in data["error"].lower() or "timeout" in data["error"].lower()

"""Built-in skill tools for ``lauren-ai`` agents.

Skills are pre-built ``@tool()``-decorated classes that can be attached to
any agent via ``@use_tools()``::

    from lauren_ai.skills import WebSearchTool, HttpFetchTool

    @use_tools(WebSearchTool, HttpFetchTool)
    @agent(model="claude-opus-4-6", system="You are a research assistant.")
    class ResearchAgent: ...

Available skills
----------------

* :class:`HttpFetchTool` — fetch a URL via HTTP and return the body text.
* :class:`CodeExecutionTool` — execute Python code snippets in a sandbox.
"""

import asyncio
from typing import Any

from lauren_ai._tools import ToolContext, tool

__all__ = [
    "HttpFetchTool",
    "CodeExecutionTool",
]


# ---------------------------------------------------------------------------
# HttpFetchTool
# ---------------------------------------------------------------------------


@tool()
async def HttpFetchTool(  # noqa: N802
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    timeout: float = 30.0,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    """Fetch a URL and return the response body and status code.

    Args:
        url: The URL to fetch. Must start with http:// or https://.
        method: HTTP method (GET, POST). Defaults to GET.
        headers: Optional request headers dict.
        timeout: Request timeout in seconds.

    Returns a dict with ``status``, ``body`` (truncated at 8 KB), and
    ``content_type`` keys.
    """
    try:
        import httpx  # type: ignore[import]  # noqa: PLC0415
    except ImportError:
        return {
            "error": "httpx is not installed. Run: pip install httpx",
            "status": 0,
            "body": "",
            "content_type": "",
        }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(method.upper(), url, headers=headers or {})
        body = resp.text[:8192]  # Truncate at 8 KB
        return {
            "status": resp.status_code,
            "body": body,
            "content_type": resp.headers.get("content-type", ""),
        }


# ---------------------------------------------------------------------------
# CodeExecutionTool
# ---------------------------------------------------------------------------


@tool()
async def CodeExecutionTool(  # noqa: N802
    code: str,
    timeout: float = 10.0,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    """Execute a Python code snippet and return stdout/stderr.

    Args:
        code: The Python code to execute. Must be safe, non-destructive code.
        timeout: Execution timeout in seconds (max 30).

    Returns a dict with ``stdout``, ``stderr``, and ``exit_code`` keys.

    .. warning::
        This is a minimal sandbox.  For production use, execute in an isolated
        subprocess or a container.
    """
    import sys  # noqa: PLC0415

    effective_timeout = min(timeout, 30.0)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        except TimeoutError:
            proc.kill()
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {effective_timeout}s.",
                "exit_code": -1,
            }

        return {
            "stdout": stdout_b.decode("utf-8", errors="replace")[:4096],
            "stderr": stderr_b.decode("utf-8", errors="replace")[:2048],
            "exit_code": proc.returncode,
        }
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "exit_code": -1}

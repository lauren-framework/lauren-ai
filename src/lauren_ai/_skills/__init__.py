"""Built-in skill tools for ``lauren-ai`` agents.

Skills are pre-built ``@tool()``-decorated classes that can be attached to
any agent via ``@use_tools()``::

    from lauren_ai.skills import WebSearchTool, HttpFetchTool

    @use_tools(WebSearchTool, HttpFetchTool)
    @agent(model="claude-opus-4-6", system="You are a research assistant.")
    class ResearchAgent: ...

Available skills
----------------

* :class:`WebSearchTool` — search the web (stub; override ``search()`` with a real API).
* :class:`HttpFetchTool` — fetch a URL via HTTP and return the body text.
* :class:`CodeExecutionTool` — execute Python code snippets in a sandbox.
* :class:`DelegateToAgentTool` — hand off to another registered agent.
"""

import asyncio
from typing import Any, Optional

from lauren_ai._tools import ToolContext, tool

__all__ = [
    "WebSearchTool",
    "HttpFetchTool",
    "CodeExecutionTool",
    "DelegateToAgentTool",
]


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


@tool()
async def WebSearchTool(  # noqa: N802
    query: str,
    max_results: int = 5,
    ctx: ToolContext | None = None,
) -> list:
    """Search the web for information about *query*.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1-20).

    Returns a list of results, each with ``title``, ``url``, and ``snippet``.
    """
    # Stub implementation — override with a real search API (e.g. SerpAPI,
    # Brave Search, Tavily) in production.
    return [
        {
            "title": f"Search result for: {query}",
            "url": "https://example.com/result-1",
            "snippet": (
                "This is a placeholder search result. "
                "Configure a real search backend to get actual results."
            ),
        }
    ]


# ---------------------------------------------------------------------------
# HttpFetchTool
# ---------------------------------------------------------------------------


@tool()
async def HttpFetchTool(  # noqa: N802
    url: str,
    method: str = "GET",
    headers: Optional[dict] = None,
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
    import subprocess  # noqa: PLC0415
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
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
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


# ---------------------------------------------------------------------------
# DelegateToAgentTool
# ---------------------------------------------------------------------------


@tool()
async def DelegateToAgentTool(  # noqa: N802
    agent_name: str,
    message: str,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    """Delegate a task to another registered agent.

    Args:
        agent_name: The class name of the target agent (e.g. 'ResearchAgent').
        message: The message to pass to the delegated agent.

    Returns the delegated agent's response as a dict with ``content`` and
    ``turns`` keys.
    """
    # The actual delegation is handled by AgentContext.delegate(); this tool
    # is a convenience wrapper for use when delegation is triggered via a
    # tool call from the LLM.
    if ctx is None:
        return {"error": "No agent context available for delegation.", "content": ""}

    agent_ctx = getattr(ctx, "agent_context", None)
    if agent_ctx is None:
        return {"error": "No agent context available for delegation.", "content": ""}

    return {
        "content": (
            f"Delegation to {agent_name!r} is not yet resolved. "
            "Implement DelegateToAgentTool with a real agent registry."
        ),
        "turns": 0,
    }

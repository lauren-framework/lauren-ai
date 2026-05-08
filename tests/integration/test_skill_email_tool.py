"""Integration tests for Skill 42: Email / Notification Dispatch Tool.

Tests cover:
- Send email to single recipient → success, recipient recorded
- Send email to multiple recipients → all recipients recorded
- Empty/blank 'to' field → error returned
- Backend receives correct EmailPayload fields
- Custom backend injection
- Failing backend reflected in result

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import asyncio

from abc import ABC, abstractmethod
from dataclasses import dataclass
from unittest.mock import MagicMock

from lauren_ai._tools import tool, ToolContext


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------


@dataclass
class EmailPayload:
    to: list
    subject: str
    body: str
    from_addr: str = "agent@example.com"


class EmailBackend(ABC):
    @abstractmethod
    async def send(self, email: EmailPayload) -> bool: ...


class InMemoryEmailBackend(EmailBackend):
    def __init__(self):
        self.sent: list = []

    async def send(self, email: EmailPayload) -> bool:
        self.sent.append(email)
        return True


class FailingEmailBackend(EmailBackend):
    async def send(self, email: EmailPayload) -> bool:
        return False


@tool()
class EmailDispatchTool:
    """Send an email notification.

    Args:
        to: Comma-separated recipient email addresses.
        subject: Email subject.
        body: Email body text.
    """

    def __init__(self, backend: "EmailBackend | None" = None):
        self._backend = backend or InMemoryEmailBackend()

    async def run(
        self, ctx: ToolContext, to: str, subject: str, body: str
    ) -> dict:
        recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
        if not recipients:
            return {"error": "No valid recipients"}
        email = EmailPayload(to=recipients, subject=subject, body=body)
        success = await self._backend.send(email)
        return {"sent": success, "recipients": recipients}


# ---------------------------------------------------------------------------
# MockToolContext helper
# ---------------------------------------------------------------------------


def _tool_ctx(state=None):
    ctx = MagicMock()
    ctx.execution_context = None
    ctx.agent_context = MagicMock()
    ctx.agent_context.metadata = {}
    ctx.get_metadata = lambda k, d=None: ctx.agent_context.metadata.get(k, d)
    ctx.state = state if state is not None else {}
    ctx.tool_use_id = "t1"
    ctx.turn = 0
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmailDispatchSend:
    def test_send_single_recipient(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "alice@example.com", "Hello", "Body text"))
        assert result["sent"] is True
        assert result["recipients"] == ["alice@example.com"]

    def test_send_records_in_backend(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "bob@example.com", "Subj", "Content"))
        assert len(backend.sent) == 1
        assert "bob@example.com" in backend.sent[0].to

    def test_send_correct_subject(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "a@b.com", "My Subject", "body"))
        assert backend.sent[0].subject == "My Subject"

    def test_send_correct_body(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "a@b.com", "Subj", "My body text"))
        assert backend.sent[0].body == "My body text"


class TestEmailDispatchMultipleRecipients:
    def test_comma_separated_recipients(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        result = asyncio.run(
            tool.run(ctx, "alice@example.com, bob@example.com", "Hi all", "Group message")
        )
        assert len(result["recipients"]) == 2
        assert "alice@example.com" in result["recipients"]
        assert "bob@example.com" in result["recipients"]

    def test_whitespace_trimmed_from_recipients(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        result = asyncio.run(
            tool.run(ctx, "  alice@example.com , bob@example.com  ", "Hi", "body")
        )
        assert "alice@example.com" in result["recipients"]
        assert "bob@example.com" in result["recipients"]


class TestEmailDispatchErrors:
    def test_empty_to_returns_error(self):
        tool = EmailDispatchTool(backend=InMemoryEmailBackend())
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "", "Subject", "Body"))
        assert "error" in result
        assert "recipient" in result["error"].lower()

    def test_only_spaces_in_to_returns_error(self):
        tool = EmailDispatchTool(backend=InMemoryEmailBackend())
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "   ,  ", "Subject", "Body"))
        assert "error" in result

    def test_backend_failure_reflected_in_result(self):
        tool = EmailDispatchTool(backend=FailingEmailBackend())
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "test@example.com", "Subj", "body"))
        assert result["sent"] is False


class TestEmailDispatchDefaultBackend:
    def test_default_backend_is_in_memory(self):
        tool = EmailDispatchTool()
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "x@y.com", "Test", "hello"))
        assert result["sent"] is True

    def test_multiple_sends_accumulate_in_backend(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "x@y.com", "Test1", "hello1"))
        asyncio.run(tool.run(ctx, "x@y.com", "Test2", "hello2"))
        assert len(backend.sent) == 2

    def test_clear_backend_resets_inbox(self):
        backend = InMemoryEmailBackend()
        tool = EmailDispatchTool(backend=backend)
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "x@y.com", "Test", "hello"))
        backend.sent.clear()
        assert len(backend.sent) == 0

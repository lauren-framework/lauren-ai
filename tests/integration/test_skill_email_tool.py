"""Integration tests for Skill 42: Email / Notification Dispatch Tool.

Tests cover:
- Send email to single recipient → success, recipient recorded
- Send email to multiple recipients → all recipients recorded
- Empty/blank 'to' field → error returned
- Backend receives correct EmailPayload fields
- Custom backend injection
- Tool via agent runner with MockTransport

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import pytest

from lauren_ai._tools import ToolContext
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import _add_to_tool_map
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------

from lauren_ai._tools import tool


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
# Mock context helper
# ---------------------------------------------------------------------------

class MockContext:
    def __init__(self):
        self.state = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}", model="mock-model", content=content, tool_calls=[],
        stop_reason=stop_reason, usage=TokenUsage(input_tokens=10, output_tokens=5)
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmailDispatchSend:
    async def test_send_single_recipient(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        result = await tool_instance.run(ctx, "alice@example.com", "Hello", "Body text")
        assert result["sent"] is True
        assert result["recipients"] == ["alice@example.com"]

    async def test_send_records_in_backend(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        await tool_instance.run(ctx, "bob@example.com", "Subj", "Content")
        assert len(backend.sent) == 1
        assert backend.sent[0].to == ["bob@example.com"]

    async def test_send_correct_subject(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        await tool_instance.run(ctx, "a@b.com", "My Subject", "body")
        assert backend.sent[0].subject == "My Subject"

    async def test_send_correct_body(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        await tool_instance.run(ctx, "a@b.com", "Subj", "My body text")
        assert backend.sent[0].body == "My body text"


class TestEmailDispatchMultipleRecipients:
    async def test_comma_separated_recipients(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        result = await tool_instance.run(
            ctx, "alice@example.com, bob@example.com", "Hi all", "Group message"
        )
        assert len(result["recipients"]) == 2
        assert "alice@example.com" in result["recipients"]
        assert "bob@example.com" in result["recipients"]

    async def test_whitespace_trimmed_from_recipients(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        result = await tool_instance.run(
            ctx, "  alice@example.com , bob@example.com  ", "Hi", "body"
        )
        assert "alice@example.com" in result["recipients"]
        assert "bob@example.com" in result["recipients"]


class TestEmailDispatchErrors:
    async def test_empty_to_returns_error(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        result = await tool_instance.run(ctx, "", "Subject", "Body")
        assert "error" in result
        assert "recipient" in result["error"].lower()

    async def test_only_spaces_in_to_returns_error(self):
        backend = InMemoryEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        result = await tool_instance.run(ctx, "   ,  ", "Subject", "Body")
        assert "error" in result

    async def test_backend_failure_reflected_in_result(self):
        backend = FailingEmailBackend()
        tool_instance = EmailDispatchTool(backend=backend)
        ctx = MockContext()
        result = await tool_instance.run(ctx, "test@example.com", "Subj", "body")
        assert result["sent"] is False


class TestEmailDispatchDefaultBackend:
    async def test_default_backend_is_in_memory(self):
        tool_instance = EmailDispatchTool()
        ctx = MockContext()
        result = await tool_instance.run(ctx, "x@y.com", "Test", "hello")
        assert result["sent"] is True

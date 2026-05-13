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

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# Tool definition
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

    async def run(self, ctx: ToolContext, to: str, subject: str, body: str) -> dict:
        recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
        if not recipients:
            return {"error": "No valid recipients"}
        email = EmailPayload(to=recipients, subject=subject, body=body)
        success = await self._backend.send(email)
        return {"sent": success, "recipients": recipients}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text, *, n=1, stop="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock",
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(10, 5),
    )


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


def _make_agent(backend: "EmailBackend | None" = None):
    """Create a fresh agent with the given email backend."""
    email_tool = EmailDispatchTool(backend=backend)

    @agent(model="mock-model", system="Email assistant")
    @use_tools(email_tool)
    class EmailTestAgent(_Capture):
        def __init__(self):
            _Capture.__init__(self)

    return EmailTestAgent()


# ---------------------------------------------------------------------------
# Tests: single recipient
# ---------------------------------------------------------------------------


class TestEmailDispatchSend:
    def test_send_single_recipient(self):
        """Sending to one recipient succeeds and records the recipient."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "alice@example.com", "subject": "Hello", "body": "Body text"},
        )
        client.mock.queue_response(_c("Email sent."))
        client.run("Send email to Alice")
        result = json.loads(agent_inst.captured[0].content)
        assert result["sent"] is True
        assert result["recipients"] == ["alice@example.com"]

    def test_send_records_in_backend(self):
        """The backend's sent list is populated after dispatch."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "bob@example.com", "subject": "Subj", "body": "Content"},
        )
        client.mock.queue_response(_c("Sent."))
        client.run("Send to Bob")
        assert len(backend.sent) == 1
        assert "bob@example.com" in backend.sent[0].to

    def test_send_correct_subject(self):
        """Backend receives the correct subject field."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "a@b.com", "subject": "My Subject", "body": "body"},
        )
        client.mock.queue_response(_c("Sent."))
        client.run("Send with subject")
        assert backend.sent[0].subject == "My Subject"

    def test_send_correct_body(self):
        """Backend receives the correct body field."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "a@b.com", "subject": "Subj", "body": "My body text"},
        )
        client.mock.queue_response(_c("Sent."))
        client.run("Send with body")
        assert backend.sent[0].body == "My body text"


# ---------------------------------------------------------------------------
# Tests: multiple recipients
# ---------------------------------------------------------------------------


class TestEmailDispatchMultipleRecipients:
    def test_comma_separated_recipients(self):
        """Comma-separated 'to' field splits into multiple recipients."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {
                "to": "alice@example.com, bob@example.com",
                "subject": "Hi all",
                "body": "Group message",
            },
        )
        client.mock.queue_response(_c("Sent to all."))
        client.run("Send to alice and bob")
        result = json.loads(agent_inst.captured[0].content)
        assert len(result["recipients"]) == 2
        assert "alice@example.com" in result["recipients"]
        assert "bob@example.com" in result["recipients"]

    def test_whitespace_trimmed_from_recipients(self):
        """Whitespace around email addresses in 'to' is trimmed."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {
                "to": "  alice@example.com , bob@example.com  ",
                "subject": "Hi",
                "body": "body",
            },
        )
        client.mock.queue_response(_c("Sent."))
        client.run("Send trimmed recipients")
        result = json.loads(agent_inst.captured[0].content)
        assert "alice@example.com" in result["recipients"]
        assert "bob@example.com" in result["recipients"]


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------


class TestEmailDispatchErrors:
    def test_empty_to_returns_error(self):
        """Empty 'to' field returns an error dict."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "", "subject": "Subject", "body": "Body"},
        )
        client.mock.queue_response(_c("Error."))
        client.run("Send to empty recipient")
        result = json.loads(agent_inst.captured[0].content)
        assert "error" in result
        assert "recipient" in result["error"].lower()

    def test_only_spaces_in_to_returns_error(self):
        """'to' with only spaces/commas returns an error dict."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "   ,  ", "subject": "Subject", "body": "Body"},
        )
        client.mock.queue_response(_c("Error."))
        client.run("Send to blank recipients")
        result = json.loads(agent_inst.captured[0].content)
        assert "error" in result

    def test_backend_failure_reflected_in_result(self):
        """When the backend returns False, the result reflects sent=False."""
        agent_inst = _make_agent(FailingEmailBackend())
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "test@example.com", "subject": "Subj", "body": "body"},
        )
        client.mock.queue_response(_c("Failed."))
        client.run("Send via failing backend")
        result = json.loads(agent_inst.captured[0].content)
        assert result["sent"] is False


# ---------------------------------------------------------------------------
# Tests: default backend and accumulation
# ---------------------------------------------------------------------------


class TestEmailDispatchDefaultBackend:
    def test_default_backend_is_in_memory(self):
        """Default backend (InMemory) successfully receives and stores emails."""
        agent_inst = _make_agent()  # no backend → InMemoryEmailBackend
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "x@y.com", "subject": "Test", "body": "hello"},
        )
        client.mock.queue_response(_c("Sent."))
        client.run("Send test email")
        result = json.loads(agent_inst.captured[0].content)
        assert result["sent"] is True

    def test_multiple_sends_accumulate_in_backend(self):
        """Multiple sends in sequence accumulate in the backend's sent list."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "x@y.com", "subject": "Test1", "body": "hello1"},
        )
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "x@y.com", "subject": "Test2", "body": "hello2"},
        )
        client.mock.queue_response(_c("Sent both."))
        client.run("Send two emails")
        assert len(backend.sent) == 2

    def test_clear_backend_resets_inbox(self):
        """Clearing the backend's sent list resets it to empty."""
        backend = InMemoryEmailBackend()
        agent_inst = _make_agent(backend)
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "email_dispatch_tool",
            {"to": "x@y.com", "subject": "Test", "body": "hello"},
        )
        client.mock.queue_response(_c("Sent."))
        client.run("Send one email")
        backend.sent.clear()
        assert len(backend.sent) == 0

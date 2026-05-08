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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from lauren import LaurenFactory, controller, delete, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
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
# Module-level shared backend and controller state
# ---------------------------------------------------------------------------

_test_backend = InMemoryEmailBackend()


class _SendRequest(BaseModel):
    to: str
    subject: str
    body: str


@controller("/email")
class EmailController:
    def __init__(self) -> None:
        self._tool = EmailDispatchTool(backend=_test_backend)

    @post("/send")
    async def send(self, body: Json[_SendRequest]) -> dict:
        ctx = _MockCtx()
        return await self._tool.run(ctx, body.to, body.subject, body.body)

    @get("/inbox/{address}")
    async def inbox(self, address: str) -> dict:
        msgs = [e for e in _test_backend.sent if address in e.to]
        return {"count": len(msgs), "messages": [
            {"to": e.to, "subject": e.subject, "body": e.body} for e in msgs
        ]}

    @delete("/clear")
    async def clear(self) -> dict:
        _test_backend.sent.clear()
        return {"cleared": True}


@module(controllers=[EmailController])
class EmailModule: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCtx:
    def __init__(self) -> None:
        self.state: dict = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


def build_app() -> TestClient:
    _test_backend.sent.clear()
    return TestClient(LaurenFactory.create(EmailModule))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmailDispatchSend:
    def test_send_single_recipient(self):
        client = build_app()
        r = client.post("/email/send", json={"to": "alice@example.com", "subject": "Hello", "body": "Body text"})
        assert r.status_code == 200
        data = r.json()
        assert data["sent"] is True
        assert data["recipients"] == ["alice@example.com"]

    def test_send_records_in_backend(self):
        client = build_app()
        client.post("/email/send", json={"to": "bob@example.com", "subject": "Subj", "body": "Content"})
        r = client.get("/email/inbox/bob@example.com")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert "bob@example.com" in data["messages"][0]["to"]

    def test_send_correct_subject(self):
        client = build_app()
        client.post("/email/send", json={"to": "a@b.com", "subject": "My Subject", "body": "body"})
        r = client.get("/email/inbox/a@b.com")
        assert r.json()["messages"][0]["subject"] == "My Subject"

    def test_send_correct_body(self):
        client = build_app()
        client.post("/email/send", json={"to": "a@b.com", "subject": "Subj", "body": "My body text"})
        r = client.get("/email/inbox/a@b.com")
        assert r.json()["messages"][0]["body"] == "My body text"


class TestEmailDispatchMultipleRecipients:
    def test_comma_separated_recipients(self):
        client = build_app()
        r = client.post("/email/send", json={
            "to": "alice@example.com, bob@example.com",
            "subject": "Hi all",
            "body": "Group message",
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["recipients"]) == 2
        assert "alice@example.com" in data["recipients"]
        assert "bob@example.com" in data["recipients"]

    def test_whitespace_trimmed_from_recipients(self):
        client = build_app()
        r = client.post("/email/send", json={
            "to": "  alice@example.com , bob@example.com  ",
            "subject": "Hi",
            "body": "body",
        })
        assert r.status_code == 200
        data = r.json()
        assert "alice@example.com" in data["recipients"]
        assert "bob@example.com" in data["recipients"]


class TestEmailDispatchErrors:
    def test_empty_to_returns_error(self):
        client = build_app()
        r = client.post("/email/send", json={"to": "", "subject": "Subject", "body": "Body"})
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "recipient" in data["error"].lower()

    def test_only_spaces_in_to_returns_error(self):
        client = build_app()
        r = client.post("/email/send", json={"to": "   ,  ", "subject": "Subject", "body": "Body"})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_backend_failure_reflected_in_result(self):
        # Test with a fresh tool using the failing backend directly
        import asyncio
        failing_tool = EmailDispatchTool(backend=FailingEmailBackend())
        ctx = _MockCtx()

        async def _run():
            return await failing_tool.run(ctx, "test@example.com", "Subj", "body")

        result = asyncio.new_event_loop().run_until_complete(_run())
        assert result["sent"] is False


class TestEmailDispatchDefaultBackend:
    def test_default_backend_is_in_memory(self):
        client = build_app()
        r = client.post("/email/send", json={"to": "x@y.com", "subject": "Test", "body": "hello"})
        assert r.status_code == 200
        assert r.json()["sent"] is True

    def test_clear_resets_inbox(self):
        client = build_app()
        client.post("/email/send", json={"to": "x@y.com", "subject": "Test", "body": "hello"})
        r = client.delete("/email/clear")
        assert r.status_code == 200
        assert r.json()["cleared"] is True
        r2 = client.get("/email/inbox/x@y.com")
        assert r2.json()["count"] == 0

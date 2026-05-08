---
name: email-notification-tool
description: Implements an agent tool for dispatching email notifications with a swappable backend. Use when building agents that send emails or notifications, with an in-memory backend for testing and a pluggable abstract backend for production SMTP or API-based senders.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Email / Notification Dispatch Tool

## Critical rule — no PEP 563 in tool files

**Never add `from __future__ import annotations` to any file that defines `@tool()`.**

---

## Overview

`EmailDispatchTool` is a class-form `@tool()` that wraps an `EmailBackend`
protocol.  `InMemoryEmailBackend` is used in tests; production code provides
an SMTP or transactional-email-API implementation.

---

## Implementation

```python
# tools/email_tool.py — NO from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from lauren_ai import tool, ToolContext

@dataclass
class EmailPayload:
    to: list[str]
    subject: str
    body: str
    from_addr: str = "agent@example.com"

class EmailBackend(ABC):
    @abstractmethod
    async def send(self, email: EmailPayload) -> bool: ...

class InMemoryEmailBackend(EmailBackend):
    def __init__(self):
        self.sent: list[EmailPayload] = []

    async def send(self, email: EmailPayload) -> bool:
        self.sent.append(email)
        return True

@tool()
class EmailDispatchTool:
    """Send an email notification.

    Args:
        to: Comma-separated recipient email addresses.
        subject: Email subject.
        body: Email body text.
    """

    def __init__(self, backend: EmailBackend | None = None):
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
```

---

## Injecting a custom backend

To use a real SMTP backend, subclass `EmailBackend` and pass an instance to
the `EmailDispatchTool` constructor or supply it through DI:

```python
class SMTPBackend(EmailBackend):
    async def send(self, email: EmailPayload) -> bool:
        # call smtplib or aiosmtplib here
        ...
        return True

tool_instance = EmailDispatchTool(backend=SMTPBackend(...))
```

---

## Attaching to an agent

```python
# agents.py — from __future__ import annotations is safe here
from __future__ import annotations
from lauren_ai import agent, use_tools
from .tools.email_tool import EmailDispatchTool

@agent(model="claude-opus-4-6", system="You are a notification assistant.")
@use_tools(EmailDispatchTool)
class NotificationAgent: ...
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `@tool()`, `ToolContext` |
| `src/lauren_ai/_tools/_executor.py` | `ToolExecutor` dispatch |

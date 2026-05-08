---
name: human-in-the-loop
description: Gate tool execution behind human approval using asyncio.Future. Use when deploying agents that perform irreversible actions (payments, deletions, emails) and need a human review step before proceeding.
---

> Use `codemap find "ApprovalGate"` after adding the pattern to your project.

# Human-in-the-Loop Approval Gate

Block a tool call until a human explicitly approves or rejects it. On timeout
the gate automatically declines.

## Pattern

```python
import asyncio
from uuid import uuid4

class ApprovalGate:
    """Blocks execution until a human approves or the timeout expires."""

    def __init__(self, timeout: float = 30.0):
        self._pending: dict[str, asyncio.Future] = {}
        self._timeout = timeout

    async def request(self, approval_id: str, details: dict) -> bool:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[approval_id] = fut
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(approval_id, None)
            return False

    def resolve(self, approval_id: str, approved: bool) -> bool:
        fut = self._pending.pop(approval_id, None)
        if fut and not fut.done():
            fut.set_result(approved)
            return True
        return False
```

## Tool integration

Inject the gate into a tool that requires confirmation:

```python
from uuid import uuid4
from lauren_ai._tools import tool, ToolContext

@tool()
class TransferFundsTool:
    """Transfer funds between accounts.

    Args:
        from_account: Source account ID.
        to_account: Destination account ID.
        amount: Amount in USD.
    """

    def __init__(self, gate: ApprovalGate):
        self._gate = gate

    async def run(
        self, ctx: ToolContext, from_account: str, to_account: str, amount: float
    ) -> dict:
        approval_id = str(uuid4())
        approved = await self._gate.request(
            approval_id,
            {"from": from_account, "to": to_account, "amount": amount},
        )
        if not approved:
            return {"error": "Transfer declined by operator"}
        # Perform the actual transfer
        return {"status": "transferred", "amount": amount}
```

## Approval workflow (banking chatbot pattern)

```
User ──► Agent ──► TransferFundsTool ──► ApprovalGate.request()
                                              │ (blocks, waiting)
Operator reviews ──────────────────────► gate.resolve(id, True/False)
                                              │
              ◄─── tool returns result ───────┘
```

## Testing

```python
import asyncio

async def test_approval_approved():
    gate = ApprovalGate(timeout=5.0)
    approval_id = "req-001"

    # Schedule approval after a short delay
    async def approve_after_delay():
        await asyncio.sleep(0.01)
        gate.resolve(approval_id, True)

    asyncio.create_task(approve_after_delay())
    result = await gate.request(approval_id, {"action": "delete"})
    assert result is True

async def test_approval_timeout():
    gate = ApprovalGate(timeout=0.05)
    result = await gate.request("req-timeout", {"action": "send"})
    assert result is False
```

## Notes

- `asyncio.shield` prevents the Future from being cancelled when
  `wait_for` raises `TimeoutError`, allowing safe cleanup.
- Each `approval_id` should be a UUID so concurrent requests don't collide.
- For multi-worker deployments, replace the in-process `Future` with a
  Redis pub/sub channel or a webhook endpoint.

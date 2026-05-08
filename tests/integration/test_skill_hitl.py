"""Integration tests for the Human-in-the-Loop approval gate pattern (Skill 34).

Tests cover:
- Approve request → returns True
- Decline request → returns False
- Timeout → returns False
- resolve() returns False when no pending request exists
- Concurrent approval/decline of different requests
"""

import asyncio
import pytest


# ---------------------------------------------------------------------------
# ApprovalGate implementation
# ---------------------------------------------------------------------------


class ApprovalGate:
    """Blocks execution until a human approves or the timeout expires."""

    def __init__(self, timeout: float = 30.0):
        self._pending: dict[str, asyncio.Future] = {}
        self._timeout = timeout

    async def request(self, approval_id: str, details: dict) -> bool:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
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


# ---------------------------------------------------------------------------
# Tests: basic approve/decline
# ---------------------------------------------------------------------------


class TestApprovalGateBasic:
    @pytest.mark.asyncio
    async def test_request_approved_returns_true(self):
        """Resolving with approved=True causes request() to return True."""
        gate = ApprovalGate(timeout=5.0)
        approval_id = "req-001"

        async def approve():
            await asyncio.sleep(0.01)
            gate.resolve(approval_id, True)

        asyncio.create_task(approve())
        result = await gate.request(approval_id, {"action": "delete"})

        assert result is True

    @pytest.mark.asyncio
    async def test_request_declined_returns_false(self):
        """Resolving with approved=False causes request() to return False."""
        gate = ApprovalGate(timeout=5.0)
        approval_id = "req-002"

        async def decline():
            await asyncio.sleep(0.01)
            gate.resolve(approval_id, False)

        asyncio.create_task(decline())
        result = await gate.request(approval_id, {"action": "send_email"})

        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_returns_true_when_pending(self):
        """resolve() returns True when it successfully resolves a pending request."""
        gate = ApprovalGate(timeout=5.0)
        approval_id = "req-003"

        async def do_request():
            await gate.request(approval_id, {})

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.01)

        resolved = gate.resolve(approval_id, True)
        assert resolved is True
        await task

    @pytest.mark.asyncio
    async def test_resolve_returns_false_when_no_pending(self):
        """resolve() returns False when the approval_id is not pending."""
        gate = ApprovalGate()
        resolved = gate.resolve("nonexistent-id", True)
        assert resolved is False


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


class TestApprovalGateTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        """When the timeout expires without resolution, request() returns False."""
        gate = ApprovalGate(timeout=0.05)
        result = await gate.request("req-timeout", {"action": "irreversible"})
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_after_timeout_is_noop(self):
        """Calling resolve() after the timeout has already expired is a no-op."""
        gate = ApprovalGate(timeout=0.05)
        approval_id = "req-late"

        await gate.request(approval_id, {})  # Times out immediately
        resolved = gate.resolve(approval_id, True)  # Already cleaned up

        assert resolved is False


# ---------------------------------------------------------------------------
# Tests: concurrent requests
# ---------------------------------------------------------------------------


class TestApprovalGateConcurrent:
    @pytest.mark.asyncio
    async def test_two_concurrent_approvals(self):
        """Two concurrent requests can be independently approved."""
        gate = ApprovalGate(timeout=5.0)

        results = {}

        async def req_a():
            results["a"] = await gate.request("req-a", {})

        async def req_b():
            results["b"] = await gate.request("req-b", {})

        task_a = asyncio.create_task(req_a())
        task_b = asyncio.create_task(req_b())
        await asyncio.sleep(0.01)

        gate.resolve("req-a", True)
        gate.resolve("req-b", False)

        await task_a
        await task_b

        assert results["a"] is True
        assert results["b"] is False

    @pytest.mark.asyncio
    async def test_resolving_one_does_not_affect_other(self):
        """Resolving one pending request does not affect another pending request."""
        gate = ApprovalGate(timeout=5.0)

        results = {}

        async def req_x():
            results["x"] = await gate.request("req-x", {})

        async def req_y():
            results["y"] = await gate.request("req-y", {})

        task_x = asyncio.create_task(req_x())
        task_y = asyncio.create_task(req_y())
        await asyncio.sleep(0.01)

        gate.resolve("req-x", True)
        gate.resolve("req-y", True)

        await task_x
        await task_y

        assert results["x"] is True
        assert results["y"] is True


# ---------------------------------------------------------------------------
# Tests: details dict
# ---------------------------------------------------------------------------


class TestApprovalGateDetails:
    @pytest.mark.asyncio
    async def test_details_dict_does_not_affect_approval(self):
        """The details dict is informational; approval result is unaffected."""
        gate = ApprovalGate(timeout=5.0)
        approval_id = "req-details"

        async def approve():
            await asyncio.sleep(0.01)
            gate.resolve(approval_id, True)

        asyncio.create_task(approve())
        result = await gate.request(
            approval_id, {"amount": 999.99, "currency": "USD", "recipient": "alice"}
        )

        assert result is True

"""Integration tests for the Human-in-the-Loop approval gate pattern (Skill 34).

Tests cover:
- Approve request -> returns True
- Decline request -> returns False
- Timeout -> returns False
- resolve() returns False when no pending request exists
- Concurrent approval/decline of different requests
"""

import asyncio
import concurrent.futures
import threading


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
# Thread-safe gate for cross-thread scenarios
# ---------------------------------------------------------------------------


class ThreadApprovalGate:
    """Approval gate backed by threading.Event — works across thread boundaries."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._events: dict[str, threading.Event] = {}
        self._results: dict[str, bool] = {}
        self._timeout = timeout
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def start(self, approval_id: str) -> concurrent.futures.Future:
        """Start a background thread that blocks until approval or timeout."""
        event = threading.Event()
        with self._lock:
            self._events[approval_id] = event

        def _wait() -> bool:
            signalled = event.wait(timeout=self._timeout)
            with self._lock:
                self._events.pop(approval_id, None)
                if signalled:
                    return self._results.pop(approval_id, False)
            return False

        return self._executor.submit(_wait)

    def resolve(self, approval_id: str, approved: bool) -> bool:
        with self._lock:
            event = self._events.get(approval_id)
            if event is None:
                return False
            self._results[approval_id] = approved
            event.set()
            return True

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._results.clear()


# ---------------------------------------------------------------------------
# Tests: basic approve/decline
# ---------------------------------------------------------------------------


class TestApprovalGateBasic:
    async def test_request_approved_returns_true(self):
        """Resolving with approved=True causes request() to return True."""
        gate = ApprovalGate(timeout=5.0)
        result_container: list[bool] = []

        async def _do_request():
            r = await gate.request("req-001", {})
            result_container.append(r)

        task = asyncio.create_task(_do_request())
        await asyncio.sleep(0)
        gate.resolve("req-001", True)
        await task
        assert result_container[0] is True

    async def test_request_declined_returns_false(self):
        """Resolving with approved=False causes request() to return False."""
        gate = ApprovalGate(timeout=5.0)
        result_container: list[bool] = []

        async def _do_request():
            r = await gate.request("req-002", {})
            result_container.append(r)

        task = asyncio.create_task(_do_request())
        await asyncio.sleep(0)
        gate.resolve("req-002", False)
        await task
        assert result_container[0] is False

    def test_resolve_returns_true_when_pending(self):
        """resolve() returns True when it successfully resolves a pending request."""
        gate = ThreadApprovalGate(timeout=5.0)
        fut = gate.start("req-003")
        resolved = gate.resolve("req-003", True)
        result = fut.result(timeout=2.0)
        assert resolved is True
        assert result is True

    async def test_resolve_returns_false_when_no_pending(self):
        """resolve() returns False when the approval_id is not pending."""
        gate = ApprovalGate(timeout=5.0)
        resolved = gate.resolve("nonexistent-id", True)
        assert resolved is False


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


class TestApprovalGateTimeout:
    async def test_timeout_returns_false(self):
        """When the timeout expires without resolution, request() returns False."""
        gate = ApprovalGate(timeout=0.05)
        result = await gate.request("req-timeout", {})
        assert result is False

    async def test_resolve_after_timeout_is_noop(self):
        """Calling resolve() after the timeout has already expired is a no-op."""
        gate = ApprovalGate(timeout=0.05)
        await gate.request("req-late", {})
        # gate is already cleaned up after timeout
        resolved = gate.resolve("req-late", True)
        assert resolved is False


# ---------------------------------------------------------------------------
# Tests: concurrent requests
# ---------------------------------------------------------------------------


class TestApprovalGateConcurrent:
    async def test_two_concurrent_approvals(self):
        """Two concurrent requests can be independently approved."""
        gate = ApprovalGate(timeout=5.0)
        results: dict[str, bool] = {}

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

    async def test_resolving_one_does_not_affect_other(self):
        """Resolving one pending request does not affect another pending request."""
        gate = ApprovalGate(timeout=5.0)
        results: dict[str, bool] = {}

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
    async def test_details_dict_does_not_affect_approval(self):
        """The details dict is informational; approval result is unaffected."""
        gate = ApprovalGate(timeout=5.0)
        result_container: list[bool] = []

        async def _do_request():
            r = await gate.request("req-details", {"action": "delete", "reason": "test"})
            result_container.append(r)

        task = asyncio.create_task(_do_request())
        await asyncio.sleep(0)
        gate.resolve("req-details", True)
        await task
        assert result_container[0] is True

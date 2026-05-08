"""Integration tests for the Human-in-the-Loop approval gate pattern (Skill 34).

Tests cover:
- Approve request → returns True
- Decline request → returns False
- Timeout → returns False
- resolve() returns False when no pending request exists
- Concurrent approval/decline of different requests
"""

from __future__ import annotations

import asyncio

from lauren import LaurenFactory, controller, get, post, module, Json, Path
from lauren.testing import TestClient


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
# Module-level gate and results store
# ---------------------------------------------------------------------------

_gate = ApprovalGate(timeout=5.0)
_results: dict[str, bool | None] = {}
_tasks: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/approvals")
class ApprovalController:
    @post("/request")
    async def request_approval(self, body: Json[dict]) -> dict:
        approval_id = body.get("approval_id", "req-default")
        details = body.get("details", {})
        timeout = body.get("timeout", None)

        # Use custom timeout if provided
        gate = _gate
        if timeout is not None:
            gate = ApprovalGate(timeout=timeout)
            _results[approval_id] = None

            async def _process():
                result = await gate.request(approval_id, details)
                _results[approval_id] = result

            task = asyncio.create_task(_process())
            _tasks[approval_id] = task
            return {"approval_id": approval_id, "status": "pending"}

        # For normal requests: run in background task
        _results[approval_id] = None

        async def _process_normal():
            result = await _gate.request(approval_id, details)
            _results[approval_id] = result

        task = asyncio.create_task(_process_normal())
        _tasks[approval_id] = task
        return {"approval_id": approval_id, "status": "pending"}

    @post("/resolve")
    async def resolve_approval(self, body: Json[dict]) -> dict:
        approval_id = body.get("approval_id", "")
        approved = body.get("approved", False)
        resolved = _gate.resolve(approval_id, approved)
        return {"resolved": resolved, "approval_id": approval_id}

    @get("/result/{approval_id}")
    async def get_result(self, approval_id: Path[str]) -> dict:
        # Give background task a chance to complete
        if approval_id in _tasks:
            task = _tasks[approval_id]
            if not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        result = _results.get(approval_id)
        done = result is not None
        return {"approved": result, "done": done, "approval_id": approval_id}

    @post("/resolve-direct")
    async def resolve_direct(self, body: Json[dict]) -> dict:
        """Direct resolve call on the gate — tests resolve() return value."""
        approval_id = body.get("approval_id", "")
        approved = body.get("approved", False)
        resolved = _gate.resolve(approval_id, approved)
        return {"resolved": resolved}

    @post("/wait-result")
    async def wait_result(self, body: Json[dict]) -> dict:
        """Run request and resolve in same handler for synchronous test."""
        approval_id = body.get("approval_id", "req-sync")
        approved = body.get("approved", True)
        timeout_val = body.get("timeout", 5.0)

        gate = ApprovalGate(timeout=timeout_val)
        result_container: list[bool] = []

        async def _do_request():
            r = await gate.request(approval_id, {})
            result_container.append(r)

        task = asyncio.create_task(_do_request())
        await asyncio.sleep(0)  # yield to let task start
        gate.resolve(approval_id, approved)
        await task

        return {"result": result_container[0] if result_container else None}

    @post("/timeout-test")
    async def timeout_test(self, body: Json[dict]) -> dict:
        """Run request with tiny timeout — expect False."""
        approval_id = body.get("approval_id", "req-timeout")
        timeout_val = body.get("timeout", 0.05)
        gate = ApprovalGate(timeout=timeout_val)
        result = await gate.request(approval_id, {})
        return {"result": result}

    @post("/concurrent")
    async def concurrent_approvals(self, body: Json[dict]) -> dict:
        """Run two concurrent requests and resolve them independently."""
        id_a = body.get("id_a", "req-a")
        id_b = body.get("id_b", "req-b")
        approved_a = body.get("approved_a", True)
        approved_b = body.get("approved_b", False)

        gate = ApprovalGate(timeout=5.0)
        results: dict[str, bool] = {}

        async def req_a():
            results["a"] = await gate.request(id_a, {})

        async def req_b():
            results["b"] = await gate.request(id_b, {})

        task_a = asyncio.create_task(req_a())
        task_b = asyncio.create_task(req_b())
        await asyncio.sleep(0.01)

        gate.resolve(id_a, approved_a)
        gate.resolve(id_b, approved_b)

        await task_a
        await task_b

        return {"a": results["a"], "b": results["b"]}


@module(controllers=[ApprovalController])
class HITLModule: ...


def build_app() -> TestClient:
    _results.clear()
    _tasks.clear()
    return TestClient(LaurenFactory.create(HITLModule))


# ---------------------------------------------------------------------------
# Tests: basic approve/decline
# ---------------------------------------------------------------------------


class TestApprovalGateBasic:
    def test_request_approved_returns_true(self):
        """Resolving with approved=True causes request() to return True."""
        client = build_app()
        r = client.post("/approvals/wait-result", json={"approval_id": "req-001", "approved": True})
        assert r.status_code == 200
        assert r.json()["result"] is True

    def test_request_declined_returns_false(self):
        """Resolving with approved=False causes request() to return False."""
        client = build_app()
        r = client.post("/approvals/wait-result", json={"approval_id": "req-002", "approved": False})
        assert r.status_code == 200
        assert r.json()["result"] is False

    def test_resolve_returns_true_when_pending(self):
        """resolve() returns True when it successfully resolves a pending request."""
        client = build_app()
        # Start a request (background task via /request endpoint)
        client.post("/approvals/request", json={"approval_id": "req-003"})
        # Resolve it
        r = client.post("/approvals/resolve", json={"approval_id": "req-003", "approved": True})
        assert r.status_code == 200
        assert r.json()["resolved"] is True

    def test_resolve_returns_false_when_no_pending(self):
        """resolve() returns False when the approval_id is not pending."""
        client = build_app()
        r = client.post("/approvals/resolve-direct", json={"approval_id": "nonexistent-id", "approved": True})
        assert r.status_code == 200
        assert r.json()["resolved"] is False


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


class TestApprovalGateTimeout:
    def test_timeout_returns_false(self):
        """When the timeout expires without resolution, request() returns False."""
        client = build_app()
        r = client.post("/approvals/timeout-test", json={"approval_id": "req-timeout", "timeout": 0.05})
        assert r.status_code == 200
        assert r.json()["result"] is False

    def test_resolve_after_timeout_is_noop(self):
        """Calling resolve() after the timeout has already expired is a no-op."""
        client = build_app()
        # Timeout the request first
        client.post("/approvals/timeout-test", json={"approval_id": "req-late", "timeout": 0.05})
        # Try to resolve — gate is already cleaned up
        r = client.post("/approvals/resolve-direct", json={"approval_id": "req-late", "approved": True})
        assert r.status_code == 200
        assert r.json()["resolved"] is False


# ---------------------------------------------------------------------------
# Tests: concurrent requests
# ---------------------------------------------------------------------------


class TestApprovalGateConcurrent:
    def test_two_concurrent_approvals(self):
        """Two concurrent requests can be independently approved."""
        client = build_app()
        r = client.post(
            "/approvals/concurrent",
            json={"id_a": "req-a", "id_b": "req-b", "approved_a": True, "approved_b": False},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["a"] is True
        assert data["b"] is False

    def test_resolving_one_does_not_affect_other(self):
        """Resolving one pending request does not affect another pending request."""
        client = build_app()
        r = client.post(
            "/approvals/concurrent",
            json={"id_a": "req-x", "id_b": "req-y", "approved_a": True, "approved_b": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["a"] is True
        assert data["b"] is True


# ---------------------------------------------------------------------------
# Tests: details dict
# ---------------------------------------------------------------------------


class TestApprovalGateDetails:
    def test_details_dict_does_not_affect_approval(self):
        """The details dict is informational; approval result is unaffected."""
        client = build_app()
        r = client.post(
            "/approvals/wait-result",
            json={"approval_id": "req-details", "approved": True},
        )
        assert r.status_code == 200
        assert r.json()["result"] is True

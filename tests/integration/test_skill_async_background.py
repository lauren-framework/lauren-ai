"""Integration tests for the async agent background worker pattern (Skill 35).

All tests go through the TestClient / HTTP layer.

Uses threading.Thread + queue.Queue instead of asyncio primitives so that
the background worker survives across separate TestClient request cycles
(each request may use a different event-loop iteration).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from uuid import uuid4

from lauren import LaurenFactory, controller, get, post, module, Json, Path
from lauren.testing import TestClient


# ---------------------------------------------------------------------------
# AgentTask and AgentTaskQueue implementation
# ---------------------------------------------------------------------------


@dataclass
class AgentTask:
    task_id: str
    prompt: str
    status: str = "pending"
    result: str | None = None
    error: str | None = None


class AgentTaskQueue:
    """Thread-safe task queue backed by stdlib threading primitives.

    Using threading (not asyncio) here means the worker and queue state
    survive across separate TestClient request cycles without event-loop
    binding issues.
    """

    _STOP = object()  # sentinel value to stop the worker thread

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._tasks: dict[str, AgentTask] = {}
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

    def submit(self, prompt: str) -> str:
        task = AgentTask(task_id=str(uuid4()), prompt=prompt)
        with self._lock:
            self._tasks[task.task_id] = task
        self._queue.put(task)
        return task.task_id

    def get_result(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def start_worker(self, handler_fn) -> None:
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            args=(handler_fn,),
            daemon=True,
        )
        self._worker_thread.start()

    def stop_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            self._queue.put(self._STOP)
            self._worker_thread.join(timeout=5.0)
        self._worker_thread = None

    def drain(self) -> None:
        self._queue.join()

    def reset(self) -> None:
        self.stop_worker()
        with self._lock:
            self._tasks.clear()
        # drain any leftover items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _run_worker(self, handler_fn) -> None:
        """Run async handlers inside a dedicated thread-local event loop."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            while True:
                task = self._queue.get()
                if task is self._STOP:
                    self._queue.task_done()
                    break
                task.status = "running"
                try:
                    task.result = loop.run_until_complete(handler_fn(task.prompt))
                    task.status = "done"
                except Exception as e:
                    task.error = str(e)
                    task.status = "failed"
                finally:
                    self._queue.task_done()
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def echo_handler(prompt: str) -> str:
    return f"result:{prompt}"


async def failing_handler(prompt: str) -> str:
    raise ValueError("handler failed")


async def slow_handler(prompt: str) -> str:
    import asyncio
    await asyncio.sleep(0.01)
    return f"slow:{prompt}"


async def sometimes_failing_handler(prompt: str) -> str:
    if prompt == "fail":
        raise RuntimeError("boom")
    return f"ok:{prompt}"


# ---------------------------------------------------------------------------
# Module-level queue + Controllers / Module
# ---------------------------------------------------------------------------

_queue = AgentTaskQueue()

_HANDLERS = {
    "echo": echo_handler,
    "failing": failing_handler,
    "slow": slow_handler,
    "sometimes_failing": sometimes_failing_handler,
}


@controller("/jobs")
class JobsController:
    @post("/submit")
    async def submit(self, body: Json[dict]) -> dict:
        task_id = _queue.submit(body.get("prompt", ""))
        return {"task_id": task_id}

    @post("/start-worker")
    async def start_worker(self, body: Json[dict]) -> dict:
        handler = _HANDLERS.get(body.get("handler", "echo"), echo_handler)
        _queue.start_worker(handler)
        return {"started": True}

    @post("/drain")
    async def drain(self, body: Json[dict]) -> dict:
        _queue.drain()
        return {"drained": True}

    @post("/stop-worker")
    async def stop_worker(self, body: Json[dict]) -> dict:
        _queue.stop_worker()
        return {"stopped": True}

    @get("/result/{task_id}")
    async def get_result(self, task_id: Path[str]) -> dict:
        task = _queue.get_result(task_id)
        if task is None:
            return {"status": "not_found"}
        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
        }

    @get("/task-status/{task_id}")
    async def task_status(self, task_id: Path[str]) -> dict:
        task = _queue.get_result(task_id)
        if task is None:
            return {"found": False}
        return {"found": True, "status": task.status}

    @post("/reset")
    async def reset(self, body: Json[dict]) -> dict:
        _queue.reset()
        return {"reset": True}


@module(controllers=[JobsController])
class AsyncBackgroundModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(AsyncBackgroundModule))


# ---------------------------------------------------------------------------
# Tests — all through TestClient
# ---------------------------------------------------------------------------


class TestAgentTaskQueueSubmit:
    def test_submit_returns_task_id_string(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.post("/jobs/submit", json={"prompt": "Hello"})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["task_id"], str)
        assert len(data["task_id"]) > 0

    def test_submitted_task_is_initially_pending(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.post("/jobs/submit", json={"prompt": "Test"})
        task_id = r.json()["task_id"]
        r2 = client.get(f"/jobs/task-status/{task_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["found"] is True
        assert data["status"] == "pending"

    def test_get_result_returns_not_found_for_unknown_id(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.get("/jobs/result/nonexistent-id")
        assert r.status_code == 200
        assert r.json()["status"] == "not_found"


class TestAgentTaskQueueWorker:
    def test_task_status_is_done_after_processing(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.post("/jobs/submit", json={"prompt": "work"})
        task_id = r.json()["task_id"]
        client.post("/jobs/start-worker", json={"handler": "echo"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        r2 = client.get(f"/jobs/result/{task_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "done"

    def test_task_result_is_set_after_processing(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.post("/jobs/submit", json={"prompt": "my-prompt"})
        task_id = r.json()["task_id"]
        client.post("/jobs/start-worker", json={"handler": "echo"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        r2 = client.get(f"/jobs/result/{task_id}")
        assert r2.status_code == 200
        assert r2.json()["result"] == "result:my-prompt"

    def test_prompt_is_forwarded_to_handler(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.post("/jobs/submit", json={"prompt": "specific prompt text"})
        task_id = r.json()["task_id"]
        client.post("/jobs/start-worker", json={"handler": "echo"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        r2 = client.get(f"/jobs/result/{task_id}")
        assert r2.status_code == 200
        assert r2.json()["result"] == "result:specific prompt text"


class TestAgentTaskQueueFailure:
    def test_failed_handler_sets_status_failed(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.post("/jobs/submit", json={"prompt": "doomed"})
        task_id = r.json()["task_id"]
        client.post("/jobs/start-worker", json={"handler": "failing"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        r2 = client.get(f"/jobs/result/{task_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "failed"

    def test_failed_handler_sets_error_message(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r = client.post("/jobs/submit", json={"prompt": "doomed"})
        task_id = r.json()["task_id"]
        client.post("/jobs/start-worker", json={"handler": "failing"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        r2 = client.get(f"/jobs/result/{task_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["error"] is not None
        assert "handler failed" in data["error"]

    def test_worker_continues_after_failure(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        r1 = client.post("/jobs/submit", json={"prompt": "fail"})
        r2 = client.post("/jobs/submit", json={"prompt": "ok-prompt"})
        id1 = r1.json()["task_id"]
        id2 = r2.json()["task_id"]
        client.post("/jobs/start-worker", json={"handler": "sometimes_failing"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        t1 = client.get(f"/jobs/result/{id1}").json()
        t2 = client.get(f"/jobs/result/{id2}").json()
        assert t1["status"] == "failed"
        assert t2["status"] == "done"
        assert t2["result"] == "ok:ok-prompt"


class TestAgentTaskQueueMultipleTasks:
    def test_multiple_tasks_all_processed(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        prompts = [f"prompt-{i}" for i in range(5)]
        task_ids = [
            client.post("/jobs/submit", json={"prompt": p}).json()["task_id"]
            for p in prompts
        ]
        client.post("/jobs/start-worker", json={"handler": "echo"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        for i, task_id in enumerate(task_ids):
            r = client.get(f"/jobs/result/{task_id}").json()
            assert r["status"] == "done"
            assert r["result"] == f"result:prompt-{i}"

    def test_slow_tasks_all_complete_after_drain(self):
        client = build_app()
        client.post("/jobs/reset", json={})
        ids = [
            client.post("/jobs/submit", json={"prompt": f"slow-{i}"}).json()["task_id"]
            for i in range(3)
        ]
        client.post("/jobs/start-worker", json={"handler": "slow"})
        client.post("/jobs/drain", json={})
        client.post("/jobs/stop-worker", json={})
        for task_id in ids:
            r = client.get(f"/jobs/result/{task_id}").json()
            assert r["status"] == "done"

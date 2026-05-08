"""Integration tests for the async agent background worker pattern (Skill 35).

All tests call AgentTaskQueue methods directly (no HTTP layer).
Uses threading.Thread + queue.Queue so the background worker can run
async handlers in a dedicated thread-local event loop.
"""

import queue
import threading
from dataclasses import dataclass
from uuid import uuid4


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
    """Thread-safe task queue backed by stdlib threading primitives."""

    _STOP = object()

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
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _run_worker(self, handler_fn) -> None:
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
# Tests
# ---------------------------------------------------------------------------


class TestAgentTaskQueueSubmit:
    def test_submit_returns_task_id_string(self):
        q = AgentTaskQueue()
        task_id = q.submit("Hello")
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        q.reset()

    def test_submitted_task_is_initially_pending(self):
        q = AgentTaskQueue()
        task_id = q.submit("Test")
        task = q.get_result(task_id)
        assert task is not None
        assert task.status == "pending"
        q.reset()

    def test_get_result_returns_none_for_unknown_id(self):
        q = AgentTaskQueue()
        task = q.get_result("nonexistent-id")
        assert task is None


class TestAgentTaskQueueWorker:
    def test_task_status_is_done_after_processing(self):
        q = AgentTaskQueue()
        task_id = q.submit("work")
        q.start_worker(echo_handler)
        q.drain()
        q.stop_worker()
        task = q.get_result(task_id)
        assert task is not None
        assert task.status == "done"

    def test_task_result_is_set_after_processing(self):
        q = AgentTaskQueue()
        task_id = q.submit("my-prompt")
        q.start_worker(echo_handler)
        q.drain()
        q.stop_worker()
        task = q.get_result(task_id)
        assert task is not None
        assert task.result == "result:my-prompt"

    def test_prompt_is_forwarded_to_handler(self):
        q = AgentTaskQueue()
        task_id = q.submit("specific prompt text")
        q.start_worker(echo_handler)
        q.drain()
        q.stop_worker()
        task = q.get_result(task_id)
        assert task is not None
        assert task.result == "result:specific prompt text"


class TestAgentTaskQueueFailure:
    def test_failed_handler_sets_status_failed(self):
        q = AgentTaskQueue()
        task_id = q.submit("doomed")
        q.start_worker(failing_handler)
        q.drain()
        q.stop_worker()
        task = q.get_result(task_id)
        assert task is not None
        assert task.status == "failed"

    def test_failed_handler_sets_error_message(self):
        q = AgentTaskQueue()
        task_id = q.submit("doomed")
        q.start_worker(failing_handler)
        q.drain()
        q.stop_worker()
        task = q.get_result(task_id)
        assert task is not None
        assert task.error is not None
        assert "handler failed" in task.error

    def test_worker_continues_after_failure(self):
        q = AgentTaskQueue()
        id1 = q.submit("fail")
        id2 = q.submit("ok-prompt")
        q.start_worker(sometimes_failing_handler)
        q.drain()
        q.stop_worker()
        t1 = q.get_result(id1)
        t2 = q.get_result(id2)
        assert t1 is not None
        assert t1.status == "failed"
        assert t2 is not None
        assert t2.status == "done"
        assert t2.result == "ok:ok-prompt"


class TestAgentTaskQueueMultipleTasks:
    def test_multiple_tasks_all_processed(self):
        q = AgentTaskQueue()
        prompts = [f"prompt-{i}" for i in range(5)]
        task_ids = [q.submit(p) for p in prompts]
        q.start_worker(echo_handler)
        q.drain()
        q.stop_worker()
        for i, task_id in enumerate(task_ids):
            task = q.get_result(task_id)
            assert task is not None
            assert task.status == "done"
            assert task.result == f"result:prompt-{i}"

    def test_slow_tasks_all_complete_after_drain(self):
        q = AgentTaskQueue()
        ids = [q.submit(f"slow-{i}") for i in range(3)]
        q.start_worker(slow_handler)
        q.drain()
        q.stop_worker()
        for task_id in ids:
            task = q.get_result(task_id)
            assert task is not None
            assert task.status == "done"

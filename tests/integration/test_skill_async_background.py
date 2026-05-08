"""Integration tests for the async agent background worker pattern (Skill 35).

Tests cover:
- Submit task → returns a task_id string
- Worker processes task, status transitions to "done"
- Worker sets result on the task
- Failed handler → status is "failed" and error is set
- drain() blocks until all tasks are processed
- Multiple tasks are all processed by a single worker
"""

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest


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
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: dict[str, AgentTask] = {}
        self._worker_task: asyncio.Task | None = None

    async def submit(self, prompt: str) -> str:
        task = AgentTask(task_id=str(uuid4()), prompt=prompt)
        self._tasks[task.task_id] = task
        await self._queue.put(task)
        return task.task_id

    async def get_result(self, task_id: str) -> AgentTask | None:
        return self._tasks.get(task_id)

    async def start_worker(self, handler) -> None:
        self._worker_task = asyncio.create_task(self._run_worker(handler))

    async def stop_worker(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def drain(self) -> None:
        await self._queue.join()

    async def _run_worker(self, handler) -> None:
        while True:
            task = await self._queue.get()
            task.status = "running"
            try:
                task.result = await handler(task.prompt)
                task.status = "done"
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Handlers used in tests
# ---------------------------------------------------------------------------


async def echo_handler(prompt: str) -> str:
    return f"result:{prompt}"


async def failing_handler(prompt: str) -> str:
    raise ValueError("handler failed")


async def slow_handler(prompt: str) -> str:
    await asyncio.sleep(0.01)
    return f"slow:{prompt}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentTaskQueueSubmit:
    @pytest.mark.asyncio
    async def test_submit_returns_task_id_string(self):
        """submit() returns a non-empty string task ID."""
        q = AgentTaskQueue()
        task_id = await q.submit("Hello")
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    @pytest.mark.asyncio
    async def test_submitted_task_is_initially_pending(self):
        """A submitted task starts with status='pending' before the worker runs."""
        q = AgentTaskQueue()
        task_id = await q.submit("Test")
        task = await q.get_result(task_id)
        assert task is not None
        assert task.status == "pending"

    @pytest.mark.asyncio
    async def test_get_result_returns_none_for_unknown_id(self):
        """get_result() returns None for an unknown task ID."""
        q = AgentTaskQueue()
        result = await q.get_result("nonexistent-id")
        assert result is None


class TestAgentTaskQueueWorker:
    @pytest.mark.asyncio
    async def test_task_status_is_done_after_processing(self):
        """After the worker processes a task, status is 'done'."""
        q = AgentTaskQueue()
        await q.start_worker(echo_handler)

        task_id = await q.submit("work")
        await q.drain()
        await q.stop_worker()

        task = await q.get_result(task_id)
        assert task.status == "done"

    @pytest.mark.asyncio
    async def test_task_result_is_set_after_processing(self):
        """The handler's return value is stored in task.result."""
        q = AgentTaskQueue()
        await q.start_worker(echo_handler)

        task_id = await q.submit("my-prompt")
        await q.drain()
        await q.stop_worker()

        task = await q.get_result(task_id)
        assert task.result == "result:my-prompt"

    @pytest.mark.asyncio
    async def test_prompt_is_forwarded_to_handler(self):
        """The exact prompt text is passed to the handler."""
        received = []

        async def capture_handler(prompt: str) -> str:
            received.append(prompt)
            return "ok"

        q = AgentTaskQueue()
        await q.start_worker(capture_handler)

        await q.submit("specific prompt text")
        await q.drain()
        await q.stop_worker()

        assert received == ["specific prompt text"]


class TestAgentTaskQueueFailure:
    @pytest.mark.asyncio
    async def test_failed_handler_sets_status_failed(self):
        """When the handler raises, status is set to 'failed'."""
        q = AgentTaskQueue()
        await q.start_worker(failing_handler)

        task_id = await q.submit("doomed")
        await q.drain()
        await q.stop_worker()

        task = await q.get_result(task_id)
        assert task.status == "failed"

    @pytest.mark.asyncio
    async def test_failed_handler_sets_error_message(self):
        """When the handler raises, the exception message is in task.error."""
        q = AgentTaskQueue()
        await q.start_worker(failing_handler)

        task_id = await q.submit("doomed")
        await q.drain()
        await q.stop_worker()

        task = await q.get_result(task_id)
        assert task.error is not None
        assert "handler failed" in task.error

    @pytest.mark.asyncio
    async def test_worker_continues_after_failure(self):
        """A failed task does not stop the worker from processing subsequent tasks."""
        call_count = 0

        async def sometimes_failing(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if prompt == "fail":
                raise RuntimeError("boom")
            return f"ok:{prompt}"

        q = AgentTaskQueue()
        await q.start_worker(sometimes_failing)

        id1 = await q.submit("fail")
        id2 = await q.submit("ok-prompt")
        await q.drain()
        await q.stop_worker()

        t1 = await q.get_result(id1)
        t2 = await q.get_result(id2)

        assert t1.status == "failed"
        assert t2.status == "done"
        assert t2.result == "ok:ok-prompt"


class TestAgentTaskQueueMultipleTasks:
    @pytest.mark.asyncio
    async def test_multiple_tasks_all_processed(self):
        """All submitted tasks are eventually processed by a single worker."""
        q = AgentTaskQueue()
        await q.start_worker(echo_handler)

        prompts = [f"prompt-{i}" for i in range(5)]
        task_ids = [await q.submit(p) for p in prompts]

        await q.drain()
        await q.stop_worker()

        for i, task_id in enumerate(task_ids):
            task = await q.get_result(task_id)
            assert task.status == "done"
            assert task.result == f"result:prompt-{i}"

    @pytest.mark.asyncio
    async def test_slow_tasks_all_complete_after_drain(self):
        """drain() waits for slow tasks to complete before returning."""
        q = AgentTaskQueue()
        await q.start_worker(slow_handler)

        ids = [await q.submit(f"slow-{i}") for i in range(3)]
        await q.drain()
        await q.stop_worker()

        for task_id in ids:
            task = await q.get_result(task_id)
            assert task.status == "done"

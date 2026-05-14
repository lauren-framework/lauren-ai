---
name: async-agent-background
description: Submit agent tasks to an asyncio queue and process them with a background worker, returning results by task ID. Use when you need non-blocking agent execution in web handlers, batch processing pipelines, or any situation where callers should not wait for agent completion.
---

> Use `codemap find "AgentTaskQueue"` after adding the pattern to your project.

# Async Agent Task Execution with Background Workers

Submit prompts as tasks, let a background worker run the agent, and poll
for results by task ID.

## Pattern

```python
import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

@dataclass
class AgentTask:
    task_id: str
    prompt: str
    status: str = "pending"   # pending | running | done | failed
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
        """Wait until all queued tasks have been processed."""
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
```

## Integration with AgentRunner

```python
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()
cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
runner = AgentRunner(transport=mock)

@agent(model="claude-sonnet-4-6", system="You are a helpful assistant.")
class BackgroundAgent: ...

agent_instance = BackgroundAgent()

async def run_agent(prompt: str) -> str:
    response = await runner.run(agent_instance, prompt)
    return response.content

queue = AgentTaskQueue()
await queue.start_worker(run_agent)

task_id = await queue.submit("Summarise this document...")
await queue.drain()
result = await queue.get_result(task_id)
print(result.status, result.result)
```

## Lauren web handler pattern

```python
@post("/tasks")
async def submit_task(req: Request) -> dict:
    body = await req.json()
    task_id = await queue.submit(body["prompt"])
    return {"task_id": task_id, "status": "pending"}

@get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    task = await queue.get_result(task_id)
    if task is None:
        raise HTTPError(404, detail="Task not found")
    return {"task_id": task.task_id, "status": task.status, "result": task.result}
```

## Notes

- `asyncio.Queue.join()` / `task_done()` ensure reliable drain-and-wait.
- For persistence across restarts, swap the in-memory queue for a Redis
  stream or a database-backed queue (e.g. pgqueue).
- Concurrency: start multiple workers with `asyncio.create_task` in a loop.

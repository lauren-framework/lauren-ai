"""TeamRunner — orchestrates multi-agent teams."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from lauren_ai._teams._decorator import TEAM_META, TeamConfigError
from lauren_ai._teams._events import (
    TeamCoordinatorDecision,
    TeamEvent,
    TeamFinalAnswer,
    TeamWorkerFinished,
    TeamWorkerStarted,
)
from lauren_ai._teams._memory import TeamMemory


@dataclass
class TeamResult:
    """Final result of a team run."""

    final_answer: str
    worker_outputs: dict[str, str] = field(default_factory=dict)
    rounds: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


_DEFAULT_COORDINATOR_PROMPT = """\
You are coordinating a team of specialist agents to complete a task.

Available workers:
{worker_descriptions}

Task: {task}

Prior outputs:
{prior_outputs}

Which worker should handle the next step, or is the task complete?
Respond with ONE of:
  ROUTE: <worker_name>
Or:
  DONE: <final answer here>
"""


class TeamRunner:
    """Orchestrates a @team() class through its workers.

    Usage::

        runner = TeamRunner(team_cls=ResearchTeam, llm=llm_service)
        result = await runner.run("Summarise recent news about AI.")
    """

    def __init__(
        self,
        team_cls: type,
        llm: Any,
        agent_runner: Any,
    ) -> None:
        meta = getattr(team_cls, TEAM_META, None)
        if meta is None:
            raise TeamConfigError(f"{team_cls.__name__} is not decorated with @team()")
        self._team_cls = team_cls
        self._meta = meta
        self._llm = llm
        self._agent_runner = agent_runner
        # Discover worker names from constructor annotations
        self._worker_names = self._discover_workers()

    def _discover_workers(self) -> list[str]:
        """Extract worker parameter names from the class constructor.

        :return: List of worker parameter names (excludes 'return').
        """
        init_method = getattr(self._team_cls, "__init__", None)
        hints = getattr(init_method, "__annotations__", None)
        if hints is None:
            return []
        return [k for k in hints if k != "return"]

    async def run(
        self,
        task: str,
        *,
        conversation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> TeamResult:
        """Run the team and return the aggregated result.

        :param task: The task string to complete.
        :param conversation_id: Optional conversation session identifier.
        :param context: Optional extra context passed to workers.
        :return: A :class:`TeamResult` with the final answer and worker outputs.
        """
        memory = TeamMemory()
        worker_outputs: dict[str, str] = {}

        if self._meta.mode == "coordinator":
            result = await self._run_coordinator(task, memory, worker_outputs)
        else:
            result = await self._run_collaborate(task, memory, worker_outputs)

        return result

    async def run_stream(
        self,
        task: str,
        *,
        conversation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[TeamEvent]:
        """Stream team execution events as an async generator.

        Yields :class:`TeamEvent` subclass instances as the team progresses.

        :param task: The task string to complete.
        :param conversation_id: Optional conversation session identifier.
        :param context: Optional extra context passed to workers.
        :return: An async iterator of :class:`TeamEvent` instances.
        """
        memory = TeamMemory()
        worker_outputs: dict[str, str] = {}

        if self._meta.mode == "coordinator":
            async for event in self._stream_coordinator(task, memory, worker_outputs):
                yield event
        else:
            async for event in self._stream_collaborate(task, memory, worker_outputs):
                yield event

    async def _call_worker(self, worker_name: str, task: str) -> str:
        """Run a single worker agent with the given task.

        :param worker_name: The name of the worker.
        :param task: The task to assign.
        :return: The worker's text output.
        """
        from lauren_ai._transport import Completion, Message  # noqa: PLC0415

        prompt = f"Worker: {worker_name}\nTask: {task}\n\nProvide your analysis:"
        result = await self._llm.complete([Message(role="user", content=prompt)])
        if isinstance(result, Completion):
            return result.content
        chunks = []
        async for chunk in result:
            if chunk.delta:
                chunks.append(chunk.delta)
        return "".join(chunks)

    async def _coordinator_decide(
        self, task: str, worker_outputs: dict[str, str], round_num: int
    ) -> tuple[str, str]:
        """Ask coordinator to route or declare done.

        :param task: The overall task.
        :param worker_outputs: Outputs from workers so far.
        :param round_num: Current round index.
        :return: A tuple of (action, content) where action is "ROUTE" or "DONE".
        """
        from lauren_ai._transport import Completion, Message  # noqa: PLC0415

        worker_desc = "\n".join(f"- {name}: A specialist agent" for name in self._worker_names)
        prior = (
            "\n".join(f"{name}: {output}" for name, output in worker_outputs.items()) or "None yet."
        )

        prompt_template = self._meta.coordinator_prompt or _DEFAULT_COORDINATOR_PROMPT
        prompt = prompt_template.format(
            worker_descriptions=worker_desc,
            task=task,
            prior_outputs=prior,
        )
        result = await self._llm.complete(
            [Message(role="user", content=prompt)],
            model=self._meta.model,
        )
        if isinstance(result, Completion):
            text = result.content.strip()
        else:
            chunks = []
            async for chunk in result:
                if chunk.delta:
                    chunks.append(chunk.delta)
            text = "".join(chunks).strip()

        if text.upper().startswith("DONE:"):
            return "DONE", text[5:].strip()
        if text.upper().startswith("DONE"):
            return "DONE", text[4:].strip()
        if "ROUTE:" in text.upper():
            parts = text.split(":", 1)
            if len(parts) == 2:
                return "ROUTE", parts[1].strip()
        return "DONE", text

    async def _run_coordinator(
        self, task: str, memory: TeamMemory, worker_outputs: dict[str, str]
    ) -> TeamResult:
        """Run coordinator mode: route to workers one at a time until DONE.

        :param task: The overall task.
        :param memory: Shared team memory.
        :param worker_outputs: Accumulator for worker outputs.
        :return: A :class:`TeamResult`.
        """
        rounds = 0
        for round_num in range(self._meta.max_rounds):
            rounds += 1
            action, content = await self._coordinator_decide(task, worker_outputs, round_num)
            if action == "DONE":
                return TeamResult(
                    final_answer=content,
                    worker_outputs=worker_outputs,
                    rounds=rounds,
                )
            # Find matching worker
            worker_name = content
            for wn in self._worker_names:
                if wn.lower() in content.lower():
                    worker_name = wn
                    break
            output = await self._call_worker(worker_name, task)
            worker_outputs[worker_name] = output
            await memory.set(worker_name, output)

        # Max rounds reached — synthesise from what we have
        final = "\n\n".join(f"{k}: {v}" for k, v in worker_outputs.items())
        return TeamResult(
            final_answer=final,
            worker_outputs=worker_outputs,
            rounds=rounds,
        )

    async def _run_collaborate(
        self, task: str, memory: TeamMemory, worker_outputs: dict[str, str]
    ) -> TeamResult:
        """Run collaborate mode: all workers in sequence, then synthesise.

        :param task: The overall task.
        :param memory: Shared team memory.
        :param worker_outputs: Accumulator for worker outputs.
        :return: A :class:`TeamResult`.
        """
        for worker_name in self._worker_names:
            output = await self._call_worker(worker_name, task)
            worker_outputs[worker_name] = output
            await memory.set(worker_name, output)

        # Synthesise
        synthesis_prompt = (
            f"Synthesise these expert outputs into a final answer for: {task}\n\n"
            + "\n\n".join(f"{k}:\n{v}" for k, v in worker_outputs.items())
        )
        from lauren_ai._transport import Completion, Message  # noqa: PLC0415

        result = await self._llm.complete(
            [Message(role="user", content=synthesis_prompt)],
            model=self._meta.model,
        )
        if isinstance(result, Completion):
            final = result.content
        else:
            chunks = []
            async for chunk in result:
                if chunk.delta:
                    chunks.append(chunk.delta)
            final = "".join(chunks)

        return TeamResult(
            final_answer=final,
            worker_outputs=worker_outputs,
            rounds=len(self._worker_names),
        )

    async def _stream_coordinator(
        self, task: str, memory: TeamMemory, worker_outputs: dict[str, str]
    ) -> AsyncIterator[TeamEvent]:
        """Async generator for coordinator-mode streaming.

        :param task: The overall task.
        :param memory: Shared team memory.
        :param worker_outputs: Accumulator for worker outputs.
        :return: Yields :class:`TeamEvent` instances.
        """
        for round_num in range(self._meta.max_rounds):
            action, content = await self._coordinator_decide(task, worker_outputs, round_num)
            yield TeamCoordinatorDecision(decision=f"{action}: {content}", round=round_num)
            if action == "DONE":
                yield TeamFinalAnswer(content=content, rounds=round_num + 1)
                return
            worker_name = content
            for wn in self._worker_names:
                if wn.lower() in content.lower():
                    worker_name = wn
                    break
            yield TeamWorkerStarted(worker_name=worker_name, task=task, round=round_num)
            output = await self._call_worker(worker_name, task)
            worker_outputs[worker_name] = output
            await memory.set(worker_name, output)
            yield TeamWorkerFinished(
                worker_name=worker_name, result_content=output, round=round_num
            )

        final = "\n\n".join(f"{k}: {v}" for k, v in worker_outputs.items())
        yield TeamFinalAnswer(content=final, rounds=self._meta.max_rounds)

    async def _stream_collaborate(
        self, task: str, memory: TeamMemory, worker_outputs: dict[str, str]
    ) -> AsyncIterator[TeamEvent]:
        """Async generator for collaborate-mode streaming.

        :param task: The overall task.
        :param memory: Shared team memory.
        :param worker_outputs: Accumulator for worker outputs.
        :return: Yields :class:`TeamEvent` instances.
        """
        for i, worker_name in enumerate(self._worker_names):
            yield TeamWorkerStarted(worker_name=worker_name, task=task, round=i)
            output = await self._call_worker(worker_name, task)
            worker_outputs[worker_name] = output
            await memory.set(worker_name, output)
            yield TeamWorkerFinished(worker_name=worker_name, result_content=output, round=i)

        synthesis_prompt = f"Synthesise: {task}\n\n" + "\n\n".join(
            f"{k}:\n{v}" for k, v in worker_outputs.items()
        )
        from lauren_ai._transport import Completion, Message  # noqa: PLC0415

        result = await self._llm.complete([Message(role="user", content=synthesis_prompt)])
        if isinstance(result, Completion):
            final = result.content
        else:
            chunks = []
            async for chunk in result:
                if chunk.delta:
                    chunks.append(chunk.delta)
            final = "".join(chunks)

        yield TeamFinalAnswer(content=final, rounds=len(self._worker_names))

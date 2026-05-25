"""Demonstrate a multi-agent workflow powered by ``AgentMessageBus``.

This example is intentionally dependency-light: it uses the messaging runtime
directly rather than requiring a live LLM provider.  That makes it useful for
both:

* end users who want to understand the messaging API quickly
* maintainers who want an executable reference for request/reply, topic
  publication, streaming responses, session scoping, and graceful shutdown

Run it from the repository root with:

    uv run python examples/inter_agent_messaging_workflow.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from lauren_ai import (
    AgentMessage,
    AgentMessageBus,
    AgentMessageRequestCompleted,
    AgentMessageSent,
    AgentMessageType,
    MessageRetryPolicy,
    SignalBus,
)

SESSION_ID = "example-session"
TASK_ID = "launch-brief"
RESEARCH_TOPIC = "research.findings"


def install_signal_logging(signals: SignalBus) -> None:
    """Print a compact trace of message activity."""

    @signals.on(AgentMessageSent)
    async def _on_sent(event: AgentMessageSent) -> None:
        target = event.to or event.topic or "*"
        print(
            f"[signal] sent {event.message_type.value:<12} "
            f"from={event.from_agent:<10} to={target:<18} "
            f"receivers={event.receiver_count} dropped={event.dropped_count}"
        )

    @signals.on(AgentMessageRequestCompleted)
    async def _on_request_completed(event: AgentMessageRequestCompleted) -> None:
        outcome = "timeout" if event.timed_out else "ok"
        print(
            f"[signal] request {outcome:<7} from={event.from_agent:<10} "
            f"target={event.target:<10} attempts={event.attempts} "
            f"elapsed_ms={event.elapsed_ms:.1f}"
        )


@dataclass(slots=True)
class BaseDemoAgent:
    """Base class for long-running agent workers."""

    name: str
    bus: AgentMessageBus
    session_id: str
    include_broadcast: bool = True
    topics: tuple[str, ...] = ()
    _subscription: Any = None

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Subscribe and register the background message loop."""
        self._subscription = await self.bus.subscribe(
            self.name,
            session_id=self.session_id,
            include_broadcast=self.include_broadcast,
            topics=self.topics,
        )
        task_group.create_task(self._run())

    async def stop(self) -> None:
        """Close the agent subscription."""
        if self._subscription is None:
            return
        await self._subscription.close()

    async def _run(self) -> None:
        assert self._subscription is not None
        try:
            while True:
                message = await self._subscription.get()
                await self.handle_message(message)
        except RuntimeError:
            return

    async def handle_message(self, message: AgentMessage) -> None:
        """Process an incoming message."""
        raise NotImplementedError


@dataclass(slots=True)
class ResearcherAgent(BaseDemoAgent):
    """Produces findings and an outline for the planner."""

    completed_requests: int = 0

    async def handle_message(self, message: AgentMessage) -> None:
        if message.message_type is AgentMessageType.NOTIFICATION:
            print(f"[researcher] workflow notification: {message.payload['status']}")
            return

        if message.message_type is not AgentMessageType.TASK_REQUEST:
            return

        topic = str(message.payload["topic"])
        findings = [
            "Users want launch updates grouped into a single digest.",
            "Operators care about rollout checkpoints and rollback windows.",
            "Executives need a brief with risks, mitigations, and clear owners.",
        ]
        outline = [
            "Audience and release goals",
            "Key milestones",
            "Risks and mitigations",
            "Owner checklist",
        ]

        self.completed_requests += 1
        print(f"[researcher] researching {topic!r} and publishing shared findings")

        await self.bus.publish(
            RESEARCH_TOPIC,
            from_agent=self.name,
            message_type=AgentMessageType.EVENT,
            payload={
                "topic": topic,
                "findings": findings,
            },
            session_id=self.session_id,
            task_id=message.task_id,
        )

        await self.bus.reply(
            message,
            from_agent=self.name,
            message_type=AgentMessageType.TASK_RESULT,
            payload={
                "topic": topic,
                "outline": outline,
                "finding_count": len(findings),
                "request_count": self.completed_requests,
            },
        )


@dataclass(slots=True)
class WriterAgent(BaseDemoAgent):
    """Consumes research events and streams a final draft."""

    findings: list[str] = field(default_factory=list)

    async def handle_message(self, message: AgentMessage) -> None:
        if message.message_type is AgentMessageType.NOTIFICATION:
            print(f"[writer] workflow notification: {message.payload['status']}")
            return

        if message.topic == RESEARCH_TOPIC:
            self.findings = list(message.payload["findings"])
            print(f"[writer] cached {len(self.findings)} findings from topic publication")
            return

        if message.message_type is not AgentMessageType.TASK_REQUEST:
            return

        topic = str(message.payload["topic"])
        outline = [str(item) for item in message.payload.get("outline", [])]
        print(f"[writer] streaming a draft for {topic!r}")

        sections = self._build_sections(topic=topic, outline=outline)
        for section in sections:
            await self.bus.stream_chunk(
                message,
                from_agent=self.name,
                payload={"chunk": section},
                stream_id="launch-brief-draft",
            )
            await asyncio.sleep(0.05)

        await self.bus.end_stream(
            message,
            from_agent=self.name,
            payload={
                "section_count": len(sections),
                "finding_count": len(self.findings),
            },
        )

    def _build_sections(self, *, topic: str, outline: list[str]) -> list[str]:
        findings_text = "; ".join(self.findings) if self.findings else "No findings were cached."
        return [
            f"# Launch Brief: {topic}",
            "## Outline\n" + "\n".join(f"- {item}" for item in outline),
            f"## Shared Findings\n{findings_text}",
            "## Next Step\nPlanner can now hand this draft to an LLM-backed agent, reviewer, or tool.",
        ]


async def run_planner_workflow(bus: AgentMessageBus) -> None:
    """Execute the planner side of the workflow."""
    planner_inbox = await bus.subscribe("planner", session_id=SESSION_ID)
    try:
        await bus.broadcast(
            from_agent="planner",
            message_type=AgentMessageType.NOTIFICATION,
            payload={
                "status": "kickoff",
                "task": TASK_ID,
            },
            session_id=SESSION_ID,
            task_id=TASK_ID,
        )

        research_request = AgentMessage(
            from_agent="planner",
            to="researcher",
            reply_to="planner",
            message_type=AgentMessageType.TASK_REQUEST,
            payload={"topic": "Q3 Launch Readiness"},
            session_id=SESSION_ID,
            task_id=TASK_ID,
            headers={"workflow": "launch-brief"},
            metadata={"step": "research"},
        )
        research_result = await bus.request(
            "researcher",
            research_request,
            timeout=1.0,
            retry_policy=MessageRetryPolicy(max_attempts=2, initial_backoff_s=0.05),
        )
        outline = [str(item) for item in research_result.payload["outline"]]
        print(f"[planner] got outline from researcher with {research_result.payload['finding_count']} findings")

        writer_request = AgentMessage(
            from_agent="planner",
            to="writer",
            reply_to="planner",
            message_type=AgentMessageType.TASK_REQUEST,
            payload={
                "topic": "Q3 Launch Readiness",
                "outline": outline,
            },
            session_id=SESSION_ID,
            task_id=TASK_ID,
            stream_id="launch-brief-draft",
            headers={"workflow": "launch-brief"},
            metadata={"step": "draft"},
        )

        streamed_sections: list[str] = []
        async for chunk in await bus.request_stream("writer", writer_request, timeout=2.0):
            section = str(chunk.payload["chunk"])
            streamed_sections.append(section)
            print(f"[planner] stream chunk received: {section.splitlines()[0]}")

        print("\n=== Final Draft Assembled By Planner ===")
        print("\n\n".join(streamed_sections))

        print("\n=== Planner Inbox Snapshot ===")
        for inbound in planner_inbox.drain():
            target = inbound.to or inbound.topic or "*"
            print(f"- {inbound.message_type.value} from {inbound.from_agent} via {target}")
    finally:
        await planner_inbox.close()


async def main() -> None:
    """Boot the bus, run the workflow, and shut everything down cleanly."""
    signals = SignalBus()
    install_signal_logging(signals)

    bus = AgentMessageBus(signals=signals)
    researcher = ResearcherAgent(name="researcher", bus=bus, session_id=SESSION_ID)
    writer = WriterAgent(
        name="writer",
        bus=bus,
        session_id=SESSION_ID,
        topics=(RESEARCH_TOPIC,),
    )

    async with asyncio.TaskGroup() as task_group:
        await researcher.start(task_group)
        await writer.start(task_group)
        await run_planner_workflow(bus)
        await researcher.stop()
        await writer.stop()

    await bus.shutdown()
    print(f"\nDead letters recorded: {len(bus.dead_letters())}")


if __name__ == "__main__":
    asyncio.run(main())

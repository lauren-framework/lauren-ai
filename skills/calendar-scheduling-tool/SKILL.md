---
name: calendar-scheduling-tool
description: Implements an in-memory calendar management tool for agents using @tool() class-form. Use when building agents that create, query, or cancel calendar events, with ISO 8601 time handling and date-based filtering. Supports @set_metadata for tool-type annotation.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Calendar / Scheduling Tool

## Critical rule — tool annotations must resolve

**`from __future__ import annotations` is supported, but every type used by `@tool()` must resolve when schema generation runs.**

---

## Overview

`CalendarTool` manages calendar events in memory using `create`, `query`, and
`cancel` actions. Events are stored as `CalendarEvent` dataclasses keyed by a
short UUID. Query filters by ISO date prefix (e.g. `"2026-01-15"`).

Use `@set_metadata("tool_type", "scheduling")` to annotate the tool's category
for agent routing and documentation.

---

## Implementation

```python
# tools/calendar_tool.py — future annotations are allowed; keep tool types importable
from dataclasses import dataclass, field
from uuid import uuid4
from lauren_ai import tool, ToolContext
from lauren_ai._tools import set_metadata

@dataclass
class CalendarEvent:
    event_id: str
    title: str
    start_time: str  # ISO 8601
    end_time: str
    attendees: list[str] = field(default_factory=list)
    description: str = ""

@set_metadata("tool_type", "scheduling")
@tool()
class CalendarTool:
    """Manage calendar events.

    Args:
        action: 'create', 'query', or 'cancel'.
        title: Event title (for create).
        start_time: ISO 8601 start time (e.g. '2026-01-15T10:00:00').
        end_time: ISO 8601 end time (for create).
        attendees: Comma-separated emails (for create).
        event_id: Event ID (for cancel).
        date: Date to query events for (YYYY-MM-DD, for query).
    """

    def __init__(self):
        self._events: dict[str, CalendarEvent] = {}

    async def run(
        self, ctx: ToolContext, action: str, title: str = "",
        start_time: str = "", end_time: str = "",
        attendees: str = "", event_id: str = "", date: str = "",
    ) -> dict:
        if action == "create":
            if not title or not start_time:
                return {"error": "title and start_time are required"}
            eid = str(uuid4())[:8]
            evt = CalendarEvent(
                event_id=eid, title=title,
                start_time=start_time, end_time=end_time or start_time,
                attendees=[a.strip() for a in attendees.split(",") if a.strip()],
            )
            self._events[eid] = evt
            return {"created": eid, "title": title, "start": start_time}
        elif action == "query":
            matching = [
                {"id": e.event_id, "title": e.title, "start": e.start_time}
                for e in self._events.values()
                if not date or e.start_time.startswith(date)
            ]
            return {"events": matching}
        elif action == "cancel":
            if event_id in self._events:
                del self._events[event_id]
                return {"cancelled": event_id}
            return {"error": f"Event {event_id!r} not found"}
        return {"error": f"Unknown action: {action}"}
```

---

## Event lifecycle

1. `create` — validates required fields, stores event, returns `event_id`.
2. `query` — returns all events, or filters by `date` prefix if supplied.
3. `cancel` — deletes by `event_id`; returns error if not found.

---

## AgentRunner test pattern

Create a fresh `CalendarTool` instance per test to ensure event storage is
isolated. Queue multiple tool calls in one run for multi-step sequences.

```python
import json
from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolResult
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


def _make_agent():
    cal_tool = CalendarTool()  # fresh instance per test

    @agent(model=None, system="Calendar agent")
    @use_tools(cal_tool)
    class CalendarTestAgent(_Capture):
        def __init__(self):
            _Capture.__init__(self)

    return CalendarTestAgent()


def _c(text):
    return Completion(id="c1", model="mock", content=text, tool_calls=[],
                      stop_reason="end_turn", usage=TokenUsage(10, 5))


def test_create_and_query():
    agent_inst = _make_agent()
    client = TestClient(agent_inst)
    client.mock.queue_tool_use(
        "calendar_tool",
        {"action": "create", "title": "Standup", "start_time": "2026-01-15T09:00:00"},
    )
    client.mock.queue_tool_use("calendar_tool", {"action": "query"})
    client.mock.queue_response(_c("Events retrieved."))
    client.run("Create standup and query all events")

    create_result = json.loads(agent_inst.captured[0].content)
    query_result = json.loads(agent_inst.captured[1].content)
    assert "created" in create_result
    assert len(query_result["events"]) == 1
```

---

## Attaching to an agent

```python
# agents.py — from __future__ import annotations is safe here
from __future__ import annotations
from lauren_ai import agent, use_tools
from .tools.calendar_tool import CalendarTool

cal_tool = CalendarTool()  # single instance with shared event store

@agent(model="claude-opus-4-6", system="You are a scheduling assistant.")
@use_tools(cal_tool)
class SchedulingAgent: ...
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `@tool()`, `ToolContext`, `set_metadata` |
| `src/lauren_ai/_tools/_executor.py` | `ToolExecutor` dispatch |

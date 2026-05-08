---
name: calendar-scheduling-tool
description: Implements an in-memory calendar management tool for agents using @tool() class-form. Use when building agents that create, query, or cancel calendar events, with ISO 8601 time handling and date-based filtering.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Calendar / Scheduling Tool

## Critical rule — no PEP 563 in tool files

**Never add `from __future__ import annotations` to any file that defines `@tool()`.**

---

## Overview

`CalendarTool` manages calendar events in memory using `create`, `query`, and
`cancel` actions.  Events are stored as `CalendarEvent` dataclasses keyed by
a short UUID.  Query filters by ISO date prefix (e.g. `"2026-01-15"`).

---

## Implementation

```python
# tools/calendar_tool.py — NO from __future__ import annotations
from dataclasses import dataclass, field
from uuid import uuid4
from lauren_ai import tool, ToolContext

@dataclass
class CalendarEvent:
    event_id: str
    title: str
    start_time: str  # ISO 8601
    end_time: str
    attendees: list[str] = field(default_factory=list)
    description: str = ""

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

## Attaching to an agent

```python
# agents.py — from __future__ import annotations is safe here
from __future__ import annotations
from lauren_ai import agent, use_tools
from .tools.calendar_tool import CalendarTool

@agent(model="claude-opus-4-6", system="You are a scheduling assistant.")
@use_tools(CalendarTool)
class SchedulingAgent: ...
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_tools/__init__.py` | `@tool()`, `ToolContext` |
| `src/lauren_ai/_tools/_executor.py` | `ToolExecutor` dispatch |

"""Integration tests for Skill 43: Calendar / Scheduling Tool.

Tests cover:
- Create event → returns event_id, title, start
- Create event without required fields → error
- Query all events → returns all
- Query by date prefix → filters correctly
- Cancel existing event → removed
- Cancel non-existent event → error
- Multiple events can be created and queried
- Cancel clears event from storage

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import asyncio

from dataclasses import dataclass, field
from unittest.mock import MagicMock
from uuid import uuid4

from lauren_ai._tools import tool, ToolContext


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------


@dataclass
class CalendarEvent:
    event_id: str
    title: str
    start_time: str
    end_time: str
    attendees: list = field(default_factory=list)
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
        self._events: dict = {}

    async def run(
        self,
        ctx: ToolContext,
        action: str,
        title: str = "",
        start_time: str = "",
        end_time: str = "",
        attendees: str = "",
        event_id: str = "",
        date: str = "",
    ) -> dict:
        if action == "create":
            if not title or not start_time:
                return {"error": "title and start_time are required"}
            eid = str(uuid4())[:8]
            evt = CalendarEvent(
                event_id=eid,
                title=title,
                start_time=start_time,
                end_time=end_time or start_time,
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


# ---------------------------------------------------------------------------
# MockToolContext helper
# ---------------------------------------------------------------------------


def _tool_ctx(state=None):
    ctx = MagicMock()
    ctx.execution_context = None
    ctx.agent_context = MagicMock()
    ctx.agent_context.metadata = {}
    ctx.get_metadata = lambda k, d=None: ctx.agent_context.metadata.get(k, d)
    ctx.state = state if state is not None else {}
    ctx.tool_use_id = "t1"
    ctx.turn = 0
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalendarCreate:
    def test_create_returns_event_id(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(
            tool.run(ctx, "create", title="Team Standup", start_time="2026-01-15T09:00:00")
        )
        assert "created" in result
        assert isinstance(result["created"], str)
        assert len(result["created"]) == 8

    def test_create_returns_title(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(
            tool.run(ctx, "create", title="My Meeting", start_time="2026-01-15T10:00:00")
        )
        assert result["title"] == "My Meeting"

    def test_create_returns_start_time(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(
            tool.run(ctx, "create", title="Event", start_time="2026-03-20T14:30:00")
        )
        assert result["start"] == "2026-03-20T14:30:00"

    def test_create_without_title_returns_error(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(
            tool.run(ctx, "create", title="", start_time="2026-01-15T10:00:00")
        )
        assert "error" in result

    def test_create_without_start_time_returns_error(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "create", title="No time", start_time=""))
        assert "error" in result

    def test_create_with_attendees(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(
            tool.run(
                ctx,
                "create",
                title="Workshop",
                start_time="2026-02-01T13:00:00",
                attendees="alice@example.com,bob@example.com",
            )
        )
        assert "created" in result


class TestCalendarQuery:
    def test_query_all_events(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "create", title="A", start_time="2026-01-10T09:00:00"))
        asyncio.run(tool.run(ctx, "create", title="B", start_time="2026-01-11T09:00:00"))
        result = asyncio.run(tool.run(ctx, "query"))
        assert len(result["events"]) == 2

    def test_query_by_date_filters(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "create", title="Jan Event", start_time="2026-01-15T09:00:00"))
        asyncio.run(tool.run(ctx, "create", title="Feb Event", start_time="2026-02-15T09:00:00"))
        result = asyncio.run(tool.run(ctx, "query", date="2026-01"))
        assert len(result["events"]) == 1
        assert result["events"][0]["title"] == "Jan Event"

    def test_query_empty_returns_empty_list(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "query"))
        assert result["events"] == []

    def test_query_no_match_date_returns_empty(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        asyncio.run(tool.run(ctx, "create", title="Event", start_time="2026-01-15T09:00:00"))
        result = asyncio.run(tool.run(ctx, "query", date="2027-06"))
        assert result["events"] == []


class TestCalendarCancel:
    def test_cancel_existing_event(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        create_result = asyncio.run(
            tool.run(ctx, "create", title="To Cancel", start_time="2026-01-01T08:00:00")
        )
        eid = create_result["created"]
        result = asyncio.run(tool.run(ctx, "cancel", event_id=eid))
        assert result["cancelled"] == eid

    def test_cancel_removes_from_storage(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        create_result = asyncio.run(
            tool.run(ctx, "create", title="Doomed", start_time="2026-01-01T08:00:00")
        )
        eid = create_result["created"]
        asyncio.run(tool.run(ctx, "cancel", event_id=eid))
        result = asyncio.run(tool.run(ctx, "query"))
        event_ids = [e["id"] for e in result["events"]]
        assert eid not in event_ids

    def test_cancel_nonexistent_returns_error(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        result = asyncio.run(tool.run(ctx, "cancel", event_id="nonexistent"))
        assert "error" in result

    def test_cancel_leaves_other_events(self):
        tool = CalendarTool()
        ctx = _tool_ctx()
        r1 = asyncio.run(tool.run(ctx, "create", title="Keep", start_time="2026-01-01T08:00:00"))
        r2 = asyncio.run(tool.run(ctx, "create", title="Delete", start_time="2026-01-02T08:00:00"))
        eid1 = r1["created"]
        eid2 = r2["created"]
        asyncio.run(tool.run(ctx, "cancel", event_id=eid2))
        result = asyncio.run(tool.run(ctx, "query"))
        event_ids = [e["id"] for e in result["events"]]
        assert eid1 in event_ids
        assert eid2 not in event_ids

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

import pytest

from lauren_ai._tools import ToolContext
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from dataclasses import dataclass, field
from uuid import uuid4


# ---------------------------------------------------------------------------
# Tool definition (module level — no future annotations)
# ---------------------------------------------------------------------------

from lauren_ai._tools import tool


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
# Mock context helper
# ---------------------------------------------------------------------------

class MockContext:
    def __init__(self):
        self.state = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalendarCreate:
    async def test_create_returns_event_id(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(ctx, "create", title="Team Standup", start_time="2026-01-15T09:00:00")
        assert "created" in result
        assert isinstance(result["created"], str)
        assert len(result["created"]) == 8

    async def test_create_returns_title(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(ctx, "create", title="My Meeting", start_time="2026-01-15T10:00:00")
        assert result["title"] == "My Meeting"

    async def test_create_returns_start_time(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(ctx, "create", title="Event", start_time="2026-03-20T14:30:00")
        assert result["start"] == "2026-03-20T14:30:00"

    async def test_create_without_title_returns_error(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(ctx, "create", start_time="2026-01-15T10:00:00")
        assert "error" in result

    async def test_create_without_start_time_returns_error(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(ctx, "create", title="No time")
        assert "error" in result

    async def test_create_with_attendees(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(
            ctx, "create",
            title="Workshop", start_time="2026-02-01T13:00:00",
            attendees="alice@example.com,bob@example.com"
        )
        assert "created" in result
        eid = result["created"]
        event = cal._events[eid]
        assert "alice@example.com" in event.attendees


class TestCalendarQuery:
    async def test_query_all_events(self):
        cal = CalendarTool()
        ctx = MockContext()
        await cal.run(ctx, "create", title="A", start_time="2026-01-10T09:00:00")
        await cal.run(ctx, "create", title="B", start_time="2026-01-11T09:00:00")
        result = await cal.run(ctx, "query")
        assert len(result["events"]) == 2

    async def test_query_by_date_filters(self):
        cal = CalendarTool()
        ctx = MockContext()
        await cal.run(ctx, "create", title="Jan Event", start_time="2026-01-15T09:00:00")
        await cal.run(ctx, "create", title="Feb Event", start_time="2026-02-15T09:00:00")
        result = await cal.run(ctx, "query", date="2026-01")
        assert len(result["events"]) == 1
        assert result["events"][0]["title"] == "Jan Event"

    async def test_query_empty_returns_empty_list(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(ctx, "query")
        assert result["events"] == []

    async def test_query_no_match_date_returns_empty(self):
        cal = CalendarTool()
        ctx = MockContext()
        await cal.run(ctx, "create", title="Event", start_time="2026-01-15T09:00:00")
        result = await cal.run(ctx, "query", date="2027-06")
        assert result["events"] == []


class TestCalendarCancel:
    async def test_cancel_existing_event(self):
        cal = CalendarTool()
        ctx = MockContext()
        create_result = await cal.run(ctx, "create", title="To Cancel", start_time="2026-01-01T08:00:00")
        eid = create_result["created"]
        cancel_result = await cal.run(ctx, "cancel", event_id=eid)
        assert cancel_result["cancelled"] == eid

    async def test_cancel_removes_from_storage(self):
        cal = CalendarTool()
        ctx = MockContext()
        create_result = await cal.run(ctx, "create", title="Doomed", start_time="2026-01-01T08:00:00")
        eid = create_result["created"]
        await cal.run(ctx, "cancel", event_id=eid)
        assert eid not in cal._events

    async def test_cancel_nonexistent_returns_error(self):
        cal = CalendarTool()
        ctx = MockContext()
        result = await cal.run(ctx, "cancel", event_id="nonexistent")
        assert "error" in result

    async def test_cancel_leaves_other_events(self):
        cal = CalendarTool()
        ctx = MockContext()
        r1 = await cal.run(ctx, "create", title="Keep", start_time="2026-01-01T08:00:00")
        r2 = await cal.run(ctx, "create", title="Delete", start_time="2026-01-02T08:00:00")
        await cal.run(ctx, "cancel", event_id=r2["created"])
        assert r1["created"] in cal._events
        assert r2["created"] not in cal._events

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

import json

from dataclasses import dataclass, field
from uuid import uuid4

from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, set_metadata, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@dataclass
class CalendarEvent:
    event_id: str
    title: str
    start_time: str
    end_time: str
    attendees: list = field(default_factory=list)
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
# Helpers
# ---------------------------------------------------------------------------


def _c(text, *, n=1, stop="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock",
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(10, 5),
    )


class _Capture:
    def __init__(self):
        self.captured: list[ToolResult] = []

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        self.captured.append(result)
        return None


def _make_agent():
    """Create a fresh CalendarTool instance and a fresh agent for isolation."""
    cal_tool = CalendarTool()

    @agent(model=None, system="Calendar agent")
    @use_tools(cal_tool)
    class CalendarTestAgent(_Capture):
        def __init__(self):
            _Capture.__init__(self)

    return CalendarTestAgent()


# ---------------------------------------------------------------------------
# Tests: create
# ---------------------------------------------------------------------------


class TestCalendarCreate:
    def test_create_returns_event_id(self):
        """create returns an 8-char event_id."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Team Standup", "start_time": "2026-01-15T09:00:00"},
        )
        client.mock.queue_response(_c("Event created."))
        client.run("Create standup event")
        result = json.loads(agent_inst.captured[0].content)
        assert "created" in result
        assert isinstance(result["created"], str)
        assert len(result["created"]) == 8

    def test_create_returns_title(self):
        """create returns the event title."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "My Meeting", "start_time": "2026-01-15T10:00:00"},
        )
        client.mock.queue_response(_c("Meeting created."))
        client.run("Create my meeting")
        result = json.loads(agent_inst.captured[0].content)
        assert result["title"] == "My Meeting"

    def test_create_returns_start_time(self):
        """create returns the correct start time."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Event", "start_time": "2026-03-20T14:30:00"},
        )
        client.mock.queue_response(_c("Event created."))
        client.run("Create event")
        result = json.loads(agent_inst.captured[0].content)
        assert result["start"] == "2026-03-20T14:30:00"

    def test_create_without_title_returns_error(self):
        """create without a title returns an error dict."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "", "start_time": "2026-01-15T10:00:00"},
        )
        client.mock.queue_response(_c("Error."))
        client.run("Create event without title")
        result = json.loads(agent_inst.captured[0].content)
        assert "error" in result

    def test_create_without_start_time_returns_error(self):
        """create without a start_time returns an error dict."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "No time", "start_time": ""},
        )
        client.mock.queue_response(_c("Error."))
        client.run("Create event without time")
        result = json.loads(agent_inst.captured[0].content)
        assert "error" in result

    def test_create_with_attendees(self):
        """create with attendees succeeds."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {
                "action": "create",
                "title": "Workshop",
                "start_time": "2026-02-01T13:00:00",
                "attendees": "alice@example.com,bob@example.com",
            },
        )
        client.mock.queue_response(_c("Workshop created."))
        client.run("Create workshop with attendees")
        result = json.loads(agent_inst.captured[0].content)
        assert "created" in result


# ---------------------------------------------------------------------------
# Tests: query
# ---------------------------------------------------------------------------


class TestCalendarQuery:
    def test_query_all_events(self):
        """query without date returns all events."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "A", "start_time": "2026-01-10T09:00:00"},
        )
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "B", "start_time": "2026-01-11T09:00:00"},
        )
        client.mock.queue_tool_use("calendar_tool", {"action": "query"})
        client.mock.queue_response(_c("Here are your events."))
        client.run("Create two events then query")
        query_result = json.loads(agent_inst.captured[2].content)
        assert len(query_result["events"]) == 2

    def test_query_by_date_filters(self):
        """query with date prefix returns only matching events."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Jan Event", "start_time": "2026-01-15T09:00:00"},
        )
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Feb Event", "start_time": "2026-02-15T09:00:00"},
        )
        client.mock.queue_tool_use("calendar_tool", {"action": "query", "date": "2026-01"})
        client.mock.queue_response(_c("January events."))
        client.run("Create Jan and Feb events, query January")
        query_result = json.loads(agent_inst.captured[2].content)
        assert len(query_result["events"]) == 1
        assert query_result["events"][0]["title"] == "Jan Event"

    def test_query_empty_returns_empty_list(self):
        """query on an empty calendar returns empty events list."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("calendar_tool", {"action": "query"})
        client.mock.queue_response(_c("No events."))
        client.run("Query empty calendar")
        result = json.loads(agent_inst.captured[0].content)
        assert result["events"] == []

    def test_query_no_match_date_returns_empty(self):
        """query with a date that has no events returns empty list."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Event", "start_time": "2026-01-15T09:00:00"},
        )
        client.mock.queue_tool_use("calendar_tool", {"action": "query", "date": "2027-06"})
        client.mock.queue_response(_c("No matching events."))
        client.run("Create Jan event, query June 2027")
        query_result = json.loads(agent_inst.captured[1].content)
        assert query_result["events"] == []


# ---------------------------------------------------------------------------
# Tests: cancel
# ---------------------------------------------------------------------------


class TestCalendarCancel:
    def test_cancel_existing_event(self):
        """cancel returns the event_id that was cancelled."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "To Cancel", "start_time": "2026-01-01T08:00:00"},
        )
        client.mock.queue_response(_c("Processing."))
        client.run("Create event to cancel")
        eid = json.loads(agent_inst.captured[0].content)["created"]

        # Second run: cancel the event
        agent_inst2 = _make_agent()
        # Copy the event into the new tool instance
        cal_tool = (
            agent_inst2.__class__.__lauren_ai_tools__[0]
            if hasattr(agent_inst2.__class__, "__lauren_ai_tools__")
            else None
        )

        # Use the same agent instance for cancel (same CalendarTool instance)
        client2 = TestClient(agent_inst)
        client2.mock.queue_tool_use("calendar_tool", {"action": "cancel", "event_id": eid})
        client2.mock.queue_response(_c("Cancelled."))
        client2.run("Cancel event")
        cancel_result = json.loads(agent_inst.captured[1].content)
        assert cancel_result["cancelled"] == eid

    def test_cancel_removes_from_storage(self):
        """After cancel, query no longer returns the event."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        # Create, cancel, then query — all in one agent run sequence
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Doomed", "start_time": "2026-01-01T08:00:00"},
        )
        client.mock.queue_response(_c("Created."))
        client.run("Create doomed event")
        eid = json.loads(agent_inst.captured[0].content)["created"]

        # Cancel it
        client.mock.queue_tool_use("calendar_tool", {"action": "cancel", "event_id": eid})
        client.mock.queue_response(_c("Cancelled."))
        client.run("Cancel doomed event")

        # Query — should be empty
        client.mock.queue_tool_use("calendar_tool", {"action": "query"})
        client.mock.queue_response(_c("No events."))
        client.run("Query events")
        query_result = json.loads(agent_inst.captured[2].content)
        event_ids = [e["id"] for e in query_result["events"]]
        assert eid not in event_ids

    def test_cancel_nonexistent_returns_error(self):
        """Cancelling a non-existent event returns an error."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use("calendar_tool", {"action": "cancel", "event_id": "nonexistent"})
        client.mock.queue_response(_c("Not found."))
        client.run("Cancel non-existent event")
        result = json.loads(agent_inst.captured[0].content)
        assert "error" in result

    def test_cancel_leaves_other_events(self):
        """Cancelling one event does not remove others."""
        agent_inst = _make_agent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Keep", "start_time": "2026-01-01T08:00:00"},
        )
        client.mock.queue_tool_use(
            "calendar_tool",
            {"action": "create", "title": "Delete", "start_time": "2026-01-02T08:00:00"},
        )
        client.mock.queue_response(_c("Two events created."))
        client.run("Create two events")
        eid1 = json.loads(agent_inst.captured[0].content)["created"]
        eid2 = json.loads(agent_inst.captured[1].content)["created"]

        # Cancel eid2, then query
        client.mock.queue_tool_use("calendar_tool", {"action": "cancel", "event_id": eid2})
        client.mock.queue_tool_use("calendar_tool", {"action": "query"})
        client.mock.queue_response(_c("Done."))
        client.run("Cancel second event and query")
        query_result = json.loads(agent_inst.captured[3].content)
        event_ids = [e["id"] for e in query_result["events"]]
        assert eid1 in event_ids
        assert eid2 not in event_ids


# ---------------------------------------------------------------------------
# Tests: @set_metadata
# ---------------------------------------------------------------------------


class TestCalendarToolMetadata:
    def test_calendar_tool_type_metadata(self):
        """@set_metadata('tool_type', 'scheduling') is stored on CalendarTool."""
        from lauren_ai._tools import TOOL_METADATA

        meta = getattr(CalendarTool, TOOL_METADATA, {})
        assert meta.get("tool_type") == "scheduling"

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

from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel

from lauren import LaurenFactory, controller, delete, get, post, module, Json, Query
from lauren.testing import TestClient
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
# Module-level mutable state to hold the current tool
# ---------------------------------------------------------------------------

_cal_state: dict = {}


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class _CreateRequest(BaseModel):
    title: str
    start_time: str
    attendees: str = ""


@controller("/calendar")
class CalendarController:
    @post("/create")
    async def create(self, body: Json[_CreateRequest]) -> dict:
        ctx = _MockCtx()
        return await _cal_state["tool"].run(
            ctx, "create", title=body.title, start_time=body.start_time, attendees=body.attendees
        )

    @get("/query")
    async def query(self, date: Query[str] = "") -> dict:
        ctx = _MockCtx()
        return await _cal_state["tool"].run(ctx, "query", date=date)

    @delete("/cancel/{event_id}")
    async def cancel(self, event_id: str) -> dict:
        ctx = _MockCtx()
        return await _cal_state["tool"].run(ctx, "cancel", event_id=event_id)


@module(controllers=[CalendarController])
class CalendarModule: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCtx:
    def __init__(self) -> None:
        self.state: dict = {}
        self.execution_context = None
        self.agent_context = None
        self.tool_use_id = "t1"
        self.turn = 0
        self.request = None

    def get_metadata(self, key, default=None):
        return default


def build_app() -> TestClient:
    _cal_state["tool"] = CalendarTool()
    return TestClient(LaurenFactory.create(CalendarModule))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalendarCreate:
    def test_create_returns_event_id(self):
        client = build_app()
        r = client.post("/calendar/create", json={"title": "Team Standup", "start_time": "2026-01-15T09:00:00"})
        assert r.status_code == 200
        data = r.json()
        assert "created" in data
        assert isinstance(data["created"], str)
        assert len(data["created"]) == 8

    def test_create_returns_title(self):
        client = build_app()
        r = client.post("/calendar/create", json={"title": "My Meeting", "start_time": "2026-01-15T10:00:00"})
        assert r.status_code == 200
        assert r.json()["title"] == "My Meeting"

    def test_create_returns_start_time(self):
        client = build_app()
        r = client.post("/calendar/create", json={"title": "Event", "start_time": "2026-03-20T14:30:00"})
        assert r.status_code == 200
        assert r.json()["start"] == "2026-03-20T14:30:00"

    def test_create_without_title_returns_error(self):
        client = build_app()
        r = client.post("/calendar/create", json={"title": "", "start_time": "2026-01-15T10:00:00"})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_create_without_start_time_returns_error(self):
        client = build_app()
        r = client.post("/calendar/create", json={"title": "No time", "start_time": ""})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_create_with_attendees(self):
        client = build_app()
        r = client.post("/calendar/create", json={
            "title": "Workshop",
            "start_time": "2026-02-01T13:00:00",
            "attendees": "alice@example.com,bob@example.com",
        })
        assert r.status_code == 200
        assert "created" in r.json()


class TestCalendarQuery:
    def test_query_all_events(self):
        client = build_app()
        client.post("/calendar/create", json={"title": "A", "start_time": "2026-01-10T09:00:00"})
        client.post("/calendar/create", json={"title": "B", "start_time": "2026-01-11T09:00:00"})
        r = client.get("/calendar/query")
        assert r.status_code == 200
        assert len(r.json()["events"]) == 2

    def test_query_by_date_filters(self):
        client = build_app()
        client.post("/calendar/create", json={"title": "Jan Event", "start_time": "2026-01-15T09:00:00"})
        client.post("/calendar/create", json={"title": "Feb Event", "start_time": "2026-02-15T09:00:00"})
        r = client.get("/calendar/query?date=2026-01")
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) == 1
        assert events[0]["title"] == "Jan Event"

    def test_query_empty_returns_empty_list(self):
        client = build_app()
        r = client.get("/calendar/query")
        assert r.status_code == 200
        assert r.json()["events"] == []

    def test_query_no_match_date_returns_empty(self):
        client = build_app()
        client.post("/calendar/create", json={"title": "Event", "start_time": "2026-01-15T09:00:00"})
        r = client.get("/calendar/query?date=2027-06")
        assert r.status_code == 200
        assert r.json()["events"] == []


class TestCalendarCancel:
    def test_cancel_existing_event(self):
        client = build_app()
        create_r = client.post("/calendar/create", json={"title": "To Cancel", "start_time": "2026-01-01T08:00:00"})
        eid = create_r.json()["created"]
        r = client.delete(f"/calendar/cancel/{eid}")
        assert r.status_code == 200
        assert r.json()["cancelled"] == eid

    def test_cancel_removes_from_storage(self):
        client = build_app()
        create_r = client.post("/calendar/create", json={"title": "Doomed", "start_time": "2026-01-01T08:00:00"})
        eid = create_r.json()["created"]
        client.delete(f"/calendar/cancel/{eid}")
        r = client.get("/calendar/query")
        event_ids = [e["id"] for e in r.json()["events"]]
        assert eid not in event_ids

    def test_cancel_nonexistent_returns_error(self):
        client = build_app()
        r = client.delete("/calendar/cancel/nonexistent")
        assert r.status_code == 200
        assert "error" in r.json()

    def test_cancel_leaves_other_events(self):
        client = build_app()
        r1 = client.post("/calendar/create", json={"title": "Keep", "start_time": "2026-01-01T08:00:00"})
        r2 = client.post("/calendar/create", json={"title": "Delete", "start_time": "2026-01-02T08:00:00"})
        eid1 = r1.json()["created"]
        eid2 = r2.json()["created"]
        client.delete(f"/calendar/cancel/{eid2}")
        r = client.get("/calendar/query")
        event_ids = [e["id"] for e in r.json()["events"]]
        assert eid1 in event_ids
        assert eid2 not in event_ids

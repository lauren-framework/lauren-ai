"""Integration tests for the database query tool pattern (Skill 37).

Tests cover:
- SELECT on read-only tool returns rows
- INSERT on read-only tool is blocked
- UPDATE on read-only tool is blocked
- DROP on read-only tool is blocked
- SELECT with read_only=False (write-enabled) still works
- INSERT with write-enabled tool modifies data
- Malformed SQL returns an error dict (does not raise)
- Row count is correct
- AgentRunner pattern: tool dispatched via real runner, result captured

NOTE: No `from __future__ import annotations` — @tool() needs live annotations.
"""

import json
import re
import sqlite3

from lauren_ai._agents import AgentContext, agent, use_tools
from lauren_ai._tools import ToolContext, ToolResult, set_metadata, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import TestClient

# ---------------------------------------------------------------------------
# SQLQueryTool implementation
# ---------------------------------------------------------------------------


WRITE_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b",
    re.IGNORECASE,
)


@set_metadata("mode", "read")
@tool(name="sql_query_tool")
class SQLQueryTool:
    """Execute a SQL query against the database.

    Args:
        query: The SQL query to execute.
        read_only: If true, only SELECT queries are allowed.
    """

    def __init__(self, connection_string: str = ":memory:", read_only: bool = True):
        self._conn = sqlite3.connect(connection_string)
        self._read_only = read_only
        self._setup()

    def _setup(self):
        self._conn.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, value REAL)")
        self._conn.execute("INSERT OR IGNORE INTO items VALUES (1, 'Widget', 9.99)")
        self._conn.execute("INSERT OR IGNORE INTO items VALUES (2, 'Gadget', 29.99)")
        self._conn.commit()

    async def run(self, ctx: ToolContext, query: str, read_only: bool = True) -> dict:
        effective_read_only = self._read_only or read_only
        if effective_read_only and WRITE_PATTERNS.search(query):
            return {"error": "Write operations are not permitted in read-only mode"}
        try:
            cursor = self._conn.execute(query)
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, row, strict=False)) for row in cursor.fetchall()]
                return {"rows": rows, "count": len(rows)}
            self._conn.commit()
            return {"affected": cursor.rowcount}
        except Exception as e:
            return {"error": str(e)}


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


# ---------------------------------------------------------------------------
# Shared DB tool instances (each test that needs write uses its own instance)
# ---------------------------------------------------------------------------

# Read-only tool shared for read tests (no writes, so reuse is safe)
_ro_db = SQLQueryTool(read_only=True)


@agent(model="mock-model", system="DB read agent")
@use_tools(_ro_db)
class DBReadAgent(_Capture):
    def __init__(self):
        _Capture.__init__(self)


# ---------------------------------------------------------------------------
# Tests: SELECT queries (read-only) — via AgentRunner
# ---------------------------------------------------------------------------


class TestSQLQueryToolSelect:
    def test_select_all_returns_rows(self):
        """SELECT * returns all seeded rows — via real runner."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "SELECT * FROM items ORDER BY id", "read_only": True},
        )
        client.mock.queue_response(_c("Rows retrieved."))
        client.run("Select all items")
        data = json.loads(agent_inst.captured[0].content)
        assert "rows" in data
        assert data["count"] == 2

    def test_select_returns_correct_columns(self):
        """Column names match the schema."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "SELECT * FROM items WHERE id = 1", "read_only": True},
        )
        client.mock.queue_response(_c("Got row."))
        client.run("Get item 1")
        row = json.loads(agent_inst.captured[0].content)["rows"][0]
        assert row["id"] == 1
        assert row["name"] == "Widget"
        assert abs(row["value"] - 9.99) < 0.001

    def test_select_with_where_clause(self):
        """WHERE clause filters rows correctly."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "SELECT * FROM items WHERE name = 'Gadget'", "read_only": True},
        )
        client.mock.queue_response(_c("Filtered."))
        client.run("Get Gadget")
        data = json.loads(agent_inst.captured[0].content)
        assert data["count"] == 1
        assert data["rows"][0]["id"] == 2

    def test_select_row_count_correct(self):
        """count field reflects the actual number of rows returned."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "SELECT * FROM items", "read_only": True},
        )
        client.mock.queue_response(_c("Done."))
        client.run("Select items")
        data = json.loads(agent_inst.captured[0].content)
        assert data["count"] == len(data["rows"])


# ---------------------------------------------------------------------------
# Tests: write operations blocked in read-only mode — via AgentRunner
# ---------------------------------------------------------------------------


class TestSQLQueryToolWriteBlocked:
    def test_insert_blocked_in_read_only_mode(self):
        """INSERT is blocked when the tool is read-only."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "INSERT INTO items VALUES (3, 'Thingamajig', 4.99)", "read_only": True},
        )
        client.mock.queue_response(_c("Blocked."))
        client.run("Insert item")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data
        assert "not permitted" in data["error"].lower()

    def test_update_blocked_in_read_only_mode(self):
        """UPDATE is blocked when the tool is read-only."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "UPDATE items SET name='X' WHERE id=1", "read_only": True},
        )
        client.mock.queue_response(_c("Blocked."))
        client.run("Update item")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data

    def test_drop_blocked_in_read_only_mode(self):
        """DROP is blocked when the tool is read-only."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "DROP TABLE items", "read_only": True},
        )
        client.mock.queue_response(_c("Blocked."))
        client.run("Drop table")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data

    def test_delete_blocked_in_read_only_mode(self):
        """DELETE is blocked when the tool is read-only."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "DELETE FROM items WHERE id=1", "read_only": True},
        )
        client.mock.queue_response(_c("Blocked."))
        client.run("Delete item")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data

    def test_per_call_read_only_flag_also_blocks_writes(self):
        """Passing read_only=True per-call blocks writes even on a write-enabled tool."""
        # Use a write-enabled DB instance with the per-call flag set to True
        write_db = SQLQueryTool(read_only=False)

        @agent(model="mock-model", system="DB write agent")
        @use_tools(write_db)
        class DBWriteAgentBlocked(_Capture):
            def __init__(self):
                _Capture.__init__(self)

        agent_inst = DBWriteAgentBlocked()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "INSERT INTO items VALUES (5, 'X', 0.01)", "read_only": True},
        )
        client.mock.queue_response(_c("Blocked."))
        client.run("Insert with per-call read-only")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: write operations allowed when not read-only — via AgentRunner
# ---------------------------------------------------------------------------


class TestSQLQueryToolWriteEnabled:
    def test_insert_succeeds_in_write_mode(self):
        """INSERT succeeds when the tool is write-enabled."""
        write_db = SQLQueryTool(read_only=False)

        @agent(model="mock-model", system="DB write agent")
        @use_tools(write_db)
        class DBWriteAgent(_Capture):
            def __init__(self):
                _Capture.__init__(self)

        agent_inst = DBWriteAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "INSERT INTO items VALUES (3, 'New Item', 1.99)", "read_only": False},
        )
        client.mock.queue_response(_c("Inserted."))
        client.run("Insert new item")
        data = json.loads(agent_inst.captured[0].content)
        assert "affected" in data
        assert data["affected"] == 1

    def test_inserted_row_is_queryable(self):
        """A row inserted via the write tool is visible in subsequent SELECTs."""
        write_db = SQLQueryTool(read_only=False)

        @agent(model="mock-model", system="DB write agent")
        @use_tools(write_db)
        class DBWriteAgent2(_Capture):
            def __init__(self):
                _Capture.__init__(self)

        agent_inst = DBWriteAgent2()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "INSERT INTO items VALUES (4, 'Inserted', 5.00)", "read_only": False},
        )
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "SELECT * FROM items WHERE id = 4", "read_only": False},
        )
        client.mock.queue_response(_c("Found inserted row."))
        client.run("Insert and query")
        # Second capture is the SELECT result
        select_data = json.loads(agent_inst.captured[1].content)
        assert select_data["count"] == 1
        assert select_data["rows"][0]["name"] == "Inserted"


# ---------------------------------------------------------------------------
# Tests: error handling — via AgentRunner
# ---------------------------------------------------------------------------


class TestSQLQueryToolErrors:
    def test_malformed_sql_returns_error_dict(self):
        """Malformed SQL returns an error dict instead of raising."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "NOT VALID SQL !!!", "read_only": True},
        )
        client.mock.queue_response(_c("Error."))
        client.run("Run bad SQL")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data

    def test_nonexistent_table_returns_error(self):
        """Querying a nonexistent table returns an error dict."""
        agent_inst = DBReadAgent()
        client = TestClient(agent_inst)
        client.mock.queue_tool_use(
            "sql_query_tool",
            {"query": "SELECT * FROM nonexistent_table", "read_only": True},
        )
        client.mock.queue_response(_c("Error."))
        client.run("Query missing table")
        data = json.loads(agent_inst.captured[0].content)
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: @set_metadata propagation
# ---------------------------------------------------------------------------


class TestSQLToolMetadata:
    def test_sql_tool_mode_metadata(self):
        """@set_metadata('mode', 'read') is stored on the tool class."""
        from lauren_ai._tools import TOOL_METADATA

        meta = getattr(SQLQueryTool, TOOL_METADATA, {})
        assert meta.get("mode") == "read"

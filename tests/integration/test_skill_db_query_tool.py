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
"""

from __future__ import annotations

import re
import sqlite3

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient


# ---------------------------------------------------------------------------
# SQLQueryTool implementation
# ---------------------------------------------------------------------------


WRITE_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b",
    re.IGNORECASE,
)


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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, value REAL)"
        )
        self._conn.execute("INSERT OR IGNORE INTO items VALUES (1, 'Widget', 9.99)")
        self._conn.execute("INSERT OR IGNORE INTO items VALUES (2, 'Gadget', 29.99)")
        self._conn.commit()

    async def run(self, ctx, query: str, read_only: bool = True) -> dict:
        effective_read_only = self._read_only or read_only
        if effective_read_only and WRITE_PATTERNS.search(query):
            return {"error": "Write operations are not permitted in read-only mode"}
        try:
            cursor = self._conn.execute(query)
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                return {"rows": rows, "count": len(rows)}
            self._conn.commit()
            return {"affected": cursor.rowcount}
        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Stub context
# ---------------------------------------------------------------------------


class _Ctx:
    def get_metadata(self, key, default=None):
        return default


_CTX = _Ctx()


# ---------------------------------------------------------------------------
# Controllers / Module
# ---------------------------------------------------------------------------


@controller("/db")
class DBController:
    def __init__(self) -> None:
        # Two separate DB instances: one read-only, one write-enabled
        self._ro_db = SQLQueryTool(read_only=True)
        self._rw_db = SQLQueryTool(read_only=False)

    @post("/query")
    async def query(self, body: Json[dict]) -> dict:
        q = body.get("query", "")
        read_only = body.get("read_only", True)
        if read_only:
            return await self._ro_db.run(_CTX, q, read_only=True)
        else:
            return await self._rw_db.run(_CTX, q, read_only=False)

    @post("/query-per-call-flag")
    async def query_per_call_flag(self, body: Json[dict]) -> dict:
        """Test per-call read_only flag on a write-enabled tool."""
        q = body.get("query", "")
        read_only_flag = body.get("read_only", True)
        return await self._rw_db.run(_CTX, q, read_only=read_only_flag)


@module(controllers=[DBController])
class DBModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(DBModule))


# ---------------------------------------------------------------------------
# Tests: SELECT queries (read-only)
# ---------------------------------------------------------------------------


class TestSQLQueryToolSelect:
    def test_select_all_returns_rows(self):
        """SELECT * returns all seeded rows."""
        client = build_app()
        r = client.post("/db/query", json={"query": "SELECT * FROM items ORDER BY id", "read_only": True})
        assert r.status_code == 200
        data = r.json()
        assert "rows" in data
        assert data["count"] == 2

    def test_select_returns_correct_columns(self):
        """Column names match the schema."""
        client = build_app()
        r = client.post("/db/query", json={"query": "SELECT * FROM items WHERE id = 1", "read_only": True})
        assert r.status_code == 200
        row = r.json()["rows"][0]
        assert row["id"] == 1
        assert row["name"] == "Widget"
        assert abs(row["value"] - 9.99) < 0.001

    def test_select_with_where_clause(self):
        """WHERE clause filters rows correctly."""
        client = build_app()
        r = client.post("/db/query", json={"query": "SELECT * FROM items WHERE name = 'Gadget'", "read_only": True})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["rows"][0]["id"] == 2

    def test_select_row_count_correct(self):
        """count field reflects the actual number of rows returned."""
        client = build_app()
        r = client.post("/db/query", json={"query": "SELECT * FROM items", "read_only": True})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == len(data["rows"])


# ---------------------------------------------------------------------------
# Tests: write operations blocked in read-only mode
# ---------------------------------------------------------------------------


class TestSQLQueryToolWriteBlocked:
    def test_insert_blocked_in_read_only_mode(self):
        """INSERT is blocked when the tool is read-only."""
        client = build_app()
        r = client.post("/db/query", json={
            "query": "INSERT INTO items VALUES (3, 'Thingamajig', 4.99)",
            "read_only": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
        assert "not permitted" in data["error"].lower()

    def test_update_blocked_in_read_only_mode(self):
        """UPDATE is blocked when the tool is read-only."""
        client = build_app()
        r = client.post("/db/query", json={
            "query": "UPDATE items SET name='X' WHERE id=1",
            "read_only": True,
        })
        assert r.status_code == 200
        assert "error" in r.json()

    def test_drop_blocked_in_read_only_mode(self):
        """DROP is blocked when the tool is read-only."""
        client = build_app()
        r = client.post("/db/query", json={"query": "DROP TABLE items", "read_only": True})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_delete_blocked_in_read_only_mode(self):
        """DELETE is blocked when the tool is read-only."""
        client = build_app()
        r = client.post("/db/query", json={"query": "DELETE FROM items WHERE id=1", "read_only": True})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_per_call_read_only_flag_also_blocks_writes(self):
        """Passing read_only=True per-call blocks writes even on a write-enabled tool."""
        client = build_app()
        r = client.post("/db/query-per-call-flag", json={
            "query": "INSERT INTO items VALUES (5, 'X', 0.01)",
            "read_only": True,
        })
        assert r.status_code == 200
        assert "error" in r.json()


# ---------------------------------------------------------------------------
# Tests: write operations allowed when not read-only
# ---------------------------------------------------------------------------


class TestSQLQueryToolWriteEnabled:
    def test_insert_succeeds_in_write_mode(self):
        """INSERT succeeds when the tool is write-enabled."""
        client = build_app()
        r = client.post("/db/query", json={
            "query": "INSERT INTO items VALUES (3, 'New Item', 1.99)",
            "read_only": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert "affected" in data
        assert data["affected"] == 1

    def test_inserted_row_is_queryable(self):
        """A row inserted via the write tool is visible in subsequent SELECTs."""
        client = build_app()
        client.post("/db/query", json={
            "query": "INSERT INTO items VALUES (4, 'Inserted', 5.00)",
            "read_only": False,
        })
        r = client.post("/db/query", json={
            "query": "SELECT * FROM items WHERE id = 4",
            "read_only": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["rows"][0]["name"] == "Inserted"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestSQLQueryToolErrors:
    def test_malformed_sql_returns_error_dict(self):
        """Malformed SQL returns an error dict instead of raising."""
        client = build_app()
        r = client.post("/db/query", json={"query": "NOT VALID SQL !!!", "read_only": True})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_nonexistent_table_returns_error(self):
        """Querying a nonexistent table returns an error dict."""
        client = build_app()
        r = client.post("/db/query", json={"query": "SELECT * FROM nonexistent_table", "read_only": True})
        assert r.status_code == 200
        assert "error" in r.json()

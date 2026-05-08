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

import pytest


# ---------------------------------------------------------------------------
# SQLQueryTool implementation (inline for test isolation)
# ---------------------------------------------------------------------------

import re
import sqlite3


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
# Tests: SELECT queries (read-only)
# ---------------------------------------------------------------------------


class TestSQLQueryToolSelect:
    @pytest.mark.asyncio
    async def test_select_all_returns_rows(self):
        """SELECT * returns all seeded rows."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "SELECT * FROM items ORDER BY id")

        assert "rows" in result
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_select_returns_correct_columns(self):
        """Column names match the schema."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "SELECT * FROM items WHERE id = 1")

        row = result["rows"][0]
        assert row["id"] == 1
        assert row["name"] == "Widget"
        assert abs(row["value"] - 9.99) < 0.001

    @pytest.mark.asyncio
    async def test_select_with_where_clause(self):
        """WHERE clause filters rows correctly."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "SELECT * FROM items WHERE name = 'Gadget'")

        assert result["count"] == 1
        assert result["rows"][0]["id"] == 2

    @pytest.mark.asyncio
    async def test_select_row_count_correct(self):
        """count field reflects the actual number of rows returned."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "SELECT * FROM items")

        assert result["count"] == len(result["rows"])


# ---------------------------------------------------------------------------
# Tests: write operations blocked in read-only mode
# ---------------------------------------------------------------------------


class TestSQLQueryToolWriteBlocked:
    @pytest.mark.asyncio
    async def test_insert_blocked_in_read_only_mode(self):
        """INSERT is blocked when the tool is read-only."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "INSERT INTO items VALUES (3, 'Thingamajig', 4.99)")

        assert "error" in result
        assert "not permitted" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_blocked_in_read_only_mode(self):
        """UPDATE is blocked when the tool is read-only."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "UPDATE items SET name='X' WHERE id=1")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_drop_blocked_in_read_only_mode(self):
        """DROP is blocked when the tool is read-only."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "DROP TABLE items")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_delete_blocked_in_read_only_mode(self):
        """DELETE is blocked when the tool is read-only."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "DELETE FROM items WHERE id=1")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_per_call_read_only_flag_also_blocks_writes(self):
        """Passing read_only=True per-call blocks writes even on a write-enabled tool."""
        db = SQLQueryTool(read_only=False)
        result = await db.run(_CTX, "INSERT INTO items VALUES (5, 'X', 0.01)", read_only=True)

        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: write operations allowed when not read-only
# ---------------------------------------------------------------------------


class TestSQLQueryToolWriteEnabled:
    @pytest.mark.asyncio
    async def test_insert_succeeds_in_write_mode(self):
        """INSERT succeeds when the tool is write-enabled."""
        db = SQLQueryTool(read_only=False)
        result = await db.run(
            _CTX, "INSERT INTO items VALUES (3, 'New Item', 1.99)", read_only=False
        )

        assert "affected" in result
        assert result["affected"] == 1

    @pytest.mark.asyncio
    async def test_inserted_row_is_queryable(self):
        """A row inserted via the write tool is visible in subsequent SELECTs."""
        db = SQLQueryTool(read_only=False)
        await db.run(_CTX, "INSERT INTO items VALUES (4, 'Inserted', 5.00)", read_only=False)
        result = await db.run(_CTX, "SELECT * FROM items WHERE id = 4")

        assert result["count"] == 1
        assert result["rows"][0]["name"] == "Inserted"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestSQLQueryToolErrors:
    @pytest.mark.asyncio
    async def test_malformed_sql_returns_error_dict(self):
        """Malformed SQL returns an error dict instead of raising."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "NOT VALID SQL !!!")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_nonexistent_table_returns_error(self):
        """Querying a nonexistent table returns an error dict."""
        db = SQLQueryTool(read_only=True)
        result = await db.run(_CTX, "SELECT * FROM nonexistent_table")

        assert "error" in result

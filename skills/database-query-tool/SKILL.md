---
name: database-query-tool
description: SQL query tool with read/write guard using SQLite in-memory (or any sqlite3-compatible connection). Use when an agent needs to query or mutate a relational database, with configurable read-only mode to prevent accidental writes.
---

> Use `codemap find "SQLQueryTool"` after adding the pattern to your project.

# Database Query Tool (SQL Agent with Read/Write Guard)

A `@tool()` class wrapping a `sqlite3` connection. Blocks `INSERT`, `UPDATE`,
`DELETE`, and DDL statements when `read_only=True`.

## Pattern

```python
import re
import sqlite3
from lauren_ai._tools import tool, ToolContext

WRITE_PATTERNS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b',
    re.IGNORECASE,
)

@tool()
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
            "CREATE TABLE IF NOT EXISTS items "
            "(id INTEGER PRIMARY KEY, name TEXT, value REAL)"
        )
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
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                return {"rows": rows, "count": len(rows)}
            self._conn.commit()
            return {"affected": cursor.rowcount}
        except Exception as e:
            return {"error": str(e)}
```

## Usage with an agent

```python
from lauren_ai import agent, use_tools

db_tool = SQLQueryTool(read_only=True)  # instance pre-configured

@agent(model="claude-sonnet-4-6", system="You query our product database.")
@use_tools(SQLQueryTool)
class DBAgent: ...
```

For a write-enabled tool, pass `read_only=False` and pass the pre-built
instance to the tool map when constructing the runner, or let DI wire it.

## Extending to other databases

Replace `sqlite3` with any DBAPI-2 driver (psycopg2, asyncpg adapter, etc.):

```python
@tool()
class AsyncSQLQueryTool:
    """Execute a SQL query against a PostgreSQL database.

    Args:
        query: The SQL query to execute.
    """

    def __init__(self, pool):  # asyncpg pool injected via DI
        self._pool = pool

    async def run(self, ctx: ToolContext, query: str) -> dict:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query)
            return {"rows": [dict(r) for r in rows], "count": len(rows)}
```

## Notes

- Always validate and sanitise queries before passing them to the tool — LLM
  output is untrusted.
- The `read_only` flag can be set at the instance level (permanent) or
  per-call (temporary override). Both flags are OR-ed: either one being
  `True` blocks writes.
- For production use, add parameterised queries to prevent SQL injection.

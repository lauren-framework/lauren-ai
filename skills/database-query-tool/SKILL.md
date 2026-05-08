---
name: database-query-tool
description: SQL query tool with read/write guard using SQLite in-memory (or any sqlite3-compatible connection). Use when an agent needs to query or mutate a relational database, with configurable read-only mode to prevent accidental writes. Supports @set_metadata for mode annotation.
---

> Use `codemap find "SQLQueryTool"` after adding the pattern to your project.

# Database Query Tool (SQL Agent with Read/Write Guard)

A `@tool()` class wrapping a `sqlite3` connection. Blocks `INSERT`, `UPDATE`,
`DELETE`, and DDL statements when `read_only=True`.

## Critical rule — name override for acronym classes

The `@tool()` snake_case converter inserts `_` before every uppercase letter,
so `SQLQueryTool` would become `s_q_l_query_tool`. Always override the name
explicitly when the class name contains consecutive uppercase letters:

```python
@set_metadata("mode", "read")
@tool(name="sql_query_tool")
class SQLQueryTool:
    ...
```

## Pattern

```python
import re
import sqlite3
from lauren_ai._tools import tool, ToolContext, set_metadata

WRITE_PATTERNS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b',
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

## AgentRunner test pattern

When passing a tool **instance** (pre-configured with connection/read_only),
pass the instance to `@use_tools`. The runner stores and calls it correctly.

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


_ro_db = SQLQueryTool(read_only=True)   # shared read-only instance


def _c(text):
    return Completion(id="c1", model="mock", content=text, tool_calls=[],
                      stop_reason="end_turn", usage=TokenUsage(10, 5))


@agent(model=None, system="DB read agent")
@use_tools(_ro_db)
class DBReadAgent(_Capture):
    def __init__(self):
        _Capture.__init__(self)


def test_select_all_returns_rows():
    agent_inst = DBReadAgent()
    client = TestClient(agent_inst)
    client.mock.queue_tool_use(
        "sql_query_tool",
        {"query": "SELECT * FROM items ORDER BY id", "read_only": True},
    )
    client.mock.queue_response(_c("Rows retrieved."))
    client.run("Select all items")
    data = json.loads(agent_inst.captured[0].content)
    assert data["count"] == 2


def test_insert_blocked_in_read_only_mode():
    agent_inst = DBReadAgent()
    client = TestClient(agent_inst)
    client.mock.queue_tool_use(
        "sql_query_tool",
        {"query": "INSERT INTO items VALUES (3, 'X', 1.0)", "read_only": True},
    )
    client.mock.queue_response(_c("Blocked."))
    client.run("Insert item")
    data = json.loads(agent_inst.captured[0].content)
    assert "error" in data
```

## Usage with an agent

```python
from lauren_ai import agent, use_tools

db_tool = SQLQueryTool(read_only=True)  # instance pre-configured

@agent(model="claude-sonnet-4-6", system="You query our product database.")
@use_tools(db_tool)
class DBAgent: ...
```

For a write-enabled tool, pass `read_only=False` to the constructor.

## Notes

- Always use `@tool(name="sql_query_tool")` to override the auto-generated name
  when the class has consecutive uppercase letters in its name.
- `@set_metadata("mode", "read")` documents the tool's operation mode and is
  readable via `ctx.get_metadata("mode")` inside the tool.
- The `read_only` flag can be set at the instance level (permanent) or
  per-call (temporary override). Both flags are OR-ed.
- For production use, add parameterised queries to prevent SQL injection.

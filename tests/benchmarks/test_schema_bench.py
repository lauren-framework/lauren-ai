"""Benchmark tests for schema generation and tool decoration.

Run with::

    pytest tests/benchmarks/ -m benchmark --benchmark-autosave
"""

from __future__ import annotations

import pytest

from lauren_ai._tools import tool
from lauren_ai._tools._schema import type_to_json_schema


@pytest.mark.benchmark
def test_type_to_json_schema_bench(benchmark):
    """Benchmark type_to_json_schema for common types."""

    def run():
        type_to_json_schema(str)
        type_to_json_schema(int)
        type_to_json_schema(float)
        type_to_json_schema(bool)
        type_to_json_schema(list)
        type_to_json_schema(dict)

    benchmark(run)


@pytest.mark.benchmark
def test_tool_decoration_bench(benchmark):
    """Benchmark @tool() decoration overhead."""

    def run():
        @tool()
        async def my_tool(name: str, count: int = 1) -> str:
            """Do something.

            Args:
                name: The name.
                count: The count.
            """
            return name * count

    benchmark(run)

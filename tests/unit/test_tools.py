"""Unit tests for the @tool() decorator and schema generation."""
from __future__ import annotations

import pytest

from lauren_ai._tools import TOOL_META, ToolContext, ToolMeta, ToolResult, tool
from lauren_ai._tools._schema import generate_tool_schema, type_to_json_schema


class TestToolDecorator:
    def test_basic_tool(self):
        @tool()
        async def my_tool(name: str) -> str:
            """Return a greeting.

            Args:
                name: Person's name.
            """
            return f"Hello, {name}!"

        meta = getattr(my_tool, TOOL_META)
        assert meta.name == "my_tool"
        assert "greeting" in meta.description.lower() or "Return" in meta.description

    def test_tool_name_override(self):
        @tool(name="custom_name")
        async def some_func(x: int) -> int:
            """A function.

            Args:
                x: Input value.
            """
            return x

        meta = getattr(some_func, TOOL_META)
        assert meta.name == "custom_name"

    def test_bare_usage_raises(self):
        with pytest.raises(Exception):  # DecoratorUsageError (from lauren or lauren_ai)
            @tool
            async def bad_tool(x: str) -> str:
                return x

    def test_tool_schema_has_parameters(self):
        @tool()
        async def greet(name: str, times: int = 1) -> str:
            """Greet someone.

            Args:
                name: Name to greet.
                times: Number of times.
            """
            return name * times

        meta = getattr(greet, TOOL_META)
        # meta.parameters is a ToolSchema with input_schema key
        input_schema = meta.parameters["input_schema"]
        assert "properties" in input_schema
        assert "name" in input_schema["properties"]
        assert "times" in input_schema["properties"]

    def test_ctx_excluded_from_schema(self):
        @tool()
        async def func_with_ctx(query: str, ctx: ToolContext | None = None) -> str:
            """Search something.

            Args:
                query: The query string.
            """
            return query

        meta = getattr(func_with_ctx, TOOL_META)
        input_schema = meta.parameters["input_schema"]
        assert "ctx" not in input_schema.get("properties", {})

    def test_required_vs_optional(self):
        @tool()
        async def func(required: str, optional: int = 5) -> str:
            """A function.

            Args:
                required: Must be provided.
                optional: Has a default.
            """
            return required

        meta = getattr(func, TOOL_META)
        input_schema = meta.parameters["input_schema"]
        required_list = input_schema.get("required", [])
        assert "required" in required_list
        assert "optional" not in required_list


class TestToolResult:
    def test_ok_string(self):
        r = ToolResult.ok("hello", tool_use_id="tu1")
        assert r.content == "hello"
        assert not r.is_error

    def test_ok_dict(self):
        r = ToolResult.ok({"key": "value"}, tool_use_id="tu1")
        assert '"key"' in r.content
        assert not r.is_error

    def test_error(self):
        r = ToolResult.error("Something went wrong", tool_use_id="tu1")
        assert r.is_error
        assert "Something went wrong" in r.content


class TestTypeToJsonSchema:
    def test_str(self):
        assert type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_of_str(self):
        result = type_to_json_schema(list[str])
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_dict_type(self):
        result = type_to_json_schema(dict)
        assert result == {"type": "object"}

    def test_optional_str(self):
        from typing import Optional
        result = type_to_json_schema(Optional[str])
        assert result == {"type": "string"}

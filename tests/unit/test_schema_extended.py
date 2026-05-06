"""Extended tests for _tools/_schema.py — covering missing type branches."""

from __future__ import annotations

import pytest

from lauren_ai._tools import ToolContext
from lauren_ai._tools._schema import (
    _parse_docstring,
    generate_tool_schema,
    type_to_json_schema,
)


class TestTypeToJsonSchemaExtended:
    def test_none_annotation(self):
        result = type_to_json_schema(None)
        assert result == {"type": "string"}

    def test_empty_annotation(self):
        import inspect

        result = type_to_json_schema(inspect.Parameter.empty)
        assert result == {"type": "string"}

    def test_bytes(self):
        result = type_to_json_schema(bytes)
        assert result["type"] == "string"
        assert "base64" in result.get("contentEncoding", "").lower()

    def test_complex(self):
        result = type_to_json_schema(complex)
        assert result == {"type": "string"}

    def test_plain_dict(self):
        result = type_to_json_schema(dict)
        assert result == {"type": "object"}

    def test_plain_list(self):
        result = type_to_json_schema(list)
        assert result == {"type": "array"}

    def test_any_type(self):
        import typing

        result = type_to_json_schema(typing.Any)
        assert result == {}

    def test_list_of_int(self):
        result = type_to_json_schema(list[int])
        assert result["type"] == "array"
        assert result["items"] == {"type": "integer"}

    def test_list_of_float(self):
        result = type_to_json_schema(list[float])
        assert result["type"] == "array"
        assert result["items"] == {"type": "number"}

    def test_dict_str_int(self):
        result = type_to_json_schema(dict[str, int])
        assert result["type"] == "object"
        assert result["additionalProperties"] == {"type": "integer"}

    def test_dict_str_str(self):
        result = type_to_json_schema(dict[str, str])
        assert result["type"] == "object"
        assert result["additionalProperties"] == {"type": "string"}

    def test_dict_no_args(self):
        result = type_to_json_schema(dict)
        assert result["type"] == "object"

    def test_literal_strings(self):
        from typing import Literal

        result = type_to_json_schema(Literal["a", "b", "c"])
        assert result["type"] == "string"
        assert result["enum"] == ["a", "b", "c"]

    def test_literal_ints(self):
        from typing import Literal

        result = type_to_json_schema(Literal[1, 2, 3])
        assert result["type"] == "integer"
        assert result["enum"] == [1, 2, 3]

    def test_literal_mixed(self):
        from typing import Literal

        result = type_to_json_schema(Literal["a", 1])
        assert "enum" in result
        # Mixed types: no type key, just enum
        assert "a" in result["enum"]
        assert 1 in result["enum"]

    def test_union_non_optional(self):
        from typing import Union

        result = type_to_json_schema(Union[str, int])
        assert "anyOf" in result
        types = {s.get("type") for s in result["anyOf"]}
        assert "string" in types
        assert "integer" in types

    def test_union_collapses_to_single(self):
        from typing import Union

        result = type_to_json_schema(Union[str, None])
        # Optional[str] → str
        assert result == {"type": "string"}

    def test_optional_int(self):
        from typing import Optional

        result = type_to_json_schema(Optional[int])
        assert result == {"type": "integer"}

    def test_optional_list_str(self):
        from typing import Optional

        result = type_to_json_schema(Optional[list[str]])
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_tuple_with_ellipsis(self):
        result = type_to_json_schema(tuple[str, ...])
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_tuple_fixed(self):
        result = type_to_json_schema(tuple[str, int, bool])
        assert result["type"] == "array"
        assert "prefixItems" in result
        assert len(result["prefixItems"]) == 3

    def test_tuple_empty(self):
        # bare tuple has no origin, falls back to string
        result = type_to_json_schema(tuple)
        assert result == {"type": "string"}

    def test_depth_guard(self):
        # At depth > 10, should return {"type": "string"}
        result = type_to_json_schema(str, depth=11)
        assert result == {"type": "string"}

    def test_pydantic_model(self):
        try:
            from pydantic import BaseModel

            class Address(BaseModel):
                street: str
                city: str

            result = type_to_json_schema(Address)
            assert "properties" in result
            assert "street" in result["properties"]
            assert "city" in result["properties"]
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_dataclass_like_class(self):
        class Point:
            x: float
            y: float

        result = type_to_json_schema(Point)
        assert result["type"] == "object"
        assert "x" in result["properties"]
        assert "y" in result["properties"]

    def test_unknown_type_fallback(self):
        class WeirdType:
            pass
            # No __annotations__

        # A class with no __annotations__ at all
        if hasattr(WeirdType, "__annotations__"):
            del WeirdType.__annotations__

        # Should return object schema (from class __annotations__ branch) or
        # fall through to the string fallback — depends on hasattr check
        result = type_to_json_schema(WeirdType)
        # Either {"type": "object", ...} with empty props or {"type": "string"}
        assert result.get("type") in ("object", "string")

    def test_python310_union_syntax(self):
        import sys

        if sys.version_info < (3, 10):
            pytest.skip("Python 3.10+ only")
        # str | int union syntax
        result = type_to_json_schema(eval("str | int"))
        assert "anyOf" in result

    def test_python310_optional_syntax(self):
        import sys

        if sys.version_info < (3, 10):
            pytest.skip("Python 3.10+ only")
        result = type_to_json_schema(eval("str | None"))
        assert result == {"type": "string"}


class TestParseDocstring:
    def test_empty_docstring(self):
        desc, params = _parse_docstring("")
        assert desc == ""
        assert params == {}

    def test_simple_description(self):
        doc = "A simple function that does something."
        desc, params = _parse_docstring(doc)
        assert desc == doc

    def test_description_with_args(self):
        doc = """Search the web.

        Args:
            query: The search query string.
            max_results: Maximum number of results.
        """
        desc, params = _parse_docstring(doc)
        assert desc == "Search the web."
        assert "query" in params
        assert "max_results" in params
        assert "search query" in params["query"].lower()

    def test_description_multiline(self):
        doc = """First line.
Second line.

Args:
    x: A parameter.
"""
        desc, params = _parse_docstring(doc)
        assert "First line" in desc
        assert "Second line" in desc

    def test_args_only_no_description(self):
        # With leading blank lines and no description before Args,
        # the description extraction stops at Args section
        doc = """Do something.

Args:
    x: Input value.
"""
        desc, params = _parse_docstring(doc)
        assert "x" in params
        assert desc == "Do something."

    def test_args_section_ends_at_new_section(self):
        doc = """Do something.

Args:
    x: Parameter x.

Returns:
    The result.
"""
        desc, params = _parse_docstring(doc)
        assert "x" in params
        assert "Returns" not in params

    def test_arguments_section_alias(self):
        doc = """Do something.

Arguments:
    x: Parameter x.
"""
        desc, params = _parse_docstring(doc)
        assert "x" in params

    def test_parameters_section_alias(self):
        doc = """Do something.

Parameters:
    x: Parameter x.
"""
        desc, params = _parse_docstring(doc)
        assert "x" in params

    def test_blank_lines_in_args(self):
        doc = """Do something.

Args:
    x: Parameter x.

    y: Parameter y.
"""
        desc, params = _parse_docstring(doc)
        assert "x" in params
        assert "y" in params


class TestGenerateToolSchema:
    def test_function_schema(self):
        async def search(query: str, max_results: int = 5) -> str:
            """Search the web.

            Args:
                query: The search query.
                max_results: Maximum results to return.
            """
            return query

        name, desc, schema = generate_tool_schema(search)
        assert name == "search"
        assert "search" in desc.lower() or "web" in desc.lower()
        assert "query" in schema["input_schema"]["properties"]
        assert "max_results" in schema["input_schema"]["properties"]
        assert "query" in schema["input_schema"].get("required", [])
        assert "max_results" not in schema["input_schema"].get("required", [])

    def test_name_override(self):
        async def my_func(x: str) -> str:
            """A function."""
            return x

        name, desc, schema = generate_tool_schema(my_func, name="custom_name")
        assert name == "custom_name"

    def test_description_override(self):
        async def my_func(x: str) -> str:
            """Original description."""
            return x

        name, desc, schema = generate_tool_schema(my_func, description="Custom description")
        assert desc == "Custom description"

    def test_class_form_schema(self):
        class SearchTool:
            """A class-based search tool."""

            def run(self, query: str, limit: int = 10) -> str:
                """Execute search.

                Args:
                    query: Search query.
                    limit: Result limit.
                """
                return query

        # Convert CamelCase to snake_case for class
        name, desc, schema = generate_tool_schema(SearchTool)
        assert name == "search_tool"
        assert "query" in schema["input_schema"]["properties"]

    def test_class_form_missing_run_raises(self):
        class BadTool:
            """A bad tool with no run method."""

            pass

        with pytest.raises(ValueError) as exc_info:
            generate_tool_schema(BadTool)
        assert "run" in str(exc_info.value)

    def test_ctx_excluded(self):
        async def tool_fn(query: str, ctx: ToolContext = None) -> str:
            """A tool. Args: query: The query."""
            return query

        name, desc, schema = generate_tool_schema(tool_fn)
        assert "ctx" not in schema["input_schema"]["properties"]

    def test_optional_params_not_required(self):

        async def opt_func(required: str, optional: int | None = None) -> str:
            """A function. Args: required: Required. optional: Optional."""
            return required

        name, desc, schema = generate_tool_schema(opt_func)
        required = schema["input_schema"].get("required", [])
        assert "required" in required
        assert "optional" not in required

    def test_self_excluded(self):
        """Ensure 'self' is excluded from schema properties."""

        class MyTool:
            """A tool."""

            def run(self, x: str) -> str:
                """Run. Args: x: Input."""
                return x

        name, desc, schema = generate_tool_schema(MyTool)
        assert "self" not in schema["input_schema"]["properties"]

    def test_param_descriptions_from_docstring(self):
        async def annotated(query: str) -> str:
            """Search something.

            Args:
                query: The search query string to use.
            """
            return query

        name, desc, schema = generate_tool_schema(annotated)
        props = schema["input_schema"]["properties"]
        assert "description" in props["query"]
        assert "search query" in props["query"]["description"].lower()

    def test_param_with_default_has_default_in_schema(self):
        async def with_default(count: int = 5) -> int:
            """A function. Args: count: Count."""
            return count

        name, desc, schema = generate_tool_schema(with_default)
        props = schema["input_schema"]["properties"]
        assert props["count"].get("default") == 5

    def test_private_params_excluded(self):
        async def func_with_private(x: str, _hidden: str = "x") -> str:
            """A function. Args: x: Input."""
            return x

        name, desc, schema = generate_tool_schema(func_with_private)
        assert "_hidden" not in schema["input_schema"]["properties"]

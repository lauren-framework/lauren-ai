"""Extended tests for ToolRegistry — covers collision, instance registration,
get_schemas, all_names, all_metas."""
from __future__ import annotations

import pytest

from lauren_ai._tools import TOOL_META, tool
from lauren_ai._tools._registry import ToolRegistry


class TestToolRegistryExtended:
    def test_register_no_tool_meta_raises(self):
        class NoMetaClass:
            pass

        registry = ToolRegistry()
        with pytest.raises(ValueError) as exc_info:
            registry.register(NoMetaClass)
        assert TOOL_META in str(exc_info.value)

    def test_register_plain_function_no_meta_raises(self):
        def plain_fn(x):
            return x

        registry = ToolRegistry()
        with pytest.raises(ValueError) as exc_info:
            registry.register(plain_fn)
        assert "@tool()" in str(exc_info.value)

    def test_register_with_instance_carrying_meta(self):
        @tool()
        class ToolClass:
            """A class-form tool."""
            def run(self, x: str) -> str:
                """Run. Args: x: Input."""
                return x

        instance = ToolClass()
        registry = ToolRegistry()
        # Pass instance as the `instance` param — tool_or_cls is the class
        registry.register(ToolClass, instance=instance)
        result = registry.get("tool_class")
        assert result is not None
        callable_or_inst, meta = result
        assert callable_or_inst is instance

    def test_register_instance_no_meta_raises(self):
        class NoMetaInstance:
            pass

        registry = ToolRegistry()
        with pytest.raises(ValueError):
            registry.register(NoMetaInstance(), instance=None)

    def test_collision_raises_value_error(self):
        @tool()
        async def tool_a(x: str) -> str:
            """Tool A. Args: x: Input."""
            return x

        @tool(name="tool_a")  # Same name!
        async def tool_b(y: int) -> int:
            """Tool B. Args: y: Number."""
            return y

        registry = ToolRegistry()
        registry.register(tool_a)
        with pytest.raises(ValueError) as exc_info:
            registry.register(tool_b)
        assert "collision" in str(exc_info.value).lower()
        assert "tool_a" in str(exc_info.value)

    def test_get_returns_none_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_get_schemas_known_tools(self):
        @tool()
        async def search(query: str) -> str:
            """Search. Args: query: Query."""
            return query

        registry = ToolRegistry()
        registry.register(search)
        schemas = registry.get_schemas(["search"])
        assert len(schemas) == 1

    def test_get_schemas_unknown_skipped_with_warning(self, caplog):
        import logging
        registry = ToolRegistry()
        with caplog.at_level(logging.WARNING, logger="lauren_ai._tools._registry"):
            schemas = registry.get_schemas(["unknown_tool"])
        assert schemas == []
        assert "unknown_tool" in caplog.text

    def test_all_names_empty(self):
        registry = ToolRegistry()
        assert registry.all_names() == []

    def test_all_names_populated(self):
        @tool()
        async def tool_x(x: str) -> str:
            """Tool. Args: x: Input."""
            return x

        @tool()
        async def tool_y(y: int) -> int:
            """Tool. Args: y: Number."""
            return y

        registry = ToolRegistry()
        registry.register(tool_x)
        registry.register(tool_y)
        names = registry.all_names()
        assert "tool_x" in names
        assert "tool_y" in names
        assert len(names) == 2

    def test_all_metas_returns_copy(self):
        @tool()
        async def my_t(x: str) -> str:
            """My tool. Args: x: Input."""
            return x

        registry = ToolRegistry()
        registry.register(my_t)
        metas = registry.all_metas()
        assert "my_t" in metas
        # Modifying the returned dict shouldn't affect the registry
        metas["new_key"] = None
        assert "new_key" not in registry.all_metas()

    def test_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0

        @tool()
        async def t1(x: str) -> str:
            """T1. Args: x: Input."""
            return x

        registry.register(t1)
        assert len(registry) == 1

    def test_contains(self):
        @tool()
        async def t_check(x: str) -> str:
            """T. Args: x: Input."""
            return x

        registry = ToolRegistry()
        assert "t_check" not in registry
        registry.register(t_check)
        assert "t_check" in registry

    def test_instance_with_meta_from_class(self):
        """Instance passed as instance param but tool_or_cls has no meta — meta from instance type."""
        @tool()
        class InjectableTool:
            """Injectable tool."""
            def run(self, msg: str) -> str:
                """Run. Args: msg: Message."""
                return msg

        instance = InjectableTool()
        registry = ToolRegistry()
        # The class has TOOL_META, the instance is provided as `instance`
        registry.register(InjectableTool, instance=instance)
        assert "injectable_tool" in registry
        result = registry.get("injectable_tool")
        assert result is not None
        stored, meta = result
        assert stored is instance

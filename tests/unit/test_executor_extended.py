"""Extended unit tests for ToolExecutor — covers cache, hooks, HITL, and dispatch."""
from __future__ import annotations

import asyncio
import pytest

from lauren_ai._tools import TOOL_META, ToolContext, ToolMeta, ToolResult, tool
from lauren_ai._tools._executor import (
    CacheBackend,
    InMemoryCacheBackend,
    ToolCall,
    ToolExecutionError,
    ToolExecutor,
    ToolPendingApprovalSignal,
)
from lauren_ai._tools._registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_call(name: str, tool_input: dict, tool_use_id: str = "tc1") -> ToolCall:
    return ToolCall(tool_use_id=tool_use_id, name=name, input=tool_input)


def make_context() -> ToolContext:
    return ToolContext(agent_context=None, tool_use_id="tc1", turn=1)


def make_registry_with_tool(func_or_class) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(func_or_class)
    return registry


# ---------------------------------------------------------------------------
# InMemoryCacheBackend tests
# ---------------------------------------------------------------------------


class TestInMemoryCacheBackend:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        cache = InMemoryCacheBackend()
        await cache.set("key1", "value1", ttl=60)
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_miss_returns_none(self):
        cache = InMemoryCacheBackend()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self):
        import time
        cache = InMemoryCacheBackend()
        await cache.set("key1", "value1", ttl=1)
        # Manually expire by setting expires_at in the past
        from lauren_ai._tools._executor import _CacheEntry
        cache._store["key1"] = _CacheEntry(value="value1", expires_at=time.monotonic() - 1)
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        cache = InMemoryCacheBackend()
        await cache.set("key1", "value1", ttl=60)
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_no_error(self):
        cache = InMemoryCacheBackend()
        await cache.delete("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = InMemoryCacheBackend()
        await cache.set("key1", "v1", ttl=60)
        await cache.set("key2", "v2", ttl=60)
        await cache.clear()
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_zero_ttl_not_stored(self):
        cache = InMemoryCacheBackend()
        await cache.set("key1", "v1", ttl=0)
        result = await cache.get("key1")
        assert result is None

    def test_len(self):
        cache = InMemoryCacheBackend()
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# ToolExecutor tests
# ---------------------------------------------------------------------------


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_simple_async_tool(self):
        @tool()
        async def greet(name: str) -> str:
            """Greet someone. Args: name: The name."""
            return f"Hello, {name}!"

        registry = make_registry_with_tool(greet)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("greet", {"name": "Alice"})

        result = await executor.execute(call, ctx)
        assert result.content == "Hello, Alice!"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        @tool()
        def sync_add(a: int, b: int) -> int:
            """Add two numbers. Args: a: First. b: Second."""
            return a + b

        registry = make_registry_with_tool(sync_add)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("sync_add", {"a": 3, "b": 4})

        result = await executor.execute(call, ctx)
        assert "7" in result.content
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("nonexistent_tool", {})

        result = await executor.execute(call, ctx)
        assert result.is_error
        assert "nonexistent_tool" in result.content

    @pytest.mark.asyncio
    async def test_tool_exception_raises_execution_error(self):
        @tool()
        async def failing_tool(x: str) -> str:
            """A failing tool. Args: x: Input."""
            raise ValueError("Something went wrong")

        registry = make_registry_with_tool(failing_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("failing_tool", {"x": "test"})

        with pytest.raises(ToolExecutionError) as exc_info:
            await executor.execute(call, ctx)
        assert "failing_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_hitl_raises_pending_approval(self):
        from lauren_ai._tools import ToolMeta
        from lauren_ai._tools._schema import generate_tool_schema

        @tool(requires_confirmation=True)
        async def dangerous_tool(action: str) -> str:
            """A dangerous tool. Args: action: The action."""
            return action

        registry = make_registry_with_tool(dangerous_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("dangerous_tool", {"action": "delete_all"})

        with pytest.raises(ToolPendingApprovalSignal) as exc_info:
            await executor.execute(call, ctx)
        assert exc_info.value.tool_name == "dangerous_tool"
        assert exc_info.value.tool_input == {"action": "delete_all"}

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self):
        call_count = 0

        @tool(cache_ttl=60)
        async def cached_tool(query: str) -> str:
            """A cached tool. Args: query: The query."""
            nonlocal call_count
            call_count += 1
            return f"Result for {query}"

        registry = make_registry_with_tool(cached_tool)
        cache = InMemoryCacheBackend()
        executor = ToolExecutor(registry=registry, cache_backend=cache)
        ctx = make_context()

        call1 = make_tool_call("cached_tool", {"query": "test"}, "tc1")
        call2 = make_tool_call("cached_tool", {"query": "test"}, "tc2")

        result1 = await executor.execute(call1, ctx)
        result2 = await executor.execute(call2, ctx)

        assert result1.content == result2.content
        assert call_count == 1  # Second call used cache

    @pytest.mark.asyncio
    async def test_pre_hook_called(self):
        hook_called = []

        def pre(call, ctx):
            hook_called.append("pre")

        @tool(pre_hook=pre)
        async def hooked_tool(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        registry = make_registry_with_tool(hooked_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("hooked_tool", {"x": "test"})

        await executor.execute(call, ctx)
        assert "pre" in hook_called

    @pytest.mark.asyncio
    async def test_post_hook_called(self):
        hook_called = []

        def post(result, ctx):
            hook_called.append("post")

        @tool(post_hook=post)
        async def hooked_tool(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        registry = make_registry_with_tool(hooked_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("hooked_tool", {"x": "test"})

        await executor.execute(call, ctx)
        assert "post" in hook_called

    @pytest.mark.asyncio
    async def test_error_hook_called_on_exception(self):
        error_hook_called = []

        def error_hook(exc, ctx):
            error_hook_called.append(str(exc))

        @tool(error_hook=error_hook)
        async def failing_tool(x: str) -> str:
            """A failing tool. Args: x: Input."""
            raise RuntimeError("tool failed")

        registry = make_registry_with_tool(failing_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("failing_tool", {"x": "test"})

        with pytest.raises(ToolExecutionError):
            await executor.execute(call, ctx)

        assert len(error_hook_called) == 1

    @pytest.mark.asyncio
    async def test_async_pre_hook_called(self):
        hook_called = []

        async def async_pre(call, ctx):
            hook_called.append("async_pre")

        @tool(pre_hook=async_pre)
        async def hooked_tool(x: str) -> str:
            """A tool. Args: x: Input."""
            return x

        registry = make_registry_with_tool(hooked_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("hooked_tool", {"x": "test"})

        await executor.execute(call, ctx)
        assert "async_pre" in hook_called

    @pytest.mark.asyncio
    async def test_tool_returns_dict_serialized(self):
        @tool()
        async def dict_tool(key: str) -> dict:
            """A tool. Args: key: The key."""
            return {"key": key, "value": 42}

        registry = make_registry_with_tool(dict_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("dict_tool", {"key": "test"})

        result = await executor.execute(call, ctx)
        assert not result.is_error
        assert "test" in result.content

    @pytest.mark.asyncio
    async def test_tool_with_ctx_injection(self):
        received_ctx = []

        @tool()
        async def ctx_tool(x: str, ctx: ToolContext = None) -> str:
            """A tool. Args: x: Input."""
            received_ctx.append(ctx)
            return x

        registry = make_registry_with_tool(ctx_tool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("ctx_tool", {"x": "hello"})

        result = await executor.execute(call, ctx)
        assert not result.is_error
        assert len(received_ctx) == 1
        assert received_ctx[0] is ctx

    def test_default_cache_key(self):
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)
        meta = ToolMeta(
            name="my_tool", description="", parameters={}, is_async=False, reads_context=False
        )
        key = executor._default_cache_key(meta, {"b": 2, "a": 1})
        # Keys should be sorted
        assert "my_tool" in key
        assert key == executor._default_cache_key(meta, {"a": 1, "b": 2})

    @pytest.mark.asyncio
    async def test_cache_key_fn_used(self):
        custom_keys = []

        def custom_key_fn(tool_input):
            key = f"custom:{tool_input.get('q', '')}"
            custom_keys.append(key)
            return key

        @tool(cache_ttl=60, cache_key_fn=custom_key_fn)
        async def custom_cached(q: str) -> str:
            """Tool. Args: q: Query."""
            return q

        registry = make_registry_with_tool(custom_cached)
        cache = InMemoryCacheBackend()
        executor = ToolExecutor(registry=registry, cache_backend=cache)
        ctx = make_context()

        call = make_tool_call("custom_cached", {"q": "hello"})
        await executor.execute(call, ctx)

        assert len(custom_keys) >= 1
        assert "custom:hello" in custom_keys

    @pytest.mark.asyncio
    async def test_class_form_tool_execution(self):
        @tool()
        class EchoTool:
            """Echo the input."""

            def run(self, message: str) -> str:
                """Echo. Args: message: The message."""
                return message

        registry = make_registry_with_tool(EchoTool)
        executor = ToolExecutor(registry=registry)
        ctx = make_context()
        call = make_tool_call("echo_tool", {"message": "hello world"})

        result = await executor.execute(call, ctx)
        assert not result.is_error
        assert "hello world" in result.content


class TestToolExecutionErrorAndPendingApproval:
    def test_tool_execution_error_str(self):
        original = ValueError("bad input")
        exc = ToolExecutionError("my_tool", original)
        s = str(exc)
        assert "my_tool" in s
        assert "ValueError" in s

    def test_tool_pending_approval_signal(self):
        exc = ToolPendingApprovalSignal(
            tool_name="delete_file",
            tool_use_id="tc1",
            tool_input={"path": "/etc/passwd"},
        )
        assert exc.tool_name == "delete_file"
        assert exc.tool_use_id == "tc1"
        assert exc.tool_input == {"path": "/etc/passwd"}
        assert "delete_file" in str(exc)
        assert "tc1" in str(exc)

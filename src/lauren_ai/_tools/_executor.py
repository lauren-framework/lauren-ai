"""Tool executor — dispatches tool calls, handles DI, hooks, caching, and HITL.

``ToolExecutor`` is the runtime dispatcher invoked once per tool call inside
the agentic loop.  It:

1. Resolves the tool callable (function or class ``run()`` method) from the
   registry.
2. Injects a ``ToolContext`` when the entry point declares a ``ctx`` parameter.
3. Runs pre/post/error hooks in the correct order.
4. Caches successful results (when ``cache_ttl`` is set on ``ToolMeta``).
5. Converts arbitrary return values to ``ToolResult`` objects.
6. Raises ``ToolPendingApprovalSignal`` for HITL-gated tools before execution.
"""

from __future__ import annotations

__all__ = [
    "CacheBackend",
    "InMemoryCacheBackend",
    "ToolExecutor",
]

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from . import ToolContext, ToolMeta, ToolResult
from ._registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolExecutionError(Exception):
    """Raised when a tool raises an unhandled exception during execution.

    :param tool_name: Name of the tool that failed.
    :type tool_name: str
    :param original: The original exception raised by the tool.
    :type original: Exception
    """

    def __init__(self, tool_name: str, original: Exception) -> None:
        self.tool_name = tool_name
        self.original = original
        super().__init__(f"Tool '{tool_name}' raised: {type(original).__name__}: {original}")


class ToolPendingApprovalSignal(Exception):
    """Raised (and caught by ``AgentRunner``) when a tool requires human approval.

    The runner suspends the agentic loop and awaits an out-of-band approval
    before resuming with ``ToolExecutor.approve()``.

    :param tool_name: Tool name requiring approval.
    :type tool_name: str
    :param tool_use_id: Provider tool-use identifier for this call.
    :type tool_use_id: str
    :param tool_input: The input dict the tool would receive.
    :type tool_input: dict[str, Any]
    """

    def __init__(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
    ) -> None:
        self.tool_name = tool_name
        self.tool_use_id = tool_use_id
        self.tool_input = tool_input
        super().__init__(
            f"Tool '{tool_name}' (id={tool_use_id}) requires human approval before execution."
        )


# ---------------------------------------------------------------------------
# ToolCall (minimal local definition — the transport layer re-exports this)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool-use request emitted by the model.

    :param tool_use_id: Provider-assigned identifier.
    :type tool_use_id: str
    :param name: The tool name to invoke.
    :type name: str
    :param input: The argument dict supplied by the model.
    :type input: dict[str, Any]
    """

    tool_use_id: str
    name: str
    input: dict[str, Any]


# ---------------------------------------------------------------------------
# CacheBackend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for result caching backends used by ``ToolExecutor``.

    Implementations are expected to be safe for concurrent async access.
    """

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value.

        :param key: Cache key.
        :type key: str
        :return: Cached value, or ``None`` on cache miss.
        :rtype: Any | None
        """
        ...

    async def set(self, key: str, value: Any, *, ttl: int) -> None:
        """Store a value in the cache.

        :param key: Cache key.
        :type key: str
        :param value: Value to cache.
        :type value: Any
        :param ttl: Time-to-live in seconds.
        :type ttl: int
        """
        ...

    async def delete(self, key: str) -> None:
        """Invalidate a single cache entry.

        :param key: Cache key to delete.
        :type key: str
        """
        ...


# ---------------------------------------------------------------------------
# InMemoryCacheBackend
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    value: Any
    expires_at: float  # monotonic clock timestamp


class InMemoryCacheBackend:
    """Simple in-memory cache with TTL expiry.

    Uses ``time.monotonic()`` for TTL tracking; no background eviction thread.
    Expired entries are lazily removed on ``get``.

    Suitable for development and single-process deployments.  Not shared
    across processes or restarts.
    """

    def __init__(self) -> None:
        self._store: dict[str, _CacheEntry] = {}

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value, returning ``None`` if expired or absent.

        :param key: Cache key.
        :type key: str
        :return: Cached value or ``None``.
        :rtype: Any | None
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    async def set(self, key: str, value: Any, *, ttl: int) -> None:
        """Store a value with the given TTL.

        :param key: Cache key.
        :type key: str
        :param value: Value to store.
        :type value: Any
        :param ttl: Expiry in seconds (must be > 0).
        :type ttl: int
        """
        if ttl <= 0:
            return
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl,
        )

    async def delete(self, key: str) -> None:
        """Delete a single entry.

        :param key: Cache key.
        :type key: str
        """
        self._store.pop(key, None)

    async def clear(self) -> None:
        """Clear all cached entries.

        :return: None
        :rtype: None
        """
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Dispatches tool calls from the agentic loop.

    Handles:

    - Resolving the tool (function or class instance) from the registry.
    - Injecting ``ToolContext`` when the entry point declares a ``ctx`` param.
    - Pre/post/error hooks (sync or async, both accepted).
    - Result caching (when ``ToolMeta.cache_ttl`` is set and a cache backend
      is provided).
    - Converting arbitrary return values to ``ToolResult``.
    - HITL: raising ``ToolPendingApprovalSignal`` before executing a tool
      whose ``ToolMeta.requires_confirmation`` is ``True``.

    :param registry: The populated ``ToolRegistry``.
    :type registry: ToolRegistry
    :param cache_backend: Optional cache backend for result caching.
    :type cache_backend: CacheBackend | None
    :param signals: Optional signal bus for emitting ``ToolCallStarted`` /
        ``ToolCallComplete`` events.  Accepts any object with a ``emit``
        method; typed as ``Any`` to avoid coupling to the full signals module.
    :type signals: Any | None
    """

    def __init__(
        self,
        registry: ToolRegistry,
        cache_backend: CacheBackend | None = None,
        signals: Any | None = None,
    ) -> None:
        self._registry = registry
        self._cache = cache_backend
        self._signals = signals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        tool_call: ToolCall,
        tool_context: ToolContext,
    ) -> ToolResult:
        """Execute a single tool call.

        :param tool_call: The tool-use request from the model.
        :type tool_call: ToolCall
        :param tool_context: Context injected into tools that declare a
            ``ctx: ToolContext`` parameter.
        :type tool_context: ToolContext
        :return: The tool result to feed back into the conversation.
        :rtype: ToolResult
        :raises ToolExecutionError: When the tool raises an unhandled exception
            and no ``error_hook`` handles it.
        :raises ToolPendingApprovalSignal: When the tool's
            ``requires_confirmation`` flag is ``True``.
        """
        name = tool_call.name
        tool_use_id = tool_call.tool_use_id
        tool_input = tool_call.input or {}

        entry = self._registry.get(name)
        if entry is None:
            logger.warning(
                "lauren_ai.ToolExecutor: unknown tool '%s' — returning error result",
                name,
            )
            return ToolResult.error(
                f"Unknown tool: '{name}'",
                tool_use_id=tool_use_id,
            )

        callable_or_instance, meta = entry

        # HITL gate — must happen before any hooks or caching
        if meta.requires_confirmation:
            raise ToolPendingApprovalSignal(
                tool_name=name,
                tool_use_id=tool_use_id,
                tool_input=tool_input,
            )

        # Cache read
        cached = await self._try_get_cache(meta, tool_input)
        if cached is not None:
            logger.debug("lauren_ai.ToolExecutor: cache hit for tool '%s'", name)
            return ToolResult.ok(cached, tool_use_id=tool_use_id)

        # Pre-hook
        if meta.pre_hook is not None:
            await self._call_hook(meta.pre_hook, tool_call, tool_context)

        # Dispatch
        try:
            raw_result = await self._dispatch(
                callable_or_instance,
                meta,
                tool_input,
                tool_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "lauren_ai.ToolExecutor: tool '%s' raised %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
            # Error hook
            if meta.error_hook is not None:
                try:
                    await self._call_hook(meta.error_hook, exc, tool_context)
                except Exception:  # noqa: BLE001
                    pass
            raise ToolExecutionError(name, exc) from exc

        # Build ToolResult
        result = ToolResult.ok(raw_result, tool_use_id=tool_use_id)

        # Post-hook — receives the ToolResult and ToolContext
        if meta.post_hook is not None:
            await self._call_hook(meta.post_hook, result, tool_context)

        # Cache write
        await self._try_set_cache(meta, tool_input, raw_result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        callable_or_instance: Any,
        meta: ToolMeta,
        tool_input: dict[str, Any],
        tool_context: ToolContext,
    ) -> Any:
        """Invoke the tool callable and return the raw result.

        :param callable_or_instance: The raw function or class instance stored
            in the registry.
        :param meta: The tool's metadata.
        :param tool_input: Argument dict from the model.
        :param tool_context: Injected context (used when ``reads_context=True``).
        :return: Raw return value from the tool function.
        """
        # Determine the actual callable
        if inspect.isclass(callable_or_instance):
            # Class form without a pre-resolved DI instance: instantiate no-arg
            try:
                instance = callable_or_instance()
            except TypeError as exc:
                raise ToolExecutionError(
                    meta.name,
                    TypeError(
                        f"Class-form tool {callable_or_instance!r} could not be instantiated "
                        f"without arguments: {exc}. Use @injectable + a pre-registered "
                        "instance for tools with constructor dependencies."
                    ),
                ) from exc
            fn = instance.run
        elif hasattr(callable_or_instance, "run") and not callable(callable_or_instance):
            # Shouldn't happen but guard against it
            fn = callable_or_instance.run
        elif inspect.ismethod(callable_or_instance) or (
            hasattr(callable_or_instance, "run")
            and not inspect.isfunction(callable_or_instance)
        ):
            # Instance from DI — use its run() method
            fn = callable_or_instance.run
        else:
            fn = callable_or_instance

        # Build kwargs from tool_input
        kwargs = dict(tool_input)

        # Inject ToolContext if needed — use the annotation-discovered param name,
        # not a hardcoded "ctx", so any parameter name works.
        if meta.reads_context and meta.context_param_name:
            kwargs[meta.context_param_name] = tool_context

        # Call (sync or async)
        if meta.is_async:
            return await fn(**kwargs)
        else:
            # Run sync function in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: fn(**kwargs))

    async def _call_hook(self, hook: Any, *args: Any) -> None:
        """Invoke a hook (sync or async) with the given positional args.

        :param hook: The hook callable.
        :param args: Positional arguments forwarded to the hook.
        """
        try:
            if inspect.iscoroutinefunction(hook):
                await hook(*args)
            else:
                hook(*args)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "lauren_ai.ToolExecutor: hook %r raised %s: %s — ignoring",
                hook,
                type(exc).__name__,
                exc,
            )

    def _default_cache_key(self, meta: ToolMeta, tool_input: dict[str, Any]) -> str:
        """Build a default cache key from tool name + sorted input items.

        :param meta: Tool metadata.
        :param tool_input: Input dict.
        :return: A deterministic string cache key.
        :rtype: str
        """
        try:
            input_repr = json.dumps(tool_input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            input_repr = str(sorted(tool_input.items()))
        return f"lauren_ai:tool:{meta.name}:{input_repr}"

    async def _try_get_cache(
        self,
        meta: ToolMeta,
        tool_input: dict[str, Any],
    ) -> Any | None:
        """Attempt to retrieve a cached result.

        Returns ``None`` on cache miss, backend absence, or any error.
        """
        if self._cache is None or meta.cache_ttl is None:
            return None
        try:
            if meta.cache_key_fn is not None:
                key = meta.cache_key_fn(tool_input)
            else:
                key = self._default_cache_key(meta, tool_input)
            return await self._cache.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("lauren_ai.ToolExecutor: cache get error: %s", exc)
            return None

    async def _try_set_cache(
        self,
        meta: ToolMeta,
        tool_input: dict[str, Any],
        value: Any,
    ) -> None:
        """Attempt to cache a successful result.  Errors are silently logged."""
        if self._cache is None or meta.cache_ttl is None:
            return
        try:
            if meta.cache_key_fn is not None:
                key = meta.cache_key_fn(tool_input)
            else:
                key = self._default_cache_key(meta, tool_input)
            await self._cache.set(key, value, ttl=meta.cache_ttl)
        except Exception as exc:  # noqa: BLE001
            logger.debug("lauren_ai.ToolExecutor: cache set error: %s", exc)

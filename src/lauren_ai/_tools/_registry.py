"""Tool registry — maps tool names to callables and their metadata.

``ToolRegistry`` is collected once per module graph at startup (analogous to
the route table or the DI container).  It provides O(1) lookup by name and
validates for name collisions at startup.
"""

from __future__ import annotations

__all__ = [
    "ToolRegistry",
]

import logging
from typing import Any

from . import TOOL_META, ToolMeta

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of all tools available to agents.

    Collected once per module graph at startup.  Provides ``O(1)`` lookup by
    name and validates for name collisions.

    :Example:

    .. code-block:: python

        registry = ToolRegistry()
        registry.register(search_web)
        tool_callable, meta = registry.get("search_web")
        schemas = registry.get_schemas(["search_web"])
    """

    def __init__(self) -> None:
        # name → (callable_or_instance, ToolMeta)
        self._tools: dict[str, Any] = {}
        self._metas: dict[str, ToolMeta] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool_or_cls: Any, *, instance: Any | None = None) -> None:
        """Register a tool function or class (or an already-resolved instance).

        The object must have a ``TOOL_META`` attribute (i.e. it must have been
        decorated with ``@tool()``).

        :param tool_or_cls: A ``@tool()``-decorated function or class.
        :type tool_or_cls: Any
        :param instance: Pre-resolved instance for class-form tools (supplied
            by the DI container at startup).  When ``None`` the raw
            function/class is stored and instantiated at dispatch time.
        :type instance: Any | None
        :raises ValueError: When the tool has no ``TOOL_META`` attribute.
        :raises ValueError: When a tool with the same name is already
            registered (collision detection).
        """
        meta: ToolMeta | None = getattr(tool_or_cls, TOOL_META, None)
        if meta is None:
            # Maybe the instance carries the meta (class-form resolved by DI)
            if instance is not None:
                meta = getattr(type(instance), TOOL_META, None) or getattr(instance, TOOL_META, None)
            if meta is None:
                raise ValueError(
                    f"Object {tool_or_cls!r} does not have a {TOOL_META!r} attribute. "
                    "Only objects decorated with @tool() can be registered."
                )

        name = meta.name

        if name in self._tools:
            existing_meta = self._metas[name]
            raise ValueError(
                f"Tool name collision: '{name}' is already registered "
                f"(existing={existing_meta!r}, new={meta!r}). "
                "Each tool must have a unique name within the registry."
            )

        # Store the instance if provided (class-form resolved from DI), else
        # store the raw function/class.
        self._tools[name] = instance if instance is not None else tool_or_cls
        self._metas[name] = meta
        logger.debug("lauren_ai.ToolRegistry: registered tool '%s'", name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> tuple[Any, ToolMeta] | None:
        """Return ``(callable_or_instance, ToolMeta)`` for *name*, or ``None``.

        :param name: The tool name to look up.
        :type name: str
        :return: A ``(callable_or_instance, ToolMeta)`` pair, or ``None`` when
            the name is not registered.
        :rtype: tuple[Any, ToolMeta] | None
        """
        if name not in self._tools:
            return None
        return self._tools[name], self._metas[name]

    def get_schemas(self, tool_names: list[str]) -> list[Any]:
        """Return the ``ToolSchema`` list for the given tool names.

        Unknown names are silently skipped (same pattern as ``use_guards``
        dropping ``None`` entries).

        :param tool_names: Ordered list of tool names to include.
        :type tool_names: list[str]
        :return: List of ``ToolSchema`` dicts ready to pass to the transport.
        :rtype: list[ToolSchema]
        """
        schemas: list[Any] = []
        for name in tool_names:
            meta = self._metas.get(name)
            if meta is not None:
                schemas.append(meta.parameters)
            else:
                logger.warning(
                    "lauren_ai.ToolRegistry.get_schemas: unknown tool '%s' — skipping",
                    name,
                )
        return schemas

    def all_names(self) -> list[str]:
        """Return all registered tool names in registration order.

        :return: Sorted list of registered tool names.
        :rtype: list[str]
        """
        return list(self._tools.keys())

    def all_metas(self) -> dict[str, ToolMeta]:
        """Return a copy of the name → ``ToolMeta`` mapping.

        :return: Shallow copy of the meta dictionary.
        :rtype: dict[str, ToolMeta]
        """
        return dict(self._metas)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of registered tools.

        :return: Number of tools.
        :rtype: int
        """
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Support ``in`` operator for tool name membership tests.

        :param name: Tool name to check.
        :type name: str
        :return: ``True`` if the tool is registered.
        :rtype: bool
        """
        return name in self._tools

    def __repr__(self) -> str:  # pragma: no cover
        names = ", ".join(self._tools)
        return f"ToolRegistry(tools=[{names}])"

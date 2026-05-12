"""Type-annotation compatibility helpers for Python 3.10+.

Provides ``is_optional`` and ``peel_optional`` that work with both the
``typing.Optional[X]`` form and the new ``X | None`` union syntax.
"""

from __future__ import annotations

__all__ = [
    "is_optional",
    "peel_optional",
]

import sys
import types
import typing
from typing import Any, get_args, get_origin


def is_optional(ann: Any) -> bool:
    """Return ``True`` if *ann* is ``Optional[X]`` or ``X | None``.

    :param ann: A type annotation to test.
    :type ann: Any
    :return: Whether the annotation allows ``None``.
    :rtype: bool
    """
    origin = get_origin(ann)
    if origin is typing.Union:
        return type(None) in get_args(ann)

    # Python 3.10+ union types
    if sys.version_info >= (3, 10) and isinstance(ann, types.UnionType):
        return type(None) in get_args(ann)

    return False


def peel_optional(ann: Any) -> Any:
    """Return the inner type of ``Optional[X]`` / ``X | None``.

    If *ann* is ``Optional[str]`` returns ``str``.  If multiple non-None
    types exist (e.g. ``str | int | None``) returns a new ``Union[str, int]``.

    :param ann: An optional type annotation.
    :type ann: Any
    :return: The non-None inner type.
    :rtype: Any
    """
    args = get_args(ann)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1:
        return non_none[0]
    # Reconstruct Union for multi-type optionals
    return typing.Union[tuple(non_none)]  # noqa: UP007 — runtime, not annotation

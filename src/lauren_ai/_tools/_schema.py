"""JSON Schema generation for tool parameters.

Walks ``inspect.signature()`` combined with type annotations and docstring
``Args:`` sections to produce the ``ToolSchema`` objects consumed by LLM
providers (Anthropic, OpenAI, etc.).

Supported type → JSON Schema mappings
--------------------------------------

- ``str``                   → ``{"type": "string"}``
- ``int``                   → ``{"type": "integer"}``
- ``float``                 → ``{"type": "number"}``
- ``bool``                  → ``{"type": "boolean"}``
- ``list[X]``               → ``{"type": "array", "items": <X schema>}``
- ``dict[str, X]``          → ``{"type": "object", "additionalProperties": <X schema>}``
- ``Optional[X]`` / ``X | None`` → same as ``X`` schema (parameter not *required*)
- Pydantic ``BaseModel``    → expanded via ``TypeAdapter(T).json_schema()``
- ``Literal["a", "b"]``     → ``{"type": "string", "enum": ["a", "b"]}``
- Everything else           → ``{"type": "string"}`` (with a warning emitted to stderr)
"""

from __future__ import annotations

__all__ = [
    "generate_tool_schema",
    "type_to_json_schema",
]

import inspect
import logging
import re
import sys
import types
import typing
from typing import Any, get_args, get_origin

from ._types_compat import is_optional, peel_optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forward-declare ToolSchema locally to avoid circular import.
# The real ToolSchema is imported from __init__ at call time.
# ---------------------------------------------------------------------------
from typing import TypedDict


class _InputSchema(TypedDict, total=False):
    type: str
    properties: dict[str, Any]
    required: list[str]
    additionalProperties: Any


# ---------------------------------------------------------------------------
# Docstring parser
# ---------------------------------------------------------------------------


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Parse a docstring into ``(description, param_descriptions)``.

    The *description* is the first paragraph (text before any section header
    or first blank line followed by a section).  Supports Google-style
    ``Args:`` / ``Arguments:`` sections.

    :param doc: Raw docstring text (may include leading/trailing whitespace).
    :type doc: str
    :return: Tuple of ``(description, {param_name: param_description})``.
    :rtype: tuple[str, dict[str, str]]
    """
    if not doc:
        return "", {}

    # Dedent manually (textwrap.dedent handles it but we keep it dependency-free)
    lines = doc.expandtabs().splitlines()
    # Remove leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return "", {}

    # Find the common indent of non-empty lines (excluding the first line which
    # may be on the same line as the opening ``"""``)
    non_empty = [ln for ln in lines[1:] if ln.strip()]
    if non_empty:
        common_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
    else:
        common_indent = 0

    # Strip common indent from all lines except the first
    stripped_lines: list[str] = [lines[0].strip()]
    for ln in lines[1:]:
        stripped_lines.append(ln[common_indent:] if len(ln) >= common_indent else ln.lstrip())

    # Extract first paragraph (up to first blank line)
    description_lines: list[str] = []
    i = 0
    while i < len(stripped_lines):
        if not stripped_lines[i].strip():
            # Blank line — end of first paragraph
            break
        description_lines.append(stripped_lines[i].strip())
        i += 1
    description = " ".join(description_lines).strip()

    # Find Args / Arguments section
    param_descriptions: dict[str, str] = {}
    args_section_re = re.compile(r"^(Args|Arguments|Parameters)\s*:\s*$")
    in_args = False
    current_param: str | None = None
    current_desc_parts: list[str] = []

    # Param line pattern: "    param_name: description" (4 spaces indent typical)
    # Allow any indent that is > 0
    param_line_re = re.compile(r"^(\s{2,})(\w+)\s*:\s*(.*)")
    # Continuation line: more indented than param line
    continuation_re = re.compile(r"^(\s{4,})(.*)")

    param_indent: int | None = None

    for line in stripped_lines:
        if args_section_re.match(line.strip()):
            # Save any in-progress param
            if current_param is not None:
                param_descriptions[current_param] = " ".join(current_desc_parts).strip()
                current_param = None
                current_desc_parts = []
            in_args = True
            param_indent = None
            continue

        if in_args:
            # Check if we've left the Args section (new top-level section)
            # A top-level section header matches a line with no leading indent
            # and ends with ":"
            stripped = line.rstrip()
            if stripped and not stripped[0].isspace() and stripped.endswith(":"):
                # Save current param and exit args section
                if current_param is not None:
                    param_descriptions[current_param] = " ".join(current_desc_parts).strip()
                    current_param = None
                    current_desc_parts = []
                in_args = False
                continue

            if not line.strip():
                # Blank line inside args section — end current param
                if current_param is not None:
                    param_descriptions[current_param] = " ".join(current_desc_parts).strip()
                    current_param = None
                    current_desc_parts = []
                    param_indent = None
                continue

            m = param_line_re.match(line)
            if m:
                indent_len = len(m.group(1))
                # First param sets the base indent for this section
                if param_indent is None:
                    param_indent = indent_len

                if indent_len <= (param_indent + 2):
                    # This is a new parameter entry
                    if current_param is not None:
                        param_descriptions[current_param] = " ".join(current_desc_parts).strip()
                    current_param = m.group(2)
                    current_desc_parts = [m.group(3).strip()] if m.group(3).strip() else []
                else:
                    # Continuation of current param
                    if current_param is not None:
                        current_desc_parts.append(line.strip())
                continue

            # Plain continuation line
            if current_param is not None and line.strip():
                current_desc_parts.append(line.strip())

    # Flush last param
    if in_args and current_param is not None:
        param_descriptions[current_param] = " ".join(current_desc_parts).strip()

    return description, param_descriptions


# ---------------------------------------------------------------------------
# Type → JSON Schema conversion
# ---------------------------------------------------------------------------


def type_to_json_schema(ann: Any, *, depth: int = 0) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema dict.

    :param ann: A type annotation (e.g. ``str``, ``list[int]``,
        ``Optional[str]``, a Pydantic ``BaseModel`` subclass, etc.).
    :type ann: Any
    :param depth: Recursion depth guard (stops at 10 to prevent infinite
        recursion on self-referential types).
    :type depth: int
    :return: A JSON Schema dict (e.g. ``{"type": "string"}``).
    :rtype: dict[str, Any]
    """
    if depth > 10:
        return {"type": "string"}

    if ann is inspect.Parameter.empty or ann is None:
        return {"type": "string"}

    # Unwrap Optional[X] / X | None → extract inner type
    if is_optional(ann):
        inner = peel_optional(ann)
        return type_to_json_schema(inner, depth=depth + 1)

    # Handle basic scalar types
    if ann is str:
        return {"type": "string"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is bool:
        return {"type": "boolean"}
    if ann is bytes:
        return {"type": "string", "contentEncoding": "base64"}
    if ann is complex:
        return {"type": "string"}
    if ann is dict:
        return {"type": "object"}
    if ann is list:
        return {"type": "array"}

    # typing.Any → {}
    if ann is typing.Any:
        return {}

    origin = get_origin(ann)
    args = get_args(ann)

    # list[X] / List[X]
    if origin is list:
        if args:
            return {"type": "array", "items": type_to_json_schema(args[0], depth=depth + 1)}
        return {"type": "array"}

    # tuple[X, ...] → array
    if origin is tuple:
        if args and len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": type_to_json_schema(args[0], depth=depth + 1)}
        if args:
            return {
                "type": "array",
                "prefixItems": [type_to_json_schema(a, depth=depth + 1) for a in args],
            }
        return {"type": "array"}

    # dict[str, X] / Dict[str, X]
    if origin is dict:
        if args and len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": type_to_json_schema(args[1], depth=depth + 1),
            }
        return {"type": "object"}

    # Literal["a", "b"]
    if origin is typing.Literal:  # type: ignore[comparison-overlap]
        values = list(args)
        if values and all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if values and all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        return {"enum": values}

    # Union[X, Y] (non-Optional)
    if origin is typing.Union:
        schemas = [type_to_json_schema(a, depth=depth + 1) for a in args if a is not type(None)]
        if len(schemas) == 1:
            return schemas[0]
        return {"anyOf": schemas}

    # Python 3.10+ union syntax: X | Y
    if sys.version_info >= (3, 10) and isinstance(ann, types.UnionType):
        union_args = get_args(ann)
        non_none = [a for a in union_args if a is not type(None)]
        if len(non_none) == 1:
            return type_to_json_schema(non_none[0], depth=depth + 1)
        return {"anyOf": [type_to_json_schema(a, depth=depth + 1) for a in non_none]}

    # Pydantic BaseModel
    try:
        from pydantic import BaseModel, TypeAdapter

        if isinstance(ann, type) and issubclass(ann, BaseModel):
            adapter = TypeAdapter(ann)
            schema = adapter.json_schema()
            # Remove $defs noise at top level when not needed by the schema body
            return schema
    except ImportError:
        pass

    # Class with __annotations__ (plain dataclass or typed dict)
    if inspect.isclass(ann) and hasattr(ann, "__annotations__"):
        properties: dict[str, Any] = {}
        required_fields: list[str] = []
        for field_name, field_ann in ann.__annotations__.items():
            if field_name.startswith("_"):
                continue
            properties[field_name] = type_to_json_schema(field_ann, depth=depth + 1)
            required_fields.append(field_name)
        return {
            "type": "object",
            "properties": properties,
            "required": required_fields,
        }

    # Fallback
    logger.warning(
        "lauren_ai: unrecognised type annotation %r — falling back to "
        '{"type": "string"} in generated JSON Schema',
        ann,
    )
    return {"type": "string"}


# ---------------------------------------------------------------------------
# Main schema generator
# ---------------------------------------------------------------------------


def generate_tool_schema(
    func_or_class: Any,
    *,
    name: str | None = None,
    description: str | None = None,
) -> tuple[str, str, Any]:
    """Generate ``(tool_name, tool_description, ToolSchema)`` from a function or class.

    For **function-form** tools the function itself is inspected.
    For **class-form** tools the ``run()`` method is inspected for parameters;
    the class docstring provides the description.

    Parameters annotated as ``ToolContext`` (or ``ToolContext | None``) are
    excluded from the schema regardless of their name — they are injected at
    runtime and must not be exposed to the model.  Parameters with a leading
    underscore are also excluded.

    :param func_or_class: A ``@tool()``-decorated function or class.
    :type func_or_class: Any
    :param name: Override the inferred tool name.
    :type name: str | None
    :param description: Override the description extracted from the docstring.
    :type description: str | None
    :return: ``(tool_name, tool_description, ToolSchema)``.
    :rtype: tuple[str, str, ToolSchema]
    """
    from . import ToolContext  # local import to avoid circular dep

    is_class = inspect.isclass(func_or_class)

    # Determine entry-point function for signature inspection
    if is_class:
        entry_fn: Any = getattr(func_or_class, "run", None)
        if entry_fn is None:
            raise ValueError(f"Class-form tool {func_or_class!r} must define a 'run' method.")
        # Description from class docstring; fallback to run() docstring
        raw_doc = inspect.getdoc(func_or_class) or inspect.getdoc(entry_fn) or ""
    else:
        entry_fn = func_or_class
        raw_doc = inspect.getdoc(func_or_class) or ""

    # Parse docstring
    parsed_desc, param_docs = _parse_docstring(raw_doc)

    # Resolve name
    if name:
        tool_name = name
    else:
        raw_name = func_or_class.__name__ if is_class else func_or_class.__name__
        # Convert CamelCase class names to snake_case
        if is_class:
            tool_name = re.sub(r"(?<!^)(?=[A-Z])", "_", raw_name).lower()
        else:
            tool_name = raw_name

    # Resolve description
    tool_description = description if description else parsed_desc

    # Inspect parameters
    try:
        sig = inspect.signature(entry_fn)
    except (ValueError, TypeError):
        sig = None

    properties: dict[str, Any] = {}
    required: list[str] = []

    if sig is not None:
        # Resolve annotations via typing.get_type_hints() so that
        # `from __future__ import annotations` (PEP 563) in the user's file
        # doesn't break schema generation. get_type_hints() evaluates the
        # string annotations in the module's global namespace, returning real
        # types instead of raw strings.
        _entry_module = sys.modules.get(entry_fn.__module__)
        _globalns = vars(_entry_module) if _entry_module is not None else {}
        try:
            _hints: dict[str, Any] = typing.get_type_hints(
                entry_fn, globalns=_globalns, include_extras=False
            )
        except Exception:
            _hints = {}

        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'cls'
            if param_name in ("self", "cls"):
                continue
            # Skip private parameters
            if param_name.startswith("_"):
                continue

            # Resolve annotation (PEP 563 safe: prefer get_type_hints result)
            ann = _hints.get(param_name, param.annotation)

            # Skip any parameter annotated as ToolContext / ToolContext | None,
            # regardless of the parameter name.
            ann_str = str(ann)
            if ann is ToolContext or "ToolContext" in ann_str:
                continue

            prop_schema = type_to_json_schema(ann)

            # Add description from docstring if available
            if param_name in param_docs and param_docs[param_name]:
                prop_schema = dict(prop_schema)  # shallow copy
                prop_schema["description"] = param_docs[param_name]

            # Add default value to schema
            if param.default is not inspect.Parameter.empty:
                prop_schema = dict(prop_schema)
                prop_schema["default"] = param.default
            else:
                # No default → required (unless Optional)
                if not is_optional(ann):
                    required.append(param_name)

            properties[param_name] = prop_schema

    input_schema: _InputSchema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required

    # Build ToolSchema
    schema: dict[str, Any] = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": input_schema,
    }

    return tool_name, tool_description, schema

"""Generate plain-Markdown API reference for the lauren-ai-website.

The lauren-ai-website renders docs with react-markdown.  The canonical
``docs/reference/*.md`` files are comprehensive hand-written Markdown tables
that are great for MkDocs.  However, if they ever add mkdocstrings ``:::``
directives, react-markdown cannot interpret them.

This script uses griffe (the same parser mkdocstrings uses internally) to
extract docstrings from the ``lauren_ai`` package and write clean Markdown
files to ``docs/generated-reference/``.  Those files are committed to the
repo so the website's production build works without requiring Python.

Usage::

    python scripts/generate_api_docs.py

Run this script whenever public API docstrings change, then commit the output.

Requirements:
    griffe>=1.0   (available via ``pip install griffe``)
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import griffe
except ImportError:
    raise SystemExit(
        "griffe is required.  Install it with:\n    pip install griffe\n"
    ) from ImportError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "lauren_ai"
OUTPUT_DIR = ROOT / "docs" / "generated-reference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Page definitions
# ---------------------------------------------------------------------------
# Each page is a list of items.  A tuple ("heading text", "") emits a raw
# Markdown heading / intro paragraph.  A string "lauren_ai.Symbol" looks up
# the symbol in the griffe tree and renders it.

PAGES: dict[str, list[str | tuple[str, str]]] = {
    "agents.md": [
        (
            "# Agents\n\nDecorators and types for building AI agents.",
            "",
        ),
        ("## Decorators", ""),
        "lauren_ai.agent",
        "lauren_ai.use_tools",
        "lauren_ai.use_knowledge_sources",
        ("## Agent types", ""),
        "lauren_ai.AgentMeta",
        "lauren_ai.AgentContext",
        "lauren_ai.AgentResponse",
        ("## Runner", ""),
        "lauren_ai.AgentRunner",
        "lauren_ai.AgentRunnerBase",
    ],
    "module.md": [
        (
            "# Modules & Services\n\n"
            "DI-wiring helpers for integrating `lauren-ai` into a Lauren application.",
            "",
        ),
        "lauren_ai.LLMModule",
        "lauren_ai.AgentModule",
        "lauren_ai.LLMService",
    ],
    "config.md": [
        (
            "# Configuration\n\n"
            "Frozen dataclasses that configure the LLM provider and agent behaviour.",
            "",
        ),
        "lauren_ai.LLMConfig",
        "lauren_ai.AgentConfig",
    ],
    "tools.md": [
        (
            "# Tools\n\nThe `@tool()` decorator and runtime context.",
            "",
        ),
        "lauren_ai.tool",
        "lauren_ai.ToolContext",
        "lauren_ai.ToolResult",
    ],
    "guardrails.md": [
        (
            "# Guardrails\n\nContent safety filters for agent inputs and outputs.",
            "",
        ),
        ("## Decorators", ""),
        "lauren_ai.guardrail",
        "lauren_ai.use_guardrails",
        ("## Decision types", ""),
        "lauren_ai.GuardrailDecision",
        "lauren_ai.GuardrailContext",
        "lauren_ai.GuardrailViolated",
        "lauren_ai.InputGuardrail",
        "lauren_ai.OutputGuardrail",
        ("## Built-in guardrails", ""),
        "lauren_ai.TopicFilter",
        "lauren_ai.PIIRedactor",
        "lauren_ai.LengthFilter",
        "lauren_ai.PromptInjectionFilter",
        "lauren_ai.LLMGuardrail",
    ],
    "memory.md": [
        (
            "# Memory\n\nShort-term memory, conversation stores, vector stores, and user memory.",
            "",
        ),
        ("## Short-term memory", ""),
        "lauren_ai.ShortTermMemory",
        ("## Conversation store", ""),
        "lauren_ai.ConversationStore",
        "lauren_ai.InMemoryConversationStore",
        ("## Vector store", ""),
        "lauren_ai.InMemoryVectorStore",
        ("## User memory", ""),
        "lauren_ai.MemoryFact",
        "lauren_ai.UserMemoryStore",
        "lauren_ai.InMemoryUserMemoryStore",
        ("## `@remember` decorator", ""),
        "lauren_ai.remember",
    ],
    "cost.md": [
        (
            "# Cost & Rate Tracking\n\nToken budgets, cost estimation, and rate limiting.",
            "",
        ),
        ("## Pricing", ""),
        "lauren_ai.ModelPricing",
        "lauren_ai.CostEstimate",
        "lauren_ai.PricingTable",
        "lauren_ai.default_pricing_table",
        ("## Cost tracker", ""),
        "lauren_ai.CostTracker",
        "lauren_ai.CostSession",
        "lauren_ai.CostReport",
        ("## Budgets & limits", ""),
        "lauren_ai.TokenBudget",
        "lauren_ai.BudgetExceededError",
        "lauren_ai.RateLimiter",
        "lauren_ai.RateLimitExhaustedError",
    ],
    "signals.md": [
        (
            "# Signals\n\nObservable lifecycle events emitted by the agent runner.",
            "",
        ),
        "lauren_ai.SignalBus",
        ("## Event types", ""),
        "lauren_ai.ModelCallStarted",
        "lauren_ai.ModelCallComplete",
        "lauren_ai.ToolCallStarted",
        "lauren_ai.ToolCallComplete",
        "lauren_ai.AgentRunComplete",
    ],
    "teams.md": [
        (
            "# Agent Teams\n\nMulti-agent coordination via coordinator and collaborate modes.",
            "",
        ),
        ("## Decorator", ""),
        "lauren_ai.team",
        ("## Metadata", ""),
        "lauren_ai.TeamMeta",
        ("## Runner & result", ""),
        "lauren_ai.TeamRunner",
        "lauren_ai.TeamResult",
        ("## Events", ""),
        "lauren_ai.TeamWorkerStarted",
        "lauren_ai.TeamWorkerFinished",
        "lauren_ai.TeamCoordinatorDecision",
        "lauren_ai.TeamFinalAnswer",
    ],
    "transport.md": [
        (
            "# Transport & Multimodal\n\nCore message types exchanged with LLM providers.",
            "",
        ),
        ("## Messages & completions", ""),
        "lauren_ai.Message",
        "lauren_ai.Completion",
        "lauren_ai.CompletionChunk",
        ("## Usage & calls", ""),
        "lauren_ai.TokenUsage",
        "lauren_ai.ToolCall",
        "lauren_ai.ToolSchema",
        "lauren_ai.Embedding",
        ("## Structured output", ""),
        "lauren_ai.StructuredLLM",
        ("## Multimodal content", ""),
        "lauren_ai.ImageContent",
        "lauren_ai.AudioContent",
        "lauren_ai.DocumentContent",
    ],
    "tracing.md": [
        (
            "# Tracing\n\nObservability spans, exporters, and the `@traced` decorator.",
            "",
        ),
        ("## Decorator", ""),
        "lauren_ai.traced",
        ("## Span types", ""),
        "lauren_ai.SpanKind",
        "lauren_ai.Span",
        "lauren_ai.Trace",
        ("## Store & config", ""),
        "lauren_ai.TraceStore",
        "lauren_ai.TracingConfig",
        "lauren_ai.set_trace_store",
        "lauren_ai.get_trace_store",
        ("## Exporters", ""),
        "lauren_ai.TraceExporter",
        "lauren_ai.InMemoryTraceExporter",
        "lauren_ai.ConsoleTraceExporter",
        "lauren_ai.FileTraceExporter",
    ],
    "output-parsers.md": [
        (
            "# Output Parsers\n\nStructured extraction from LLM text responses.",
            "",
        ),
        "lauren_ai.StrOutputParser",
        "lauren_ai.JSONOutputParser",
        "lauren_ai.RegexParser",
        "lauren_ai.CommaSeparatedListParser",
        "lauren_ai.MarkdownCodeBlockParser",
        "lauren_ai.PydanticOutputParser",
        "lauren_ai.RetryOutputParser",
        "lauren_ai.MaxRetryError",
    ],
    "prompts.md": [
        (
            "# Prompt Templates\n\nReusable, composable prompt builders.",
            "",
        ),
        "lauren_ai.PromptTemplate",
        "lauren_ai.ChatPromptTemplate",
        "lauren_ai.FewShotPromptTemplate",
        "lauren_ai.FewShotExample",
    ],
    "router.md": [
        (
            "# Semantic Router\n\nIntent-based routing using embedding similarity.",
            "",
        ),
        "lauren_ai.SemanticRouter",
        "lauren_ai.Route",
        "lauren_ai.RouteMatch",
    ],
    "exceptions.md": [
        (
            "# Exceptions\n\nAll exception classes raised by `lauren-ai`.",
            "",
        ),
        ("## Base", ""),
        "lauren_ai.LaurenAIError",
        ("## Transport errors", ""),
        "lauren_ai.TransportError",
        ("## Agent errors", ""),
        "lauren_ai.AgentMaxTurnsError",
        "lauren_ai.AgentBudgetExceededError",
        "lauren_ai.AgentConfigError",
        ("## Tool errors", ""),
        "lauren_ai.ToolExecutionError",
        ("## Decorator errors", ""),
        "lauren_ai.DecoratorUsageError",
        ("## Parser errors", ""),
        "lauren_ai.OutputParserError",
        ("## Memory errors", ""),
        "lauren_ai.MemoryConfigError",
        ("## Tracing errors", ""),
        "lauren_ai.TracingError",
    ],
}

# ---------------------------------------------------------------------------
# Griffe loading
# ---------------------------------------------------------------------------


def _load_package() -> griffe.Module:
    # Use sphinx parser so :param:/:type:/:returns: directives are parsed
    # into structured sections rather than emitted as raw RST text.
    loader = griffe.GriffeLoader(docstring_parser=griffe.Parser.sphinx)
    return loader.load(SRC)


# ---------------------------------------------------------------------------
# Rendering helpers (identical to lauren-framework/scripts/generate_api_docs.py)
# ---------------------------------------------------------------------------


def _resolve(pkg: griffe.Module, dotted: str) -> griffe.Object | None:
    # Strip "lauren_ai." prefix — pkg IS the lauren_ai module
    if dotted.startswith("lauren_ai."):
        dotted = dotted[len("lauren_ai.") :]

    parts = dotted.split(".")
    obj: griffe.Object | None = pkg  # type: ignore[assignment]
    for part in parts:
        if obj is None:
            return None
        try:
            obj = obj.get_member(part)
            if isinstance(obj, griffe.Alias):
                import contextlib

                with contextlib.suppress(Exception):
                    obj = obj.target  # type: ignore[assignment]
        except Exception:
            return None
    return obj


def _fmt_annotation(ann: object | None) -> str:
    if ann is None:
        return ""
    try:
        return str(ann)
    except Exception:
        return ""


def _fmt_default(default: object | None) -> str:
    if default is None:
        return ""
    s = str(default)
    if s.startswith("<") and ">" in s:
        return ""
    return s


def _render_signature(obj: griffe.Function | griffe.Class) -> str:
    try:
        if isinstance(obj, griffe.Function):
            params = []
            for p in obj.parameters:
                ann = _fmt_annotation(p.annotation)
                default = _fmt_default(p.default)
                part = p.name
                if ann:
                    part += f": {ann}"
                if default:
                    part += f" = {default}"
                params.append(part)
            ret = _fmt_annotation(obj.returns)
            sig = f"def {obj.name}({', '.join(params)})"
            if ret:
                sig += f" -> {ret}"
            return f"```python\n{sig}\n```\n"
        elif isinstance(obj, griffe.Class):
            init = obj.members.get("__init__")
            if init and isinstance(init, griffe.Function):
                params = []
                for p in init.parameters:
                    if p.name == "self":
                        continue
                    ann = _fmt_annotation(p.annotation)
                    default = _fmt_default(p.default)
                    part = p.name
                    if ann:
                        part += f": {ann}"
                    if default:
                        part += f" = {default}"
                    params.append(part)
                sig = f"class {obj.name}({', '.join(params)})"
            else:
                sig = f"class {obj.name}"
            return f"```python\n{sig}\n```\n"
    except Exception:
        pass
    return ""


def _clean_rst_inline(text: str) -> str:
    """Convert RST inline cross-references to plain Markdown backtick syntax."""
    # :class:`~module.ClassName` → `ClassName`  (strip module prefix)
    text = re.sub(r":class:`~[^`]*\.([^`]+)`", r"`\1`", text)
    # :class:`ClassName` → `ClassName`
    text = re.sub(r":class:`([^`]+)`", r"`\1`", text)
    # :attr:`obj.attr` → `obj.attr`
    text = re.sub(r":attr:`([^`]+)`", r"`\1`", text)
    # :meth:`method` → `method()`
    text = re.sub(r":meth:`([^`]+)`", r"`\1()`", text)
    # :data:`CONST` → `CONST`
    text = re.sub(r":data:`([^`]+)`", r"`\1`", text)
    # :func:`fn` → `fn()`
    text = re.sub(r":func:`([^`]+)`", r"`\1()`", text)
    # :exc:`Error`
    text = re.sub(r":exc:`([^`]+)`", r"`\1`", text)
    # :mod:`module` → `module`
    text = re.sub(r":mod:`([^`]+)`", r"`\1`", text)
    # ``double_backtick`` → `single_backtick`
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    return text


def _render_docstring(obj: griffe.Object) -> str:
    """Render a docstring to clean Markdown using griffe's parsed sections."""
    if obj.docstring is None:
        return ""

    try:
        sections = obj.docstring.parsed
    except Exception:
        sections = []

    if not sections:
        raw = obj.docstring.value.strip()
        return (_clean_rst_inline(raw) + "\n\n") if raw else ""

    parts: list[str] = []
    for section in sections:
        kind_name: str = getattr(section.kind, "value", str(section.kind))
        val = section.value

        if kind_name == "text":
            text = _clean_rst_inline(str(val).strip())
            if text:
                parts.append(text + "\n\n")

        elif kind_name == "parameters":
            rows = []
            for param in val or []:
                name = f"`{param.name}`"
                ann = str(param.annotation) if param.annotation else ""
                type_cell = f"`{ann}`" if ann else ""
                desc = _clean_rst_inline(param.description.strip()) if param.description else ""
                rows.append(f"| {name} | {type_cell} | {desc} |")
            if rows:
                parts.append(
                    "**Parameters:**\n\n"
                    "| Name | Type | Description |\n|---|---|---|\n" + "\n".join(rows) + "\n\n"
                )

        elif kind_name == "returns":
            for ret in val or []:
                ann = str(ret.annotation) if ret.annotation else ""
                desc = _clean_rst_inline(ret.description.strip()) if ret.description else ""
                type_prefix = f"`{ann}` — " if ann else ""
                if desc or type_prefix:
                    parts.append(f"**Returns:** {type_prefix}{desc}\n\n")

        elif kind_name == "raises":
            rows = []
            for exc in val or []:
                exc_name = f"`{exc.annotation}`" if exc.annotation else ""
                desc = _clean_rst_inline(exc.description.strip()) if exc.description else ""
                rows.append(f"| {exc_name} | {desc} |")
            if rows:
                parts.append(
                    "**Raises:**\n\n"
                    "| Exception | Description |\n|---|---|\n" + "\n".join(rows) + "\n\n"
                )

        elif kind_name in ("examples", "example"):
            example_text = str(val).strip() if val else ""
            if example_text:
                parts.append(f"**Example:**\n\n{example_text}\n\n")

        # Other section kinds (notes, warnings, etc.) — render as plain text
        else:
            try:
                raw = str(val).strip()
                if raw:
                    parts.append(_clean_rst_inline(raw) + "\n\n")
            except Exception:
                pass

    return "".join(parts)


def _render_object(pkg: griffe.Module, dotted: str, heading_level: int = 3) -> str:
    obj = _resolve(pkg, dotted)
    if obj is None:
        print(f"  ⚠  {dotted}: not found — skipping")
        return f"> **`{dotted}`** — symbol not found in the installed package.\n\n"

    name = obj.name
    heading = "#" * heading_level
    parts: list[str] = [f"{heading} `{name}`\n\n"]

    if isinstance(obj, (griffe.Function, griffe.Class)):
        sig = _render_signature(obj)
        if sig:
            parts.append(sig + "\n")

    parts.append(_render_docstring(obj))

    # Render public methods for classes
    if isinstance(obj, griffe.Class):
        for member_name, member in obj.members.items():
            if member_name.startswith("_"):
                continue
            if not isinstance(member, griffe.Function):
                continue
            sub = "#" * (heading_level + 1)
            parts.append(f"{sub} `{name}.{member_name}`\n\n")
            sig = _render_signature(member)
            if sig:
                parts.append(sig + "\n")
            parts.append(_render_docstring(member))

    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate() -> None:
    print(f"Loading lauren_ai from {SRC} …")
    pkg = _load_package()
    print("Package loaded.  Generating reference pages …\n")

    for filename, entries in PAGES.items():
        out_path = OUTPUT_DIR / filename
        sections: list[str] = []

        for entry in entries:
            if isinstance(entry, tuple):
                heading_text, _ = entry
                sections.append(heading_text + "\n\n")
            else:
                print(f"  {entry}")
                sections.append(_render_object(pkg, entry, heading_level=3))

        out_path.write_text("".join(sections), encoding="utf-8")
        print(f"  → {out_path.relative_to(ROOT)}\n")

    print(f"Done.  Generated {len(PAGES)} files in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    generate()

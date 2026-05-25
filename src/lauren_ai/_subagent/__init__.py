"""Subagent runtime primitives for isolated child-agent execution."""

__all__ = [
    "BriefCompiler",
    "LlmCompiler",
    "PassThroughCompiler",
    "ReturnMode",
    "SubagentConfig",
    "SubagentPool",
    "SubagentTool",
    "TemplateCompiler",
]

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

from lauren_ai._agents import AGENT_META
from lauren_ai._config import AgentConfig
from lauren_ai._exceptions import AgentConfigError, OutputParserError
from lauren_ai._memory import ShortTermMemory
from lauren_ai._module import LLMService
from lauren_ai._signals import SubagentCompleted, SubagentStarted
from lauren_ai._tools import ToolContext, tool
from lauren_ai._transport import Completion, CompletionChunk, Message

_T = TypeVar("_T", bound=BaseModel)
_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class ReturnMode(StrEnum):
    """How a subagent's final text should be parsed into a typed result."""

    DIRECT_JSON = "direct_json"
    STRUCTURED_LLM = "structured_llm"


@runtime_checkable
class BriefCompiler(Protocol):
    """Strategy that produces the subagent's opening user message."""

    async def compile(self, tool_input: dict[str, Any], ctx: ToolContext) -> str:
        """Compile the subagent brief from tool input and the parent context."""
        ...


@dataclass(slots=True)
class PassThroughCompiler:
    """Use ``tool_input['task']`` verbatim as the subagent brief."""

    async def compile(self, tool_input: dict[str, Any], ctx: ToolContext) -> str:
        return str(tool_input.get("task", ""))


@dataclass(slots=True)
class TemplateCompiler:
    """Fill a simple ``{{ variable }}`` template from tool input and metadata."""

    template: str
    metadata_keys: tuple[str, ...] = ()

    async def compile(self, tool_input: dict[str, Any], ctx: ToolContext) -> str:
        values: dict[str, str] = {key: _stringify(value) for key, value in tool_input.items()}
        for key in self.metadata_keys:
            values[key] = _stringify(ctx.get_metadata(key, ""))
        return _TEMPLATE_RE.sub(lambda match: values.get(match.group(1), ""), self.template)


@dataclass(slots=True)
class LlmCompiler:
    """Use an LLM to summarize the parent context into a focused brief."""

    llm: Any
    window: int = 10

    async def compile(self, tool_input: dict[str, Any], ctx: ToolContext) -> str:
        parent_memory = getattr(ctx.agent_context, "memory", None)
        recent_messages = []
        if parent_memory is not None and hasattr(parent_memory, "messages"):
            recent_messages = list(parent_memory.messages())[-self.window :]
        history = "\n".join(_format_memory_message(message) for message in recent_messages) or "(no prior history)"
        prompt = (
            "Write a concise subagent brief from the parent agent context.\n\n"
            f"Tool input:\n{json.dumps(tool_input, default=str, indent=2)}\n\n"
            f"Parent metadata:\n{json.dumps(getattr(ctx.agent_context, 'metadata', {}), default=str, indent=2)}\n\n"
            f"Recent conversation:\n{history}\n"
        )
        completion = await self.llm.complete([Message.user(prompt)])
        return await _completion_to_text(completion)


@dataclass(slots=True)
class SubagentConfig:
    """Per-tool configuration controlling how the subagent run is executed."""

    max_turns: int | None = None
    max_cost_usd: float | None = None
    max_tokens_per_turn: int | None = None
    model: str | None = None
    brief_compiler: BriefCompiler = field(default_factory=PassThroughCompiler)
    return_mode: ReturnMode = ReturnMode.STRUCTURED_LLM
    extraction_model: str | None = None


class SubagentPool:
    """Run multiple subagent tasks concurrently using structured concurrency."""

    def __init__(self, tool_instance: Any, *, max_concurrent: int = 0) -> None:
        self._tool_instance = tool_instance
        self._max_concurrent = max_concurrent

    async def run_all(self, tasks: list[str], ctx: ToolContext) -> list[dict[str, Any]]:
        """Run all tasks and return results in input order."""
        if not tasks:
            return []

        results: list[dict[str, Any] | None] = [None] * len(tasks)
        semaphore = asyncio.Semaphore(self._max_concurrent) if self._max_concurrent > 0 else None

        async def _run_one(index: int, task: str) -> None:
            try:
                if semaphore is None:
                    result = await self._tool_instance.run(ctx, task)
                else:
                    async with semaphore:
                        result = await self._tool_instance.run(ctx, task)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc), "task": task}
            results[index] = cast(dict[str, Any], result)

        async with asyncio.TaskGroup() as task_group:
            for index, task in enumerate(tasks):
                task_group.create_task(_run_one(index, task))

        return [cast(dict[str, Any], result) for result in results]


def SubagentTool(
    *,
    subagent_cls: type,
    return_type: type[_T],
    name: str,
    description: str,
    config: SubagentConfig | None = None,
) -> type:
    """Create a class-form ``@tool()`` that runs *subagent_cls* with a blank slate."""

    resolved_config = config or SubagentConfig()
    class_name = "".join(part.capitalize() for part in re.split(r"[^a-zA-Z0-9]+", name) if part) + "SubagentTool"

    @tool(name=name, description=description)
    class _GeneratedSubagentTool:
        """Run a subagent and return a structured result.

        Args:
            task: The isolated task brief sent to the subagent.
        """

        def __init__(
            self,
            agent: Any,
            llm: LLMService,
        ) -> None:
            self._agent = agent
            self._llm = llm

        async def run(self, ctx: ToolContext, task: str) -> dict[str, Any]:
            """Run the configured subagent with fresh memory and typed output."""
            parent_ctx = ctx.agent_context
            parent_runner = ctx.runner
            if parent_runner is None:
                raise RuntimeError("SubagentTool requires ToolContext.runner to be populated by AgentRunner.")
            brief = await resolved_config.brief_compiler.compile({"task": task}, ctx)
            config_override = _build_config_override(self._agent, resolved_config)
            started = time.monotonic()
            success = False
            error_message: str | None = None

            await _emit_signal(
                getattr(parent_ctx, "signals", None),
                SubagentStarted(
                    parent_agent_name=_parent_agent_name(parent_ctx),
                    subagent_name=_agent_name(self._agent),
                    parent_run_id=getattr(parent_ctx, "agent_run_id", ""),
                    conversation_id=getattr(parent_ctx, "conversation_id", None),
                    brief_length_chars=len(brief),
                ),
            )

            try:
                response = await parent_runner.run(
                    self._agent,
                    brief,
                    request=getattr(parent_ctx, "request", None),
                    execution_context=getattr(parent_ctx, "execution_context", None),
                    memory=ShortTermMemory(),
                    config_override=config_override,
                    model_override=resolved_config.model,
                )
                result = await _parse_return(
                    response.content,
                    return_type=return_type,
                    mode=resolved_config.return_mode,
                    llm=self._llm,
                    extraction_model=resolved_config.extraction_model,
                )
                success = True
                return result.model_dump()
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                return {
                    "error": error_message,
                    "subagent": _agent_name(self._agent),
                    "task": task,
                }
            finally:
                await _emit_signal(
                    getattr(parent_ctx, "signals", None),
                    SubagentCompleted(
                        parent_agent_name=_parent_agent_name(parent_ctx),
                        subagent_name=_agent_name(self._agent),
                        parent_run_id=getattr(parent_ctx, "agent_run_id", ""),
                        conversation_id=getattr(parent_ctx, "conversation_id", None),
                        elapsed_ms=(time.monotonic() - started) * 1000,
                        success=success,
                        error=error_message,
                    ),
                )

    _GeneratedSubagentTool.__name__ = class_name
    _GeneratedSubagentTool.__qualname__ = class_name
    _GeneratedSubagentTool.__init__.__annotations__["agent"] = subagent_cls
    return _GeneratedSubagentTool


def _build_config_override(agent: Any, config: SubagentConfig) -> AgentConfig | None:
    """Return a full per-run config override when the subagent config changes it."""
    agent_cls = type(agent) if not isinstance(agent, type) else agent
    meta = getattr(agent_cls, AGENT_META, None)
    if meta is None:
        raise AgentConfigError(
            f"{agent_cls!r} is not decorated with @agent(). Only agent classes can be used as subagents.",
            agent_class=agent_cls,
        )
    base_config = meta.config
    overrides: dict[str, Any] = {}
    if config.max_turns is not None:
        overrides["max_turns"] = config.max_turns
    if config.max_cost_usd is not None:
        overrides["max_cost_usd"] = config.max_cost_usd
    if config.max_tokens_per_turn is not None:
        overrides["max_tokens_per_turn"] = config.max_tokens_per_turn
    if not overrides:
        return None
    return replace(base_config, **overrides)


async def _parse_return(
    content: str,
    *,
    return_type: type[_T],
    mode: ReturnMode,
    llm: LLMService,
    extraction_model: str | None,
) -> _T:
    """Parse subagent text into the configured structured result type."""
    if mode is ReturnMode.DIRECT_JSON:
        return return_type.model_validate_json(_extract_json_block(content))

    structured_llm = llm
    if extraction_model is not None:
        structured_llm = LLMService(
            transport=llm._transport,
            config=replace(llm._config, model=extraction_model),
        )
    return await structured_llm.with_structured_output(return_type).complete([Message.user(content)])


def _extract_json_block(content: str) -> str:
    """Extract a JSON object from the model output."""
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
    if fenced is not None:
        return fenced.group(1)

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise OutputParserError("No JSON object found in subagent output.", raw_output=content)
    return content[start : end + 1]


async def _completion_to_text(result: Completion | Any) -> str:
    """Convert a completion or completion stream into plain text."""
    if isinstance(result, Completion):
        return result.content

    chunks: list[str] = []
    async for chunk in cast(AsyncIterator[CompletionChunk], result):
        if isinstance(chunk, CompletionChunk) and chunk.delta:
            chunks.append(chunk.delta)
    return "".join(chunks)


def _format_memory_message(message: Any) -> str:
    role = getattr(message, "role", "unknown")
    content = getattr(message, "content", "")
    return f"{role}: {_stringify(content)}"


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _parent_agent_name(parent_ctx: Any) -> str:
    if parent_ctx is None:
        return ""
    return getattr(parent_ctx, "agent_name", "")


def _agent_name(agent: Any) -> str:
    agent_cls = type(agent) if not isinstance(agent, type) else agent
    meta = getattr(agent_cls, AGENT_META, None)
    if meta is not None and meta.name:
        return meta.name
    return agent_cls.__name__


async def _emit_signal(signals: Any, event: Any) -> None:
    if signals is None:
        return
    await signals.emit(event)

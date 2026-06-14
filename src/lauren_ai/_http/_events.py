"""Agent lifecycle events for wire serialisation (PRD 11)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentHttpTokenEvent:
    """A single streamed text fragment from the LLM."""

    type: Literal["token"] = field(default="token", init=False)
    content: str = ""

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "content": self.content})


@dataclass
class AgentToolStartEvent:
    """Emitted when the agent invokes a tool (before execution)."""

    type: Literal["tool_start"] = field(default="tool_start", init=False)
    tool_name: str = ""
    tool_use_id: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type,
                "tool_name": self.tool_name,
                "tool_use_id": self.tool_use_id,
                "input": self.input,
            }
        )


@dataclass
class AgentToolResultEvent:
    """Emitted after a tool call completes (success or error)."""

    type: Literal["tool_result"] = field(default="tool_result", init=False)
    tool_name: str = ""
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False
    duration_ms: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type,
                "tool_name": self.tool_name,
                "tool_use_id": self.tool_use_id,
                "content": self.content,
                "is_error": self.is_error,
                "duration_ms": self.duration_ms,
            }
        )


@dataclass
class AgentToolProgressEvent:
    """Emitted when an MCP tool sends a progress notification."""

    type: Literal["tool_progress"] = field(default="tool_progress", init=False)
    tool_name: str = ""
    tool_use_id: str = ""
    progress: float = 0.0
    total: float | None = None
    message: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "type": self.type,
            "tool_name": self.tool_name,
            "tool_use_id": self.tool_use_id,
            "progress": self.progress,
        }
        if self.total is not None:
            payload["total"] = self.total
        if self.message is not None:
            payload["message"] = self.message
        return json.dumps(payload)


@dataclass
class AgentDoneEvent:
    """Final event — the completed agent response."""

    type: Literal["done"] = field(default="done", init=False)
    content: str = ""
    turn_count: int = 0
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type,
                "content": self.content,
                "turn_count": self.turn_count,
                "stop_reason": self.stop_reason,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }
        )


# Discriminated union — every possible event yielded by AgentStreamService
AgentEvent = AgentHttpTokenEvent | AgentToolStartEvent | AgentToolResultEvent | AgentToolProgressEvent | AgentDoneEvent

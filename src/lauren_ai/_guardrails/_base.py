from __future__ import annotations

"""Base types for the guardrails system."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class GuardrailDecision:
    """Result of a guardrail check."""
    action: Literal["pass", "block", "modify"]
    modified_content: str | None = None
    violation: str | None = None
    guardrail_name: str = ""


@dataclass
class GuardrailContext:
    """Per-call context passed to each guardrail check."""
    agent_name: str = ""
    conversation_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailViolated:
    """Signal emitted when a guardrail fires."""
    guardrail_name: str = ""
    phase: Literal["input", "output"] = "input"
    action: Literal["block", "modify"] = "block"
    violation: str | None = None
    agent_name: str = ""
    conversation_id: str | None = None


@runtime_checkable
class InputGuardrail(Protocol):
    """Protocol for input guardrails -- check messages before LLM call."""

    async def check(self, message: str, context: GuardrailContext) -> GuardrailDecision: ...


@runtime_checkable
class OutputGuardrail(Protocol):
    """Protocol for output guardrails -- check/modify LLM responses."""

    async def check(self, response: str, context: GuardrailContext) -> GuardrailDecision: ...

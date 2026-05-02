from __future__ import annotations

"""@guardrail() decorator for @agent() classes."""

from dataclasses import dataclass, field
from typing import Any, TypeVar

from lauren_ai._exceptions import DecoratorUsageError

GUARDRAIL_META = "__lauren_ai_guardrail__"
C = TypeVar("C", bound=type)


@dataclass
class GuardrailMeta:
    """Attached to @agent() class by @guardrail()."""
    input_guardrails: list[Any] = field(default_factory=list)
    output_guardrails: list[Any] = field(default_factory=list)


def guardrail(
    *args: Any,
    input: list[Any] | None = None,
    output: list[Any] | None = None,
) -> Any:
    """Attach input/output safety guardrails to an @agent() class.

    Must be applied BELOW @agent()::

        @agent(model="claude-haiku-4-5")
        @guardrail(
            input=[TopicFilter(allowed_topics=["cooking"])],
            output=[PIIRedactor(entities=["EMAIL"])],
        )
        class CookingAssistant: ...

    Input guardrails run before each LLM call; if any block, the LLM is
    never called and the violation message is returned directly.

    Output guardrails run after the LLM response; they may block or modify
    the response before it reaches the caller.
    """
    if args:
        raise DecoratorUsageError(
            "@guardrail must be called with parentheses: @guardrail(input=[...], output=[...])"
        )

    def _apply(cls: C) -> C:
        meta = GuardrailMeta(
            input_guardrails=list(input or []),
            output_guardrails=list(output or []),
        )
        setattr(cls, GUARDRAIL_META, meta)
        return cls

    return _apply

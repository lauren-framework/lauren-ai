from __future__ import annotations

from lauren_ai._guardrails._base import (
    GuardrailContext,
    GuardrailDecision,
    GuardrailViolated,
    InputGuardrail,
    OutputGuardrail,
)
from lauren_ai._guardrails._builtin import (
    LengthFilter,
    PIIRedactor,
    PromptInjectionFilter,
    TopicFilter,
)
from lauren_ai._guardrails._decorator import (
    GUARDRAIL_CLASS_META,
    USE_GUARDRAILS_META,
    GuardrailClassMeta,
    UseGuardrailsMeta,
    guardrail,
    use_guardrails,
)
from lauren_ai._guardrails._llm import LLMGuardrail

__all__ = [
    # Base types
    "GuardrailDecision",
    "GuardrailContext",
    "InputGuardrail",
    "OutputGuardrail",
    "GuardrailViolated",
    # Class decorator — makes a guardrail DI-injectable
    "guardrail",
    "GuardrailClassMeta",
    "GUARDRAIL_CLASS_META",
    # Agent decorator — attaches guardrail instances to an agent
    "use_guardrails",
    "UseGuardrailsMeta",
    "USE_GUARDRAILS_META",
    # Built-in guardrails
    "TopicFilter",
    "PIIRedactor",
    "LengthFilter",
    "PromptInjectionFilter",
    "LLMGuardrail",
]

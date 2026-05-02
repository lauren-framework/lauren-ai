from __future__ import annotations

"""Built-in guardrail implementations."""

import re
from typing import Any

from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision


class TopicFilter:
    """Block messages not related to allowed topics using keyword/pattern matching.

    For production use, pass embed_fn for embedding-based similarity.
    Without embed_fn, uses simple keyword matching.

    Usage::

        guard = TopicFilter(
            allowed_topics=["cooking", "recipes", "food"],
            violation_message="I only discuss cooking.",
        )
    """

    def __init__(
        self,
        allowed_topics: list[str],
        violation_message: str = "I can only help with allowed topics.",
        embed_fn: Any = None,
        min_similarity: float = 0.6,
    ) -> None:
        self._topics = [t.lower() for t in allowed_topics]
        self._violation_message = violation_message
        self._embed_fn = embed_fn
        self._min_similarity = min_similarity

    async def check(self, message: str, context: GuardrailContext) -> GuardrailDecision:
        if self._embed_fn is not None:
            return await self._embedding_check(message, context)
        return self._keyword_check(message, context)

    def _keyword_check(self, message: str, context: GuardrailContext) -> GuardrailDecision:
        msg_lower = message.lower()
        for topic in self._topics:
            if topic in msg_lower:
                return GuardrailDecision(
                    action="pass",
                    guardrail_name=type(self).__name__,
                )
        return GuardrailDecision(
            action="block",
            violation=self._violation_message,
            guardrail_name=type(self).__name__,
        )

    async def _embedding_check(self, message: str, context: GuardrailContext) -> GuardrailDecision:
        import math
        results = await self._embed_fn([message] + self._topics)
        query_vec = results[0].vector if hasattr(results[0], "vector") else results[0]

        best_score = 0.0
        for r in results[1:]:
            topic_vec = r.vector if hasattr(r, "vector") else r
            # cosine similarity
            dot = sum(a * b for a, b in zip(query_vec, topic_vec))
            mag_a = math.sqrt(sum(x * x for x in query_vec))
            mag_b = math.sqrt(sum(x * x for x in topic_vec))
            if mag_a > 0 and mag_b > 0:
                score = dot / (mag_a * mag_b)
                best_score = max(best_score, score)

        if best_score >= self._min_similarity:
            return GuardrailDecision(action="pass", guardrail_name=type(self).__name__)
        return GuardrailDecision(
            action="block",
            violation=self._violation_message,
            guardrail_name=type(self).__name__,
        )


class PIIRedactor:
    """Redact PII patterns from LLM outputs.

    Uses regex patterns for EMAIL, PHONE, SSN, and CREDIT_CARD.

    Usage::

        guard = PIIRedactor(entities=["EMAIL", "PHONE"], replacement="[REDACTED]")
    """

    PATTERNS: dict[str, str] = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "PHONE": r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "CREDIT_CARD": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }

    def __init__(
        self,
        entities: list[str] | None = None,
        replacement: str = "[REDACTED]",
    ) -> None:
        self._entities = entities or list(self.PATTERNS.keys())
        self._replacement = replacement
        self._compiled = [
            re.compile(self.PATTERNS[e])
            for e in self._entities
            if e in self.PATTERNS
        ]

    async def check(self, response: str, context: GuardrailContext) -> GuardrailDecision:
        modified = response
        found_any = False
        for pattern in self._compiled:
            new, count = pattern.subn(self._replacement, modified)
            if count > 0:
                found_any = True
                modified = new

        if found_any:
            return GuardrailDecision(
                action="modify",
                modified_content=modified,
                violation="PII detected and redacted.",
                guardrail_name=type(self).__name__,
            )
        return GuardrailDecision(action="pass", guardrail_name=type(self).__name__)


class LengthFilter:
    """Block messages outside min/max length limits.

    Usage::

        guard = LengthFilter(min_chars=1, max_chars=2000)
    """

    def __init__(
        self,
        min_chars: int | None = None,
        max_chars: int | None = None,
        violation_message: str | None = None,
    ) -> None:
        self._min = min_chars
        self._max = max_chars
        self._violation_message = violation_message

    async def check(self, text: str, context: GuardrailContext) -> GuardrailDecision:
        if self._min is not None and len(text) < self._min:
            return GuardrailDecision(
                action="block",
                violation=self._violation_message or f"Message too short (min {self._min} chars).",
                guardrail_name=type(self).__name__,
            )
        if self._max is not None and len(text) > self._max:
            return GuardrailDecision(
                action="block",
                violation=self._violation_message or f"Message too long (max {self._max} chars).",
                guardrail_name=type(self).__name__,
            )
        return GuardrailDecision(action="pass", guardrail_name=type(self).__name__)


class PromptInjectionFilter:
    """Detect common prompt injection patterns in user input.

    Usage::

        guard = PromptInjectionFilter(violation_message="Prompt injection detected.")
    """

    INJECTION_PATTERNS = [
        r'ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions',
        r'disregard\s+(?:all\s+)?(?:previous|prior)\s+instructions',
        r'forget\s+(?:all\s+)?(?:previous|prior)\s+instructions',
        r'you\s+are\s+now\s+(?:a\s+)?(?:different|new)',
        r'act\s+as\s+(?:a\s+)?(?:different|new|evil|dan)',
        r'jailbreak',
        r'do\s+anything\s+now',
        r'system\s*:\s*you\s+are',
        r'\[system\]',
        r'<\|im_start\|>',
        r'###\s*instruction',
    ]

    def __init__(self, violation_message: str = "Potential prompt injection detected.") -> None:
        self._violation_message = violation_message
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    async def check(self, message: str, context: GuardrailContext) -> GuardrailDecision:
        for pattern in self._compiled:
            if pattern.search(message):
                return GuardrailDecision(
                    action="block",
                    violation=self._violation_message,
                    guardrail_name=type(self).__name__,
                )
        return GuardrailDecision(action="pass", guardrail_name=type(self).__name__)

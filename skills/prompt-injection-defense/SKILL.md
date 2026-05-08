---
name: prompt-injection-defense
description: Detects and blocks prompt injection attempts in user input before they reach the LLM. Use when agents accept free-text user input, when handling untrusted data in RAG pipelines, or when compliance requires logging and blocking jailbreak attempts.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Input Sanitization for Prompt Injection Defense

## Built-in: `PromptInjectionFilter`

Lauren-AI ships `PromptInjectionFilter` in `lauren_ai._guardrails._builtin`.
It detects the most common injection patterns with compiled regexes:

```python
# NOTE: Do NOT add `from __future__ import annotations` to tool files.
from lauren_ai import agent, use_guardrails, PromptInjectionFilter

@agent(model="claude-opus-4-6", system="You are a helpful assistant.")
@use_guardrails(input=[PromptInjectionFilter()])
class SafeAgent: ...
```

Built-in patterns cover: `ignore previous instructions`, `jailbreak`, `DAN`,
`[system]`, `###instruction`, `<|im_start|>`, `system: you are`, and more.

## Custom injection guard with extended patterns

```python
import re
from lauren_ai import guardrail
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision

INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"forget\s+(everything|your\s+instructions)",
    r"you\s+are\s+now\s+(?:a\s+)?(?!helpful)",
    r"disregard\s+(?:your\s+)?(?:previous|prior|all)\s+",
    r"system\s*prompt\s*:\s*",
    r"<\s*/?(?:system|instruction)\s*>",
    r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+)?(?:different|another|evil|jailbreak)",
]

@guardrail(kind="input")
class PromptInjectionGuard:
    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
        for pattern in self._patterns:
            if pattern.search(message):
                return GuardrailDecision(
                    action="block",
                    violation="Potential prompt injection detected.",
                    guardrail_name="PromptInjectionGuard",
                )
        return GuardrailDecision(action="pass", guardrail_name="PromptInjectionGuard")
```

## RAG pipeline defense

When user-provided content is injected into prompts (e.g. from retrieved
documents), sanitize the document text before embedding it in the prompt:

```python
def sanitize_retrieved_text(text: str) -> str:
    """Strip HTML/XML-style instruction tags from retrieved text."""
    import re
    # Remove <system>, <instruction>, <|...|> style tags
    text = re.sub(r'<[^>]{0,100}>', '', text)
    text = re.sub(r'<\|[^|]{0,100}\|>', '', text)
    return text.strip()
```

## Logging blocked attempts

Use `GuardrailViolated` signal or wrap in a custom guard that logs to your audit
trail:

```python
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision
import logging

logger = logging.getLogger("security.prompt_injection")


@guardrail(kind="input")
class AuditingInjectionGuard:
    def __init__(self):
        self._inner = PromptInjectionFilter()

    async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
        decision = await self._inner.check(message, ctx)
        if decision.action == "block":
            logger.warning(
                "Prompt injection blocked",
                extra={
                    "user_id": ctx.user_id,
                    "agent": ctx.agent_name,
                    "message_preview": message[:200],
                },
            )
        return decision
```

## Decorator placement

Input guardrails must be in the `input=` list of `@use_guardrails()`:

```python
@agent(model="claude-opus-4-6")
@use_guardrails(input=[PromptInjectionGuard()])
class ProtectedAgent: ...
```

## Pitfalls

- Regex-based detection has false positives (e.g. "ignore previous chapters" in
  a book summary request).  Tune patterns or add allow-listing for known-good phrases.
- No guardrail is 100% effective — defense in depth is essential.  Combine with:
  - A fixed, non-modifiable system prompt.
  - Output guardrails to catch any slip-through.
  - Least-privilege tool access.
- Adversarial prompts in retrieved documents (indirect injection) are harder to
  catch — always sanitize retrieved text before injection.

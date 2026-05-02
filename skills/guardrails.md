# Guardrails

Guardrails filter agent inputs and outputs for content safety, privacy
compliance, and length limits.

---

## Applying guardrails

```python
from lauren_ai import agent, guardrail, use_tools, PromptInjectionFilter, PIIRedactor, LengthFilter

@agent(model="openai/gpt-4o-mini")
@guardrail(
    input=[PromptInjectionFilter(), PIIRedactor()],
    output=[LengthFilter(max_chars=8000)],
)
@use_tools(my_tool)
class SafeAgent: ...
```

`@guardrail()` must be stacked **between** `@agent()` (above) and `@use_tools()`
(below).

---

## Built-in guardrails

### `PromptInjectionFilter()`

Blocks jailbreak and prompt-override attempts.  Detects patterns like:
"ignore previous instructions", "you are now DAN", "repeat after me", etc.

```python
from lauren_ai import PromptInjectionFilter

guard = PromptInjectionFilter()
# Direction: input
```

### `PIIRedactor()`

Redacts sensitive personally identifiable information before the text reaches
the LLM.  Targets: email addresses, US phone numbers, SSNs, credit card numbers.

```python
from lauren_ai import PIIRedactor

guard = PIIRedactor()
# Direction: input (recommended); can also be used as output
```

Redacted values are replaced with `[REDACTED]` tokens.

### `LengthFilter(max_chars=N)`

Truncates output to at most `N` characters.

```python
from lauren_ai import LengthFilter

guard = LengthFilter(max_chars=4000)
# Direction: output
```

### `TopicFilter(blocked=[...])`

Blocks messages that contain any of the listed topic keywords (case-insensitive).

```python
from lauren_ai import TopicFilter

guard = TopicFilter(blocked=["competitor_name", "lawsuit", "confidential"])
# Direction: input or output
```

### `LLMGuardrail(policy="...")`

Uses a separate LLM call to evaluate content against a policy.  More flexible
but slower than rule-based guards.

```python
from lauren_ai import LLMGuardrail

guard = LLMGuardrail(
    policy="Block any request that asks for medical diagnosis or treatment advice.",
    model="openai/gpt-4o-mini",  # Optional — defaults to agent model
)
# Direction: input or output
```

---

## Custom guardrail

Implement the `InputGuardrail` or `OutputGuardrail` protocol:

```python
from lauren_ai import InputGuardrail, OutputGuardrail, GuardrailContext, GuardrailDecision

class KeywordFilter(InputGuardrail):
    def __init__(self, blocked_words: list[str]) -> None:
        self._blocked = [w.lower() for w in blocked_words]

    async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
        for word in self._blocked:
            if word in text.lower():
                return GuardrailDecision.block(
                    f"Message contains blocked keyword: {word!r}"
                )
        return GuardrailDecision.allow()


class SanitiseOutput(OutputGuardrail):
    async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
        sanitised = text.replace("<script>", "").replace("</script>", "")
        if sanitised != text:
            # Return modified text via allow() with replacement
            return GuardrailDecision.allow(replacement=sanitised)
        return GuardrailDecision.allow()
```

Then use in `@guardrail()`:

```python
@guardrail(
    input=[KeywordFilter(blocked_words=["forbidden", "secret"])],
    output=[SanitiseOutput()],
)
```

---

## `GuardrailDecision` API

```python
GuardrailDecision.allow()                           # Pass through unchanged
GuardrailDecision.allow(replacement="clean text")   # Pass through with replacement
GuardrailDecision.block("Reason for blocking")      # Raises GuardrailViolated
```

When a guard returns `block(...)`, `GuardrailViolated` is raised.  The agent
runner catches this and returns an error to the caller.

---

## Stacking multiple guardrails

All guards in a list are checked in order.  The first `block` decision short-circuits
the rest:

```python
@guardrail(
    input=[
        PromptInjectionFilter(),   # checked first
        PIIRedactor(),             # checked if injection guard passes
        TopicFilter(blocked=["lawsuit"]),
    ],
    output=[
        LengthFilter(max_chars=6000),
    ],
)
```

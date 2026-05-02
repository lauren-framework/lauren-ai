# Guardrails & Content Safety

`lauren-ai` provides a composable guardrails system in `lauren_ai._guardrails`.
Guardrails are callables that inspect or modify text before it is sent to an LLM
(input guardrails) or after the model responds (output guardrails).

## Core types

```python
from lauren_ai._guardrails import (
    GuardrailDecision,
    GuardrailContext,
    InputGuardrail,
    OutputGuardrail,
)
```

### GuardrailDecision

Every guardrail `check()` call returns a `GuardrailDecision`:

| Field | Type | Meaning |
|---|---|---|
| `action` | `"pass"` / `"block"` / `"modify"` | What to do with the content |
| `modified_content` | `str \| None` | Replacement content when `action="modify"` |
| `violation` | `str \| None` | Human-readable reason for block or modify |
| `guardrail_name` | `str` | Class name of the guardrail that fired |

### GuardrailContext

Runtime context passed to every `check()` call:

```python
ctx = GuardrailContext(
    agent_name="MyAgent",
    conversation_id="conv-42",
    user_id="user-1",
    metadata={"source": "api"},
)
```

## @guardrail() decorator

Apply `@guardrail()` **below** `@agent()` to attach safety checks to an agent
class.  Input guardrails run before the LLM is called; output guardrails run
after:

```python
from lauren_ai._agents import agent
from lauren_ai._guardrails import guardrail, TopicFilter, PIIRedactor

@agent(model="claude-haiku-4-5")
@guardrail(
    input=[TopicFilter(allowed_topics=["cooking", "recipes", "food"])],
    output=[PIIRedactor(entities=["EMAIL", "PHONE"])],
)
class CookingAssistant:
    """A cooking assistant that only discusses food topics."""
```

`@guardrail` must be called with parentheses.  Using the bare form raises
`DecoratorUsageError`.

## Built-in guardrails

### TopicFilter

Blocks messages not matching a list of allowed topics.  By default uses
keyword matching; pass `embed_fn` for embedding-based similarity:

```python
guard = TopicFilter(
    allowed_topics=["cooking", "recipes", "food"],
    violation_message="I can only help with cooking topics.",
    # embed_fn=my_embed_function,  # optional
    # min_similarity=0.6,          # used with embed_fn
)
```

Keyword matching is case-insensitive and checks whether any topic string appears
literally in the message.  For semantic matching, supply an `embed_fn` that
accepts a list of strings and returns objects with a `.vector` attribute (or
plain lists).

### PIIRedactor

Redacts personally-identifiable information from LLM responses using regex
patterns.  Returns `action="modify"` with the cleaned text when PII is found,
or `action="pass"` when the text is clean.

```python
guard = PIIRedactor(
    entities=["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS"],
    replacement="[REDACTED]",
)
```

Available entity types: `EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`, `IP_ADDRESS`.
Omit `entities` to redact all of them.

### LengthFilter

Blocks text outside configurable character length bounds:

```python
# Input guard: reject empty messages and very long inputs
guard = LengthFilter(min_chars=1, max_chars=2000)

# Output guard: reject unexpectedly short responses
guard = LengthFilter(min_chars=20)
```

### PromptInjectionFilter

Detects common prompt injection patterns in user input:

```python
guard = PromptInjectionFilter(
    violation_message="Potential prompt injection detected.",
)
```

Detected patterns include:
- "ignore all previous instructions"
- "disregard prior instructions"
- "jailbreak"
- "act as [DAN/evil/different]"
- System-prompt injection markers (`[system]`, `<|im_start|>`, `### Instruction`)

### LLMGuardrail

Uses a secondary LLM call to judge whether content is safe.  The `prompt` must
contain `{content}` which is replaced with the text being evaluated:

```python
from lauren_ai._guardrails import LLMGuardrail

guard = LLMGuardrail(
    llm=llm_service,
    prompt=(
        "Is the following message harmful, offensive, or inappropriate?\n\n"
        "{content}\n\n"
        "Answer YES or NO only."
    ),
    block_if="YES",
    violation_message="Content blocked by safety classifier.",
)
```

The `block_if` comparison is case-insensitive.  The secondary LLM call is made
with the same `LLMService` you pass in; use a cheap, fast model.

## Writing a custom guardrail

Any object with a `check(text, context) -> GuardrailDecision` coroutine method
satisfies the `InputGuardrail` / `OutputGuardrail` protocol:

```python
from lauren_ai._guardrails import GuardrailDecision, GuardrailContext

class ProfanityFilter:
    WORDS = {"badword1", "badword2"}

    async def check(self, message: str, context: GuardrailContext) -> GuardrailDecision:
        for word in self.WORDS:
            if word in message.lower():
                return GuardrailDecision(
                    action="block",
                    violation="Profanity detected.",
                    guardrail_name=type(self).__name__,
                )
        return GuardrailDecision(action="pass", guardrail_name=type(self).__name__)
```

## GuardrailViolated signal

`GuardrailViolated` is a signal dataclass that can be emitted on a `SignalBus`
when a guardrail fires, enabling centralized audit logging:

```python
from lauren_ai._guardrails import GuardrailViolated
from lauren_ai._signals import SignalBus

bus = SignalBus()

@bus.on(GuardrailViolated)
async def log_violation(event: GuardrailViolated) -> None:
    print(
        f"[{event.phase}] {event.guardrail_name} {event.action}: "
        f"{event.violation} (agent={event.agent_name})"
    )
```

## Error types

| Error | Module | Raised when |
|---|---|---|
| `DecoratorUsageError` | `_exceptions` | `@guardrail` used without parentheses |

Guardrail violations are expressed as `GuardrailDecision(action="block")` values,
not exceptions — the caller (typically the agent runner) decides how to handle
them.

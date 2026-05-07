# Guardrails & Content Safety

`lauren-ai` provides a composable guardrails system in `lauren_ai._guardrails`.
Guardrails are callables that inspect or modify text before it is sent to an LLM
(input guardrails) or after the model responds (output guardrails).

Two decorators are provided:

| Decorator | Purpose |
|---|---|
| `@guardrail()` | **Class decorator** — marks a class as a DI-injectable guardrail provider.  Analogous to `@injectable()`. |
| `@use_guardrails()` | **Agent decorator** — attaches pre-built guardrail instances to an `@agent()` class.  Analogous to `@use_guards()`. |

---

## Core types

```python
from lauren_ai import (
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

---

## @use_guardrails() — attach instances to an agent

Apply `@use_guardrails()` **below** `@agent()` to attach safety checks to an
agent class.  Input guardrails run before the LLM is called; output guardrails
run after:

```python
from lauren_ai import agent, use_guardrails, TopicFilter, PIIRedactor

@agent(model="claude-haiku-4-5")
@use_guardrails(
    input=[TopicFilter(allowed_topics=["cooking", "recipes", "food"])],
    output=[PIIRedactor(entities=["EMAIL", "PHONE"])],
)
class CookingAssistant:
    """A cooking assistant that only discusses food topics."""
```

`None` entries are silently dropped, enabling conditional selection:

```python
@agent(model="claude-opus-4-6")
@use_guardrails(
    input=[
        PromptInjectionFilter(),
        TopicFilter(allowed_topics=allowed) if allowed else None,
    ],
)
class DynamicAgent: ...
```

`@use_guardrails` must be called with parentheses.  Using the bare form raises
`DecoratorUsageError`.

---

## @guardrail() — DI-injectable guardrail class

Use `@guardrail()` to mark a class as a DI provider.  This registers it with
the Lauren DI container (via `@injectable()`) so it can be resolved, injected,
and lifecycle-managed by the framework — exactly the same as `@injectable()`,
but also stamps the class with `GUARDRAIL_CLASS_META` so the system knows it is
a guardrail implementation.

```python
from lauren_ai import guardrail, GuardrailDecision, GuardrailContext

@guardrail(kind="input")
class ProfanityFilter:
    """Block messages containing profanity.

    DI-injectable: resolved as a SINGLETON by the Lauren DI container.
    """

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

`kind` hints at the intended position (`"input"` / `"output"` / `"any"`); it
does not affect runtime behaviour but aids static analysis and documentation.

### Wiring DI guardrails into agents

Because `@guardrail()` classes are registered as DI singletons, you can inject
them into a wiring class that attaches them to agents at startup — the same
pattern used for delegation tools:

```python
# guardrail_wiring.py — HAS from __future__ import annotations
from __future__ import annotations

import logging
from lauren import Scope, injectable
from lauren_ai import USE_GUARDRAILS_META, UseGuardrailsMeta

from .filters import ProfanityFilter
from .my_agent import MyAgent

logger = logging.getLogger(__name__)


@injectable(scope=Scope.SINGLETON)
class GuardrailWiring:
    """Wires DI-resolved guardrails into agent metadata at startup."""

    def __init__(
        self,
        profanity_filter: ProfanityFilter,
        my_agent: MyAgent,
    ) -> None:
        meta: UseGuardrailsMeta = getattr(my_agent, USE_GUARDRAILS_META, None)
        if meta is None:
            meta = UseGuardrailsMeta()
            setattr(my_agent, USE_GUARDRAILS_META, meta)
        meta.input_guardrails.append(profanity_filter)
        logger.debug("GuardrailWiring: ProfanityFilter wired into MyAgent")
```

Register `GuardrailWiring` in your module's `providers=[...]` list alongside
the guardrail class:

```python
@module(
    imports=[LLMProvider, AgentProvider],
    providers=[ProfanityFilter, GuardrailWiring],
)
class AppModule: ...
```

---

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
literally in the message.

### PIIRedactor

Redacts personally-identifiable information using regex patterns.  Returns
`action="modify"` with the cleaned text when PII is found, `action="pass"`
otherwise.

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
from lauren_ai import LLMGuardrail

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

The `block_if` comparison is case-insensitive.  Use a cheap, fast model for the
secondary call.

#### All parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `llm` | `Any` | required | `LLMService` or any object with a `.complete(messages, **kwargs)` method |
| `prompt` | `str` | required | Judgment prompt; must contain `{content}` |
| `block_if` | `str` | required | String that triggers the guardrail (case-insensitive) |
| `violation_message` | `str` | `"Content blocked by safety filter."` | User-facing message; becomes `modified_content` when `action="modify"` |
| `action` | `"block"` \| `"modify"` | `"block"` | What to do on trigger — see below |
| `system` | `str \| None` | `None` | System prompt for the judgment call |
| `max_tokens` | `int \| None` | `None` | Max tokens for the judgment response — set to `5` for YES/NO |
| `temperature` | `float \| None` | `None` | Sampling temperature — `0.0` for deterministic YES/NO |
| `guardrail_name` | `str` | `"LLMGuardrail"` | Label attached to every `GuardrailDecision` emitted |

#### `action="modify"` — graceful redirect

The default `action="block"` causes the runner to raise `GuardrailViolated`.
Use `action="modify"` to silently replace the agent's response with
`violation_message` instead — useful for redirecting users to the correct agent
without surfacing an error:

```python
guard = LLMGuardrail(
    llm=llm_service,
    prompt="Is this response outside the agent's allowed scope?\n\n{content}\n\nYES or NO.",
    block_if="YES",
    action="modify",                   # replace response, don't raise
    violation_message=(
        "I can't help with that. Would you like me to transfer you to our CRM agent?"
    ),
    system="Answer with YES or NO only.",
    max_tokens=5,                      # YES/NO needs at most 1 token
    temperature=0.0,                   # deterministic
    guardrail_name="ScopeGuard",
)
```

---

## Writing a custom guardrail

Any object with a `check(text, context) -> GuardrailDecision` coroutine method
satisfies the `InputGuardrail` / `OutputGuardrail` protocol.  To make it
DI-injectable, add `@guardrail()`:

```python
from lauren_ai import guardrail, GuardrailDecision, GuardrailContext

# Plain class — instantiate manually and pass to @use_guardrails()
class ProfanityFilter:
    async def check(self, message: str, context: GuardrailContext) -> GuardrailDecision:
        ...

# DI-injectable class — resolved by the Lauren DI container
@guardrail(kind="input")
class ProfanityFilter:
    async def check(self, message: str, context: GuardrailContext) -> GuardrailDecision:
        ...
```

---

## GuardrailViolated signal

`GuardrailViolated` is a signal dataclass that can be emitted on a `SignalBus`
when a guardrail fires, enabling centralized audit logging:

```python
from lauren_ai import GuardrailViolated, SignalBus

bus = SignalBus()

@bus.on(GuardrailViolated)
async def log_violation(event: GuardrailViolated) -> None:
    print(
        f"[{event.phase}] {event.guardrail_name} {event.action}: "
        f"{event.violation} (agent={event.agent_name})"
    )
```

---

## Decorator ordering — mandatory

```
@agent()            ← outermost
@remember()         ← optional
@use_guardrails()   ← optional (attaches guardrail instances)
@use_tools()        ← innermost
class MyAgent: ...
```

---

## Error reference

| Error | Raised when |
|---|---|
| `DecoratorUsageError` | `@use_guardrails` or `@guardrail` used without parentheses |
| `GuardrailViolated` | A guardrail returns `action="block"` and the runner propagates it |

Guardrail block decisions are expressed as `GuardrailDecision(action="block")`,
not exceptions — the agent runner decides how to handle them.

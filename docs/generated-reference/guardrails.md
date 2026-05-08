# Guardrails

Content safety filters for agent inputs and outputs.

## Decorators

### `guardrail`

Mark a class as a DI-injectable guardrail and register it as a provider.

Applying ``@guardrail()`` to a class does two things:

1. Sets :data:`GUARDRAIL_CLASS_META` on the class (a :class:`GuardrailClassMeta`
   instance) so the framework knows it is a guardrail implementation.
2. Calls ``@injectable(scope=scope)`` from the Lauren framework, registering
   the class as a DI singleton (or the requested scope) so it can be
   injected into other components via the DI container.

Must be called **with parentheses**.  Bare ``@guardrail`` raises
:class:`~lauren_ai._exceptions.DecoratorUsageError`.

Example — a custom DI-injectable input guardrail::

    from lauren_ai import guardrail, GuardrailDecision, GuardrailContext

    @guardrail(kind="input")
    class ProfanityFilter:
        """Block messages containing profanity."""

        async def check(
            self, message: str, context: GuardrailContext
        ) -> GuardrailDecision:
            if any(w in message.lower() for w in ("badword",)):
                return GuardrailDecision(
                    action="block",
                    violation="Profanity detected.",
                    guardrail_name="ProfanityFilter",
                )
            return GuardrailDecision(
                action="pass", guardrail_name="ProfanityFilter"
            )

The class is now resolvable from the Lauren DI container and can be
injected by type into other providers or wiring classes::

    @injectable(scope=Scope.SINGLETON)
    class GuardrailWiring:
        def __init__(
            self,
            profanity_filter: ProfanityFilter,
            my_agent: MyAgent,
        ) -> None:
            # Attach the DI-resolved filter instance to the agent at startup
            meta = getattr(my_agent, USE_GUARDRAILS_META, None)
            if meta:
                meta.input_guardrails.append(profanity_filter)

:param kind: Hint for which position this guardrail is intended —
    ``"input"`` (runs before the model call), ``"output"`` (runs after),
    or ``"any"`` (either position).  Does not affect runtime behaviour;
    used for documentation and static analysis only.
:type kind: Literal["input", "output", "any"]
:param scope: The DI scope to register the class under.  Defaults to
    ``Scope.SINGLETON`` (the ``scope`` is resolved lazily from
    ``lauren.Scope`` to avoid a hard import at module load time).
:type scope: Any
:raises DecoratorUsageError: When called without parentheses (bare
    ``@guardrail``).

### `use_guardrails`

Attach input/output guardrail instances to an ``@agent()``-decorated class.

Analogous to ``@use_guards()`` in the Lauren framework — attaches
pre-built guardrail objects to the agent so the runner can execute them
before and after each LLM call.

Must be applied **below** ``@agent()`` (closer to the class body)::

    @agent(model="claude-haiku-4-5")
    @use_guardrails(
        input=[TopicFilter(allowed_topics=["cooking"])],
        output=[PIIRedactor(entities=["EMAIL"])],
    )
    class CookingAssistant: ...

``None`` entries are silently dropped, enabling conditional selection::

    @agent(model="claude-opus-4-6")
    @use_guardrails(
        input=[
            PromptInjectionFilter(),
            TopicFilter(allowed_topics=topics) if topics else None,
        ],
    )
    class DynamicAgent: ...

Input guardrails run before each LLM call.  If any guardrail returns
``action="block"`` the model is never called and the violation message is
returned to the caller.  A ``"modify"`` decision replaces the user message
before it is sent to the model.

Output guardrails run after the LLM response.  A ``"block"`` decision
raises :class:`~lauren_ai._guardrails._base.GuardrailViolated`.  A
``"modify"`` decision replaces the response content before it reaches the
caller.

Must be called **with parentheses**.  Bare ``@use_guardrails`` raises
:class:`~lauren_ai._exceptions.DecoratorUsageError`.

:param input: List of :class:`~lauren_ai._guardrails._base.InputGuardrail`
    instances (or ``None`` entries which are silently dropped) to run
    before each LLM call.
:type input: list[Any] | None
:param output: List of :class:`~lauren_ai._guardrails._base.OutputGuardrail`
    instances (or ``None`` entries which are silently dropped) to run
    after each LLM call.
:type output: list[Any] | None
:raises DecoratorUsageError: When called without parentheses (bare
    ``@use_guardrails``).

## Decision types

### `GuardrailDecision`

Result of a guardrail check.

### `GuardrailContext`

Per-call context passed to each guardrail check.

### `GuardrailViolated`

Signal emitted when a guardrail fires.

### `InputGuardrail`

Protocol for input guardrails -- check messages before LLM call.

### `OutputGuardrail`

Protocol for output guardrails -- check/modify LLM responses.

## Built-in guardrails

### `TopicFilter`

Block messages not related to allowed topics using keyword/pattern matching.

For production use, pass embed_fn for embedding-based similarity.
Without embed_fn, uses simple keyword matching.

Usage::

    guard = TopicFilter(
        allowed_topics=["cooking", "recipes", "food"],
        violation_message="I only discuss cooking.",
    )

### `PIIRedactor`

Redact PII patterns from LLM outputs.

Uses regex patterns for EMAIL, PHONE, SSN, and CREDIT_CARD.

Usage::

    guard = PIIRedactor(entities=["EMAIL", "PHONE"], replacement="[REDACTED]")

### `LengthFilter`

Block messages outside min/max length limits.

Usage::

    guard = LengthFilter(min_chars=1, max_chars=2000)

### `PromptInjectionFilter`

Detect common prompt injection patterns in user input.

Usage::

    guard = PromptInjectionFilter(violation_message="Prompt injection detected.")

### `LLMGuardrail`

Use a secondary LLM call to judge whether content is safe.

The prompt must contain ``{content}`` which will be replaced with the text
being evaluated.

:param llm: An ``LLMService`` (or any object with a compatible ``.complete()``
    method) used to run the judgment call.
:param prompt: Judgment prompt; must contain the ``{content}`` placeholder.
:param block_if: String that, when found in the LLM's response (case-insensitive),
    triggers the guardrail action.
:param violation_message: Text returned to the caller on a trigger.  When
    ``action="modify"`` this becomes the replacement content.
:param action: What to do when the guardrail triggers.

    ``"block"`` (default) — returns a ``GuardrailDecision(action="block", ...)``
    which causes the runner to raise ``GuardrailViolated``.

    ``"modify"`` — returns a ``GuardrailDecision(action="modify",
    modified_content=violation_message, ...)`` which replaces the agent's
    response without raising; useful for graceful redirects.
:param system: Optional system prompt passed to the judgment call.  Use this to
    set concise instructions such as ``"Answer YES or NO only."`` without
    baking them into the main prompt template.
:param max_tokens: Maximum tokens for the judgment response.  Set to a small
    value (e.g. ``5``) when you only need a YES/NO answer — significantly
    reduces cost and latency.
:param temperature: Sampling temperature for the judgment call.  ``0.0``
    produces deterministic YES/NO answers.
:param guardrail_name: Label attached to every ``GuardrailDecision`` emitted by
    this instance.  Defaults to ``"LLMGuardrail"`` (previously was
    ``type(self).__name__``).

Example::

    guard = LLMGuardrail(
        llm=llm_service,
        prompt="Is this response off-topic?\n\n{content}\n\nAnswer YES or NO.",
        block_if="YES",
        action="modify",
        violation_message="I can't help with that. Let me redirect you.",
        system="Answer with YES or NO only.",
        max_tokens=5,
        temperature=0.0,
        guardrail_name="OffTopicGuard",
    )


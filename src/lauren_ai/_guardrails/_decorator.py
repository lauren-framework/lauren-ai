from __future__ import annotations

"""Guardrail decorators for ``lauren-ai``.

Two decorators are provided:

* :func:`guardrail` — class decorator.  Marks a class as a DI-injectable
  guardrail implementation (registers it as a ``lauren`` singleton provider
  via ``@injectable()``).  Use this when you want the Lauren DI container to
  instantiate and inject your custom guardrail class.

* :func:`use_guardrails` — agent decorator.  Attaches pre-built guardrail
  instances to an ``@agent()``-decorated class, identical to how
  ``@use_guards()`` attaches guard classes to a controller.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from lauren_ai._exceptions import DecoratorUsageError

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

#: Attribute set on a class decorated with ``@guardrail()`` to store
#: :class:`GuardrailClassMeta`.
GUARDRAIL_CLASS_META: str = "__lauren_ai_guardrail_class__"

#: Attribute set on an ``@agent()``-decorated class by ``@use_guardrails()``
#: to store :class:`UseGuardrailsMeta`.
USE_GUARDRAILS_META: str = "__lauren_ai_use_guardrails__"

C = TypeVar("C", bound=type)

# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GuardrailClassMeta:
    """Attached to a class decorated with ``@guardrail()``.

    :param kind: Whether this is an ``input`` guardrail, an ``output``
        guardrail, or ``"any"`` (can be used in either position).
    :type kind: Literal["input", "output", "any"]
    :param scope: The DI scope the class was registered with.
    :type scope: Any
    """

    kind: Literal["input", "output", "any"] = "any"
    scope: Any = None


@dataclass
class UseGuardrailsMeta:
    """Attached to an ``@agent()``-decorated class by ``@use_guardrails()``.

    :param input_guardrails: Guardrail instances to run before each LLM call.
    :type input_guardrails: list[Any]
    :param output_guardrails: Guardrail instances to run after each LLM call.
    :type output_guardrails: list[Any]
    """

    input_guardrails: list[Any] = field(default_factory=list)
    output_guardrails: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# @guardrail() — class decorator (injectable guardrail provider)
# ---------------------------------------------------------------------------


def guardrail(
    *args: Any,
    kind: Literal["input", "output", "any"] = "any",
    scope: Any = None,
) -> Any:
    """Mark a class as a DI-injectable guardrail and register it as a provider.

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
            \"\"\"Block messages containing profanity.\"\"\"

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
    """
    if args:
        raise DecoratorUsageError(
            "@guardrail must be called with parentheses: "
            "@guardrail() or @guardrail(kind='input')"
        )

    def _apply(cls: C) -> C:
        # Lazy import — avoids requiring lauren-framework at import time for
        # users who only use guardrails outside the DI context.
        from lauren import Scope, injectable  # noqa: PLC0415

        resolved_scope = scope if scope is not None else Scope.SINGLETON

        meta = GuardrailClassMeta(kind=kind, scope=resolved_scope)
        setattr(cls, GUARDRAIL_CLASS_META, meta)

        # Register with DI unless already decorated with @injectable().
        _INJECTABLE_META = "__lauren_injectable__"
        if _INJECTABLE_META not in cls.__dict__:
            cls = injectable(scope=resolved_scope)(cls)

        return cls

    return _apply


# ---------------------------------------------------------------------------
# @use_guardrails() — agent decorator (attach guardrail instances to an agent)
# ---------------------------------------------------------------------------


def use_guardrails(
    *args: Any,
    input: list[Any] | None = None,
    output: list[Any] | None = None,
) -> Any:
    """Attach input/output guardrail instances to an ``@agent()``-decorated class.

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
    """
    if args:
        raise DecoratorUsageError(
            "@use_guardrails must be called with parentheses: "
            "@use_guardrails(input=[...], output=[...])"
        )

    def _apply(cls: C) -> C:
        meta = UseGuardrailsMeta(
            input_guardrails=[g for g in (input or []) if g is not None],
            output_guardrails=[g for g in (output or []) if g is not None],
        )
        setattr(cls, USE_GUARDRAILS_META, meta)
        return cls

    return _apply

"""LLM-based guardrail — uses a secondary model to judge content."""

from __future__ import annotations

from typing import Any, Literal

from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision
from lauren_ai._transport import Completion, Message


class LLMGuardrail:
    """Use a secondary LLM call to judge whether content is safe.

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
            prompt="Is this response off-topic?\\n\\n{content}\\n\\nAnswer YES or NO.",
            block_if="YES",
            action="modify",
            violation_message="I can't help with that. Let me redirect you.",
            system="Answer with YES or NO only.",
            max_tokens=5,
            temperature=0.0,
            guardrail_name="OffTopicGuard",
        )
    """

    def __init__(
        self,
        llm: Any,
        prompt: str,
        block_if: str,
        violation_message: str = "Content blocked by safety filter.",
        action: Literal["block", "modify"] = "block",
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        guardrail_name: str = "LLMGuardrail",
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._block_if = block_if.strip().upper()
        self._violation_message = violation_message
        self._action = action
        self._system = system
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._guardrail_name = guardrail_name

    async def check(self, content: str, context: GuardrailContext) -> GuardrailDecision:
        prompt_text = self._prompt.replace("{content}", content)

        # Build kwargs only for params that are explicitly set so callers that
        # don't accept unknown kwargs keep working.
        kwargs: dict[str, Any] = {}
        if self._system is not None:
            kwargs["system"] = self._system
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            **kwargs,
        )

        if isinstance(result, Completion):
            answer = result.content.strip().upper()
        else:
            chunks = []
            async for chunk in result:
                if chunk.delta:
                    chunks.append(chunk.delta)
            answer = "".join(chunks).strip().upper()

        if self._block_if in answer:
            if self._action == "modify":
                return GuardrailDecision(
                    action="modify",
                    modified_content=self._violation_message,
                    violation=self._violation_message,
                    guardrail_name=self._guardrail_name,
                )
            return GuardrailDecision(
                action="block",
                violation=self._violation_message,
                guardrail_name=self._guardrail_name,
            )
        return GuardrailDecision(action="pass", guardrail_name=self._guardrail_name)

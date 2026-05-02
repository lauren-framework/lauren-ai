from __future__ import annotations

"""LLM-based guardrail -- uses a secondary model to judge content."""

from typing import Any

from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision
from lauren_ai._transport import Completion, Message


class LLMGuardrail:
    """Use a secondary LLM call to judge whether content is safe.

    The prompt must contain {content} which will be replaced with the text.

    Usage::

        guard = LLMGuardrail(
            llm=llm_service,
            prompt="Is this harmful? {content}\\n\\nAnswer YES or NO only.",
            block_if="YES",
            violation_message="Content blocked by safety filter.",
        )
    """

    def __init__(
        self,
        llm: Any,
        prompt: str,
        block_if: str,
        violation_message: str = "Content blocked by safety filter.",
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._block_if = block_if.strip().upper()
        self._violation_message = violation_message

    async def check(self, content: str, context: GuardrailContext) -> GuardrailDecision:
        prompt_text = self._prompt.replace("{content}", content)
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
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
            return GuardrailDecision(
                action="block",
                violation=self._violation_message,
                guardrail_name=type(self).__name__,
            )
        return GuardrailDecision(action="pass", guardrail_name=type(self).__name__)

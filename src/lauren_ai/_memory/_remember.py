"""@remember() decorator for agents — automatic memory extraction and injection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from lauren_ai._exceptions import LaurenAIError

REMEMBER_META = "__lauren_ai_remember__"

C = TypeVar("C", bound=type)


class MemoryConfigError(LaurenAIError):
    """Raised when memory configuration is invalid or user_id is missing."""


@dataclass
class RememberMeta:
    """Metadata attached by @remember() to an @agent() class."""

    store_token: str | None  # DI token name; None = inject UserMemoryStore directly
    extract: bool
    inject: bool
    top_k: int
    extraction_model: str | None


def remember(
    *args: Any,
    store: str | None = None,
    extract: bool = True,
    inject: bool = True,
    top_k: int = 5,
    extraction_model: str | None = None,
) -> Callable[[C], C]:
    """Opt a @agent() class into automatic user memory extraction/injection.

    Must be applied BELOW @agent()::

        @agent(model="claude-haiku-4-5")
        @remember(store="user_memory", extract=True, inject=True, top_k=5)
        class PersonalAssistant: ...

    When inject=True, relevant memories are prepended to the system prompt
    before each LLM call.

    When extract=True, new facts are extracted from each conversation turn
    and stored in the UserMemoryStore.

    :param store: DI token name for UserMemoryStore (None = auto-inject).
    :param extract: Extract new facts after each turn.
    :param inject: Inject relevant memories before each turn.
    :param top_k: Number of memories to inject.
    :param extraction_model: Model for fact extraction (defaults to agent model).
    """
    if args:
        # Bare @remember usage without parentheses — provide helpful error
        raise MemoryConfigError("@remember must be called with parentheses: @remember()")

    def _apply(cls: C) -> C:
        meta = RememberMeta(
            store_token=store,
            extract=extract,
            inject=inject,
            top_k=top_k,
            extraction_model=extraction_model,
        )
        setattr(cls, REMEMBER_META, meta)
        return cls

    return _apply


async def extract_facts(
    user_id: str,
    conversation_turn: str,
    llm: Any,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Use LLM to extract facts from a conversation turn.

    Returns list of {"content": str, "topics": list[str]} dicts.
    """
    import json

    from lauren_ai._transport import Completion, Message

    prompt = (
        "Extract any facts about the user from this conversation turn. "
        "Return a JSON array of objects with 'content' (fact) and "
        "'topics' (list of topic strings). "
        "If no new facts, return []. "
        "Facts should be in third person: 'User prefers X' or 'User works on Y'.\n\n"
        f"Conversation:\n{conversation_turn}\n\n"
        "New facts (JSON array only, no explanation):"
    )

    result = await llm.complete(
        [Message(role="user", content=prompt)],  # type: ignore[arg-type]
        model=model,
    )

    if isinstance(result, Completion):
        text = result.content
    else:
        chunks = []
        async for chunk in result:
            if chunk.delta:
                chunks.append(chunk.delta)
        text = "".join(chunks)

    # Try to parse JSON from response
    import re

    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return []


def build_memory_context(facts: list[Any]) -> str:
    """Format memory facts as a system prompt block."""
    if not facts:
        return ""
    lines = ["## What I remember about you:"]
    for f in facts:
        confidence_label = (
            "high" if f.confidence >= 0.8 else "medium" if f.confidence >= 0.5 else "low"
        )
        lines.append(f"- {f.content} (confidence: {confidence_label})")
    return "\n".join(lines)

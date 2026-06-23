"""Transport protocol and canonical message / completion types for ``lauren-ai``.

This module defines the provider-agnostic data model shared by all transport
implementations (:mod:`~lauren_ai._transport._anthropic`,
:mod:`~lauren_ai._transport._openai`, :mod:`~lauren_ai._transport._ollama`,
:mod:`~lauren_ai._transport._mock`).

Key types
---------
* :class:`Message` — a single conversation turn (user or assistant).
* :class:`ContentBlock` — a typed block within a message.
* :class:`Completion` — a finished non-streaming model response.
* :class:`CompletionChunk` — a single chunk from a streaming response.
* :class:`TokenUsage` — token accounting with cost estimation.
* :class:`ToolSchema` — JSON Schema descriptor for a tool.
* :class:`Transport` — the protocol every transport must satisfy.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, runtime_checkable

from typing_extensions import Protocol

__all__ = [
    # Primitive data types
    "TokenUsage",
    "ContentBlock",
    "Message",
    "ToolCall",
    "ToolCallDelta",
    "ThinkingBlock",
    "RedactedThinkingBlock",
    "PendingApproval",
    "Completion",
    "CompletionChunk",
    "CompletionCall",
    "Embedding",
    "ToolSchema",
    "ToolChoice",
    # Protocol
    "Transport",
    # Token estimation
    "estimate_message_tokens",
]

# ---------------------------------------------------------------------------
# Price table (approximate, per 1 million tokens, input / output)
# ---------------------------------------------------------------------------

_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4-5": (0.25, 1.25),
    "claude-haiku-4": (0.25, 1.25),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o3": (15.0, 60.0),
    "o3-mini": (1.1, 4.4),
}

_DEFAULT_PRICE: tuple[float, float] = (1.0, 3.0)


def _price_for_model(model: str) -> tuple[float, float]:
    """Return ``(input_price_per_1m, output_price_per_1m)`` for *model*.

    Performs a prefix-match on the model string so that version suffixes like
    ``-20241022`` are handled gracefully.

    :param model: Model identifier string.
    :type model: str
    :return: ``(input_price, output_price)`` in USD per million tokens.
    :rtype: tuple[float, float]
    """
    # Exact match first.
    if model in _PRICE_TABLE:
        return _PRICE_TABLE[model]
    # Prefix match (e.g. "claude-opus-4-6-20260201" still hits "claude-opus-4-6").
    for key, prices in _PRICE_TABLE.items():
        if model.startswith(key):
            return prices
    return _DEFAULT_PRICE


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token accounting for a single model call.

    :param input_tokens: Number of tokens in the prompt / input messages.
    :type input_tokens: int
    :param output_tokens: Number of tokens in the completion.
    :type output_tokens: int
    :param cache_read_tokens: Tokens read from the prompt cache (Anthropic).
    :type cache_read_tokens: int
    :param cache_write_tokens: Tokens written to the prompt cache (Anthropic).
    :type cache_write_tokens: int
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output).

        :return: Sum of input and output token counts.
        :rtype: int
        """
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: str) -> float:
        """Estimate the cost in USD for this usage against *model*.

        Uses a bundled price table with prefix-matching.  Falls back to
        ``$1/$3`` per million tokens when the model is unknown.

        :param model: Model identifier used for the completion.
        :type model: str
        :return: Estimated cost in USD.
        :rtype: float
        """
        input_price, output_price = _price_for_model(model)
        cost = (self.input_tokens / 1_000_000) * input_price
        cost += (self.output_tokens / 1_000_000) * output_price
        # Cache reads are typically discounted; cache writes are similar to input cost.
        # We keep it simple and only count input + output here.
        return cost

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two :class:`TokenUsage` instances together.

        :param other: The other usage to add.
        :type other: TokenUsage
        :return: A new :class:`TokenUsage` with summed fields.
        :rtype: TokenUsage
        """
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


# ---------------------------------------------------------------------------
# ContentBlock
# ---------------------------------------------------------------------------


@dataclass
class ContentBlock:
    """A single typed block within a :class:`Message`'s content list.

    The *type* discriminator determines which other fields are populated:

    * ``"text"`` — ``text`` is set.
    * ``"tool_use"`` — ``tool_use_id``, ``name``, and ``input`` are set.
    * ``"tool_result"`` — ``tool_use_id`` and ``content`` are set.
    * ``"image"`` — ``source`` is set.

    :param type: Block type discriminator.
    :type type: Literal["text", "tool_use", "tool_result", "image"]
    :param text: Text content (for ``type="text"``).
    :type text: str | None
    :param tool_use_id: Provider-assigned tool call identifier.
    :type tool_use_id: str | None
    :param name: Tool name (for ``type="tool_use"``).
    :type name: str | None
    :param input: Tool input arguments (for ``type="tool_use"``).
    :type input: dict[str, Any] | None
    :param content: Tool result payload (for ``type="tool_result"``).
        Either a plain string or a list of content blocks.
    :type content: str | list[Any] | None
    :param source: Image source descriptor (for ``type="image"``).
        Follows the Anthropic image source format, e.g.
        ``{"type": "base64", "media_type": "image/png", "data": "..."}``.
    :type source: dict[str, Any] | None
    """

    type: Literal["text", "tool_use", "tool_result", "image"]
    text: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[Any] | None = None
    source: dict[str, Any] | None = None

    @classmethod
    def text_block(cls, text: str) -> ContentBlock:
        """Convenience factory for a plain text block.

        :param text: The text content.
        :type text: str
        :return: A new text :class:`ContentBlock`.
        :rtype: ContentBlock
        """
        return cls(type="text", text=text)

    @classmethod
    def tool_use_block(
        cls,
        name: str,
        tool_input: dict[str, Any],
        tool_use_id: str | None = None,
    ) -> ContentBlock:
        """Convenience factory for a tool-use block.

        :param name: The tool name.
        :type name: str
        :param tool_input: The tool call arguments.
        :type tool_input: dict[str, Any]
        :param tool_use_id: Optional provider-assigned identifier.  A random
            UUID is generated when *None*.
        :type tool_use_id: str | None
        :return: A new tool-use :class:`ContentBlock`.
        :rtype: ContentBlock
        """
        return cls(
            type="tool_use",
            name=name,
            input=tool_input,
            tool_use_id=tool_use_id or f"toolu_{uuid.uuid4().hex[:16]}",
        )

    @classmethod
    def tool_result_block(
        cls,
        tool_use_id: str,
        content: str | list[Any],
    ) -> ContentBlock:
        """Convenience factory for a tool-result block.

        :param tool_use_id: The ID of the tool call this result corresponds to.
        :type tool_use_id: str
        :param content: The result content.
        :type content: str | list[Any]
        :return: A new tool-result :class:`ContentBlock`.
        :rtype: ContentBlock
        """
        return cls(type="tool_result", tool_use_id=tool_use_id, content=content)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a conversation.

    :param role: The sender role.  Either ``"user"`` or ``"assistant"``.
    :type role: Literal["user", "assistant"]
    :param content: Message content.  Either a plain string or a list of
        :class:`ContentBlock` instances for multi-modal / tool-use messages.
    :type content: str | list[ContentBlock]
    """

    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]

    @classmethod
    def user(cls, content: str | list[ContentBlock]) -> Message:
        """Convenience factory for a user message.

        :param content: Message content.
        :type content: str | list[ContentBlock]
        :return: A new user :class:`Message`.
        :rtype: Message
        """
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | list[ContentBlock]) -> Message:
        """Convenience factory for an assistant message.

        :param content: Message content.
        :type content: str | list[ContentBlock]
        :return: A new assistant :class:`Message`.
        :rtype: Message
        """
        return cls(role="assistant", content=content)

    @classmethod
    def from_multimodal(
        cls,
        role: str,
        parts: list[Any],
    ) -> Message:
        """Create a Message from a list of text strings and content objects.

        Accepts any mix of plain strings, :class:`~lauren_ai._transport._multimodal.ImageContent`,
        :class:`~lauren_ai._transport._multimodal.AudioContent`, and
        :class:`~lauren_ai._transport._multimodal.DocumentContent` instances.

        :param role: The sender role — ``"user"`` or ``"assistant"``.
        :type role: str
        :param parts: Mixed list of text strings and multimodal content objects.
        :type parts: list[Any]
        :return: A new :class:`Message` whose ``content`` is the *parts* list.
        :rtype: Message
        """
        from lauren_ai._transport._multimodal import ContentPart  # noqa: F401

        return cls(role=role, content=parts)  # type: ignore[arg-type]

    def text(self) -> str:
        """Extract all text from this message as a single concatenated string.

        :return: All text content concatenated together.
        :rtype: str
        """
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, ContentBlock) and block.type == "text" and block.text:
                parts.append(block.text)
        return "".join(parts)


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A completed tool call extracted from a model response.

    :param tool_use_id: Provider-assigned identifier for this tool call.
    :type tool_use_id: str
    :param name: The registered tool name.
    :type name: str
    :param input: The parsed JSON input arguments.
    :type input: dict[str, Any]
    """

    tool_use_id: str
    name: str
    input: dict[str, Any]


# ---------------------------------------------------------------------------
# ToolCallDelta (streaming)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallDelta:
    """A partial tool call update in a streaming response.

    :param tool_use_id: The provider-assigned identifier for this tool call.
    :type tool_use_id: str
    :param name: The tool name, populated only in the first delta chunk.
    :type name: str | None
    :param input_delta: Partial JSON string fragment for the input arguments.
    :type input_delta: str
    """

    tool_use_id: str
    name: str | None
    input_delta: str


# ---------------------------------------------------------------------------
# Thinking blocks (Anthropic extended thinking)
# ---------------------------------------------------------------------------


@dataclass
class ThinkingBlock:
    """An extended-thinking block from an Anthropic model.

    :param thinking: The model's reasoning text.
    :type thinking: str
    :param signature: Cryptographic signature from Anthropic confirming the
        thinking is genuine.
    :type signature: str
    :param type: Block type discriminator.  Always ``"thinking"``.
    :type type: Literal["thinking"]
    """

    thinking: str
    signature: str
    type: Literal["thinking"] = "thinking"


@dataclass
class RedactedThinkingBlock:
    """A redacted extended-thinking block (opaque data from Anthropic).

    :param data: Opaque base64-encoded redacted thinking data.
    :type data: str
    :param type: Block type discriminator.  Always ``"redacted_thinking"``.
    :type type: Literal["redacted_thinking"]
    """

    data: str
    type: Literal["redacted_thinking"] = "redacted_thinking"


# ---------------------------------------------------------------------------
# PendingApproval (HITL)
# ---------------------------------------------------------------------------


@dataclass
class PendingApproval:
    """Signals that a tool call is awaiting human approval.

    Emitted as a :class:`CompletionChunk` in streaming mode when a
    human-in-the-loop confirmation step is required before the tool is
    executed.

    :param tool_name: The name of the tool pending approval.
    :type tool_name: str
    :param tool_use_id: The provider-assigned identifier for the pending call.
    :type tool_use_id: str
    :param input: The tool input arguments that the model wants to invoke.
    :type input: dict[str, Any]
    """

    tool_name: str
    tool_use_id: str
    input: dict[str, Any]


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    """A finished (non-streaming) model completion.

    :param id: Provider-assigned completion identifier.
    :type id: str
    :param model: The model that produced this completion.
    :type model: str
    :param content: The primary text response from the model.
    :type content: str
    :param tool_calls: All tool calls requested by the model in this turn.
    :type tool_calls: list[ToolCall]
    :param stop_reason: Why the model stopped generating:

        * ``"end_turn"`` — natural end of response.
        * ``"tool_use"`` — one or more tool calls were requested.
        * ``"max_tokens"`` — the configured token limit was reached.
        * ``"stop_sequence"`` — a stop sequence was hit.
    :type stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
    :param usage: Token usage statistics for this completion.
    :type usage: TokenUsage
    :param thinking_blocks: Extended-thinking blocks (Anthropic only).
    :type thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock]
    """

    id: str
    model: str
    content: str
    tool_calls: list[ToolCall]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
    usage: TokenUsage
    thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CompletionChunk (streaming)
# ---------------------------------------------------------------------------


@dataclass
class CompletionChunk:
    """A single chunk from a streaming model completion.

    Only one of *delta*, *thinking_delta*, or *tool_call_delta* is populated
    per chunk (though they are not mutually exclusive in the schema).

    :param delta: Text content delta for this chunk.
    :type delta: str
    :param thinking_delta: Reasoning / thinking text delta (extended thinking).
    :type thinking_delta: str | None
    :param tool_call_delta: Partial tool call update for this chunk.
    :type tool_call_delta: ToolCallDelta | None
    :param stop_reason: Stop reason, populated only in the final chunk.
    :type stop_reason: str | None
    :param usage: Token usage, populated only in the final chunk.
    :type usage: TokenUsage | None
    :param pending_approval: Human-in-the-loop pending-approval signal.
        Present when a tool call requires confirmation before execution.
    :type pending_approval: PendingApproval | None
    :param guardrail_override: When set, an output guardrail fired and this
        string is the replacement content.  The runner emits one sentinel
        chunk with only this field populated (``delta=""`` etc.) after all
        normal chunks have been yielded.  Callers should replace the
        accumulated streaming text with this value.
    :type guardrail_override: str | None
    """

    delta: str = ""
    thinking_delta: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    stop_reason: str | None = None
    usage: TokenUsage | None = None
    pending_approval: PendingApproval | None = None
    guardrail_override: str | None = None


# ---------------------------------------------------------------------------
# CompletionCall (for MockTransport assertions)
# ---------------------------------------------------------------------------


@dataclass
class CompletionCall:
    """A recorded invocation of :meth:`Transport.complete`.

    Stored by :class:`~lauren_ai._transport._mock.MockTransport` so tests can
    assert on what the transport was called with.

    :param messages: The messages that were passed.
    :type messages: list[Message]
    :param model: The model identifier.
    :type model: str
    :param system: The system prompt, if provided.
    :type system: str | None
    :param tools: Tool schemas provided to the model.
    :type tools: list[ToolSchema] | None
    :param tool_choice: Tool choice constraint.
    :type tool_choice: ToolChoice | None
    :param max_tokens: Maximum tokens requested.
    :type max_tokens: int
    :param temperature: Sampling temperature.
    :type temperature: float
    :param stop_sequences: Stop sequences, if any.
    :type stop_sequences: list[str] | None
    :param stream: Whether streaming was requested.
    :type stream: bool
    :param thinking: Whether extended thinking was requested.
    :type thinking: bool
    :param thinking_budget_tokens: Thinking token budget.
    :type thinking_budget_tokens: int
    """

    messages: list[Message]
    model: str
    system: str | None = None
    tools: list[ToolSchema] | None = None
    tool_choice: ToolChoice | None = None
    max_tokens: int = 4096
    temperature: float = 1.0
    stop_sequences: list[str] | None = None
    stream: bool = False
    thinking: bool = False
    thinking_budget_tokens: int = 8000


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@dataclass
class Embedding:
    """A single embedding vector.

    :param index: Zero-based index of this embedding in the batch.
    :type index: int
    :param vector: The floating-point embedding vector.
    :type vector: list[float]
    """

    index: int
    vector: list[float]


# ---------------------------------------------------------------------------
# ToolSchema
# ---------------------------------------------------------------------------


@dataclass
class ToolSchema:
    """JSON Schema descriptor for a tool exposed to the model.

    :param name: The tool's registered name (snake_case).
    :type name: str
    :param description: Human-readable description used in the model's system
        context.
    :type description: str
    :param input_schema: JSON Schema ``object`` describing the tool's input
        parameters.
    :type input_schema: dict[str, Any]
    """

    name: str
    description: str
    input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# ToolChoice
# ---------------------------------------------------------------------------


@dataclass
class ToolChoice:
    """Constraint on which tool(s) the model may call.

    :param type: Constraint type:

        * ``"auto"`` — model decides whether to call a tool.
        * ``"any"`` — model must call at least one tool.
        * ``"tool"`` — model must call the specific tool named in *name*.
    :type type: Literal["auto", "any", "tool"]
    :param name: Required when *type* is ``"tool"``; the name of the tool
        that must be called.
    :type name: str | None
    """

    type: Literal["auto", "any", "tool"]
    name: str | None = None

    @classmethod
    def auto(cls) -> ToolChoice:
        """Create an *auto* tool choice (model decides).

        :return: A :class:`ToolChoice` with ``type="auto"``.
        :rtype: ToolChoice
        """
        return cls(type="auto")

    @classmethod
    def required(cls) -> ToolChoice:
        """Create a *required* tool choice (model must call a tool).

        Maps to ``type="any"`` in the Anthropic API.

        :return: A :class:`ToolChoice` with ``type="any"``.
        :rtype: ToolChoice
        """
        return cls(type="any")

    @classmethod
    def specific(cls, name: str) -> ToolChoice:
        """Create a *specific* tool choice (model must call exactly this tool).

        :param name: The name of the tool that must be called.
        :type name: str
        :return: A :class:`ToolChoice` with ``type="tool"`` and *name* set.
        :rtype: ToolChoice
        """
        if not name:
            raise ValueError("name must be a non-empty string for ToolChoice.specific()")
        return cls(type="tool", name=name)


# ---------------------------------------------------------------------------
# Transport Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Provider-agnostic interface for LLM backends.

    All transport implementations (:class:`~lauren_ai._transport._anthropic.AnthropicTransport`,
    :class:`~lauren_ai._transport._openai.OpenAITransport`,
    :class:`~lauren_ai._transport._ollama.OllamaTransport`,
    :class:`~lauren_ai._transport._mock.MockTransport`) must satisfy this
    protocol.

    Every method is a coroutine so the same code path works for streaming and
    non-streaming calls.
    """

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
        thinking: bool = False,
        thinking_budget_tokens: int = 8000,
    ) -> Completion | AsyncIterator[CompletionChunk]:
        """Send *messages* to the model and return the result.

        When *stream* is ``False`` (the default), returns a :class:`Completion`.
        When *stream* is ``True``, returns an :class:`AsyncIterator` of
        :class:`CompletionChunk` objects.

        :param messages: Conversation history including the latest user turn.
        :type messages: list[Message]
        :param model: Model identifier.
        :type model: str
        :param system: System prompt prepended to the conversation.
        :type system: str | None
        :param tools: Tool schemas made available to the model.
        :type tools: list[ToolSchema] | None
        :param tool_choice: Constraint on tool selection.
        :type tool_choice: ToolChoice | None
        :param max_tokens: Maximum output tokens to generate.
        :type max_tokens: int
        :param temperature: Sampling temperature.
        :type temperature: float
        :param stop_sequences: Custom stop sequences.
        :type stop_sequences: list[str] | None
        :param stream: Return an async iterator of chunks instead of a
            complete :class:`Completion`.
        :type stream: bool
        :param thinking: Enable extended thinking (Anthropic only).
        :type thinking: bool
        :param thinking_budget_tokens: Token budget for extended thinking.
        :type thinking_budget_tokens: int
        :return: A :class:`Completion` or an
            ``AsyncIterator[CompletionChunk]``.
        :rtype: Completion | AsyncIterator[CompletionChunk]
        """
        ...

    async def embed(
        self,
        inputs: list[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[Embedding]:
        """Generate embeddings for *inputs*.

        :param inputs: List of text strings to embed.
        :type inputs: list[str]
        :param model: Embedding model identifier.
        :type model: str
        :param dimensions: Desired embedding dimensionality.  When *None*
            the model's default is used.
        :type dimensions: int | None
        :return: A list of :class:`Embedding` objects in input order.
        :rtype: list[Embedding]
        """
        ...

    async def count_tokens(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> int:
        """Estimate the token count for *messages* without generating.

        :param messages: Conversation messages to count.
        :type messages: list[Message]
        :param model: Model identifier (affects tokenisation).
        :type model: str
        :param system: System prompt to include in the count.
        :type system: str | None
        :param tools: Tool schemas to include in the count.
        :type tools: list[ToolSchema] | None
        :return: Estimated token count.
        :rtype: int
        """
        ...


# ---------------------------------------------------------------------------
# Heuristic token estimation (shared fallback)
# ---------------------------------------------------------------------------

_HEURISTIC_CHARS_PER_TOKEN: int = 4


def _block_field(block: object, name: str) -> Any:
    """Read *name* from a content block in either dict or dataclass form."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _tool_chars(tool: object) -> int:
    """Character length of a tool schema in either dict or :class:`ToolSchema` form.

    The runner passes tool schemas as plain JSON dicts (``{"name", "description",
    "input_schema"}``), while direct transport callers pass :class:`ToolSchema`
    dataclasses — both must be supported here.
    """
    if isinstance(tool, dict):
        name = tool.get("name", "") or ""
        desc = tool.get("description", "") or ""
        schema = tool.get("input_schema", tool.get("parameters", {})) or {}
    else:
        name = getattr(tool, "name", "") or ""
        desc = getattr(tool, "description", "") or ""
        schema = getattr(tool, "input_schema", {}) or {}
    return len(name) + len(desc) + len(json.dumps(schema, default=str))


def estimate_message_tokens(
    messages: list[Any],
    system: str | None = None,
    tools: list[ToolSchema] | None = None,
) -> int:
    """Estimate the token count of a request using the 4-chars-per-token rule.

    This is the shared fallback used by every transport's ``count_tokens`` when
    no exact provider endpoint is available.  Unlike the provider SDKs it is
    tolerant of **both** message representations the codebase uses: the
    dataclass :class:`Message` / :class:`ContentBlock` form *and* the plain
    ``{"role": ..., "content": ...}`` dict form produced by
    :class:`~lauren_ai._memory.ShortTermMemory`.  The runner counts dict
    messages directly, so a dict-blind estimator would raise ``AttributeError``
    on exactly the non-native-endpoint path that needs the guard most.

    :param messages: Conversation messages (dicts or :class:`Message`).
    :type messages: list[Any]
    :param system: Optional system prompt.
    :type system: str | None
    :param tools: Optional tool schemas included in the request.
    :type tools: list[ToolSchema] | None
    :return: Estimated token count.
    :rtype: int
    """
    cpt = _HEURISTIC_CHARS_PER_TOKEN
    total = 0
    if system:
        total += max(1, len(system) // cpt)
    if tools:
        for t in tools:
            total += max(1, _tool_chars(t) // cpt)
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            total += max(1, len(content) // cpt)
        elif isinstance(content, list):
            for block in content:
                text = _block_field(block, "text")
                inp = _block_field(block, "input")
                bcontent = _block_field(block, "content")
                if text:
                    total += max(1, len(text) // cpt)
                if inp:
                    total += max(1, len(json.dumps(inp, default=str)) // cpt)
                if isinstance(bcontent, str):
                    total += max(1, len(bcontent) // cpt)
                elif isinstance(bcontent, list):
                    total += max(1, len(json.dumps(bcontent, default=str)) // cpt)
    return total

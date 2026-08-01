"""Anthropic transport implementation for ``lauren-ai``.

Wraps the official ``anthropic`` Python SDK (async client).  The SDK is an
optional dependency — it is imported lazily so that the rest of the package
can be used without it installed.

Install with::

    pip install lauren-ai[anthropic]
    # or
    pip install anthropic>=0.35

:class:`AnthropicTransport` implements the
:class:`~lauren_ai._transport.Transport` protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from lauren_ai._config import LLMConfig
from lauren_ai._exceptions import AuthTransportError, TransientTransportError, TransportError
from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    Embedding,
    Message,
    RedactedThinkingBlock,
    RequestOptions,
    ThinkingBlock,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolSchema,
    estimate_message_tokens,
)

if TYPE_CHECKING:
    pass

__all__ = ["AnthropicTransport"]
logger = logging.getLogger(__name__)

_ANTHROPIC_IMPORT_ERROR = (
    "The 'anthropic' package is required to use AnthropicTransport.\n"
    "Install it with: pip install anthropic>=0.35\n"
    "Or install the extras: pip install lauren-ai[anthropic]"
)


def _require_anthropic() -> Any:
    """Import and return the ``anthropic`` module, raising a helpful error if absent.

    :return: The ``anthropic`` module.
    :raises ImportError: If ``anthropic`` is not installed.
    """
    try:
        import anthropic  # noqa: PLC0415

        return anthropic
    except ImportError as exc:
        raise ImportError(_ANTHROPIC_IMPORT_ERROR) from exc


# ---------------------------------------------------------------------------
# Message / schema translation helpers
# ---------------------------------------------------------------------------


def _content_block_to_anthropic(block: Any) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.ContentBlock` (or plain dict)
    to the Anthropic API dict.

    Accepts both ``ContentBlock`` dataclass instances and plain dicts (as stored
    by ``ShortTermMemory`` at runtime).

    :param block: The content block to convert.
    :return: Anthropic-compatible dict.
    :rtype: dict[str, Any]
    """
    if not isinstance(block, (dict,)) and not hasattr(block, "type"):
        from lauren_ai._transport._multimodal import (  # noqa: PLC0415
            AudioContent,
            DocumentContent,
            ImageContent,
            UnsupportedContentError,
        )

        if isinstance(block, (ImageContent, DocumentContent)):
            return block.to_anthropic_block()
        if isinstance(block, AudioContent):
            raise UnsupportedContentError("Anthropic Messages does not support AudioContent")
    block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
    if block_type == "text":
        text = block.get("text", "") if isinstance(block, dict) else (block.text or "")
        return {"type": "text", "text": text}
    if block_type == "tool_use":
        if isinstance(block, dict):
            return {
                "type": "tool_use",
                "id": block.get("id") or block.get("tool_use_id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }
        return {
            "type": "tool_use",
            "id": block.tool_use_id,
            "name": block.name,
            "input": block.input or {},
        }
    if block_type == "tool_result":
        if isinstance(block, dict):
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id", ""),
            }
            blk_content = block.get("content")
            if blk_content is not None:
                result["content"] = blk_content
        else:
            result = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
            }
            if isinstance(block.content, (str, list)):
                result["content"] = block.content
        return result
    if block_type == "image":
        source = block.get("source", {}) if isinstance(block, dict) else (block.source or {})
        if not isinstance(source, dict) and hasattr(source, "to_anthropic_block"):
            return source.to_anthropic_block()
        return {"type": "image", "source": source}
    if block_type == "document":
        source = block.get("source", {}) if isinstance(block, dict) else getattr(block, "source", {})
        if hasattr(source, "to_anthropic_block"):
            return source.to_anthropic_block()
        return {"type": "document", "source": source}
    if block_type == "thinking":
        # Extended-thinking block: thinking text + signature must be sent back
        # verbatim (Anthropic verifies the signature against the exact text).
        if isinstance(block, dict):
            return {
                "type": "thinking",
                "thinking": block.get("thinking", ""),
                "signature": block.get("signature", ""),
            }
        return {
            "type": "thinking",
            "thinking": getattr(block, "thinking", ""),
            "signature": getattr(block, "signature", ""),
        }
    if block_type == "redacted_thinking":
        if isinstance(block, dict):
            return {"type": "redacted_thinking", "data": block.get("data", "")}
        return {"type": "redacted_thinking", "data": getattr(block, "data", "")}
    raise ValueError(f"Unknown ContentBlock type: {block_type!r}")


def _message_to_anthropic(message: Any) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.Message` (or plain dict)
    to an Anthropic message dict.

    Accepts both ``Message`` dataclass instances and plain dicts (as stored by
    ``ShortTermMemory`` at runtime).

    :param message: The message to convert.
    :return: Anthropic-compatible message dict.
    :rtype: dict[str, Any]
    """
    if isinstance(message, dict):
        role: str = message.get("role", "user")
        content: Any = message.get("content", "")
    else:
        role = message.role
        content = message.content

    # Defensive conversion: role:"tool" is OpenAI format for a tool result.
    # Anthropic rejects this role — convert to role:"user" with a tool_result
    # content block so that legacy or mis-healed messages don't cause a 400.
    if role == "tool":
        tool_use_id: str = (
            message.get("tool_call_id", "") if isinstance(message, dict) else getattr(message, "tool_call_id", "")
        )
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content if isinstance(content, str) else str(content),
                }
            ],
        }

    if isinstance(content, str):
        return {"role": role, "content": content}
    return {
        "role": role,
        "content": [_content_block_to_anthropic(b) for b in content],
    }


def _apply_conversation_cache(messages: list[dict[str, Any]]) -> None:
    """Mark the conversation prefix as a prompt-cache breakpoint (PRD-132 L0).

    Adds ``cache_control: ephemeral`` to the last content block of the last
    message, so the provider serves the (file-heavy) history prefix from cache
    on the next turn instead of re-billing it at full input price.  String
    content is normalised to a single text block so the marker can attach.

    Mutates *messages* in place.  No-op on an empty list.
    """
    if not messages:
        return
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        last["content"] = content
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = {"type": "ephemeral"}


def _tool_schema_to_anthropic(
    schema: Any,
    *,
    cache: bool = False,
) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.ToolSchema` to the Anthropic format.

    Accepts both the ``ToolSchema`` dataclass and plain dicts (as returned by
    ``ToolRegistry.get_schemas()`` at runtime).

    :param schema: The tool schema to convert.
    :param cache: Whether to attach prompt-caching headers to this tool.
    :type cache: bool
    :return: Anthropic-compatible tool dict.
    :rtype: dict[str, Any]
    """
    if isinstance(schema, dict):
        name = schema.get("name", "")
        description = schema.get("description", "")
        input_schema = schema.get("input_schema", {})
        kind = schema.get("kind", "function")
        provider_options = schema.get("provider_options", {})
    else:
        name = schema.name
        description = schema.description
        input_schema = schema.input_schema
        kind = schema.kind
        provider_options = schema.provider_options
    if kind != "function":
        provider_result = dict(provider_options)
        provider_result.setdefault("type", kind)
        if name:
            provider_result.setdefault("name", name)
        if description:
            provider_result.setdefault("description", description)
        if input_schema:
            provider_result.setdefault("input_schema", input_schema)
        return provider_result
    result: dict[str, Any] = {
        "name": name,
        "description": description,
        "input_schema": input_schema,
    }
    if cache:
        result["cache_control"] = {"type": "ephemeral"}
    return result


def _tool_choice_to_anthropic(tool_choice: ToolChoice) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.ToolChoice` to the Anthropic format.

    :param tool_choice: The tool choice to convert.
    :type tool_choice: ToolChoice
    :return: Anthropic-compatible tool_choice dict.
    :rtype: dict[str, Any]
    """
    if tool_choice.type == "tool":
        return {"type": "tool", "name": tool_choice.name}
    return {"type": tool_choice.type}


def _parse_stop_reason(
    raw: str | None,
) -> str:
    """Normalise Anthropic stop reason strings to the canonical set.

    :param raw: Raw stop reason from the Anthropic SDK.
    :type raw: str | None
    :return: One of ``"end_turn"``, ``"tool_use"``, ``"max_tokens"``,
        ``"stop_sequence"``.
    :rtype: str
    """
    mapping = {
        "end_turn": "end_turn",
        "tool_use": "tool_use",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
    }
    return mapping.get(raw or "end_turn", "end_turn")


def _extract_text_from_response(blocks: list[Any]) -> str:
    """Extract and concatenate all text from an Anthropic content block list.

    :param blocks: List of Anthropic SDK content block objects.
    :type blocks: list[Any]
    :return: Concatenated text content.
    :rtype: str
    """
    parts: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _extract_tool_calls_from_response(blocks: list[Any]) -> list[ToolCall]:
    """Extract :class:`~lauren_ai._transport.ToolCall` objects from an Anthropic response.

    :param blocks: List of Anthropic SDK content block objects.
    :type blocks: list[Any]
    :return: List of extracted tool calls.
    :rtype: list[ToolCall]
    """
    calls: list[ToolCall] = []
    for block in blocks:
        if getattr(block, "type", None) == "tool_use":
            raw_input = getattr(block, "input", {})
            calls.append(
                ToolCall(
                    tool_use_id=getattr(block, "id", f"toolu_{uuid.uuid4().hex[:16]}"),
                    name=getattr(block, "name", ""),
                    input=dict(raw_input) if raw_input else {},
                )
            )
    return calls


def _extract_thinking_blocks(
    blocks: list[Any],
) -> list[ThinkingBlock | RedactedThinkingBlock]:
    """Extract thinking blocks from an Anthropic content block list.

    :param blocks: List of Anthropic SDK content block objects.
    :type blocks: list[Any]
    :return: List of thinking / redacted-thinking blocks.
    :rtype: list[ThinkingBlock | RedactedThinkingBlock]
    """
    result: list[ThinkingBlock | RedactedThinkingBlock] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "thinking":
            result.append(
                ThinkingBlock(
                    thinking=getattr(block, "thinking", ""),
                    signature=getattr(block, "signature", ""),
                )
            )
        elif block_type == "redacted_thinking":
            result.append(RedactedThinkingBlock(data=getattr(block, "data", "")))
    return result


# ---------------------------------------------------------------------------
# AnthropicTransport
# ---------------------------------------------------------------------------


class AnthropicTransport:
    """Anthropic SDK transport — wraps the async ``anthropic`` client.

    Handles:

    * Message/schema translation between the Transport protocol and the Anthropic
      API format.
    * Prompt caching (``cache_system_prompt`` / ``cache_tools`` in
      :class:`~lauren_ai._config.LLMConfig`).
    * Extended thinking (``thinking=True``).
    * Automatic retries with exponential back-off on
      :class:`~lauren_ai._exceptions.TransientTransportError`.
    * Streaming via ``client.messages.stream``.
    * ``count_tokens`` via the beta messages token-counting endpoint.

    :param config: LLM configuration for this transport.
    :type config: LLMConfig
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialise the transport.  The SDK client is created lazily.

        :param config: LLM configuration.
        :type config: LLMConfig
        """
        self._config = config
        self._client: Any = client  # created lazily in _get_client()
        self._client_factory = client_factory
        self._owns_client = client is None and client_factory is None

    def _get_client(self) -> Any:
        """Return the cached async ``anthropic.AsyncAnthropic`` client.

        Creates the client on first call.

        :return: The async Anthropic client.
        :raises ImportError: If ``anthropic`` is not installed.
        """
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
                return self._client
            anthropic = _require_anthropic()
            kwargs: dict[str, Any] = dict(self._config.client_options or {})
            kwargs.setdefault("max_retries", 0)  # We handle retries ourselves.
            kwargs.setdefault("timeout", self._config.timeout)
            if self._config.api_key is not None:
                kwargs.setdefault("api_key", self._config.api_key)
            if self._config.base_url is not None:
                kwargs.setdefault("base_url", self._config.base_url)
            if self._config.default_headers is not None:
                kwargs.setdefault("default_headers", dict(self._config.default_headers))
            if self._config.default_query is not None:
                kwargs.setdefault("default_query", dict(self._config.default_query))
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    def get_client(self) -> Any:
        """Return the underlying async client, creating it lazily."""

        return self._get_client()

    async def close(self) -> None:
        """Close a Lauren-owned async client, if it has been created."""

        if self._client is None or not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if isinstance(result, Awaitable):
                await result
        self._client = None

    def _build_system(self, system: str | None) -> Any:
        """Build the system prompt parameter for the Anthropic API.

        When ``config.cache_system_prompt`` is ``True``, wraps the system text
        in the prompt-cache ephemeral format.

        :param system: The system prompt text, or ``None``.
        :type system: str | None
        :return: A string or a list dict suitable for the ``system`` param.
        :rtype: Any
        """
        if not system:
            return None
        if self._config.cache_system_prompt:
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system

    def _build_tools(
        self,
        tools: list[ToolSchema] | None,
    ) -> list[dict[str, Any]] | None:
        """Build the tools list for the Anthropic API.

        Attaches cache-control headers to the last tool when
        ``config.cache_tools`` is ``True``.

        :param tools: Tool schemas to convert.
        :type tools: list[ToolSchema] | None
        :return: Anthropic-compatible tools list or ``None``.
        :rtype: list[dict[str, Any]] | None
        """
        if not tools:
            return None
        result = [_tool_schema_to_anthropic(t) for t in tools]
        if self._config.cache_tools and result:
            # Apply cache_control to the last tool only.
            result[-1]["cache_control"] = {"type": "ephemeral"}
        return result

    async def _call_with_retry(
        self,
        coro_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call an async function with exponential back-off retry.

        Retries on :class:`~lauren_ai._exceptions.TransientTransportError` up to
        ``config.max_retries`` times.

        :param coro_fn: The coroutine function to call.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: The return value of *coro_fn*.
        :raises TransientTransportError: After all retries are exhausted.
        :raises AuthTransportError: Immediately on 401/403.
        :raises TransportError: For other unretryable API errors.
        """
        max_retries = max(0, self._config.max_retries)
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                classified = self._classify_exception(exc)
                if classified is not None:
                    raise classified from exc
                # Check if this is a transient error we surfaced already.
                if isinstance(exc, TransientTransportError):
                    last_exc = exc
                    if attempt < max_retries:
                        wait = (2**attempt) * 0.5
                        await asyncio.sleep(wait)
                    continue
                raise
        # All retries exhausted.
        if last_exc is not None:
            raise last_exc
        raise TransportError("Unexpected retry loop exit")

    def _classify_exception(self, exc: Exception) -> TransportError | None:
        """Inspect an SDK exception and return the appropriate transport error.

        :param exc: The exception raised by the Anthropic SDK.
        :type exc: Exception
        :return: A classified :class:`TransportError` subclass, or ``None``
            if the exception is not a known Anthropic API error.
        :rtype: TransportError | None
        """
        try:
            import anthropic as _anthropic  # noqa: PLC0415
        except ImportError:
            return None

        status_code: int | None = getattr(exc, "status_code", None)

        if isinstance(exc, _anthropic.RateLimitError):
            return TransientTransportError(
                str(exc),
                status_code=429,
                provider="anthropic",
                cause=exc,
            )
        if isinstance(exc, _anthropic.InternalServerError):
            code = status_code or 500
            return TransientTransportError(
                str(exc),
                status_code=code,
                provider="anthropic",
                cause=exc,
            )
        if isinstance(exc, _anthropic.AuthenticationError):
            return AuthTransportError(
                str(exc),
                status_code=status_code or 401,
                provider="anthropic",
                cause=exc,
            )
        if isinstance(exc, _anthropic.PermissionDeniedError):
            return AuthTransportError(
                str(exc),
                status_code=status_code or 403,
                provider="anthropic",
                cause=exc,
            )
        if isinstance(exc, _anthropic.APIStatusError):
            code = status_code or 0
            if code in (429, 529):
                return TransientTransportError(
                    str(exc),
                    status_code=code,
                    provider="anthropic",
                    cause=exc,
                )
            if code in (401, 403):
                return AuthTransportError(
                    str(exc),
                    status_code=code,
                    provider="anthropic",
                    cause=exc,
                )
            return TransportError(
                str(exc),
                status_code=code,
                provider="anthropic",
                cause=exc,
            )
        if isinstance(exc, _anthropic.APIConnectionError):
            return TransientTransportError(
                str(exc),
                provider="anthropic",
                cause=exc,
            )
        return None

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
        top_p: float | None = None,
        max_completion_tokens: int | None = None,
        request_options: RequestOptions | None = None,
    ) -> Completion | AsyncIterator[CompletionChunk]:
        """Send messages to Anthropic and return the completion.

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param model: Anthropic model identifier.
        :type model: str
        :param system: System prompt.
        :type system: str | None
        :param tools: Tool schemas.
        :type tools: list[ToolSchema] | None
        :param tool_choice: Tool choice constraint.
        :type tool_choice: ToolChoice | None
        :param max_tokens: Maximum output tokens.
        :type max_tokens: int
        :param temperature: Sampling temperature.
        :type temperature: float
        :param stop_sequences: Custom stop sequences.
        :type stop_sequences: list[str] | None
        :param stream: Whether to return a streaming iterator.
        :type stream: bool
        :param thinking: Enable extended thinking.
        :type thinking: bool
        :param thinking_budget_tokens: Token budget for thinking.
        :type thinking_budget_tokens: int
        :return: A :class:`~lauren_ai._transport.Completion` or async iterator
            of :class:`~lauren_ai._transport.CompletionChunk`.
        :rtype: Completion | AsyncIterator[CompletionChunk]
        :raises TransientTransportError: On 429/529/5xx responses after retries.
        :raises AuthTransportError: On 401/403 responses.
        :raises TransportError: On other provider errors.
        """
        client = self._get_client()
        anthropic_messages = [_message_to_anthropic(m) for m in messages]
        if self._config.cache_conversation:
            _apply_conversation_cache(anthropic_messages)
        built_system = self._build_system(system)
        built_tools = self._build_tools(tools)

        # Build the kwargs dict to pass to the API.
        effective_options = (self._config.request_options or RequestOptions()).merged(request_options)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if max_completion_tokens is not None:
            call_kwargs["max_tokens"] = max_completion_tokens
        if built_system is not None:
            call_kwargs["system"] = built_system
        if built_tools is not None:
            call_kwargs["tools"] = built_tools
        if tool_choice is not None:
            call_kwargs["tool_choice"] = _tool_choice_to_anthropic(tool_choice)
        if stop_sequences is not None:
            call_kwargs["stop_sequences"] = stop_sequences
        provider_options = dict(effective_options.provider or {})
        if thinking and "thinking" in provider_options:
            raise ValueError("Duplicate Anthropic request option 'thinking'")
        if thinking:
            call_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
        elif "thinking" in provider_options:
            call_kwargs["thinking"] = provider_options.pop("thinking")
        else:
            call_kwargs["temperature"] = temperature

        for key, value in provider_options.items():
            if key in call_kwargs and call_kwargs[key] != value:
                raise ValueError(f"Duplicate Anthropic request option {key!r}")
            if effective_options.extra_body and key in effective_options.extra_body:
                raise ValueError(f"Duplicate Anthropic request option {key!r} in provider and extra_body")
            call_kwargs[key] = value
        if top_p is not None:
            if "top_p" in call_kwargs:
                raise ValueError("Duplicate Anthropic request option 'top_p'")
            call_kwargs["top_p"] = top_p
        if effective_options.extra_headers:
            call_kwargs["extra_headers"] = dict(effective_options.extra_headers)
        if effective_options.extra_query:
            call_kwargs["extra_query"] = dict(effective_options.extra_query)
        if effective_options.extra_body:
            duplicate_keys = set(effective_options.extra_body).intersection(call_kwargs)
            if duplicate_keys:
                names = ", ".join(sorted(duplicate_keys))
                raise ValueError(f"Duplicate Anthropic request option(s) in extra_body: {names}")
            call_kwargs["extra_body"] = dict(effective_options.extra_body)
        if effective_options.timeout is not None:
            call_kwargs["timeout"] = effective_options.timeout

        if stream:
            return self._stream(client, call_kwargs, include_raw_response=effective_options.include_raw_response)
        else:
            return await self._complete_sync(
                client,
                call_kwargs,
                model=model,
                max_retries=effective_options.max_retries,
                include_raw_response=effective_options.include_raw_response,
            )

    async def _complete_sync(
        self,
        client: Any,
        call_kwargs: dict[str, Any],
        *,
        model: str,
        max_retries: int | None = None,
        include_raw_response: bool = False,
    ) -> Completion:
        """Non-streaming completion with retry.

        :param client: The async Anthropic client.
        :param call_kwargs: Arguments for the API call.
        :param model: Model name for the resulting :class:`~lauren_ai._transport.Completion`.
        :return: A :class:`~lauren_ai._transport.Completion`.
        :rtype: Completion
        """
        max_retries = max(0, self._config.max_retries if max_retries is None else max_retries)
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await client.messages.create(**call_kwargs)
                return self._response_to_completion(
                    response,
                    model=model,
                    include_raw_response=include_raw_response,
                )
            except Exception as exc:  # noqa: BLE001
                classified = self._classify_exception(exc)
                if classified is not None:
                    if isinstance(classified, TransientTransportError) and attempt < max_retries:
                        last_exc = classified
                        wait = (2**attempt) * 0.5
                        await asyncio.sleep(wait)
                        continue
                    raise classified from exc
                raise

        if last_exc is not None:
            raise last_exc
        raise TransportError("Unexpected retry loop exit", provider="anthropic")

    def _response_to_completion(
        self,
        response: Any,
        *,
        model: str,
        include_raw_response: bool = False,
    ) -> Completion:
        """Convert an Anthropic SDK ``Message`` response to a
        :class:`~lauren_ai._transport.Completion`.

        :param response: The Anthropic SDK response object.
        :param model: Model name.
        :return: Canonical :class:`~lauren_ai._transport.Completion`.
        :rtype: Completion
        """
        usage_obj = getattr(response, "usage", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_obj, "input_tokens", 0),
            output_tokens=getattr(usage_obj, "output_tokens", 0),
            cache_read_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
            provider_metadata={"estimated": usage_obj is None} if usage_obj is not None else {"estimated": True},
        )
        content_blocks = getattr(response, "content", [])
        text = _extract_text_from_response(content_blocks)
        tool_calls = _extract_tool_calls_from_response(content_blocks)
        thinking_blocks = _extract_thinking_blocks(content_blocks)
        stop_reason = _parse_stop_reason(getattr(response, "stop_reason", None))
        return Completion(
            id=getattr(response, "id", f"msg_{uuid.uuid4().hex[:16]}"),
            model=getattr(response, "model", model),
            content=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,  # type: ignore[arg-type]
            usage=usage,
            thinking_blocks=thinking_blocks,
            provider="anthropic",
            request_id=getattr(response, "id", None),
            provider_metadata=self._response_metadata(response),
            raw_response=response if include_raw_response else None,
        )

    @staticmethod
    def _response_metadata(response: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for name in ("model", "stop_reason", "stop_sequence", "container"):
            value = getattr(response, name, None)
            if value is not None:
                metadata[name] = value
        return metadata

    async def _stream(
        self,
        client: Any,
        call_kwargs: dict[str, Any],
        *,
        include_raw_response: bool = False,
    ) -> AsyncIterator[CompletionChunk]:
        """Perform a streaming completion and yield
        :class:`~lauren_ai._transport.CompletionChunk` objects.

        :param client: The async Anthropic client.
        :param call_kwargs: Arguments for the API streaming call.
        :return: Async iterator of chunks.
        :rtype: AsyncIterator[CompletionChunk]
        """
        # We need to use the streaming context manager from the Anthropic SDK.
        # The pattern is: async with client.messages.stream(**kwargs) as stream:
        try:
            async with client.messages.stream(**call_kwargs) as stream:
                _current_tool_use_id: str | None = None
                _current_tool_name: str | None = None
                # Tool blocks where the name was empty on content_block_start.
                # Tracked so we can fall back to a non-streaming call to recover names.
                _empty_name_tool_ids: list[str] = []

                async for event in stream:
                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta is None:
                            continue
                        delta_type = getattr(delta, "type", None)

                        if delta_type == "text_delta":
                            yield CompletionChunk(
                                delta=getattr(delta, "text", ""),
                                provider_metadata={"provider": "anthropic"},
                                raw_event=event if include_raw_response else None,
                            )

                        elif delta_type == "thinking_delta":
                            yield CompletionChunk(
                                thinking_delta=getattr(delta, "thinking", ""),
                                provider_metadata={"provider": "anthropic"},
                                raw_event=event if include_raw_response else None,
                            )

                        elif delta_type == "signature_delta":
                            # Finalises the current thinking block: its signature
                            # must be round-tripped verbatim (PRD-137 A).
                            yield CompletionChunk(
                                thinking_signature=getattr(delta, "signature", ""),
                                provider_metadata={"provider": "anthropic"},
                                raw_event=event if include_raw_response else None,
                            )

                        elif delta_type == "input_json_delta" and _current_tool_use_id is not None:
                            yield CompletionChunk(
                                tool_call_delta=ToolCallDelta(
                                    tool_use_id=_current_tool_use_id,
                                    name=None,
                                    input_delta=getattr(delta, "partial_json", ""),
                                )
                            )

                    elif event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block is None:
                            continue
                        block_type = getattr(block, "type", None)
                        if block_type == "redacted_thinking":
                            # Delivered whole (opaque data, no deltas).  Preserve
                            # verbatim for passback (PRD-137 A).
                            yield CompletionChunk(
                                redacted_thinking_data=getattr(block, "data", ""),
                                provider_metadata={"provider": "anthropic"},
                                raw_event=event if include_raw_response else None,
                            )
                        elif block_type == "tool_use":
                            tc_id = getattr(block, "id", None)
                            tc_name = getattr(block, "name", None) or ""
                            if tc_id:
                                if tc_name:
                                    _current_tool_use_id = tc_id
                                    _current_tool_name = tc_name
                                    yield CompletionChunk(
                                        tool_call_delta=ToolCallDelta(
                                            tool_use_id=tc_id,
                                            name=tc_name,
                                            input_delta="",
                                        )
                                    )
                                else:
                                    # Empty name — suppress header and input_json_delta
                                    # chunks for this block; record for fallback.
                                    _empty_name_tool_ids.append(tc_id)
                                    _current_tool_use_id = None
                                    _current_tool_name = None

                    elif event_type == "content_block_stop":
                        _current_tool_use_id = None
                        _current_tool_name = None

                    elif event_type == "message_delta":
                        delta = getattr(event, "delta", None)
                        usage_obj = getattr(event, "usage", None)
                        stop_reason = _parse_stop_reason(getattr(delta, "stop_reason", None) if delta else None)
                        usage = None
                        if usage_obj is not None:
                            usage = TokenUsage(
                                input_tokens=0,
                                output_tokens=getattr(usage_obj, "output_tokens", 0),
                            )

                        # Fall back to a non-streaming call when any tool block had
                        # an empty name — mirrors the OpenAI transport's behaviour.
                        if stop_reason == "tool_use" and _empty_name_tool_ids:
                            logger.debug(
                                "anthropic_transport: tool call name missing after "
                                "streaming — fetching non-streaming response"
                            )
                            try:
                                sync_resp = await client.messages.create(**call_kwargs)
                                for tc_block in getattr(sync_resp, "content", []):
                                    if getattr(tc_block, "type", None) == "tool_use":
                                        tc_name = getattr(tc_block, "name", "") or ""
                                        tc_input = getattr(tc_block, "input", {}) or {}
                                        if tc_name:
                                            logger.debug(
                                                "anthropic_transport: fallback tool call name=%r",
                                                tc_name,
                                            )
                                            yield CompletionChunk(
                                                tool_call_delta=ToolCallDelta(
                                                    tool_use_id=f"fb_{uuid.uuid4().hex[:16]}",
                                                    name=tc_name,
                                                    input_delta=json.dumps(tc_input),
                                                )
                                            )
                            except Exception as _fb_exc:  # noqa: BLE001
                                logger.warning(
                                    "anthropic_transport: fallback non-streaming call failed: %s",
                                    _fb_exc,
                                )

                        yield CompletionChunk(
                            stop_reason=stop_reason,
                            usage=usage,
                            provider_metadata={"provider": "anthropic"},
                            raw_event=event if include_raw_response else None,
                        )

        except Exception as exc:  # noqa: BLE001
            classified = self._classify_exception(exc)
            if classified is not None:
                raise classified from exc
            raise

    async def embed(
        self,
        inputs: list[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[Embedding]:
        """Generate embeddings using the Anthropic API.

        .. note::
            Anthropic's embedding API availability varies by contract.  This
            method raises :class:`NotImplementedError` if the client does not
            expose an embeddings endpoint.

        :param inputs: Strings to embed.
        :type inputs: list[str]
        :param model: Embedding model identifier.
        :type model: str
        :param dimensions: Desired dimensions (may not be supported).
        :type dimensions: int | None
        :return: List of :class:`~lauren_ai._transport.Embedding` objects.
        :rtype: list[Embedding]
        :raises NotImplementedError: If embeddings are not available via this
            client.
        """
        client = self._get_client()
        if not hasattr(client, "embeddings"):
            raise NotImplementedError(
                "Anthropic embeddings are not available with the installed SDK version "
                "or account tier.  Use a different provider for embeddings, "
                "e.g. LLMConfig.for_openai()."
            )
        try:
            response = await client.embeddings.create(
                model=model,
                input=inputs,
                **({"dimensions": dimensions} if dimensions is not None else {}),
            )
            return [Embedding(index=i, vector=item.embedding) for i, item in enumerate(response.data)]
        except Exception as exc:  # noqa: BLE001
            classified = self._classify_exception(exc)
            if classified is not None:
                raise classified from exc
            raise

    async def structured_complete(
        self,
        messages: list[Message],
        *,
        model: str,
        model_cls: type[Any],
    ) -> Any:
        """Use Anthropic native structured parsing when available."""

        client = self._get_client()
        parse = getattr(getattr(client, "messages", None), "parse", None)
        if parse is None:
            raise NotImplementedError("The installed Anthropic SDK does not expose messages.parse")
        anthropic_messages = [_message_to_anthropic(message) for message in messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._config.max_tokens,
            "messages": anthropic_messages,
            "output_format": model_cls,
        }
        result = await parse(**kwargs)
        if isinstance(result, model_cls):
            return result
        for name in ("parsed_output", "parsed", "output"):
            parsed = getattr(result, name, None)
            if isinstance(parsed, model_cls):
                return parsed
        raise ValueError("Anthropic structured response did not contain a parsed output")

    async def count_tokens(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> int:
        """Count the tokens that would be consumed by these messages.

        Uses the Anthropic ``messages.count_tokens`` beta endpoint when
        available; falls back to the 4-chars-per-token heuristic otherwise.

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param model: Anthropic model identifier.
        :type model: str
        :param system: System prompt.
        :type system: str | None
        :param tools: Tool schemas to include in the count.
        :type tools: list[ToolSchema] | None
        :return: Estimated token count.
        :rtype: int
        """
        client = self._get_client()
        anthropic_messages = [_message_to_anthropic(m) for m in messages]
        built_system = self._build_system(system)
        built_tools = self._build_tools(tools)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
        }
        if built_system is not None:
            call_kwargs["system"] = built_system
        if built_tools is not None:
            call_kwargs["tools"] = built_tools

        # Prefer the stable endpoint. Older SDKs exposed the same operation
        # below beta.messages, so keep that as a compatibility fallback.
        messages_client = getattr(client, "messages", None)
        count_fn = getattr(messages_client, "count_tokens", None) if messages_client else None
        if count_fn is None:
            beta_client = getattr(client, "beta", None)
            messages_client = getattr(beta_client, "messages", None) if beta_client else None
            count_fn = getattr(messages_client, "count_tokens", None) if messages_client else None

        if count_fn is not None:
            try:
                response = await count_fn(**call_kwargs)
                return getattr(response, "input_tokens", 0)
            except Exception:  # noqa: BLE001
                pass  # Fall through to heuristic.

        # Heuristic fallback: 4 chars ≈ 1 token.
        return _heuristic_count(messages, system, tools)


# ---------------------------------------------------------------------------
# Heuristic token count (fallback)
# ---------------------------------------------------------------------------


def _heuristic_count(
    messages: list[Message],
    system: str | None,
    tools: list[ToolSchema] | None,
) -> int:
    """Estimate token count using the shared 4-chars-per-token heuristic.

    Delegates to :func:`~lauren_ai._transport.estimate_message_tokens`, which
    tolerates both :class:`Message` dataclasses and the plain dict messages the
    runner passes — so the fallback never raises ``AttributeError`` when the
    Anthropic count-tokens endpoint is unavailable (e.g. a proxied gateway).

    :param messages: Conversation messages.
    :type messages: list[Message]
    :param system: System prompt.
    :type system: str | None
    :param tools: Tool schemas.
    :type tools: list[ToolSchema] | None
    :return: Estimated token count.
    :rtype: int
    """
    return estimate_message_tokens(messages, system, tools)

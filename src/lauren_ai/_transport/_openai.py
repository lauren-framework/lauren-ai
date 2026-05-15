"""OpenAI Chat Completions transport for ``lauren-ai``.

Wraps the official ``openai`` Python SDK (async client).  The SDK is an
optional dependency — it is imported lazily so that the rest of the package
can be used without it installed.

Install with::

    pip install lauren-ai[openai]
    # or
    pip install openai>=1.35

:class:`OpenAITransport` implements the
:class:`~lauren_ai._transport.Transport` protocol.

Notes
-----
* ``o1`` / ``o3`` models do not support ``temperature`` — it is omitted
  automatically.
* ``reasoning_effort`` (``"low"`` / ``"medium"`` / ``"high"``) is supported
  for ``o1`` / ``o3`` models via the ``reasoning_effort`` parameter.
* Tool calls use OpenAI's ``tool_calls`` / ``tools`` API (not the legacy
  ``function_call`` API).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from lauren_ai._config import LLMConfig
from lauren_ai._exceptions import AuthTransportError, TransientTransportError, TransportError
from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    Embedding,
    Message,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolSchema,
)

__all__ = ["OpenAITransport"]

_OPENAI_IMPORT_ERROR = (
    "The 'openai' package is required to use OpenAITransport.\n"
    "Install it with: pip install openai>=1.35\n"
    "Or install the extras: pip install lauren-ai[openai]"
)

# Models that do not support temperature.
_NO_TEMPERATURE_MODELS = frozenset({"o1", "o1-mini", "o1-preview", "o3", "o3-mini"})


def _require_openai() -> Any:
    """Import and return the ``openai`` module, raising a helpful error if absent.

    :return: The ``openai`` module.
    :raises ImportError: If ``openai`` is not installed.
    """
    try:
        import openai  # noqa: PLC0415

        return openai
    except ImportError as exc:
        raise ImportError(_OPENAI_IMPORT_ERROR) from exc


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def _content_block_to_openai(block: Any) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.ContentBlock` (or plain dict) to OpenAI format.

    Accepts both ``ContentBlock`` dataclass instances and plain dicts (as stored
    by ``ShortTermMemory`` at runtime).

    :param block: The content block to convert.
    :return: OpenAI-compatible content dict.
    :rtype: dict[str, Any]
    """
    block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
    if block_type == "text":
        text = block.get("text", "") if isinstance(block, dict) else (block.text or "")
        return {"type": "text", "text": text}
    if block_type == "image":
        source = block.get("source", {}) if isinstance(block, dict) else (block.source or {})
        return {"type": "image_url", "image_url": source}
    if block_type == "tool_result":
        # Tool results are handled as separate messages in OpenAI's format.
        content = block.get("content", "") if isinstance(block, dict) else block.content
        if isinstance(content, list):
            content = json.dumps(content)
        return {"type": "text", "text": str(content or "")}
    # tool_use blocks don't appear in outgoing user messages normally.
    return {"type": "text", "text": ""}


def _message_to_openai(message: Any) -> list[dict[str, Any]]:
    """Convert a :class:`~lauren_ai._transport.Message` (or plain dict) to OpenAI message dict(s).

    Accepts both ``Message`` dataclass instances and plain dicts (as stored by
    ``ShortTermMemory`` at runtime).  OpenAI uses a flat message list; tool
    results are emitted as separate ``role="tool"`` messages.

    :param message: The message to convert.
    :return: List of OpenAI-compatible message dicts (usually length 1).
    :rtype: list[dict[str, Any]]
    """
    if isinstance(message, dict):
        role: str = message.get("role", "user")
        content: Any = message.get("content", "")
    else:
        role = message.role
        content = message.content

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    result: list[dict[str, Any]] = []
    # Separate tool_result blocks from regular content.
    regular_blocks: list[Any] = []
    tool_result_blocks: list[Any] = []

    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
        if block_type == "tool_result":
            tool_result_blocks.append(block)
        else:
            regular_blocks.append(block)

    # Emit tool result messages (role="tool") for user messages.
    for block in tool_result_blocks:
        if isinstance(block, dict):
            blk_content = block.get("content", "")
            blk_tool_use_id = block.get("tool_use_id", "")
        else:
            blk_content = block.content
            blk_tool_use_id = block.tool_use_id or ""
        if isinstance(blk_content, list):
            blk_content = json.dumps(blk_content)
        result.append(
            {
                "role": "tool",
                "tool_call_id": blk_tool_use_id,
                "content": str(blk_content or ""),
            }
        )

    # Build the main message.
    if regular_blocks or (not tool_result_blocks):
        content_parts: list[dict[str, Any]] = []
        tool_calls_list: list[dict[str, Any]] = []
        for block in regular_blocks:
            block_type = (
                block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
            )
            if block_type == "tool_use":
                if isinstance(block, dict):
                    # ShortTermMemory stores tool_use blocks with an "id" key
                    blk_id = (
                        block.get("id")
                        or block.get("tool_use_id")
                        or f"call_{uuid.uuid4().hex[:16]}"
                    )
                    blk_name = block.get("name", "")
                    blk_input = block.get("input", {})
                else:
                    blk_id = block.tool_use_id or f"call_{uuid.uuid4().hex[:16]}"
                    blk_name = block.name or ""
                    blk_input = block.input or {}
                tool_calls_list.append(
                    {
                        "id": blk_id,
                        "type": "function",
                        "function": {
                            "name": blk_name,
                            "arguments": json.dumps(blk_input),
                        },
                    }
                )
            else:
                content_parts.append(_content_block_to_openai(block))

        msg: dict[str, Any] = {"role": role}
        if content_parts:
            if len(content_parts) == 1 and content_parts[0].get("type") == "text":
                msg["content"] = content_parts[0]["text"]
            else:
                msg["content"] = content_parts
        else:
            msg["content"] = ""
        if tool_calls_list:
            msg["tool_calls"] = tool_calls_list
        result.insert(0, msg)

    return result


def _tool_schema_to_openai(schema: Any) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.ToolSchema` to the OpenAI tools format.

    Accepts both the ``ToolSchema`` dataclass and plain dicts (as returned by
    ``ToolRegistry.get_schemas()`` at runtime).

    :param schema: The tool schema to convert.
    :return: OpenAI-compatible tool dict.
    :rtype: dict[str, Any]
    """
    if isinstance(schema, dict):
        name = schema.get("name", "")
        description = schema.get("description", "")
        input_schema = schema.get("input_schema", {})
    else:
        name = schema.name
        description = schema.description
        input_schema = schema.input_schema
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
    }


def _tool_choice_to_openai(tool_choice: ToolChoice) -> Any:
    """Convert a :class:`~lauren_ai._transport.ToolChoice` to OpenAI format.

    :param tool_choice: The tool choice constraint.
    :type tool_choice: ToolChoice
    :return: OpenAI tool_choice value (str or dict).
    :rtype: Any
    """
    if tool_choice.type == "auto":
        return "auto"
    if tool_choice.type == "any":
        return "required"
    # type == "tool"
    return {"type": "function", "function": {"name": tool_choice.name or ""}}


def _parse_stop_reason(
    raw: str | None,
) -> Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]:
    """Normalise OpenAI finish_reason to canonical stop reasons.

    :param raw: The OpenAI ``finish_reason`` string.
    :type raw: str | None
    :return: Canonical stop reason.
    :rtype: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
    """
    mapping: dict[str, Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]] = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(raw or "stop", "end_turn")


# ---------------------------------------------------------------------------
# OpenAITransport
# ---------------------------------------------------------------------------


class OpenAITransport:
    """OpenAI Chat Completions transport.

    Handles:

    * Message translation between the Transport protocol and the OpenAI
      Chat Completions format (including tool-result messages as ``role="tool"``).
    * Tool calls using the ``tools`` / ``tool_calls`` API.
    * Streaming via ``stream=True`` on the async client.
    * Automatic omission of ``temperature`` for ``o1`` / ``o3`` models.
    * ``reasoning_effort`` for ``o1`` / ``o3`` models.
    * Automatic retries with exponential back-off on transient errors.

    :param config: LLM configuration for this transport.
    :type config: LLMConfig
    :param reasoning_effort: Optional reasoning effort for ``o1`` / ``o3``
        models.  Overrides any value from :class:`~lauren_ai._config.AgentConfig`.
    :type reasoning_effort: Literal["low", "medium", "high"] | None
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        """Initialise the transport.  The SDK client is created lazily.

        :param config: LLM configuration.
        :type config: LLMConfig
        :param reasoning_effort: Reasoning effort for o1/o3 models.
        :type reasoning_effort: Literal["low", "medium", "high"] | None
        """
        self._config = config
        self._reasoning_effort = reasoning_effort
        self._client: Any = None

    def _get_client(self) -> Any:
        """Return the cached async ``openai.AsyncOpenAI`` client.

        :return: The async OpenAI client.
        :raises ImportError: If ``openai`` is not installed.
        """
        if self._client is None:
            openai = _require_openai()
            kwargs: dict[str, Any] = {
                "max_retries": 0,
                "timeout": self._config.timeout,
            }
            if self._config.api_key is not None:
                kwargs["api_key"] = self._config.api_key
            if self._config.base_url is not None:
                kwargs["base_url"] = self._config.base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    def _classify_exception(self, exc: Exception) -> TransportError | None:
        """Classify an OpenAI SDK exception into a transport error.

        :param exc: The exception from the OpenAI SDK.
        :type exc: Exception
        :return: Classified :class:`TransportError` or ``None``.
        :rtype: TransportError | None
        """
        try:
            import openai as _openai  # noqa: PLC0415
        except ImportError:
            return None

        status_code: int | None = getattr(exc, "status_code", None)

        if isinstance(exc, _openai.RateLimitError):
            return TransientTransportError(
                str(exc),
                status_code=429,
                provider="openai",
                cause=exc,
            )
        if isinstance(exc, _openai.InternalServerError):
            return TransientTransportError(
                str(exc),
                status_code=status_code or 500,
                provider="openai",
                cause=exc,
            )
        if isinstance(exc, _openai.AuthenticationError):
            return AuthTransportError(
                str(exc),
                status_code=status_code or 401,
                provider="openai",
                cause=exc,
            )
        if isinstance(exc, _openai.PermissionDeniedError):
            return AuthTransportError(
                str(exc),
                status_code=status_code or 403,
                provider="openai",
                cause=exc,
            )
        if isinstance(exc, _openai.APIStatusError):
            code = status_code or 0
            if code in (429,):
                return TransientTransportError(
                    str(exc), status_code=code, provider="openai", cause=exc
                )
            if code >= 500:
                return TransientTransportError(
                    str(exc), status_code=code, provider="openai", cause=exc
                )
            if code in (401, 403):
                return AuthTransportError(str(exc), status_code=code, provider="openai", cause=exc)
            return TransportError(str(exc), status_code=code, provider="openai", cause=exc)
        if isinstance(exc, _openai.APIConnectionError):
            return TransientTransportError(str(exc), provider="openai", cause=exc)
        return None

    def _build_call_kwargs(
        self,
        messages_raw: list[dict[str, Any]],
        model: str,
        tools: list[ToolSchema] | None,
        tool_choice: ToolChoice | None,
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the keyword arguments dict for the OpenAI chat completions API.

        :param messages_raw: Pre-translated message dicts.
        :param model: Model identifier.
        :param tools: Tool schemas.
        :param tool_choice: Tool choice constraint.
        :param max_tokens: Maximum output tokens.
        :param temperature: Sampling temperature.
        :param stop_sequences: Stop sequences.
        :param stream: Whether to stream.
        :return: Keyword arguments dict.
        :rtype: dict[str, Any]
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages_raw,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # o1/o3 models don't support temperature.
        is_reasoning_model = any(model.startswith(m) for m in _NO_TEMPERATURE_MODELS)
        if not is_reasoning_model:
            kwargs["temperature"] = temperature
        else:
            # reasoning_effort for o1/o3.
            effort = self._reasoning_effort
            if effort is not None:
                kwargs["reasoning_effort"] = effort

        if tools:
            kwargs["tools"] = [_tool_schema_to_openai(t) for t in tools]
        if tool_choice is not None and tools:
            kwargs["tool_choice"] = _tool_choice_to_openai(tool_choice)
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        return kwargs

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
        """Send messages to OpenAI and return the completion.

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param model: OpenAI model identifier.
        :type model: str
        :param system: System prompt (prepended as a ``role="system"`` message).
        :type system: str | None
        :param tools: Tool schemas.
        :type tools: list[ToolSchema] | None
        :param tool_choice: Tool choice constraint.
        :type tool_choice: ToolChoice | None
        :param max_tokens: Maximum output tokens.
        :type max_tokens: int
        :param temperature: Sampling temperature (ignored for o1/o3 models).
        :type temperature: float
        :param stop_sequences: Custom stop sequences.
        :type stop_sequences: list[str] | None
        :param stream: Whether to return a streaming iterator.
        :type stream: bool
        :param thinking: Ignored for OpenAI; included for protocol compatibility.
        :type thinking: bool
        :param thinking_budget_tokens: Ignored for OpenAI.
        :type thinking_budget_tokens: int
        :return: A :class:`~lauren_ai._transport.Completion` or async iterator
            of :class:`~lauren_ai._transport.CompletionChunk`.
        :rtype: Completion | AsyncIterator[CompletionChunk]
        :raises TransientTransportError: On 429/5xx after retries.
        :raises AuthTransportError: On 401/403.
        :raises TransportError: On other provider errors.
        """
        client = self._get_client()

        # Translate all messages.
        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.extend(_message_to_openai(msg))

        call_kwargs = self._build_call_kwargs(
            openai_messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            stream=stream,
        )

        if stream:
            return self._stream(client, call_kwargs, model=model)
        else:
            return await self._complete_sync(client, call_kwargs, model=model)

    async def _complete_sync(
        self,
        client: Any,
        call_kwargs: dict[str, Any],
        *,
        model: str,
    ) -> Completion:
        """Non-streaming completion with retry.

        :param client: OpenAI async client.
        :param call_kwargs: API call kwargs (without ``stream``).
        :param model: Model name.
        :return: :class:`~lauren_ai._transport.Completion`.
        :rtype: Completion
        """
        max_retries = max(0, self._config.max_retries)
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(**call_kwargs)
                return self._response_to_completion(response, model=model)
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
        raise TransportError("Unexpected retry loop exit", provider="openai")

    def _response_to_completion(self, response: Any, *, model: str) -> Completion:
        """Convert an OpenAI chat completion response to a
        :class:`~lauren_ai._transport.Completion`.

        :param response: OpenAI SDK ``ChatCompletion`` object.
        :param model: Model name.
        :return: Canonical :class:`~lauren_ai._transport.Completion`.
        :rtype: Completion
        """
        usage_obj = getattr(response, "usage", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
        )

        choices = getattr(response, "choices", [])
        if not choices:
            return Completion(
                id=getattr(response, "id", f"chatcmpl_{uuid.uuid4().hex[:16]}"),
                model=getattr(response, "model", model),
                content="",
                tool_calls=[],
                stop_reason="end_turn",
                usage=usage,
            )

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        stop_reason = _parse_stop_reason(finish_reason)

        msg = getattr(choice, "message", None)
        content: str = ""
        tool_calls: list[ToolCall] = []

        if msg is not None:
            content = getattr(msg, "content", "") or ""
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                arguments_str = getattr(fn, "arguments", "{}") or "{}"
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments_str}
                tool_calls.append(
                    ToolCall(
                        tool_use_id=getattr(tc, "id", f"call_{uuid.uuid4().hex[:16]}"),
                        name=getattr(fn, "name", ""),
                        input=arguments,
                    )
                )

        return Completion(
            id=getattr(response, "id", f"chatcmpl_{uuid.uuid4().hex[:16]}"),
            model=getattr(response, "model", model),
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def _stream(
        self,
        client: Any,
        call_kwargs: dict[str, Any],
        *,
        model: str,
    ) -> AsyncIterator[CompletionChunk]:
        """Perform a streaming completion and yield
        :class:`~lauren_ai._transport.CompletionChunk` objects.

        :param client: OpenAI async client.
        :param call_kwargs: API call kwargs (``stream=True`` must be set).
        :param model: Model name.
        :return: Async iterator of chunks.
        :rtype: AsyncIterator[CompletionChunk]
        """
        try:
            # Partial tool call accumulation state.
            _tool_call_state: dict[int, dict[str, Any]] = {}  # index -> {id, name, args}

            async with await client.chat.completions.create(**call_kwargs) as stream:
                async for chunk in stream:
                    choices = getattr(chunk, "choices", [])
                    usage_obj = getattr(chunk, "usage", None)

                    if not choices and usage_obj is not None:
                        # Final usage chunk.
                        usage = TokenUsage(
                            input_tokens=getattr(usage_obj, "prompt_tokens", 0),
                            output_tokens=getattr(usage_obj, "completion_tokens", 0),
                        )
                        yield CompletionChunk(usage=usage)
                        continue

                    if not choices:
                        continue

                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    finish_reason = getattr(choice, "finish_reason", None)

                    if delta is not None:
                        # Text delta.
                        content = getattr(delta, "content", None)
                        if content:
                            yield CompletionChunk(delta=content)

                        # Tool call deltas.
                        raw_tc_deltas = getattr(delta, "tool_calls", None) or []
                        for tc_delta in raw_tc_deltas:
                            idx = getattr(tc_delta, "index", 0)
                            tc_id = getattr(tc_delta, "id", None)
                            fn = getattr(tc_delta, "function", None)
                            fn_name = getattr(fn, "name", None) if fn else None
                            fn_args = getattr(fn, "arguments", "") if fn else ""

                            if idx not in _tool_call_state:
                                _tool_call_state[idx] = {
                                    "id": tc_id or f"call_{uuid.uuid4().hex[:16]}",
                                    "name": fn_name or "",
                                    "args": "",
                                }
                            else:
                                if tc_id:
                                    _tool_call_state[idx]["id"] = tc_id
                                if fn_name:
                                    _tool_call_state[idx]["name"] = fn_name

                            _tool_call_state[idx]["args"] += fn_args or ""

                            yield CompletionChunk(
                                tool_call_delta=ToolCallDelta(
                                    tool_use_id=_tool_call_state[idx]["id"],
                                    name=fn_name,
                                    input_delta=fn_args or "",
                                )
                            )

                    if finish_reason is not None:
                        stop_reason = _parse_stop_reason(finish_reason)
                        # Include usage if present in the same chunk.
                        chunk_usage: TokenUsage | None = None
                        if usage_obj is not None:
                            chunk_usage = TokenUsage(
                                input_tokens=getattr(usage_obj, "prompt_tokens", 0),
                                output_tokens=getattr(usage_obj, "completion_tokens", 0),
                            )
                        yield CompletionChunk(stop_reason=stop_reason, usage=chunk_usage)

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
        """Generate embeddings using the OpenAI Embeddings API.

        :param inputs: Strings to embed.
        :type inputs: list[str]
        :param model: OpenAI embedding model identifier (e.g. ``"text-embedding-3-small"``).
        :type model: str
        :param dimensions: Desired embedding dimensions (supported on
            ``text-embedding-3-*`` models).
        :type dimensions: int | None
        :return: List of :class:`~lauren_ai._transport.Embedding` objects.
        :rtype: list[Embedding]
        :raises TransportError: On provider errors.
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {"input": inputs, "model": model}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        try:
            response = await client.embeddings.create(**kwargs)
            return [Embedding(index=item.index, vector=item.embedding) for item in response.data]
        except Exception as exc:  # noqa: BLE001
            classified = self._classify_exception(exc)
            if classified is not None:
                raise classified from exc
            raise

    async def count_tokens(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> int:
        """Estimate token count using the 4-chars-per-token heuristic.

        OpenAI does not expose a token-counting endpoint, so this method uses
        a character-based heuristic (4 chars ≈ 1 token).

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param model: Model identifier (informational).
        :type model: str
        :param system: System prompt.
        :type system: str | None
        :param tools: Tool schemas.
        :type tools: list[ToolSchema] | None
        :return: Estimated token count.
        :rtype: int
        """
        total = 0
        if system:
            total += max(1, len(system) // 4)
        if tools:
            for t in tools:
                total += max(
                    1, (len(t.name) + len(t.description) + len(json.dumps(t.input_schema))) // 4
                )
        for msg in messages:
            if isinstance(msg.content, str):
                total += max(1, len(msg.content) // 4)
            else:
                for block in msg.content:
                    if block.text:
                        total += max(1, len(block.text) // 4)
                    if block.input:
                        total += max(1, len(json.dumps(block.input)) // 4)
                    if isinstance(block.content, str):
                        total += max(1, len(block.content) // 4)
                    elif isinstance(block.content, list):
                        total += max(1, len(json.dumps(block.content)) // 4)
        return total

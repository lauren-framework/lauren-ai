"""Ollama transport for ``lauren-ai``.

Communicates directly with a locally-running Ollama server over HTTP using
``httpx``.  No Ollama-specific Python package is required — ``httpx`` is
already a core dependency of ``lauren-ai``.

Install Ollama from https://ollama.com and start a model::

    ollama pull llama3.2
    ollama serve   # starts at http://localhost:11434 by default

Then use this transport::

    from lauren_ai._config import LLMConfig

    cfg = LLMConfig.for_ollama(model="llama3.2")

:class:`OllamaTransport` implements the
:class:`~lauren_ai._transport.Transport` protocol.

API endpoints used
------------------
* ``POST /api/chat`` — chat completions (streaming + non-streaming)
* ``POST /api/embed`` — embeddings (Ollama ≥ 0.5)
* ``POST /api/embeddings`` — legacy embedding endpoint (Ollama < 0.5)

Tool call support requires Ollama ≥ 0.3 with a function-calling capable model.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from lauren_ai._config import LLMConfig
from lauren_ai._exceptions import AuthTransportError, TransientTransportError, TransportError
from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    ContentBlock,
    Embedding,
    Message,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolSchema,
)

__all__ = ["OllamaTransport"]

_HTTPX_IMPORT_ERROR = (
    "The 'httpx' package is required to use OllamaTransport.\n"
    "Install it with: pip install httpx>=0.27"
)


def _require_httpx() -> Any:
    """Import and return the ``httpx`` module, raising a helpful error if absent.

    :return: The ``httpx`` module.
    :raises ImportError: If ``httpx`` is not installed.
    """
    try:
        import httpx  # noqa: PLC0415

        return httpx
    except ImportError as exc:
        raise ImportError(_HTTPX_IMPORT_ERROR) from exc


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def _content_block_to_ollama_text(block: ContentBlock) -> str:
    """Extract text from a :class:`~lauren_ai._transport.ContentBlock` for Ollama.

    :param block: The content block.
    :type block: ContentBlock
    :return: Text representation.
    :rtype: str
    """
    if block.type == "text":
        return block.text or ""
    if block.type == "tool_result":
        content = block.content
        if isinstance(content, list):
            return json.dumps(content)
        return str(content or "")
    if block.type == "tool_use":
        return json.dumps({"tool_use": {"name": block.name, "input": block.input}})
    return ""


def _message_to_ollama(message: Message) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.Message` to Ollama's API format.

    :param message: The message to convert.
    :type message: Message
    :return: Ollama-compatible message dict.
    :rtype: dict[str, Any]
    """
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}

    # Gather text parts and handle tool results / tool use.
    tool_calls_list: list[dict[str, Any]] = []
    text_parts: list[str] = []
    tool_call_id_content: list[tuple[str, str]] = []  # (id, content)

    for block in message.content:
        if block.type == "tool_use":
            tool_calls_list.append(
                {
                    "function": {
                        "name": block.name or "",
                        "arguments": block.input or {},
                    }
                }
            )
        elif block.type == "tool_result":
            content = block.content
            if isinstance(content, list):
                content_str = json.dumps(content)
            else:
                content_str = str(content or "")
            tool_call_id_content.append((block.tool_use_id or "", content_str))
        elif block.type == "text":
            text_parts.append(block.text or "")

    result: dict[str, Any] = {"role": message.role}

    if tool_call_id_content and message.role == "user":
        # For Ollama, tool results come back in a "tool" role message.
        # We only support one tool result per message in this simplified path.
        # Return the first one as a tool role message.
        first_id, first_content = tool_call_id_content[0]
        result["role"] = "tool"
        result["content"] = first_content
    elif tool_calls_list:
        result["content"] = " ".join(text_parts)
        result["tool_calls"] = tool_calls_list
    else:
        result["content"] = " ".join(text_parts)

    return result


def _messages_to_ollama(
    messages: list[Message],
    system: str | None,
) -> list[dict[str, Any]]:
    """Translate all messages to Ollama format, prepending system if given.

    :param messages: Conversation messages.
    :type messages: list[Message]
    :param system: System prompt text.
    :type system: str | None
    :return: List of Ollama message dicts.
    :rtype: list[dict[str, Any]]
    """
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})
    for msg in messages:
        result.append(_message_to_ollama(msg))
    return result


def _tool_schema_to_ollama(schema: ToolSchema) -> dict[str, Any]:
    """Convert a :class:`~lauren_ai._transport.ToolSchema` to Ollama format.

    :param schema: The tool schema to convert.
    :type schema: ToolSchema
    :return: Ollama-compatible tool dict.
    :rtype: dict[str, Any]
    """
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.input_schema,
        },
    }


def _parse_stop_reason(
    raw: str | None,
    tool_calls: list[ToolCall],
) -> Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]:
    """Derive the canonical stop reason from Ollama response fields.

    :param raw: Ollama ``done_reason`` string (e.g. ``"stop"``).
    :type raw: str | None
    :param tool_calls: Parsed tool calls from the response.
    :type tool_calls: list[ToolCall]
    :return: Canonical stop reason.
    :rtype: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
    """
    if tool_calls:
        return "tool_use"
    mapping: dict[str, Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]] = {
        "stop": "end_turn",
        "length": "max_tokens",
    }
    return mapping.get(raw or "stop", "end_turn")


def _classify_http_error(status_code: int, text: str) -> TransportError:
    """Build the appropriate :class:`TransportError` from an HTTP status code.

    :param status_code: HTTP response status code.
    :type status_code: int
    :param text: Response body text.
    :type text: str
    :return: A :class:`TransportError` subclass.
    :rtype: TransportError
    """
    msg = f"Ollama HTTP {status_code}: {text[:200]}"
    if status_code in (401, 403):
        return AuthTransportError(msg, status_code=status_code, provider="ollama")
    if status_code == 429 or status_code >= 500:
        return TransientTransportError(msg, status_code=status_code, provider="ollama")
    return TransportError(msg, status_code=status_code, provider="ollama")


# ---------------------------------------------------------------------------
# OllamaTransport
# ---------------------------------------------------------------------------


class OllamaTransport:
    """Ollama HTTP transport.

    Communicates with a running Ollama server using ``httpx.AsyncClient``.
    No additional Python packages beyond ``httpx`` are required.

    :param config: LLM configuration for this transport.
    :type config: LLMConfig
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialise the transport.

        :param config: LLM configuration.
        :type config: LLMConfig
        """
        self._config = config
        self._base_url: str = (config.base_url or "http://localhost:11434").rstrip("/")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Return the cached ``httpx.AsyncClient``.

        :return: An ``httpx.AsyncClient`` instance.
        :raises ImportError: If ``httpx`` is not installed.
        """
        if self._client is None:
            httpx = _require_httpx()
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._config.timeout,
            )
        return self._client

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
        """Send messages to Ollama and return the completion.

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param model: Ollama model tag.
        :type model: str
        :param system: System prompt.
        :type system: str | None
        :param tools: Tool schemas (requires Ollama ≥ 0.3).
        :type tools: list[ToolSchema] | None
        :param tool_choice: Tool choice (informational; Ollama handles tool
            selection internally).
        :type tool_choice: ToolChoice | None
        :param max_tokens: Maximum tokens to generate (``num_predict`` option).
        :type max_tokens: int
        :param temperature: Sampling temperature (``temperature`` option).
        :type temperature: float
        :param stop_sequences: Custom stop sequences.
        :type stop_sequences: list[str] | None
        :param stream: Whether to return a streaming iterator.
        :type stream: bool
        :param thinking: Ignored for Ollama.
        :type thinking: bool
        :param thinking_budget_tokens: Ignored for Ollama.
        :type thinking_budget_tokens: int
        :return: :class:`~lauren_ai._transport.Completion` or async iterator of
            :class:`~lauren_ai._transport.CompletionChunk`.
        :rtype: Completion | AsyncIterator[CompletionChunk]
        :raises TransientTransportError: On 429 / 5xx responses.
        :raises AuthTransportError: On 401 / 403 responses.
        :raises TransportError: On other HTTP errors.
        """
        ollama_messages = _messages_to_ollama(messages, system)

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        if tools:
            payload["tools"] = [_tool_schema_to_ollama(t) for t in tools]

        if stream:
            return self._stream(payload)
        else:
            return await self._complete_sync(payload, model=model)

    async def _complete_sync(
        self,
        payload: dict[str, Any],
        *,
        model: str,
    ) -> Completion:
        """Non-streaming POST to ``/api/chat``.

        :param payload: JSON payload.
        :param model: Model name for the resulting Completion.
        :return: :class:`~lauren_ai._transport.Completion`.
        :rtype: Completion
        :raises TransportError: On HTTP errors.
        """
        client = self._get_client()
        try:
            response = await client.post("/api/chat", json=payload)
        except Exception as exc:  # noqa: BLE001
            httpx = _require_httpx()
            if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
                raise TransientTransportError(
                    f"Ollama connection error: {exc}",
                    provider="ollama",
                    cause=exc,
                ) from exc
            raise TransportError(str(exc), provider="ollama", cause=exc) from exc

        if response.status_code != 200:
            raise _classify_http_error(response.status_code, response.text)

        data = response.json()
        return self._parse_response(data, model=model)

    def _parse_response(self, data: dict[str, Any], *, model: str) -> Completion:
        """Parse an Ollama ``/api/chat`` JSON response into a :class:`~lauren_ai._transport.Completion`.

        :param data: Parsed JSON response dict.
        :param model: Model name.
        :return: Canonical :class:`~lauren_ai._transport.Completion`.
        :rtype: Completion
        """
        message_data = data.get("message", {})
        content: str = message_data.get("content", "") or ""

        # Parse tool calls if present.
        raw_tool_calls = message_data.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(
                ToolCall(
                    tool_use_id=tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                    name=fn.get("name", ""),
                    input=args,
                )
            )

        done_reason = data.get("done_reason")
        stop_reason = _parse_stop_reason(done_reason, tool_calls)

        prompt_eval_count = data.get("prompt_eval_count", 0) or 0
        eval_count = data.get("eval_count", 0) or 0
        usage = TokenUsage(
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
        )

        return Completion(
            id=f"ollama_{uuid.uuid4().hex[:16]}",
            model=data.get("model", model),
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def _stream(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[CompletionChunk]:
        """Stream from Ollama's ``/api/chat`` endpoint.

        Ollama streams newline-delimited JSON objects.

        :param payload: JSON payload (``stream`` must be ``True``).
        :return: Async iterator of :class:`~lauren_ai._transport.CompletionChunk`.
        :rtype: AsyncIterator[CompletionChunk]
        :raises TransportError: On HTTP errors.
        """
        client = self._get_client()
        httpx = _require_httpx()

        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise _classify_http_error(response.status_code, body.decode())

                _current_tool_use_id: str | None = None
                _current_tool_name: str | None = None

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message_data = data.get("message", {}) or {}
                    content_delta: str = message_data.get("content", "") or ""
                    done: bool = data.get("done", False)

                    # Check for tool calls in the message.
                    raw_tool_calls = message_data.get("tool_calls") or []
                    for tc in raw_tool_calls:
                        fn = tc.get("function", {})
                        args = fn.get("arguments", {})
                        if isinstance(args, dict):
                            args_str = json.dumps(args)
                        else:
                            args_str = str(args)
                        tc_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"
                        _current_tool_use_id = tc_id
                        _current_tool_name = fn.get("name", "")
                        yield CompletionChunk(
                            tool_call_delta=ToolCallDelta(
                                tool_use_id=tc_id,
                                name=_current_tool_name,
                                input_delta=args_str,
                            )
                        )

                    if content_delta:
                        yield CompletionChunk(delta=content_delta)

                    if done:
                        done_reason = data.get("done_reason", "stop")
                        tool_calls_present = bool(raw_tool_calls) or (
                            _current_tool_use_id is not None
                        )
                        stop_reason = _parse_stop_reason(
                            done_reason, [ToolCall("", "", {})] if tool_calls_present else []
                        )

                        prompt_eval_count = data.get("prompt_eval_count", 0) or 0
                        eval_count = data.get("eval_count", 0) or 0
                        usage = TokenUsage(
                            input_tokens=prompt_eval_count,
                            output_tokens=eval_count,
                        )
                        yield CompletionChunk(stop_reason=stop_reason, usage=usage)
                        break

        except TransportError:
            raise
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, httpx.TimeoutException) or isinstance(exc, httpx.ConnectError):
                raise TransientTransportError(
                    f"Ollama connection error: {exc}",
                    provider="ollama",
                    cause=exc,
                ) from exc
            raise TransportError(str(exc), provider="ollama", cause=exc) from exc

    async def embed(
        self,
        inputs: list[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[Embedding]:
        """Generate embeddings using the Ollama ``/api/embed`` endpoint.

        Falls back to ``/api/embeddings`` for older Ollama versions.

        :param inputs: Strings to embed.
        :type inputs: list[str]
        :param model: Ollama model tag (must support embeddings).
        :type model: str
        :param dimensions: Ignored — Ollama does not support dimension
            truncation.
        :type dimensions: int | None
        :return: List of :class:`~lauren_ai._transport.Embedding` objects.
        :rtype: list[Embedding]
        :raises TransportError: On HTTP errors.
        """
        client = self._get_client()

        # Try the newer /api/embed endpoint (batch support, Ollama ≥ 0.5).
        try:
            response = await client.post(
                "/api/embed",
                json={"model": model, "input": inputs},
            )
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                return [Embedding(index=i, vector=vec) for i, vec in enumerate(embeddings)]
        except Exception:  # noqa: BLE001
            pass

        # Fallback: legacy /api/embeddings (single string at a time).
        results: list[Embedding] = []
        for i, text in enumerate(inputs):
            try:
                response = await client.post(
                    "/api/embeddings",
                    json={"model": model, "prompt": text},
                )
                if response.status_code != 200:
                    raise _classify_http_error(response.status_code, response.text)
                data = response.json()
                results.append(Embedding(index=i, vector=data.get("embedding", [])))
            except TransportError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise TransportError(
                    f"Ollama embed error: {exc}",
                    provider="ollama",
                    cause=exc,
                ) from exc
        return results

    async def count_tokens(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> int:
        """Estimate token count using the 4-chars-per-token heuristic.

        Ollama does not expose a token-counting endpoint; this method uses
        a character-based heuristic.

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

"""Deterministic mock transport for unit / integration tests.

:class:`MockTransport` satisfies the :class:`~lauren_ai._transport.Transport`
protocol without making any network calls.  Queue canned
:class:`~lauren_ai._transport.Completion` objects or lists of
:class:`~lauren_ai._transport.CompletionChunk` objects before running code
under test; the transport dequeues one entry per ``complete()`` call.

Example::

    from lauren_ai._transport import (
        Completion, CompletionChunk, TokenUsage, Embedding,
    )
    from lauren_ai._transport._mock import MockTransport

    mock = MockTransport()
    mock.queue_response(
        Completion(
            id="c1",
            model="mock",
            content="Hello!",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
    )

    result = await mock.complete(
        [Message(role="user", content="Hi")],
        model="mock",
    )
    assert result.content == "Hello!"
    assert len(mock.calls) == 1
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from lauren_ai._exceptions import EmptyQueueError
from lauren_ai._transport import (
    Completion,
    CompletionCall,
    CompletionChunk,
    Embedding,
    Message,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolSchema,
)

__all__ = ["MockTransport"]


def _heuristic_token_count(text: str) -> int:
    """Estimate token count using the 4-chars-per-token heuristic.

    :param text: Input text to estimate.
    :type text: str
    :return: Estimated token count (minimum 1).
    :rtype: int
    """
    return max(1, len(text) // 4)


def _messages_token_count(
    messages: list[Message],
    system: str | None,
    tools: list[ToolSchema] | None,
) -> int:
    """Estimate total token count for a set of messages plus optional context.

    :param messages: Conversation messages.
    :type messages: list[Message]
    :param system: Optional system prompt.
    :type system: str | None
    :param tools: Optional tool schemas included in context.
    :type tools: list[ToolSchema] | None
    :return: Estimated token count.
    :rtype: int
    """
    total = 0
    if system:
        total += _heuristic_token_count(system)
    if tools:
        for tool in tools:
            total += _heuristic_token_count(tool.name)
            total += _heuristic_token_count(tool.description)
            total += _heuristic_token_count(json.dumps(tool.input_schema))
    for msg in messages:
        if isinstance(msg.content, str):
            total += _heuristic_token_count(msg.content)
        else:
            for block in msg.content:
                if block.text:
                    total += _heuristic_token_count(block.text)
                if block.input:
                    total += _heuristic_token_count(json.dumps(block.input))
                if isinstance(block.content, str):
                    total += _heuristic_token_count(block.content)
                elif isinstance(block.content, list):
                    total += _heuristic_token_count(json.dumps(block.content))
    return total


async def _iter_chunks(
    chunks: list[CompletionChunk],
) -> AsyncIterator[CompletionChunk]:
    """Yield each :class:`~lauren_ai._transport.CompletionChunk` from *chunks*.

    :param chunks: List of chunks to stream.
    :type chunks: list[CompletionChunk]
    :return: Async iterator over the chunks.
    :rtype: AsyncIterator[CompletionChunk]
    """
    for chunk in chunks:
        yield chunk


async def _completion_as_stream(
    completion: Completion,
) -> AsyncIterator[CompletionChunk]:
    """Wrap a :class:`~lauren_ai._transport.Completion` as a single-item stream.

    Yields two chunks: one with the text content and one with the stop reason
    and usage.

    :param completion: The completion to stream.
    :type completion: Completion
    :return: Async iterator of :class:`~lauren_ai._transport.CompletionChunk`.
    :rtype: AsyncIterator[CompletionChunk]
    """
    if completion.content:
        yield CompletionChunk(delta=completion.content)
    # Final chunk with stop reason and usage.
    yield CompletionChunk(
        delta="",
        stop_reason=completion.stop_reason,
        usage=completion.usage,
    )


class MockTransport:
    """Deterministic transport for tests — no network calls are ever made.

    Queue canned responses using :meth:`queue_response`, :meth:`queue_stream`,
    or :meth:`queue_tool_use` before calling code that uses this transport.
    After your test, inspect :attr:`calls` to assert on what was passed.

    :example:

    .. code-block:: python

        mock = MockTransport()
        mock.queue_response(
            Completion(
                id="1",
                model="mock",
                content="Hi there!",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=10),
            )
        )
        result = await mock.complete(
            [Message.user("Hello")], model="mock"
        )
        assert result.content == "Hi there!"
    """

    def __init__(self) -> None:
        """Initialise the mock transport with empty queues."""
        self._queue: deque[Completion | list[CompletionChunk]] = deque()
        self._calls: list[CompletionCall] = []
        self._embed_responses: deque[list[Embedding]] = deque()

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------

    def queue_response(self, completion: Completion) -> None:
        """Enqueue a canned :class:`~lauren_ai._transport.Completion`.

        The next call to :meth:`complete` (streaming or not) will consume
        this entry.

        :param completion: The completion to enqueue.
        :type completion: Completion
        """
        self._queue.append(completion)

    def queue_stream(self, chunks: list[CompletionChunk]) -> None:
        """Enqueue a list of :class:`~lauren_ai._transport.CompletionChunk` objects.

        When :meth:`complete` is called with ``stream=True``, the chunks are
        yielded in order.  When called with ``stream=False``, the chunks are
        aggregated into a single :class:`~lauren_ai._transport.Completion`.

        :param chunks: The list of chunks to enqueue.
        :type chunks: list[CompletionChunk]
        :raises ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("chunks must be a non-empty list")
        self._queue.append(chunks)

    def queue_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_use_id: str | None = None,
        model: str = "mock-model",
        content: str = "",
    ) -> None:
        """Convenience method to queue a tool-use completion.

        Builds a :class:`~lauren_ai._transport.Completion` with
        ``stop_reason="tool_use"`` and a single :class:`~lauren_ai._transport.ToolCall`.

        :param tool_name: The tool the model should call.
        :type tool_name: str
        :param tool_input: The tool call arguments.
        :type tool_input: dict[str, Any]
        :param tool_use_id: Provider-assigned identifier.  A random ID is
            generated when *None*.
        :type tool_use_id: str | None
        :param model: Model name to set on the completion.  Defaults to
            ``"mock-model"``.
        :type model: str
        :param content: Optional text content preceding the tool call.
        :type content: str
        """
        resolved_id = tool_use_id or f"toolu_{uuid.uuid4().hex[:16]}"
        completion = Completion(
            id=f"msg_{uuid.uuid4().hex[:16]}",
            model=model,
            content=content,
            tool_calls=[
                ToolCall(
                    tool_use_id=resolved_id,
                    name=tool_name,
                    input=tool_input,
                )
            ],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=50, output_tokens=20),
        )
        self._queue.append(completion)

    def queue_structured(self, instance: Any) -> None:
        """Queue a structured output instance for the next complete() call.

        The instance is serialized to a JSON dict and queued as a tool-use
        completion so that :class:`~lauren_ai._transport._structured.StructuredLLM`
        can deserialize it back into a typed object.

        :param instance: A Pydantic model instance (or any object with
            ``model_dump()`` or ``__dict__``).
        :type instance: Any
        """
        import json  # noqa: PLC0415

        input_data: dict[str, Any] = instance.model_dump() if hasattr(instance, "model_dump") else vars(instance)
        tc = ToolCall(
            tool_use_id="struct_001",
            name="structured_output",
            input=input_data,
        )
        completion = Completion(
            id="struct_001",
            model="mock",
            content=json.dumps(input_data),
            tool_calls=[tc],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
        self._queue.append(completion)

    def queue_embed(self, embeddings: list[Embedding]) -> None:
        """Enqueue a canned embedding response.

        :param embeddings: The embeddings to return.
        :type embeddings: list[Embedding]
        :raises ValueError: If *embeddings* is empty.
        """
        if not embeddings:
            raise ValueError("embeddings must be a non-empty list")
        self._embed_responses.append(embeddings)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def calls(self) -> list[CompletionCall]:
        """All :class:`~lauren_ai._transport.CompletionCall` records, oldest first.

        :return: List of recorded calls.
        :rtype: list[CompletionCall]
        """
        return list(self._calls)

    def reset(self) -> None:
        """Clear the response queue, embed queue, and call history.

        Use this between test cases when reusing a single :class:`MockTransport`
        instance.
        """
        self._queue.clear()
        self._calls.clear()
        self._embed_responses.clear()

    # ------------------------------------------------------------------
    # Transport interface
    # ------------------------------------------------------------------

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
        """Dequeue and return the next canned response.

        Records a :class:`~lauren_ai._transport.CompletionCall` for later
        inspection.

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param model: Model identifier (stored in call record).
        :type model: str
        :param system: System prompt (stored in call record).
        :type system: str | None
        :param tools: Tool schemas (stored in call record).
        :type tools: list[ToolSchema] | None
        :param tool_choice: Tool choice constraint (stored in call record).
        :type tool_choice: ToolChoice | None
        :param max_tokens: Max tokens (stored in call record).
        :type max_tokens: int
        :param temperature: Temperature (stored in call record).
        :type temperature: float
        :param stop_sequences: Stop sequences (stored in call record).
        :type stop_sequences: list[str] | None
        :param stream: Whether to return a streaming iterator.
        :type stream: bool
        :param thinking: Whether thinking was requested.
        :type thinking: bool
        :param thinking_budget_tokens: Thinking budget.
        :type thinking_budget_tokens: int
        :return: A :class:`~lauren_ai._transport.Completion` or an async
            iterator of :class:`~lauren_ai._transport.CompletionChunk`.
        :rtype: Completion | AsyncIterator[CompletionChunk]
        :raises EmptyQueueError: If no response has been queued.
        """
        if not self._queue:
            raise EmptyQueueError(f"MockTransport has no queued responses for call #{len(self._calls) + 1}")

        # Record the call.
        self._calls.append(
            CompletionCall(
                messages=messages,
                model=model,
                system=system,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                stream=stream,
                thinking=thinking,
                thinking_budget_tokens=thinking_budget_tokens,
            )
        )

        item = self._queue.popleft()

        if stream:
            # The caller wants an async iterator.
            if isinstance(item, list):
                return _iter_chunks(item)
            else:
                # Wrap the Completion as a single-item stream.
                return _completion_as_stream(item)
        else:
            # The caller wants a Completion.
            if isinstance(item, Completion):
                return item
            else:
                # Aggregate chunks into a Completion.
                return _aggregate_chunks(item, model=model)

    async def embed(
        self,
        inputs: list[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[Embedding]:
        """Return the next queued embedding response, or zeros if none queued.

        When no embedding response has been queued, returns a list of
        zero-filled vectors of dimension 1536 (one per input).

        :param inputs: Strings to embed.
        :type inputs: list[str]
        :param model: Embedding model identifier (informational).
        :type model: str
        :param dimensions: Desired dimensions (informational; not enforced).
        :type dimensions: int | None
        :return: List of :class:`~lauren_ai._transport.Embedding` objects.
        :rtype: list[Embedding]
        """
        if self._embed_responses:
            return self._embed_responses.popleft()
        # Default: zero vectors.
        dim = dimensions or 1536
        return [Embedding(index=i, vector=[0.0] * dim) for i in range(len(inputs))]

    async def count_tokens(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> int:
        """Estimate token count using the 4-chars-per-token heuristic.

        :param messages: Conversation messages.
        :type messages: list[Message]
        :param model: Model identifier (used for bookkeeping only).
        :type model: str
        :param system: Optional system prompt.
        :type system: str | None
        :param tools: Optional tool schemas.
        :type tools: list[ToolSchema] | None
        :return: Estimated token count.
        :rtype: int
        """
        return _messages_token_count(messages, system, tools)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_chunks(
    chunks: list[CompletionChunk],
    *,
    model: str,
) -> Completion:
    """Collapse a list of :class:`~lauren_ai._transport.CompletionChunk` objects
    into a single :class:`~lauren_ai._transport.Completion`.

    :param chunks: The chunks to aggregate.
    :type chunks: list[CompletionChunk]
    :param model: Model identifier for the resulting completion.
    :type model: str
    :return: A :class:`~lauren_ai._transport.Completion` with aggregated content.
    :rtype: Completion
    """
    content_parts: list[str] = []
    stop_reason: str = "end_turn"
    usage: TokenUsage = TokenUsage(input_tokens=0, output_tokens=0)

    # Tool call accumulation.
    _tool_calls: dict[str, dict[str, Any]] = {}  # tool_use_id -> {name, input_json}

    for chunk in chunks:
        if chunk.delta:
            content_parts.append(chunk.delta)
        if chunk.stop_reason:
            stop_reason = chunk.stop_reason
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.tool_call_delta is not None:
            td = chunk.tool_call_delta
            if td.tool_use_id not in _tool_calls:
                _tool_calls[td.tool_use_id] = {
                    "name": td.name or "",
                    "input_json": "",
                }
            _tool_calls[td.tool_use_id]["input_json"] += td.input_delta
            if td.name and not _tool_calls[td.tool_use_id]["name"]:
                _tool_calls[td.tool_use_id]["name"] = td.name

    tool_calls: list[ToolCall] = []
    for tool_use_id, data in _tool_calls.items():
        try:
            parsed_input: dict[str, Any] = json.loads(data["input_json"]) if data["input_json"] else {}
        except json.JSONDecodeError:
            parsed_input = {"_raw": data["input_json"]}
        tool_calls.append(
            ToolCall(
                tool_use_id=tool_use_id,
                name=data["name"],
                input=parsed_input,
            )
        )

    valid_stop_reasons = {"end_turn", "tool_use", "max_tokens", "stop_sequence"}
    if stop_reason not in valid_stop_reasons:
        stop_reason = "end_turn"

    return Completion(
        id=f"mock_{uuid.uuid4().hex[:16]}",
        model=model,
        content="".join(content_parts),
        tool_calls=tool_calls,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=usage,
    )

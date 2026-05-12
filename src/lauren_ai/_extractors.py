"""Handler parameter extractor markers for ``lauren-ai``.

These :class:`~lauren.extractors.ExtractionMarker` subclasses let ``lauren``
HTTP handler parameters be resolved from AI services rather than the raw
HTTP request.

Provided markers
----------------

* :class:`Agent` — resolve a registered ``@agent()``-decorated class instance
  from the DI container.
* :class:`Completion` — run a single structured LLM call and return a
  validated ``T`` instance.
* :class:`Embed` — compute an embedding vector for the request body text.
* :class:`StreamCompletion` — return a streaming completion iterator.

Usage::

    from lauren_ai.extractors import Agent, Completion, Embed, StreamCompletion

    @post("/research")
    async def research(
        self,
        body: Json[ResearchRequest],
        agent: Agent[ResearchAgent],
    ) -> dict:
        response = await agent.run(body.query)
        return {"answer": response.content, "turns": response.turns}
"""

from __future__ import annotations

__all__ = [
    "Agent",
    "Completion",
    "Embed",
    "StreamCompletion",
]

import logging
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from lauren import Scope, injectable
from lauren.extractors import ExtractionMarker

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Agent[T] extractor
# ---------------------------------------------------------------------------


@injectable(scope=Scope.SINGLETON)
class Agent(ExtractionMarker):
    """Resolve an ``@agent()``-decorated class instance from the DI container.

    Use as a handler parameter annotation to inject a resolved agent
    instance that can immediately be called with ``agent.run(message)``::

        @post("/research")
        async def research(
            self,
            body: Json[ResearchRequest],
            agent: Agent[ResearchAgent],
        ) -> dict:
            response = await agent.run(body.query)
            return {"answer": response.content}

    :param runner: The :class:`~lauren_ai._agents._runner.AgentRunner`
        singleton (injected by the DI container).
    :type runner: Any
    :param container: The DI container used to resolve the agent class.
    :type container: Any
    """

    source: str = "lauren_ai.agent"

    def __init__(self, runner: Any, container: Any) -> None:
        self._runner = runner
        self._container = container

    async def extract(
        self,
        execution_context: Any,
        extraction: Any,
    ) -> Any:
        """Resolve the agent class from the DI container.

        :param execution_context: The current request execution context.
        :type execution_context: ExecutionContext
        :param extraction: The extraction descriptor (provides ``inner_type``
            which is the agent class ``T`` in ``Agent[T]``).
        :type extraction: Extraction
        :return: A resolved instance of the ``@agent()``-decorated class.
        :rtype: Any
        :raises MissingProviderError: When the agent class is not registered
            in the DI container.
        """
        agent_cls = extraction.inner_type
        return await self._container.resolve(agent_cls)

    def __class_getitem__(cls, item: Any) -> Any:
        """Support ``Agent[MyAgent]`` subscript syntax.

        :param item: The agent class to resolve.
        :type item: Any
        :return: An ``Annotated[item, Agent]`` type hint.
        :rtype: Any
        """
        from typing import Annotated  # noqa: PLC0415

        return Annotated[item, cls]  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Completion[T] extractor
# ---------------------------------------------------------------------------


@injectable(scope=Scope.SINGLETON)
class Completion(ExtractionMarker):
    """Run a single structured LLM completion and return a typed ``T`` instance.

    The prompt is read (in priority order) from:

    1. ``execution_context.get_metadata("completion_prompt")`` — a string
       prompt set by middleware or ``@set_metadata``.
    2. ``execution_context.get_metadata("completion_prompt_field")`` — a
       field name; the value is read from the parsed request body at that
       field.
    3. Fallback: ``repr`` of ``request.body`` (or an empty string).

    Example::

        @post("/classify")
        async def classify(
            self,
            body: Json[ClassifyRequest],
            result: Completion[SentimentLabel],
        ) -> SentimentLabel:
            ...

    :param transport: The LLM transport.
    :type transport: Any
    :param config: The LLM configuration.
    :type config: LLMConfig
    """

    source: str = "lauren_ai.completion"

    def __init__(self, transport: Any, config: Any) -> None:
        self._transport = transport
        self._config = config

    async def extract(
        self,
        execution_context: Any,
        extraction: Any,
    ) -> Any:
        """Run one LLM completion call and return a validated ``T`` instance.

        :param execution_context: The current request execution context.
        :type execution_context: ExecutionContext
        :param extraction: The extraction descriptor; ``inner_type`` is the
            target Pydantic model or type ``T``.
        :type extraction: Extraction
        :return: A validated instance of ``T``.
        :rtype: Any
        :raises TransportError: When the LLM provider returns an error.
        """
        from lauren_ai._transport import Message  # noqa: PLC0415

        prompt = self._resolve_prompt(execution_context)
        inner_type = extraction.inner_type

        # Build messages
        messages = [Message(role="user", content=prompt)]

        # Try structured output via tool use
        try:
            from pydantic import TypeAdapter  # noqa: PLC0415

            adapter = TypeAdapter(inner_type)
            schema = adapter.json_schema()
            structured_tool: list[Any] = [
                {
                    "name": "structured_output",
                    "description": "Return a structured response matching the requested schema.",
                    "input_schema": schema,
                }
            ]

            completion = await self._transport.complete(
                messages,
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                tools=structured_tool,
                tool_choice={"type": "tool", "name": "structured_output"},
                stream=False,
            )

            # Extract the tool call input and validate
            if completion.tool_calls:
                raw_input = completion.tool_calls[0].input
                return adapter.validate_python(raw_input)

            # Fallback: parse JSON from content
            content = completion.content.strip()
            return adapter.validate_json(content)

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "lauren_ai.Completion.extract: structured output failed, "
                "falling back to raw content: %s",
                exc,
            )
            # Last resort: return raw completion content
            completion = await self._transport.complete(
                messages,
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                stream=False,
            )
            return completion.content

    @staticmethod
    def _resolve_prompt(execution_context: Any) -> str:
        """Determine the completion prompt from the execution context.

        :param execution_context: The current request execution context.
        :type execution_context: ExecutionContext
        :return: The prompt string.
        :rtype: str
        """
        # Priority 1: explicit prompt in metadata
        prompt = None
        if hasattr(execution_context, "get_metadata"):
            prompt = execution_context.get_metadata("completion_prompt")

        if prompt is not None:
            return str(prompt)

        # Priority 2: prompt field name — read from request state
        if hasattr(execution_context, "get_metadata"):
            field_name = execution_context.get_metadata("completion_prompt_field")
            if field_name is not None:
                request = getattr(execution_context, "request", None)
                if request is not None:
                    state = getattr(request, "state", None)
                    if state is not None:
                        body = getattr(state, "body", None) or getattr(state, "parsed_body", None)
                        if body is not None and isinstance(body, dict):
                            value = body.get(field_name)
                            if value is not None:
                                return str(value)

        # Priority 3: repr of request body
        request = getattr(execution_context, "request", None)
        if request is not None:
            body = getattr(request, "body", None)
            if body is not None:
                return repr(body)

        return ""

    def __class_getitem__(cls, item: Any) -> Any:
        """Support ``Completion[T]`` subscript syntax.

        :param item: The return type.
        :type item: Any
        :return: An ``Annotated[item, Completion]`` type hint.
        :rtype: Any
        """
        from typing import Annotated  # noqa: PLC0415

        return Annotated[item, cls]  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Embed[T] extractor
# ---------------------------------------------------------------------------


@injectable(scope=Scope.SINGLETON)
class Embed(ExtractionMarker):
    """Compute an embedding vector for the request body text.

    Reads the text to embed from (in priority order):

    1. ``request.state.embed_input`` — set by middleware or the caller.
    2. The raw request body decoded as UTF-8.
    3. Empty string fallback.

    Example::

        @post("/semantic-search")
        async def search(
            self,
            body: Json[SearchRequest],
            query_embedding: Embed[list[float]],
        ) -> list[SearchResult]:
            ...

    :param transport: The LLM transport (used for embedding calls).
    :type transport: Any
    :param config: The LLM configuration.
    :type config: LLMConfig
    """

    source: str = "lauren_ai.embed"

    def __init__(self, transport: Any, config: Any) -> None:
        self._transport = transport
        self._config = config

    async def extract(
        self,
        execution_context: Any,
        extraction: Any,
    ) -> list[float]:
        """Compute an embedding and return the first vector.

        :param execution_context: The current request execution context.
        :type execution_context: ExecutionContext
        :param extraction: The extraction descriptor.
        :type extraction: Extraction
        :return: The embedding vector as a list of floats.
        :rtype: list[float]
        :raises TransportError: When the embedding call fails.
        """
        embed_input = self._resolve_embed_input(execution_context)
        embed_model = self._config.embed_model or self._config.model

        embeddings = await self._transport.embed(
            [embed_input],
            model=embed_model,
            dimensions=self._config.embed_dimensions,
        )

        if embeddings:
            return embeddings[0].vector
        return []

    @staticmethod
    def _resolve_embed_input(execution_context: Any) -> str:
        """Determine the text to embed from the execution context.

        :param execution_context: The current request execution context.
        :type execution_context: ExecutionContext
        :return: The text to embed.
        :rtype: str
        """
        request = getattr(execution_context, "request", None)
        if request is not None:
            state = getattr(request, "state", None)
            if state is not None:
                embed_input = getattr(state, "embed_input", None)
                if embed_input is not None:
                    return str(embed_input)

        # Fallback: try to read the raw body
        if request is not None:
            body = getattr(request, "body", None)
            if body is not None:
                if isinstance(body, bytes):
                    return body.decode("utf-8", errors="replace")
                return str(body)

        return ""

    def __class_getitem__(cls, item: Any) -> Any:
        """Support ``Embed[T]`` subscript syntax.

        :param item: The return type annotation.
        :type item: Any
        :return: An ``Annotated[item, Embed]`` type hint.
        :rtype: Any
        """
        from typing import Annotated  # noqa: PLC0415

        return Annotated[item, cls]  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# StreamCompletion[T] extractor
# ---------------------------------------------------------------------------


@injectable(scope=Scope.SINGLETON)
class StreamCompletion(ExtractionMarker):
    """Return a streaming completion iterator for use with ``EventStream``.

    Example::

        @post("/chat")
        async def chat(
            self,
            body: Json[ChatRequest],
            stream: StreamCompletion[str],
        ) -> EventStream:
            return EventStream(stream, keep_alive=15.0)

    The returned iterator yields
    :class:`~lauren_ai._transport.CompletionChunk` items as they arrive
    from the provider.

    :param transport: The LLM transport.
    :type transport: Any
    :param config: The LLM configuration.
    :type config: LLMConfig
    """

    source: str = "lauren_ai.stream_completion"

    def __init__(self, transport: Any, config: Any) -> None:
        self._transport = transport
        self._config = config

    async def extract(
        self,
        execution_context: Any,
        extraction: Any,
    ) -> AsyncIterator[Any]:
        """Return a streaming completion async iterator.

        Reads the prompt using the same priority order as
        :class:`Completion`.

        :param execution_context: The current request execution context.
        :type execution_context: ExecutionContext
        :param extraction: The extraction descriptor.
        :type extraction: Extraction
        :return: An async iterator of
            :class:`~lauren_ai._transport.CompletionChunk` items.
        :rtype: AsyncIterator[CompletionChunk]
        :raises TransportError: When the streaming call cannot be started.
        """
        from lauren_ai._transport import Message  # noqa: PLC0415

        # Reuse the Completion prompt resolution logic
        prompt = self._resolve_prompt(execution_context)
        messages = [Message(role="user", content=prompt)]

        stream = await self._transport.complete(
            messages,
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            stream=True,
        )
        return stream

    @staticmethod
    def _resolve_prompt(execution_context: Any) -> str:
        """Determine the completion prompt from the execution context.

        Mirrors :meth:`Completion._resolve_prompt`.

        :param execution_context: The current request execution context.
        :type execution_context: ExecutionContext
        :return: The prompt string.
        :rtype: str
        """
        prompt = None
        if hasattr(execution_context, "get_metadata"):
            prompt = execution_context.get_metadata("completion_prompt")

        if prompt is not None:
            return str(prompt)

        if hasattr(execution_context, "get_metadata"):
            field_name = execution_context.get_metadata("completion_prompt_field")
            if field_name is not None:
                request = getattr(execution_context, "request", None)
                if request is not None:
                    state = getattr(request, "state", None)
                    if state is not None:
                        body = getattr(state, "body", None) or getattr(state, "parsed_body", None)
                        if body is not None and isinstance(body, dict):
                            value = body.get(field_name)
                            if value is not None:
                                return str(value)

        request = getattr(execution_context, "request", None)
        if request is not None:
            body = getattr(request, "body", None)
            if body is not None:
                return repr(body)

        return ""

    def __class_getitem__(cls, item: Any) -> Any:
        """Support ``StreamCompletion[T]`` subscript syntax.

        :param item: The element type annotation.
        :type item: Any
        :return: An ``Annotated[item, StreamCompletion]`` type hint.
        :rtype: Any
        """
        from typing import Annotated  # noqa: PLC0415

        return Annotated[item, cls]  # type: ignore[valid-type]

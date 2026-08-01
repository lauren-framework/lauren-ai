"""Native async provider services.

These services intentionally preserve official SDK request, response, and
streaming objects. They complement the portable Transport protocol for APIs
whose state machines or event taxonomies cannot be represented losslessly by a
Completion.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from lauren_ai._config import LLMConfig
from lauren_ai._signals import ModelCallComplete, ModelCallStarted

__all__ = [
    "OpenAIClient",
    "ResponsesRequest",
    "OpenAIResponsesService",
    "OpenAIRealtimeService",
    "AnthropicClient",
    "AnthropicMessagesService",
    "AnthropicBatchService",
    "AnthropicModelsService",
]


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _emit_started(signals: Any | None, *, model: str, messages_count: int) -> float:
    """Emit a common model-start signal and return the monotonic start time."""

    started = time.monotonic()
    if signals is not None:
        await signals.emit(ModelCallStarted(model=model, messages_count=messages_count))
    return started


async def _emit_complete(
    signals: Any | None,
    *,
    model: str,
    started: float,
    result: Any = None,
) -> None:
    """Emit a common model-complete signal for a native provider result."""

    if signals is None:
        return
    usage = getattr(result, "usage", None)
    await signals.emit(
        ModelCallComplete(
            model=getattr(result, "model", model),
            usage=usage,
            duration_ms=(time.monotonic() - started) * 1000,
            stop_reason=getattr(result, "stop_reason", "unknown") or "unknown",
        )
    )


def _request_kwargs(request: Any) -> dict[str, Any]:
    if isinstance(request, Mapping):
        return {str(key): value for key, value in request.items() if value is not None}
    return {
        item.name: getattr(request, item.name) for item in fields(request) if getattr(request, item.name) is not None
    }


def _messages_count(value: Any) -> int:
    """Return a safe message count for native signal metadata."""

    return len(value) if isinstance(value, (list, tuple)) else 0


class OpenAIClient:
    """Lazy, DI-friendly wrapper around the official AsyncOpenAI client.

    Applications may use the wrapper directly for resources not yet wrapped by
    Lauren. Attribute access delegates to the official client, while the
    raw property makes the ownership boundary explicit.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._client_factory = client_factory
        self._owns_client = client is None and client_factory is None

    @property
    def raw(self) -> Any:
        """Return the official async client, creating it lazily."""

        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                if self._config is None:
                    raise RuntimeError("OpenAIClient requires config, client, or client_factory")
                from lauren_ai._transport._openai import _require_openai  # noqa: PLC0415

                kwargs: dict[str, Any] = dict(self._config.client_options or {})
                kwargs.setdefault("max_retries", 0)
                kwargs.setdefault("timeout", self._config.timeout)
                if self._config.api_key is not None:
                    kwargs.setdefault("api_key", self._config.api_key)
                if self._config.base_url is not None:
                    kwargs.setdefault("base_url", self._config.base_url)
                if self._config.default_headers is not None:
                    kwargs.setdefault("default_headers", dict(self._config.default_headers))
                if self._config.default_query is not None:
                    kwargs.setdefault("default_query", dict(self._config.default_query))
                self._client = _require_openai().AsyncOpenAI(**kwargs)
        return self._client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    async def close(self) -> None:
        """Close the client only when this wrapper created it."""

        if self._client is None or not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if close is not None:
            await _await_if_needed(close())
        self._client = None


@dataclass(frozen=True, slots=True)
class ResponsesRequest:
    """Typed common subset of an OpenAI Responses create request."""

    model: str
    input: Any
    instructions: str | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    conversation: str | None = None
    reasoning: Mapping[str, Any] | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    include: list[str] | None = None
    store: bool | None = None
    extra_headers: Mapping[str, str] | None = None
    extra_query: Mapping[str, Any] | None = None
    extra_body: Mapping[str, Any] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        """Convert the typed request to SDK keyword arguments."""

        kwargs = _request_kwargs(self)
        for key in ("extra_headers", "extra_query", "extra_body"):
            value = kwargs.pop(key, None)
            if value:
                kwargs[key] = dict(value)
        return kwargs


class OpenAIResponsesService:
    """Native service for OpenAI Responses create, stream, and retrieve."""

    def __init__(
        self,
        client: OpenAIClient | Any | None = None,
        *,
        config: LLMConfig | None = None,
        client_factory: Callable[[], Any] | None = None,
        signals: Any | None = None,
    ) -> None:
        self._client = (
            client
            if isinstance(client, OpenAIClient)
            else OpenAIClient(config, client=client, client_factory=client_factory)
        )
        self._signals = signals

    @property
    def client(self) -> OpenAIClient:
        return self._client

    async def create(
        self,
        request: ResponsesRequest | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a native Responses object."""

        params = request.to_kwargs() if isinstance(request, ResponsesRequest) else _request_kwargs(request or {})
        params.update({key: value for key, value in kwargs.items() if value is not None})
        model = str(params.get("model", ""))
        started = await _emit_started(self._signals, model=model, messages_count=1)
        result = await _await_if_needed(self._client.raw.responses.create(**params))
        await _emit_complete(self._signals, model=model, started=started, result=result)
        return result

    async def retrieve(self, response_id: str, **kwargs: Any) -> Any:
        """Retrieve a stored response by provider response ID."""

        return await _await_if_needed(self._client.raw.responses.retrieve(response_id, **kwargs))

    async def stream(
        self,
        request: ResponsesRequest | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Yield native Responses streaming events without normalization."""

        params = request.to_kwargs() if isinstance(request, ResponsesRequest) else _request_kwargs(request or {})
        params.update({key: value for key, value in kwargs.items() if value is not None})
        params["stream"] = True
        model = str(params.get("model", ""))
        started = await _emit_started(self._signals, model=model, messages_count=1)
        result = await _await_if_needed(self._client.raw.responses.create(**params))
        completed = False
        try:
            async for event in result:
                yield event
            completed = True
        finally:
            if completed:
                await _emit_complete(self._signals, model=model, started=started)

    async def close(self) -> None:
        await self._client.close()


class OpenAIRealtimeService:
    """Native session-oriented OpenAI Realtime boundary.

    Realtime sessions are intentionally returned as provider-native objects;
    they are not adapted to LLMService.complete or AgentRunner.run_stream.
    The SDK has exposed this surface under different namespaces over time, so
    the wrapper checks the stable client namespace first and the beta namespace
    second.
    """

    def __init__(
        self,
        client: OpenAIClient | Any | None = None,
        *,
        config: LLMConfig | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._client = (
            client
            if isinstance(client, OpenAIClient)
            else OpenAIClient(config, client=client, client_factory=client_factory)
        )

    @property
    def client(self) -> OpenAIClient:
        return self._client

    async def connect(self, **kwargs: Any) -> Any:
        """Open and return an SDK-native Realtime session."""

        raw = self._client.raw
        resource = getattr(raw, "realtime", None)
        if resource is None:
            beta = getattr(raw, "beta", None)
            resource = getattr(beta, "realtime", None) if beta is not None else None
        connect = getattr(resource, "connect", None) if resource is not None else None
        if connect is None:
            raise NotImplementedError("The installed OpenAI SDK does not expose a Realtime connect method")
        return await _await_if_needed(connect(**kwargs))

    async def close(self) -> None:
        await self._client.close()


class AnthropicClient:
    """Lazy, DI-friendly wrapper around the official AsyncAnthropic client."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._client_factory = client_factory
        self._owns_client = client is None and client_factory is None

    @property
    def raw(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                if self._config is None:
                    raise RuntimeError("AnthropicClient requires config, client, or client_factory")
                from lauren_ai._transport._anthropic import _require_anthropic  # noqa: PLC0415

                kwargs: dict[str, Any] = dict(self._config.client_options or {})
                kwargs.setdefault("max_retries", 0)
                kwargs.setdefault("timeout", self._config.timeout)
                if self._config.api_key is not None:
                    kwargs.setdefault("api_key", self._config.api_key)
                if self._config.base_url is not None:
                    kwargs.setdefault("base_url", self._config.base_url)
                if self._config.default_headers is not None:
                    kwargs.setdefault("default_headers", dict(self._config.default_headers))
                if self._config.default_query is not None:
                    kwargs.setdefault("default_query", dict(self._config.default_query))
                self._client = _require_anthropic().AsyncAnthropic(**kwargs)
        return self._client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    async def close(self) -> None:
        if self._client is None or not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if close is not None:
            await _await_if_needed(close())
        self._client = None


class AnthropicMessagesService:
    """Native Anthropic Messages create, stream, parse, and token count API."""

    def __init__(
        self,
        client: AnthropicClient | Any | None = None,
        *,
        config: LLMConfig | None = None,
        client_factory: Callable[[], Any] | None = None,
        signals: Any | None = None,
    ) -> None:
        self._client = (
            client
            if isinstance(client, AnthropicClient)
            else AnthropicClient(config, client=client, client_factory=client_factory)
        )
        self._signals = signals

    @property
    def client(self) -> AnthropicClient:
        return self._client

    async def create(self, **kwargs: Any) -> Any:
        """Create and return the native Anthropic Message."""

        model = str(kwargs.get("model", ""))
        started = await _emit_started(
            self._signals,
            model=model,
            messages_count=_messages_count(kwargs.get("messages")),
        )
        result = await _await_if_needed(self._client.raw.messages.create(**kwargs))
        await _emit_complete(self._signals, model=model, started=started, result=result)
        return result

    async def parse(self, *, output_format: Any, **kwargs: Any) -> Any:
        """Use native structured parsing when supported by the SDK."""

        parse = getattr(self._client.raw.messages, "parse", None)
        if parse is None:
            raise NotImplementedError("The installed Anthropic SDK does not expose messages.parse")
        model = str(kwargs.get("model", ""))
        started = await _emit_started(
            self._signals,
            model=model,
            messages_count=_messages_count(kwargs.get("messages")),
        )
        result = await _await_if_needed(parse(output_format=output_format, **kwargs))
        await _emit_complete(self._signals, model=model, started=started, result=result)
        return result

    async def count_tokens(self, **kwargs: Any) -> int:
        """Call stable message token counting and return input token count."""

        count_tokens = getattr(self._client.raw.messages, "count_tokens", None)
        if count_tokens is None:
            raise NotImplementedError("The installed Anthropic SDK does not expose messages.count_tokens")
        result = await _await_if_needed(count_tokens(**kwargs))
        if isinstance(result, Mapping):
            return int(result.get("input_tokens", 0))
        return int(getattr(result, "input_tokens", 0))

    async def stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Yield native Anthropic stream events."""

        model = str(kwargs.get("model", ""))
        started = await _emit_started(
            self._signals,
            model=model,
            messages_count=_messages_count(kwargs.get("messages")),
        )
        stream = self._client.raw.messages.stream(**kwargs)
        stream = await _await_if_needed(stream)
        completed = False
        try:
            if hasattr(stream, "__aenter__"):
                async with stream as events:
                    async for event in events:
                        yield event
            else:
                async for event in stream:
                    yield event
            completed = True
        finally:
            if completed:
                await _emit_complete(self._signals, model=model, started=started)

    async def close(self) -> None:
        await self._client.close()


class AnthropicBatchService:
    """Native Anthropic Message Batches lifecycle service."""

    def __init__(self, client: AnthropicClient | AnthropicMessagesService) -> None:
        self._client = client.client if isinstance(client, AnthropicMessagesService) else client

    async def create(self, **kwargs: Any) -> Any:
        return await _await_if_needed(self._client.raw.messages.batches.create(**kwargs))

    async def retrieve(self, batch_id: str) -> Any:
        return await _await_if_needed(self._client.raw.messages.batches.retrieve(batch_id))

    async def list(self, **kwargs: Any) -> Any:
        return await _await_if_needed(self._client.raw.messages.batches.list(**kwargs))

    async def cancel(self, batch_id: str) -> Any:
        return await _await_if_needed(self._client.raw.messages.batches.cancel(batch_id))

    async def delete(self, batch_id: str) -> Any:
        return await _await_if_needed(self._client.raw.messages.batches.delete(batch_id))

    async def results(self, batch_id: str) -> Any:
        return await _await_if_needed(self._client.raw.messages.batches.results(batch_id))


class AnthropicModelsService:
    """Native Anthropic model listing and retrieval service."""

    def __init__(self, client: AnthropicClient | AnthropicMessagesService) -> None:
        self._client = client.client if isinstance(client, AnthropicMessagesService) else client

    async def list(self, **kwargs: Any) -> Any:
        return await _await_if_needed(self._client.raw.models.list(**kwargs))

    async def retrieve(self, model_id: str) -> Any:
        return await _await_if_needed(self._client.raw.models.retrieve(model_id))

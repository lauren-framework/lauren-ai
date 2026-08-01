"""Offline tests for the OpenAI/Anthropic SDK compatibility surface."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lauren_ai import (
    AnthropicBatchService,
    AnthropicMessagesService,
    AnthropicModelsService,
    AnthropicRequestOptions,
    LLMConfig,
    LLMModule,
    LLMService,
    Message,
    OpenAIRealtimeService,
    OpenAIRequestOptions,
    OpenAIResponsesService,
    RequestOptions,
    ResponsesRequest,
)
from lauren_ai._providers import AnthropicClient, OpenAIClient
from lauren_ai._signals import ModelCallComplete, ModelCallStarted, SignalBus
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._anthropic import AnthropicTransport
from lauren_ai._transport._mock import MockTransport
from lauren_ai._transport._multimodal import AudioContent, ImageContent, UnsupportedContentError
from lauren_ai._transport._openai import OpenAITransport


class FakeOpenAICompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = SimpleNamespace(
            id="chatcmpl_test",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3, audio_tokens=1),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2, audio_tokens=4),
            ),
        )
        return response


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = FakeOpenAICompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return SimpleNamespace(
            id="msg_test",
            model=kwargs["model"],
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=5,
                cache_read_input_tokens=2,
                cache_creation_input_tokens=4,
            ),
        )

    async def count_tokens(self, **kwargs: Any) -> Any:
        self.count_calls.append(kwargs)
        return SimpleNamespace(input_tokens=42)


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()


def test_request_options_are_immutable_and_redacted() -> None:
    options = RequestOptions(
        extra_headers={"X-Trace": "trace"},
        extra_body={"api_key": "secret", "visible": {"value": 1}},
        provider={"reasoning_effort": "none"},
    )
    merged = options.merged(RequestOptions(extra_body={"visible": {"value": 2}}))

    assert dict(options.extra_headers or {}) == {"X-Trace": "trace"}
    assert merged.extra_body == {"api_key": "secret", "visible": {"value": 2}}
    assert merged.as_diagnostic()["extra_body"]["api_key"] == "[REDACTED]"
    with pytest.raises(TypeError):
        options.extra_headers["X-New"] = "value"  # type: ignore[index]


def test_request_options_reject_header_injection() -> None:
    with pytest.raises(ValueError, match="newlines"):
        RequestOptions(extra_headers={"X-Test": "ok\r\nX-Injected: yes"})


def test_typed_provider_options_compile_to_common_options() -> None:
    openai_options = OpenAIRequestOptions(reasoning_effort="none", seed=7).to_request_options()
    anthropic_options = AnthropicRequestOptions(top_k=10, thinking={"type": "adaptive"}).to_request_options()
    assert openai_options.provider == {"reasoning_effort": "none", "seed": 7}
    assert anthropic_options.provider == {"top_k": 10, "thinking": {"type": "adaptive"}}


@pytest.mark.asyncio
async def test_llm_service_preserves_zero_and_forwards_request_options() -> None:
    mock = MockTransport()
    mock.queue_response(
        Completion(
            id="c1",
            model="mock",
            content="ok",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )
    service = LLMService(
        transport=mock,
        config=LLMConfig.for_openai(
            api_key="test",
            request_options=RequestOptions(provider={"reasoning_effort": "none"}),
        ),
    )

    await service.complete(
        [Message.user("hello")],
        temperature=0.0,
        top_p=0.0,
        max_tokens=0,
        request_options=RequestOptions(extra_body={"vendor_trace": True}),
    )

    call = mock.calls[0]
    assert call.temperature == 0.0
    assert call.top_p == 0.0
    assert call.max_tokens == 0
    assert call.request_options is not None
    assert call.request_options.provider == {"reasoning_effort": "none"}
    assert call.request_options.extra_body == {"vendor_trace": True}


@pytest.mark.asyncio
async def test_openai_transport_forwards_custom_options_and_usage() -> None:
    client = FakeOpenAIClient()
    transport = OpenAITransport(
        LLMConfig.for_openai(
            model="moonshotai/Kimi-K3",
            api_key="unused",
            default_headers={"Modal-Key": "key"},
        ),
        client=client,
    )

    result = await transport.complete(
        [Message.user("hello")],
        model="moonshotai/Kimi-K3",
        temperature=0.3,
        top_p=0.95,
        max_tokens=2048,
        max_completion_tokens=0,
        request_options=RequestOptions(
            provider={"reasoning_effort": "none"},
            extra_headers={"X-Request": "request"},
            extra_query={"tenant": "test"},
            extra_body={"vendor_trace": True},
            include_raw_response=True,
        ),
    )

    assert result.content == "ok"
    call = client.completions.calls[0]
    assert call["top_p"] == 0.95
    assert call["max_completion_tokens"] == 0
    assert "max_tokens" not in call
    assert call["reasoning_effort"] == "none"
    assert call["extra_headers"] == {"X-Request": "request"}
    assert call["extra_query"] == {"tenant": "test"}
    assert call["extra_body"] == {"vendor_trace": True}
    assert result.provider == "openai"
    assert result.raw_response is not None
    assert result.usage.cache_read_tokens == 3
    assert result.usage.reasoning_tokens == 2
    assert result.usage.audio_input_tokens == 1
    assert result.usage.audio_output_tokens == 4


def test_openai_client_constructor_receives_gateway_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class SDK:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(AsyncOpenAI=SDK))
    transport = OpenAITransport(
        LLMConfig.for_openai(
            api_key="unused",
            base_url="https://gateway.example/v1",
            default_headers={"X-Gateway": "gateway"},
            default_query={"tenant": "acme"},
            client_options={"http_client": "fake"},
        )
    )
    transport.get_client()

    assert captured["api_key"] == "unused"
    assert captured["base_url"] == "https://gateway.example/v1"
    assert captured["default_headers"] == {"X-Gateway": "gateway"}
    assert captured["default_query"] == {"tenant": "acme"}
    assert captured["http_client"] == "fake"
    assert captured["max_retries"] == 0


@pytest.mark.asyncio
async def test_openai_transport_rejects_duplicate_provider_body_field() -> None:
    client = FakeOpenAIClient()
    transport = OpenAITransport(LLMConfig.for_openai(api_key="test"), client=client)
    with pytest.raises(ValueError, match="Duplicate OpenAI"):
        await transport.complete(
            [Message.user("hello")],
            model="gpt-4o",
            request_options=RequestOptions(
                provider={"reasoning_effort": "none"},
                extra_body={"reasoning_effort": "none"},
            ),
        )


@pytest.mark.asyncio
async def test_anthropic_transport_uses_stable_token_count_and_options() -> None:
    client = FakeAnthropicClient()
    transport = AnthropicTransport(
        LLMConfig.for_anthropic(model="claude-opus-4-6", api_key="test"),
        client=client,
    )

    count = await transport.count_tokens(
        [Message.user("hello")],
        model="claude-opus-4-6",
    )
    result = await transport.complete(
        [Message.user("hello")],
        model="claude-opus-4-6",
        top_p=0.0,
        request_options=RequestOptions(
            provider={"top_k": 10},
            extra_headers={"X-Request": "request"},
            extra_body={"vendor_trace": True},
        ),
    )

    assert count == 42
    assert client.messages.count_calls[0]["model"] == "claude-opus-4-6"
    call = client.messages.create_calls[0]
    assert call["top_p"] == 0.0
    assert call["top_k"] == 10
    assert call["extra_headers"] == {"X-Request": "request"}
    assert call["extra_body"] == {"vendor_trace": True}
    assert result.provider == "anthropic"
    assert result.usage.cache_read_tokens == 2


@pytest.mark.asyncio
async def test_multimodal_objects_are_not_dropped() -> None:
    openai_client = FakeOpenAIClient()
    openai_transport = OpenAITransport(LLMConfig.for_openai(api_key="test"), client=openai_client)
    await openai_transport.complete(
        [
            Message.from_multimodal(
                "user",
                [
                    ImageContent.from_url("https://example.com/image.png"),
                    AudioContent.from_bytes(b"audio", "audio/wav"),
                ],
            )
        ],
        model="gpt-4o",
    )
    parts = openai_client.completions.calls[0]["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[1]["type"] == "input_audio"

    anthropic_client = FakeAnthropicClient()
    anthropic_transport = AnthropicTransport(
        LLMConfig.for_anthropic(api_key="test"),
        client=anthropic_client,
    )
    with pytest.raises(UnsupportedContentError):
        await anthropic_transport.complete(
            [Message.from_multimodal("user", [AudioContent.from_bytes(b"audio", "audio/wav")])],
            model="claude-opus-4-6",
        )


@pytest.mark.asyncio
async def test_native_openai_responses_service_preserves_events() -> None:
    events = [SimpleNamespace(type="response.output_text.delta", delta="hello")]

    async def response_create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):

            async def iterator() -> Any:
                for event in events:
                    yield event

            return iterator()
        return SimpleNamespace(id="resp_1", kwargs=kwargs)

    raw = SimpleNamespace(
        responses=SimpleNamespace(
            create=response_create,
            retrieve=lambda response_id, **kwargs: {"id": response_id, **kwargs},
        )
    )
    service = OpenAIResponsesService(OpenAIClient(client=raw))
    result = await service.create(ResponsesRequest(model="gpt-5", input="hello"))
    streamed = [event async for event in service.stream(ResponsesRequest(model="gpt-5", input="stream"))]

    assert result.id == "resp_1"
    assert streamed == events


@pytest.mark.asyncio
async def test_native_realtime_service_keeps_session_boundary() -> None:
    session = object()

    async def connect(**kwargs: Any) -> Any:
        return session, kwargs

    raw = SimpleNamespace(realtime=SimpleNamespace(connect=connect))
    service = OpenAIRealtimeService(OpenAIClient(client=raw))
    result = await service.connect(model="gpt-realtime")

    assert result == (session, {"model": "gpt-realtime"})


@pytest.mark.asyncio
async def test_llm_module_registers_native_services_and_capabilities() -> None:
    provider = LLMModule.for_root(LLMConfig.for_openai(api_key="test"))
    assert "OpenAIClient" in provider.native_services
    assert isinstance(provider.native_services["OpenAIResponsesService"], OpenAIResponsesService)
    capabilities = await provider.llm_service_instance.capabilities()
    assert capabilities.supports_responses is True


@pytest.mark.asyncio
async def test_native_anthropic_services_cover_parse_batch_and_models() -> None:
    class BatchResource:
        async def create(self, **kwargs: Any) -> Any:
            return {"kind": "created", **kwargs}

        async def retrieve(self, batch_id: str) -> Any:
            return {"id": batch_id}

        async def list(self, **kwargs: Any) -> Any:
            return {"page": kwargs.get("after_id")}

        async def cancel(self, batch_id: str) -> Any:
            return {"cancelled": batch_id}

        async def delete(self, batch_id: str) -> Any:
            return {"deleted": batch_id}

        async def results(self, batch_id: str) -> Any:
            return {"results": batch_id}

    class ModelsResource:
        async def list(self, **kwargs: Any) -> Any:
            return {"models": kwargs}

        async def retrieve(self, model_id: str) -> Any:
            return {"id": model_id}

    async def parse(**kwargs: Any) -> Any:
        return kwargs["output_format"]()

    raw = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: kwargs,
            parse=parse,
            count_tokens=lambda **kwargs: SimpleNamespace(input_tokens=1),
            batches=BatchResource(),
        ),
        models=ModelsResource(),
    )
    client = AnthropicClient(client=raw)
    messages = AnthropicMessagesService(client)
    batches = AnthropicBatchService(client)
    models = AnthropicModelsService(client)

    class Output:
        pass

    assert isinstance(await messages.parse(output_format=Output), Output)
    assert await batches.retrieve("batch_1") == {"id": "batch_1"}
    assert await models.retrieve("claude-opus-4-6") == {"id": "claude-opus-4-6"}


@pytest.mark.asyncio
async def test_native_messages_service_emits_model_lifecycle_signals() -> None:
    bus = SignalBus()
    started: list[ModelCallStarted] = []
    completed: list[ModelCallComplete] = []

    @bus.on(ModelCallStarted)
    async def capture_started(event: ModelCallStarted) -> None:
        started.append(event)

    @bus.on(ModelCallComplete)
    async def capture_completed(event: ModelCallComplete) -> None:
        completed.append(event)

    service = AnthropicMessagesService(AnthropicClient(client=FakeAnthropicClient()), signals=bus)
    result = await service.create(model="claude-opus-4-6", messages=[{"role": "user", "content": "hello"}])

    assert result.id == "msg_test"
    assert started[0].model == "claude-opus-4-6"
    assert started[0].messages_count == 1
    assert completed[0].model == "claude-opus-4-6"
    assert completed[0].stop_reason == "end_turn"

# OpenAI and Anthropic SDK compatibility

Lauren AI has two complementary provider layers:

1. The portable LLMService and Transport APIs for agent loops, tools,
   guardrails, memory, and streaming.
2. Native provider services for SDK APIs whose request or event model cannot be
   represented losslessly by a portable completion.

Install only the provider SDKs your application uses:

~~~bash
pip install "lauren-ai[openai]"
pip install "lauren-ai[anthropic]"
~~~

The SDKs are optional and imported lazily.

## OpenAI-compatible endpoints

Custom gateways and self-hosted servers can use the OpenAI transport without a
custom Lauren transport:

~~~python
from lauren_ai import LLMConfig, LLMModule, LLMService, Message, RequestOptions

config = LLMConfig.for_openai(
    model="moonshotai/Kimi-K3",
    api_key="unused",
    base_url="https://your-modal-endpoint.modal.run/v1",
    default_headers={
        "Modal-Key": "...",
        "Modal-Secret": "...",
    },
    timeout=60.0,
    max_retries=0,
)
provider = LLMModule.for_root(config)
llm = provider.llm_service_instance

response = await llm.complete(
    [Message.user("Explain this chart.")],
    temperature=0.3,
    top_p=0.95,
    max_tokens=2048,
    request_options=RequestOptions(
        provider={"reasoning_effort": "none"},
        extra_body={"vendor_trace": True},
    ),
)
~~~

Client defaults and request options are copied when they enter the
configuration object. A request cannot mutate defaults used by another
concurrent request.

Precedence is:

~~~text
per-call override > service/agent default > LLMConfig default > SDK default
~~~

Use extra_headers, extra_query, and extra_body for fields that are not yet
modeled by Lauren. Known fields should be passed as direct arguments or in the
provider namespace. Duplicate fields are rejected instead of silently
overwritten.

## Common request options

The common completion API supports:

- top_p;
- max_tokens and provider-native max_completion_tokens;
- RequestOptions.extra_headers;
- RequestOptions.extra_query;
- RequestOptions.extra_body;
- RequestOptions.provider;
- request-level timeout and retry overrides;
- optional raw response retention.

None means omit a field. Valid falsey values such as 0, False, and empty lists
are preserved.

Request options can be configured once:

~~~python
config = LLMConfig.for_openai(
    model="gpt-5",
    request_options=RequestOptions(
        provider={"reasoning_effort": "medium"},
        extra_headers={"X-Tenant": "acme"},
    ),
)
~~~

or overridden for one call:

~~~python
response = await llm.complete(
    [Message.user("Hello")],
    request_options=RequestOptions(extra_query={"trace": "1"}),
)
~~~

Lauren redacts credential-like keys from diagnostic request summaries. API
keys, authorization headers, cookies, tokens, and custom secret fields must
not be logged or included in signals.

## OpenAI Responses

Responses has a different input-item, output-item, state, and streaming-event
model from Chat Completions. Use the native service when those semantics
matter:

~~~python
from lauren_ai import OpenAIResponsesService, ResponsesRequest

responses = provider.native_services["OpenAIResponsesService"]

result = await responses.create(
    ResponsesRequest(
        model="gpt-5",
        input="Research quantum computing.",
        instructions="Cite the important limitations.",
        reasoning={"effort": "medium"},
    )
)

async for event in responses.stream(
    ResponsesRequest(model="gpt-5", input="Stream the answer.")
):
    # event is the native OpenAI Responses event.
    print(event)
~~~

The service also exposes retrieve(response_id). Native events are not reduced
to text deltas. A conversion to a portable completion should be explicit and
is necessarily lossy.

When using Lauren DI, inject OpenAIResponsesService into a controller or tool.
For resources that do not yet have a Lauren wrapper, inject OpenAIClient;
attribute access delegates to the official async SDK client.

## Anthropic Messages

The existing Anthropic transport supports tool use, streaming, prompt caching,
and extended thinking. Additional Messages fields can be passed through
RequestOptions:

~~~python
from lauren_ai import LLMConfig, RequestOptions

config = LLMConfig.for_anthropic(
    model="claude-opus-4-6",
    request_options=RequestOptions(
        provider={
            "top_k": 20,
            "thinking": {"type": "adaptive"},
        },
        extra_headers={"X-Trace": "trace-id"},
    ),
)
~~~

Token counting prefers the stable client.messages.count_tokens SDK method and
falls back to Lauren's documented heuristic only when the installed SDK or
endpoint does not provide it.

Use thinking=True and thinking_budget_tokens for backwards-compatible enabled
thinking. New code can use provider options for adaptive or disabled thinking.
Thinking and redacted-thinking blocks are preserved with their signatures when
the agent conversation is stored and sent back.

### Anthropic SDK 0.x, 1.x and 2.x

The transport works with any `anthropic` release from 0.35 on, including the
1.x and 2.x lines. The lines differ in how they treat the deprecated sampling
parameters `temperature`, `top_p` and `top_k`:

| `anthropic` SDK | API version sent | Sampling parameters |
| --------------- | ---------------- | ------------------- |
| 0.x             | `2023-06-01`     | Sent as ordinary request fields. |
| 1.x             | `2023-06-01`     | Removed from the SDK method signatures. Lauren sends a non-default value through `extra_body`, which models that still honour it receive. |
| 2.x             | `2026-10-21`     | Rejected by the API with HTTP 400, even at the default value. Lauren does not send them. |

On 1.x and 2.x Lauren omits `temperature=1.0`, because that is what the API
applies when the field is absent. Any other `temperature`, `top_p` or `top_k`
is forwarded or dropped as the table says. When Lauren drops a value it logs
one warning per parameter, and the model's default sampling applies. Models
that accept only the default sampling, such as Claude 4.7 and later, behave the
same either way.

Lauren's own `temperature=0.0` summarisation calls follow the same rule, so
they keep working on 2.x.

To keep sending a non-default value to a model that still honours it while
using the 2.x SDK, pin the older API version on the client:

~~~python
config = LLMConfig.for_anthropic(
    model="claude-sonnet-4-6",
    temperature=0.2,
    default_headers={"anthropic-version": "2023-06-01"},
)
~~~

Lauren reads the effective `anthropic-version` header, so the pinned value is
honoured and the parameter is forwarded as on 1.x.

SDK 1.0 also moved its HTTP layer from `httpx` to `httpx2`. If you pass your
own client through `client_options={"http_client": ...}`, build it from
`httpx2` on SDK 1.x and later; the SDK rejects a plain `httpx` client. Lauren
does not construct an `httpx` client for Anthropic itself.

## Native Anthropic services

LLMModule.for_root registers these services for an Anthropic configuration:

- AnthropicClient;
- AnthropicMessagesService;
- AnthropicBatchService;
- AnthropicModelsService.

The message service supports native create, stream, parse, and token counting:

~~~python
message = await anthropic_messages.create(
    model="claude-opus-4-6",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Return a JSON object."}],
)

structured = await anthropic_messages.parse(
    output_format=MyPydanticModel,
    model="claude-opus-4-6",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Return the object."}],
)

batch = await anthropic_batches.create(requests=[...])
models = await anthropic_models.list()
~~~

Native services return official SDK objects. They do not silently translate
batch lifecycle or provider-managed tool execution into an ordinary agent
completion.

## Structured output

llm.with_structured_output(Model) selects native provider parsing when the
installed transport exposes it. If native parsing is unavailable, Lauren
falls back to the portable synthetic-tool strategy. Provider/API errors are
not hidden by an automatic second request.

The selected strategy can differ in refusal behavior, streaming support, token
usage, and validation semantics; applications should treat it as a provider
capability.

## Multimodal inputs

Image and audio objects are translated for OpenAI when supported. Image and
document objects are translated for Anthropic. Unsupported combinations raise
UnsupportedContentError rather than being converted to empty text.

~~~python
from lauren_ai import AudioContent, ImageContent, Message

message = Message.from_multimodal(
    "user",
    [
        ImageContent.from_url("https://example.com/chart.png"),
        AudioContent.from_bytes(audio_bytes, "audio/wav"),
    ],
)
~~~

Binary payloads are never included in normal Lauren signals or diagnostics.

## Client ownership and testing

Lauren-created clients are cached per transport and closed by the transport's
close() method. Clients supplied directly or through a factory remain owned by
the caller. This supports custom HTTP clients, multi-tenant routing, and
deterministic fakes:

~~~python
transport = OpenAITransport(config, client=fake_async_openai_client)
~~~

Provider compatibility tests should capture SDK request keyword arguments
without making live calls. Live tests are opt-in and must provide credentials
through the environment.

## Compatibility boundaries

Use the portable transport for:

- ordinary model completions;
- client-side function tools;
- agent loops;
- guardrails and memory;
- common signal handling.

Use native services for:

- OpenAI Responses and Realtime;
- Anthropic batches and beta resources;
- provider-hosted/server tools;
- provider-native event streams;
- resource APIs unrelated to a single completion.

This boundary keeps common Lauren applications portable while making advanced
provider features available without requiring a transport fork.

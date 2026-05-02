# Structured Output

`StructuredLLM` wraps any `LLMService` so that every completion is guaranteed
to deserialize into a Pydantic model instance.  Under the hood it uses native
tool-calling to constrain the model's response to a specific JSON schema —
no brittle prompt engineering required.

## Quick start

```python
from pydantic import BaseModel
from lauren_ai import LLMConfig
from lauren_ai._module import LLMModule

class SentimentResult(BaseModel):
    sentiment: str        # "positive" | "negative" | "neutral"
    confidence: float     # 0.0 – 1.0

cfg, mock = LLMConfig.for_testing()
LLMProviderModule = LLMModule.for_root(cfg, transport_override=mock)

# Obtain the LLMService from DI or create it directly
from lauren_ai._module import LLMService
llm = LLMService(transport=mock, config=cfg)

structured = llm.with_structured_output(SentimentResult)
result: SentimentResult = await structured.complete([
    Message.user("This product is absolutely fantastic!")
])

print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95
```

## How it works

1. `with_structured_output(MyModel)` builds a `ToolSchema` whose
   `input_schema` is derived from `MyModel.model_json_schema()`.
2. The underlying `LLMService.complete()` call includes that tool with
   `tool_choice=ToolChoice.specific("structured_output")`, forcing the model
   to emit exactly one tool call.
3. The tool call's `input` dict is unpacked into `MyModel(**input)` and
   returned to the caller.
4. If the model returns plain JSON text instead of a tool call (some
   providers), the content is parsed as JSON and used as the input dict.

## Using StructuredLLM in chains

`StructuredLLM` supports the `|` operator so it composes naturally with
`PromptTemplate`:

```python
from lauren_ai._prompts import PromptTemplate
from lauren_ai._chains import Chain

pipeline: Chain = (
    PromptTemplate(template="Analyse the sentiment of: {text}")
    | llm.with_structured_output(SentimentResult)
)

# Render the prompt, run the LLM, and get a validated SentimentResult
result = await pipeline.run(text="I love this!")
```

## Testing with MockTransport

Use `MockTransport.queue_structured()` to pre-load a response without making
any network calls:

```python
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig

transport = MockTransport()
config, _ = LLMConfig.for_testing()
llm = LLMService(transport=transport, config=config)

# Queue an instance — it will be returned by the next complete() call
transport.queue_structured(SentimentResult(sentiment="negative", confidence=0.8))

structured = llm.with_structured_output(SentimentResult)
result = await structured.complete([Message.user("This is terrible.")])

assert result.sentiment == "negative"
assert result.confidence == 0.8
```

`queue_structured` accepts any Pydantic model instance and internally builds a
`Completion` whose `tool_calls` list contains a single `ToolCall` with the
serialized data.

## Schema introspection

Access the generated JSON Schema directly:

```python
structured = llm.with_structured_output(SentimentResult)
print(structured._schema)
# {'properties': {'sentiment': {'title': 'Sentiment', 'type': 'string'},
#                 'confidence': {'title': 'Confidence', 'type': 'number'}},
#  'required': ['sentiment', 'confidence'],
#  'title': 'SentimentResult',
#  'type': 'object'}
```

## Error handling

When the model's response cannot be deserialized into the requested model,
`OutputParserError` is raised:

```python
from lauren_ai._output_parsers import OutputParserError

try:
    result = await structured.complete(messages)
except OutputParserError as exc:
    print(f"Parse failed: {exc}")
```

## Supported providers

| Provider   | Mechanism            | Notes                               |
|------------|----------------------|-------------------------------------|
| Anthropic  | Tool-forcing         | Full support                        |
| OpenAI     | Tool-forcing         | Full support                        |
| Ollama     | Tool-forcing         | Requires a model with tool support  |
| LiteLLM    | Tool-forcing         | Delegates to the upstream provider  |

## Pydantic requirements

`with_structured_output` calls `model_cls.model_json_schema()`, which is a
Pydantic v2 API.  Pydantic v1 models are not supported.  If the class does not
expose `model_json_schema()`, an empty schema is used and the model may not
constrain its output correctly.

## Limitations

- Streaming is not supported through `StructuredLLM.complete()`.  If the
  underlying transport returns a streaming iterator, the chunks are collected
  and the accumulated JSON is parsed — this buffers the full response.
- Recursive schemas (self-referential models) work but may confuse some
  provider tool implementations.
- `StructuredLLM` is not directly injectable via the lauren DI container.
  Obtain it by calling `llm_service.with_structured_output(Model)` where
  `LLMService` is injected normally.

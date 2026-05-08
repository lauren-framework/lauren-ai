---
name: json-schema-enforcement
description: Enforces JSON schema compliance on LLM responses using system prompt instructions, Pydantic validation, and RetryOutputParser. Use when the LLM must return a specific JSON structure and the response must be validated and retried on parse failure.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# JSON Schema Enforcement for LLM Responses

## Pattern: system prompt + Pydantic validation

```python
from __future__ import annotations

from pydantic import BaseModel, ValidationError

class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool

JSON_SYSTEM = """Always respond with valid JSON matching this schema:
{"name": "string", "price": number, "in_stock": boolean}
Do not include any text outside the JSON object."""

def parse_json_response(content: str, schema: type[BaseModel]) -> BaseModel:
    try:
        return schema.model_validate_json(content.strip())
    except ValidationError as exc:
        raise ValueError(f"Invalid JSON schema response: {exc}") from exc
```

---

## Pattern: JSONOutputParser

```python
from lauren_ai import JSONOutputParser

parser = JSONOutputParser()
data: dict = parser.parse('{"name": "Widget", "price": 9.99}')
```

---

## Pattern: RetryOutputParser (retry on failure)

```python
from lauren_ai import PydanticOutputParser, RetryOutputParser, LLMConfig

class OrderInfo(BaseModel):
    order_id: str
    total: float

base_parser = PydanticOutputParser(model=OrderInfo)
retry_parser = RetryOutputParser(parser=base_parser, max_retries=3)

# On parse failure, RetryOutputParser re-prompts the LLM with the error
result = await retry_parser.aparse_with_retry(
    completion_fn=llm_complete_fn,
    raw_output=initial_output,
)
```

---

## Testing valid and invalid JSON

```python
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()

# Valid JSON response
mock.queue_response(Completion(
    id="c1", model="mock-model",
    content='{"name": "Widget", "price": 9.99, "in_stock": true}',
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=10, output_tokens=15),
))

# Invalid JSON (for retry tests)
mock.queue_response(Completion(
    id="c2", model="mock-model",
    content="Sorry, I cannot provide that information.",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=10, output_tokens=10),
))
```

---

## StructuredLLM — the recommended alternative

When you control the full request, prefer `StructuredLLM` over prompt hacks:
it binds the Pydantic schema to a tool call, guaranteeing valid JSON parsing
without retry logic.

```python
from lauren_ai import StructuredLLM, LLMConfig

cfg = LLMConfig.for_anthropic()
llm = StructuredLLM(config=cfg, schema=ProductInfo)
result = await llm.generate("Describe product #42")
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_output_parsers/` | `JSONOutputParser`, `PydanticOutputParser`, `RetryOutputParser` |
| `src/lauren_ai/_transport/_structured.py` | `StructuredLLM` — schema-enforced completions |

---
name: structured-output-parsing
description: Parses LLM responses into typed Pydantic models using StructuredLLM, PydanticOutputParser, or manual JSON parsing. Use when an agent must return structured data (e.g. a classification result, extracted entity, or form field set) rather than free-form text.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Structured Output Parsing

## Option 1 — StructuredLLM (recommended)

```python
from __future__ import annotations

from pydantic import BaseModel
from lauren_ai import StructuredLLM, LLMConfig

class SentimentResult(BaseModel):
    label: str       # "positive" | "negative" | "neutral"
    score: float
    reasoning: str

cfg = LLMConfig.for_anthropic(model="claude-opus-4-6")
llm = StructuredLLM(config=cfg, schema=SentimentResult)

result: SentimentResult = await llm.generate("Analyze: 'I love this product!'")
print(result.label, result.score)
```

`StructuredLLM` builds a tool schema from the Pydantic model and instructs the
LLM to call it, then deserializes the tool input back into the model.

---

## Option 2 — PydanticOutputParser

```python
from __future__ import annotations

from pydantic import BaseModel
from lauren_ai import PydanticOutputParser, RetryOutputParser

class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool

parser = PydanticOutputParser(model=ProductInfo)
# Wrap with retry for robustness:
retry_parser = RetryOutputParser(parser=parser, max_retries=2)

result = parser.parse('{"name": "Widget", "price": 9.99, "in_stock": true}')
```

---

## Option 3 — Manual JSON parse in an agent

```python
from __future__ import annotations

import json
from pydantic import BaseModel, ValidationError

class ExtractedData(BaseModel):
    name: str
    count: int

def parse_structured_response(content: str, schema: type[BaseModel]) -> BaseModel:
    try:
        data = json.loads(content.strip())
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Agent output is not valid {schema.__name__}: {exc}") from exc
```

---

## Testing with MockTransport

```python
from lauren_ai._transport._mock import MockTransport

mock = MockTransport()
# queue_structured serializes the Pydantic instance as a tool-use completion:
mock.queue_structured(SentimentResult(label="positive", score=0.9, reasoning="Positive tone"))
```

Or queue raw JSON for manual parse tests:

```python
from lauren_ai._transport import Completion, TokenUsage

mock.queue_response(Completion(
    id="c1", model="mock-model",
    content='{"label": "positive", "score": 0.9, "reasoning": "Great!"}',
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=10, output_tokens=20),
))
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_transport/_structured.py` | `StructuredLLM` implementation |
| `src/lauren_ai/_output_parsers/` | `PydanticOutputParser`, `RetryOutputParser`, etc. |
| `src/lauren_ai/_transport/_mock.py` | `queue_structured()` helper |

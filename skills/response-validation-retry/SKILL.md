---
name: response-validation-retry
description: Validates agent responses against rules or schemas and retries with a corrective prompt when validation fails. Use when the agent must return JSON, structured data, or text meeting specific criteria; when using a weaker model that occasionally produces malformed output; or when reliability is more important than latency.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# LLM Response Validation & Retry Logic

## Built-in: `RetryOutputParser`

Lauren-AI ships `RetryOutputParser` in `lauren_ai._output_parsers`:

```python
# NOTE: future annotations are allowed, but tool signature types must resolve at import time.
from lauren_ai import RetryOutputParser, JSONOutputParser, PydanticOutputParser
```

Use `RetryOutputParser` to wrap any parser with automatic retry:

```python
from lauren_ai import RetryOutputParser, JSONOutputParser, AgentRunnerBase, LLMConfig
from lauren_ai._transport._mock import MockTransport

json_parser = JSONOutputParser()
retrying_parser = RetryOutputParser(parser=json_parser, max_retries=3)
```

## Custom `ResponseValidator` with exponential backoff

For cases where you need custom validation logic beyond parsing:

```python
import asyncio
from typing import Any, Callable


class ResponseValidator:
    def __init__(self, validators: list[Callable[[str], bool]], max_retries: int = 3):
        self._validators = validators
        self._max_retries = max_retries

    async def validate_and_retry(self, run_fn: Callable, prompt: str, **kwargs) -> Any:
        last_response = None
        for attempt in range(self._max_retries):
            response = await run_fn(prompt, **kwargs)
            content = response.content if hasattr(response, "content") else str(response)
            if all(v(content) for v in self._validators):
                return response
            last_response = response
            if attempt < self._max_retries - 1:
                correction = (
                    f"\nPrevious response was invalid. Please try again.\n"
                    f"Previous: {content}"
                )
                prompt = prompt + correction
                await asyncio.sleep(0.1 * (2 ** attempt))
        return last_response
```

## Validator factories

```python
def is_non_empty(response: str) -> bool:
    return bool(response.strip())

def is_valid_json(response: str) -> bool:
    import json
    try:
        json.loads(response.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False

def max_length(n: int) -> Callable[[str], bool]:
    return lambda r: len(r) <= n

def contains_required_keys(*keys: str) -> Callable[[str], bool]:
    import json
    def check(r: str) -> bool:
        try:
            data = json.loads(r)
            return all(k in data for k in keys)
        except Exception:
            return False
    return check
```

## Full example

```python
from lauren_ai import agent, AgentRunnerBase, LLMConfig
from lauren_ai._transport._mock import MockTransport

@agent(model="claude-haiku-4-5", system="Always respond with valid JSON only.")
class JSONAgent: ...

cfg = LLMConfig(provider="anthropic", model="claude-haiku-4-5", api_key="...")
runner = AgentRunnerBase(transport=transport)

validator = ResponseValidator(
    validators=[is_non_empty, is_valid_json, contains_required_keys("result", "confidence")],
    max_retries=3,
)

async def run_json_agent(query: str) -> str:
    import json
    response = await validator.validate_and_retry(
        lambda prompt: runner.run(JSONAgent(), prompt),
        query,
    )
    return response.content
```

## Using `PydanticOutputParser`

For structured output, `PydanticOutputParser` validates against a Pydantic model:

```python
from pydantic import BaseModel
from lauren_ai import PydanticOutputParser

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    summary: str

parser = PydanticOutputParser(model=AnalysisResult)
# parser.parse(response_content) -> AnalysisResult instance
```

## Pitfalls

- Retries increase latency and cost — use only when correctness matters more
  than speed.
- The corrective prompt ("previous response was invalid") uses tokens; for
  very long conversations this can exceed the context window.
- `RetryOutputParser` in lauren-ai wraps a parser; `ResponseValidator` here
  wraps an entire agent run.  Choose based on whether you need custom validation
  logic beyond parsing.
- Always set a reasonable `max_retries` (2–3); infinite retries against a
  broken prompt will loop forever.

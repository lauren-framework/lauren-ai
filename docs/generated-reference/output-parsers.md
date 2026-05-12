# Output Parsers

Structured extraction from LLM text responses.

### `StrOutputParser`

Strip whitespace from the LLM response and return it as a plain string.

### `JSONOutputParser`

Parse a JSON value from LLM text.

Strips Markdown fenced code blocks (```json ... ```) before parsing
so the LLM can wrap its JSON in a code block without breaking the parser.

### `RegexParser`

Extract named capture groups from LLM text using a regex pattern.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `pattern` | `str` | A regular expression string containing at least one
named group `(?P<name>...)`. |

### `CommaSeparatedListParser`

Parse a comma-separated list from LLM text.

### `MarkdownCodeBlockParser`

Extract the first fenced code block from LLM text.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `language` | `str` | Optional language tag to match (e.g. `"python"`).
When provided the parser first attempts to find a block with that
language tag before falling back to any fenced block. |

### `PydanticOutputParser`

Parse LLM text as a Pydantic model instance.

Strips Markdown fenced code blocks before JSON-decoding, then validates
the resulting dict against the provided Pydantic model class.

Usage::

    class UserInfo(BaseModel):
        name: str
        age: int

    parser = PydanticOutputParser(model=UserInfo)
    user = parser.parse('{"name": "Alice", "age": 30}')

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `type[T]` | The Pydantic model class to validate against. |

### `RetryOutputParser`

Wrap any `OutputParser` and
automatically retry with error feedback when parsing fails.

On each failure a correction turn is appended to the conversation and the
LLM is asked to produce a valid response.  Retries continue up to
*max_retries* additional attempts before `MaxRetryError` is raised.

Usage::

    parser = RetryOutputParser(
        parser=PydanticOutputParser(model=UserInfo),
        llm=llm_service,
        max_retries=3,
    )
    result = await parser.parse_with_retry(
        original_messages=messages,
        completion=completion,
    )

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `parser` | `Any` | The underlying parser to wrap. |
| `llm` | `Any` | The `LLMService` to call for
correction turns. |
| `max_retries` | `int` | Maximum number of additional retry attempts after the
first failure.  Defaults to `3`. |

### `MaxRetryError`

Raised when `RetryOutputParser` exhausts its retry budget.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the exhaustion. |
| `attempts` | `int` | Total number of parse attempts made. |

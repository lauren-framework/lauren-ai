# Output Parsers

Structured extraction from LLM text responses.

### `StrOutputParser`

Strip whitespace from the LLM response and return it as a plain string.

### `JSONOutputParser`

Parse a JSON value from LLM text.

Strips Markdown fenced code blocks (````json ... ````) before parsing
so the LLM can wrap its JSON in a code block without breaking the parser.

### `RegexParser`

Extract named capture groups from LLM text using a regex pattern.

:param pattern: A regular expression string containing at least one
    named group ``(?P<name>...)``.
:type pattern: str

### `CommaSeparatedListParser`

Parse a comma-separated list from LLM text.

### `MarkdownCodeBlockParser`

Extract the first fenced code block from LLM text.

:param language: Optional language tag to match (e.g. ``"python"``).
    When provided the parser first attempts to find a block with that
    language tag before falling back to any fenced block.
:type language: str

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

:param model: The Pydantic model class to validate against.
:type model: type[T]

### `RetryOutputParser`

Wrap any :class:`~lauren_ai._output_parsers._base.OutputParser` and
automatically retry with error feedback when parsing fails.

On each failure a correction turn is appended to the conversation and the
LLM is asked to produce a valid response.  Retries continue up to
*max_retries* additional attempts before :class:`MaxRetryError` is raised.

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

:param parser: The underlying parser to wrap.
:type parser: Any
:param llm: The :class:`~lauren_ai._module.LLMService` to call for
    correction turns.
:type llm: Any
:param max_retries: Maximum number of additional retry attempts after the
    first failure.  Defaults to ``3``.
:type max_retries: int

### `MaxRetryError`

Raised when :class:`RetryOutputParser` exhausts its retry budget.

:param message: Human-readable description of the exhaustion.
:type message: str
:param attempts: Total number of parse attempts made.
:type attempts: int


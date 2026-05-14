# Output Parsers

Output parsers transform the raw text returned by an LLM into typed Python
values. Every parser satisfies the `OutputParser` protocol:

```python
class OutputParser(Protocol):
    def parse(self, text: str) -> Any: ...

    @property
    def format_instructions(self) -> str: ...
```

The `format_instructions` property returns a short instruction string you can
embed in your prompt to tell the model what format to use.

All parsers implement `__or__` so they compose cleanly with a `Chain` via the
`|` operator.

---

## Built-in parsers

### StrOutputParser

The simplest parser: strip whitespace and return the text as-is.

```python
from lauren_ai._output_parsers import StrOutputParser

parser = StrOutputParser()
result = parser.parse("  Hello, world!  ")
# "Hello, world!"
```

**format_instructions:** `"Respond with plain text."`

---

### JSONOutputParser

Parse a JSON value (dict, list, number, …) from LLM text.  Markdown fenced
code blocks (` ```json … ``` `) are stripped automatically before parsing.

```python
from lauren_ai._output_parsers import JSONOutputParser

parser = JSONOutputParser()
result = parser.parse('{"name": "Alice", "score": 42}')
# {"name": "Alice", "score": 42}

# Also handles markdown fences:
result = parser.parse("```json\n[1, 2, 3]\n```")
# [1, 2, 3]
```

Raises `OutputParserError` on invalid JSON.

**format_instructions:** `"Respond with valid JSON."`

---

### RegexParser

Extract named capture groups from LLM text using a regular expression.

```python
from lauren_ai._output_parsers import RegexParser

parser = RegexParser(r"Score: (?P<score>\d+)/10, Verdict: (?P<verdict>\w+)")
result = parser.parse("Score: 8/10, Verdict: Excellent")
# {"score": "8", "verdict": "Excellent"}
```

Raises `OutputParserError` when the pattern does not match.

**format_instructions:** Shows the required pattern string.

---

### CommaSeparatedListParser

Parse a comma-separated list of items, stripping whitespace from each.

```python
from lauren_ai._output_parsers import CommaSeparatedListParser

parser = CommaSeparatedListParser()
result = parser.parse("Python, JavaScript, Rust, Go")
# ["Python", "JavaScript", "Rust", "Go"]
```

Empty items are discarded.  An empty string returns an empty list.

**format_instructions:** `"Respond with a comma-separated list of values."`

---

### MarkdownCodeBlockParser

Extract the contents of the first fenced code block from LLM text.  Useful
when asking the model to produce code.

```python
from lauren_ai._output_parsers import MarkdownCodeBlockParser

parser = MarkdownCodeBlockParser(language="python")
code = parser.parse(
    "Here is the solution:\n```python\nprint('Hello!')\n```"
)
# "print('Hello!')"
```

When no `language` is given, the parser matches any fenced block. Raises
`OutputParserError` when no fenced block is found.

**format_instructions:** `"Wrap your response in a ```python``` fenced code block."`

---

### PydanticOutputParser

Parse LLM text as a Pydantic v2 model instance.  Markdown fences are stripped
before JSON parsing; the resulting dict is validated by Pydantic.

```python
from pydantic import BaseModel
from lauren_ai._output_parsers import PydanticOutputParser

class UserInfo(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(model=UserInfo)
user = parser.parse('{"name": "Alice", "age": 30}')
assert isinstance(user, UserInfo)
assert user.name == "Alice"
```

Raises `OutputParserError` for invalid JSON or Pydantic validation failures.

**format_instructions:** Returns a JSON Schema description of the model, e.g.:

```
Respond with a JSON object matching this schema:
{
  "properties": {
    "name": {"title": "Name", "type": "string"},
    "age": {"title": "Age", "type": "integer"}
  },
  "required": ["name", "age"],
  ...
}
```

Embed this in your prompt to guide the model:

```python
tpl = PromptTemplate(
    template="Extract user info.\n\n{fmt}\n\nText: {text}",
    input_variables=["fmt", "text"],
)
msg = tpl.render(fmt=parser.format_instructions, text="Alice is 30 years old.")
```

---

## RetryOutputParser

Wraps any parser and automatically retries on failure by sending a correction
turn to the LLM.  On each failure the parser error is forwarded back to the
model along with the format instructions, and the model is asked to produce a
corrected response.

```python
from lauren_ai._output_parsers import PydanticOutputParser, RetryOutputParser

parser = RetryOutputParser(
    parser=PydanticOutputParser(model=UserInfo),
    llm=llm_service,
    max_retries=3,
)

result = await parser.parse_with_retry(
    original_messages=messages,   # list[Message] sent to the LLM
    completion=completion,         # Completion returned by the LLM
)
```

`parse_with_retry` is the async entry point.  The synchronous `parse(text)`
method delegates directly to the wrapped parser without retry.

Raises `MaxRetryError` after all attempts are exhausted. The `attempts`
attribute on the exception records
how many total attempts were made:

```python
from lauren_ai._output_parsers import MaxRetryError

try:
    result = await parser.parse_with_retry(...)
except MaxRetryError as e:
    print(f"Gave up after {e.attempts} attempts")
```

---

## Integration with Chain

Parsers work seamlessly as the last step in a `Chain`:

```python
from lauren_ai._prompts import PromptTemplate
from lauren_ai._output_parsers import PydanticOutputParser

tpl = PromptTemplate(
    template="Extract user info from: {text}\n\n{fmt}",
    input_variables=["text", "fmt"],
)
parser = PydanticOutputParser(model=UserInfo)

chain = tpl | llm_service | parser

user = await chain.invoke(
    text="Bob is 25 years old.",
    fmt=parser.format_instructions,
)
assert isinstance(user, UserInfo)
```

The chain passes the `Completion.content` string to `parser.parse()` and
returns the typed result.

---

## Error hierarchy

```
LaurenAIError
└── OutputParserError      — parse failure
└── MaxRetryError          — retry budget exhausted (has .attempts: int)
```

Both errors are importable from `lauren_ai._output_parsers`.

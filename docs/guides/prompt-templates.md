# Prompt Templates

Prompt templates give you a structured, reusable way to build LLM messages.
Instead of assembling strings by hand in every handler, you define a template
once and call `.render(**kwargs)` to get a typed `Message` (or a list of
messages) ready to send to `LLMService`.

Three template types are available:

| Class | Produces |
|---|---|
| `PromptTemplate` | A single `Message` with `{variable}` interpolation |
| `ChatPromptTemplate` | A `list[Message]` for multi-turn conversations |
| `FewShotPromptTemplate` | A single `Message` with injected examples |

---

## PromptTemplate

The simplest template: one message, one role, any number of
`{variable}` placeholders.

```python
from lauren_ai._prompts import PromptTemplate

tpl = PromptTemplate(
    template="Summarise this article in {words} words:\n\n{article}",
    input_variables=["words", "article"],
)

msg = tpl.render(words=50, article="The quick brown fox…")
# Message(role="user", content="Summarise this article in 50 words:\n\nThe quick brown fox…")
```

### Auto-detected variables

When `input_variables` is omitted, the template engine scans the template
string and treats every `{name}` as required:

```python
tpl = PromptTemplate(template="Hello {name}, you are {age} years old.")
msg = tpl.render(name="Alice", age=30)
```

Missing variables raise `PromptRenderError` at render time, not at request time.

### Role

The `role` field defaults to `"user"`.  Set it to `"system"` when building
system-prompt templates that you want to keep separate from request templates:

```python
system_tpl = PromptTemplate(
    template="You are a helpful {language} assistant.",
    role="system",
)
```

---

## ChatPromptTemplate

When you need a multi-turn conversation scaffold — for example a fixed system
message followed by a variable user turn — use `ChatPromptTemplate`.

```python
from lauren_ai._prompts import ChatPromptTemplate

tpl = ChatPromptTemplate(
    messages=[
        ("system", "You are an expert in {domain}."),
        ("human", "{question}"),
    ],
    input_variables=["domain", "question"],
)

msgs = tpl.render(domain="astronomy", question="How far away is Proxima Centauri?")
# [
#     Message(role="system", content="You are an expert in astronomy."),
#     Message(role="user",   content="How far away is Proxima Centauri?"),
# ]
```

### Role aliases

The following aliases are resolved automatically:

| Alias | Canonical role |
|---|---|
| `human` | `user` |
| `ai` | `assistant` |
| `system` | `system` |

### Mixing tuples and Message objects

You can mix `("role", "template")` tuples and pre-built `Message` instances
in the same list.  Plain `Message` instances are passed through unchanged:

```python
from lauren_ai._transport import Message

fixed_system = Message(role="user", content="Always respond in JSON.")

tpl = ChatPromptTemplate(
    messages=[fixed_system, ("human", "{q}")],
)
```

---

## FewShotPromptTemplate

Few-shot prompts inject a list of `(input, output)` examples into the prompt
to guide the model.  The rendered content is assembled as:

```
<prefix><separator><example_1><separator>…<separator><suffix>
```

```python
from lauren_ai._prompts import FewShotPromptTemplate, FewShotExample

tpl = FewShotPromptTemplate(
    prefix="Classify the sentiment of each review:\n",
    examples=[
        FewShotExample(input="Great product!", output="positive"),
        FewShotExample(input="Terrible quality.", output="negative"),
    ],
    example_template="{input} -> {output}",
    suffix="Review: {review}\nSentiment:",
    input_variables=["review"],
)

msg = tpl.render(review="Absolutely love it!")
```

The rendered content looks like:

```
Classify the sentiment of each review:

Great product! -> positive

Terrible quality. -> negative

Review: Absolutely love it!
Sentiment:
```

### Dynamic extra examples

You can pass additional examples at render time — useful when you are
selecting few-shot examples dynamically (e.g. from a vector store):

```python
dynamic_examples = [
    FewShotExample("Mediocre at best.", "negative"),
]
msg = tpl.render(review="Not bad!", extra_examples=dynamic_examples)
```

### Custom separator

The default separator between sections is `"\n\n"`.  Override it with
`example_separator`:

```python
tpl = FewShotPromptTemplate(
    ...,
    example_separator="\n---\n",
)
```

---

## Chain composition with the `|` operator

All three template classes implement `__or__` so they can be composed into a
`Chain` pipeline using the `|` operator:

```python
from lauren_ai._prompts import PromptTemplate
from lauren_ai._output_parsers import JSONOutputParser

# Build a chain: render template, call LLM, parse JSON
chain = tpl | llm_service | JSONOutputParser()

result = await chain.invoke(words=50, article="…")
# result is the parsed JSON dict
```

The first step in the chain receives the initial `**kwargs` from `invoke`;
subsequent steps receive the output of the previous step.

For LLM steps, the chain automatically normalises `Message`, `list[Message]`,
and arbitrary values into the correct `list[Message]` format before calling
`LLMService.complete()`.

---

## Using `format_instructions` with output parsers

Output parsers expose a `format_instructions` property that returns a
human-readable description of the expected output format.  Embed this in your
template's suffix to guide the model:

```python
from lauren_ai._output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Review(BaseModel):
    sentiment: str
    score: int

parser = PydanticOutputParser(model=Review)

tpl = PromptTemplate(
    template=(
        "Analyse this review and respond in JSON.\n\n"
        "{format_instructions}\n\n"
        "Review: {review}"
    ),
    input_variables=["format_instructions", "review"],
)

msg = tpl.render(
    format_instructions=parser.format_instructions,
    review="Excellent build quality!",
)
```

The same technique works with `FewShotPromptTemplate` by including
`{format_instructions}` in the `suffix` string.

---

## Error handling

All templates raise `PromptRenderError` when required variables are missing.
The error message lists both the missing and the provided variable names to
make debugging straightforward:

```python
from lauren_ai._prompts import PromptTemplate, PromptRenderError

tpl = PromptTemplate(template="Hello {name}!", input_variables=["name"])
try:
    tpl.render()   # missing "name"
except PromptRenderError as e:
    print(e)
    # PromptTemplate missing variables: ['name']. Provided: []
```

# Workflows & Chains

Chains let you compose templates, LLM calls, and output parsers into a
declarative pipeline using the `|` operator.

---

## Core types

```python
from lauren_ai import Chain, Runnable, RunnableLambda, chain
```

| Type | Purpose |
|---|---|
| `Chain` | An ordered pipeline of steps; drives left-to-right execution |
| `Runnable` | Protocol — any object with `async invoke(input) -> Any` |
| `RunnableLambda` | Wraps a sync or async callable as a `Runnable` |
| `chain()` | Factory function: `chain(step1, step2, ...)` → `Chain` |

---

## Building a chain

### With the | operator

Any `PromptTemplate`, `ChatPromptTemplate`, `LLMService`, or output parser
that implements `Runnable` supports `|`:

```python
from lauren_ai._prompts import PromptTemplate
from lauren_ai._output_parsers import StrOutputParser
from lauren_ai._module import LLMService

template = PromptTemplate("Translate the following to French:\n\n{text}")
llm: LLMService = ...   # injected via DI
parser = StrOutputParser()

pipeline = template | llm | parser
```

### With the chain() factory

```python
from lauren_ai import chain

pipeline = chain(template, llm, parser)
```

Both forms produce an identical `Chain` object.

---

## Invoking a chain

`Chain.invoke()` runs all steps left to right and returns the final output.
Pass template variables as keyword arguments or a dict:

```python
# Keyword arguments
result = await pipeline.invoke(text="Hello, world!")

# Dict
result = await pipeline.invoke({"text": "Hello, world!"})
```

The return type depends on the final step.  With a `StrOutputParser`, you get
a plain `str`.  With no parser, you get the raw `Completion` object.

---

## Prompt templates

`PromptTemplate` renders a single user message:

```python
from lauren_ai._prompts import PromptTemplate

template = PromptTemplate("Summarise the following article:\n\n{article}")
message = template.render(article="...")   # returns a Message object
```

`ChatPromptTemplate` renders a multi-message conversation:

```python
from lauren_ai._prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant that speaks {language}."),
    ("user", "{question}"),
])

pipeline = chat_template | llm | StrOutputParser()
result = await pipeline.invoke(language="Spanish", question="What is 2 + 2?")
```

---

## Output parsers

Chain an output parser as the final step to convert the `Completion` to a
usable type:

```python
from lauren_ai._output_parsers import StrOutputParser, JSONOutputParser, PydanticOutputParser
from pydantic import BaseModel

# Plain string
pipeline = template | llm | StrOutputParser()

# JSON dict
pipeline = template | llm | JSONOutputParser()

# Pydantic model
class ReviewSummary(BaseModel):
    sentiment: str
    score: int

pipeline = template | llm | PydanticOutputParser(ReviewSummary)
result: ReviewSummary = await pipeline.invoke(review="This was amazing!")
```

---

## RunnableLambda — inline transforms

Wrap any callable as a step using `RunnableLambda`:

```python
from lauren_ai import RunnableLambda

upper = RunnableLambda(lambda x: x.upper())
pipeline = template | llm | StrOutputParser() | upper

result = await pipeline.invoke(text="hello")
# result is "HELLO" (upper-cased)
```

Both sync and async callables are accepted:

```python
async def enrich(text: str) -> dict:
    citations = await fetch_citations(text)
    return {"text": text, "citations": citations}

pipeline = template | llm | StrOutputParser() | RunnableLambda(enrich)
```

---

## Custom Runnable steps

Any class with `async invoke(self, input: Any) -> Any` satisfies the
`Runnable` protocol and can be composed directly with `|`:

```python
from lauren_ai import Runnable

class SentimentFilter:
    async def invoke(self, input: str) -> str:
        if "hate" in input.lower():
            return "[content removed]"
        return input

pipeline = template | llm | StrOutputParser() | SentimentFilter()
```

---

## Streaming a chain

`Chain.stream()` renders the template steps and streams from the `LLMService`
step.  Output parsers are **not** applied during streaming.

```python
stream = await pipeline.stream(text="Write a haiku about autumn")
async for chunk in stream:
    print(chunk.delta, end="", flush=True)
```

---

## Structured output in chains

Use `LLMService.with_structured_output()` as a step to force the model to
return a valid Pydantic instance:

```python
from pydantic import BaseModel
from lauren_ai._module import LLMService

class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool

llm: LLMService = ...
structured = llm.with_structured_output(ProductInfo)

pipeline = template | structured
result: ProductInfo = await pipeline.invoke(description="...")
```

---

## Full example

```python
import os
from lauren import module, LaurenFactory, get
from lauren_ai import LLMConfig
from lauren_ai._module import LLMModule, LLMService
from lauren_ai._prompts import PromptTemplate
from lauren_ai._output_parsers import StrOutputParser

template = PromptTemplate(
    "You are a chef. Suggest a recipe using these ingredients: {ingredients}"
)
parser = StrOutputParser()

class RecipeController:
    def __init__(self, llm: LLMService) -> None:
        self._pipeline = template | llm | parser

    @get("/recipe")
    async def suggest(self, request: Request) -> dict:
        ingredients = request.query_params["ingredients"]
        recipe = await self._pipeline.invoke(ingredients=ingredients)
        return {"recipe": recipe}

LLMProvider = LLMModule.for_root(
    LLMConfig.for_anthropic(model="claude-opus-4-6")
)

@module(controllers=[RecipeController], imports=[LLMProvider])
class AppModule: ...

app = LaurenFactory.create(AppModule)
```

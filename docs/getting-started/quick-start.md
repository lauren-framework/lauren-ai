# Quick Start

## Minimal setup — single LLM call

The simplest integration: inject `LLMService` into a controller and call the model once.

```python
import os
from lauren import controller, get, module, LaurenFactory
from lauren_ai import LLMModule, LLMConfig, LLMService, Message


@controller("/ai")
class AIController:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    @get("/hello")
    async def hello(self) -> dict:
        completion = await self._llm.complete(
            [Message(role="user", content="Say hello in one sentence.")]
        )
        return {"message": completion.content}


LLMProviderModule = LLMModule.for_root(
    LLMConfig.for_anthropic(
        model="claude-haiku-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
)


@module(controllers=[AIController], imports=[LLMProviderModule])
class AppModule: ...


app = LaurenFactory.create(AppModule)
# Run with: uvicorn myapp:app
```

## Structured output with `Completion[T]`

Get a Pydantic model back from a single LLM call — no agent loop needed.

```python
from typing import Literal
from lauren import controller, post, set_metadata
from lauren.types import Json
from lauren_ai import Completion
from pydantic import BaseModel


class SentimentResult(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float
    reasoning: str


class ReviewRequest(BaseModel):
    text: str


@controller("/reviews")
class ReviewController:
    @post("/sentiment")
    @set_metadata("completion_prompt_field", "text")
    async def analyse(
        self,
        body: Json[ReviewRequest],
        result: Completion[SentimentResult],
    ) -> SentimentResult:
        return result  # already validated by the extractor
```

## Agent with a tool

Build an agent that can call a tool and reason over the results.

```python
from lauren_ai import agent, use_tools, tool, Agent, AgentModule

@tool()
async def get_stock_price(ticker: str) -> dict:
    """Get the current stock price.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.
    """
    # real implementation would call a finance API
    return {"ticker": ticker, "price": 175.42, "currency": "USD"}


@agent(model="claude-opus-4-6", system="You are a helpful finance assistant.")
@use_tools(get_stock_price)
class FinanceAgent: ...
```

See [First Agent](first-agent.md) for a complete end-to-end example.

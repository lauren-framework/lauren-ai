---
name: fallback-model-chain
description: Implements a provider fallback chain that tries multiple LLM providers in order, returning the first success. Use when you need resilience against provider outages, want cost-based fallback (expensive to cheap), or need graceful degradation to a rule-based response.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Fallback Model Chain

A `FallbackChain` tries each provider callable in order, returning the first
successful response. When every provider fails, a configurable fallback string
is returned.

## Pattern

```python
import asyncio
from typing import Any, Callable

class FallbackChain:
    """Tries providers in order, returning first success."""

    def __init__(
        self,
        providers: list[Callable],
        fallback_response: str = "I'm sorry, I cannot process this request right now.",
    ):
        self._providers = providers
        self._fallback = fallback_response

    async def execute(self, prompt: str) -> dict:
        last_error = None
        for i, provider in enumerate(self._providers):
            try:
                result = await provider(prompt)
                return {"content": result, "provider_index": i, "success": True}
            except Exception as e:
                last_error = e
                continue
        return {
            "content": self._fallback,
            "provider_index": -1,
            "success": False,
            "error": str(last_error),
        }
```

## Usage

```python
async def primary_provider(prompt: str) -> str:
    # Calls claude-opus-4-6 (expensive, high quality)
    response = await runner.run(primary_agent, prompt)
    return response.content

async def secondary_provider(prompt: str) -> str:
    # Calls claude-haiku-4 (cheap, fast)
    response = await runner.run(fallback_agent, prompt)
    return response.content

chain = FallbackChain(
    providers=[primary_provider, secondary_provider],
    fallback_response="Service temporarily unavailable.",
)

result = await chain.execute("What is the capital of France?")
print(result["content"])       # "Paris"
print(result["provider_index"])  # 0 (primary succeeded)
```

## Resilience patterns

- **Cost-based fallback**: expensive model → cheaper model → rule-based
- **Provider diversity**: Anthropic → OpenAI → Ollama local
- **Timeout wrapping**: wrap each provider with `asyncio.wait_for`

```python
async def provider_with_timeout(prompt: str, timeout: float = 10.0) -> str:
    return await asyncio.wait_for(call_provider(prompt), timeout=timeout)
```

## Integration with AgentRunner

Each provider can be a thin wrapper around `AgentRunnerBase.run`:

```python
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig

def make_runner(mock: MockTransport) -> AgentRunner:
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    return AgentRunner(transport=mock, tools={}, config=cfg)
```

The `FallbackChain` is provider-agnostic — it accepts any `async (str) -> str`
callable, so you can mix agents, LLM service calls, or even local rule engines.

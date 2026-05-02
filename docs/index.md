# Lauren AI

**`lauren-ai`** is the official AI/LLM companion to the [Lauren web framework](https://github.com/lauren-framework/lauren-framework). It brings large language model agents into the same decorator-first, DI-driven, module-scoped programming model that the rest of Lauren uses.

```python
@tool()
async def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: The city name, e.g. 'London'.
    """
    return {"city": city, "temperature_c": 18, "condition": "cloudy"}


@agent(model="claude-opus-4-6", system="You are a helpful travel assistant.")
@use_tools(get_weather)
class TravelAgent: ...
```

## Why lauren-ai?

- **No global singletons** — all LLM clients, agent runners, and vector stores live in the DI container
- **Provider-agnostic** — swap Anthropic, OpenAI, Ollama, or LiteLLM behind an identical `Transport` interface
- **Streaming is first-class** — every token, tool result, and agent turn streams through `EventStream` or SSE
- **Testable without network calls** — `MockTransport` + `AgentTestClient` = fast, deterministic CI
- **Same patterns you already know** — `@tool()` feels like `@injectable()`, `@agent()` feels like `@interceptor()`

## Get started

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quick-start.md)
- [First Agent](getting-started/first-agent.md)

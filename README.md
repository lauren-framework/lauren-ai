# lauren-ai

First-party AI/LLM companion to the [Lauren web framework](https://github.com/lauren-framework/lauren-framework). Brings large language model agents into the same decorator-first, DI-driven, module-scoped programming model the rest of Lauren uses.

## Features

- **Provider-agnostic transport** — Anthropic, OpenAI, Ollama, LiteLLM, and `MockTransport` for zero-network-call tests
- **`@tool()` decorator** — function-form and class-form (DI-injected), auto-generates JSON Schema from annotations + docstrings
- **`@agent()` decorator** — autonomous agentic loop with `use_tools()`, lifecycle hooks, and budget guards
- **Three memory tiers** — `ShortTermMemory` (sliding window), `MemoryStore` protocol (vector-backed), `ConversationStore` (persist history across requests)
- **Typed extractors** — `Agent[T]`, `Completion[T]`, `Embed[T]`, `StreamCompletion[T]` as handler parameters
- **Module factories** — `LLMModule.for_root()` and `AgentModule.for_root()` feel like `LoggingModule.configure()`
- **Knowledge Base & Agentic RAG** — `KnowledgeBase` with document loaders, hybrid BM25+vector retrieval, and `kb.as_tool()`
- **Structured Workflows** — `Workflow`, `Step`, `Parallel`, `Condition`, `Loop` for deterministic multi-agent pipelines
- **Tool enhancements** — HITL confirmation, pre/post hooks, result caching
- **Extended thinking** — first-class support for Claude extended thinking and OpenAI reasoning models
- **Evaluation framework** — `AccuracyEval`, `AgentJudge` (LLM-as-judge), `TrajectoryEval`, `PerformanceEval`
- **Signals** — `ModelCallComplete`, `ToolCallComplete`, `AgentRunComplete` for observability
- **Pre-built skills** — `WebSearchTool`, `CodeExecutionTool`, `HttpFetchTool`
- **Testable** — zero API calls needed in unit tests via `MockTransport`

## Installation

```bash
# Core (no LLM provider)
pip install lauren-ai

# With Anthropic
pip install "lauren-ai[anthropic]"

# With OpenAI
pip install "lauren-ai[openai]"

# With everything
pip install "lauren-ai[all]"
```

## Quick Start

```python
import os
from lauren import controller, post, module, LaurenFactory
from lauren.types import Json
from lauren_ai import agent, use_tools, tool, Agent, LLMModule, AgentModule, LLMConfig
from pydantic import BaseModel


@tool()
async def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: The city name.
    """
    return {"city": city, "temperature_c": 18, "condition": "cloudy"}


@agent(model="claude-opus-4-6", system="You are a helpful travel assistant.")
@use_tools(get_weather)
class TravelAgent: ...


class AskRequest(BaseModel):
    question: str


@controller("/travel")
class TravelController:
    def __init__(self, runner) -> None:
        self._runner = runner

    @post("/ask")
    async def ask(self, body: Json[AskRequest], agent: Agent[TravelAgent]) -> dict:
        response = await self._runner.run(agent, body.question)
        return {"answer": response.content, "turns": response.turns}


LLMProviderModule = LLMModule.for_root(
    LLMConfig.for_anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
)
AIModule = AgentModule.for_root(agents=[TravelAgent], tools=[get_weather])


@module(controllers=[TravelController], imports=[LLMProviderModule, AIModule])
class AppModule: ...


app = LaurenFactory.create(AppModule)
```

## Documentation

Full documentation: https://lauren-framework.github.io/lauren-ai/

## Development

```bash
# Install with dev dependencies
uv sync --extra dev --extra anthropic

# Run tests
uv run nox -s tests

# Run linter
uv run nox -s lint

# Build docs
uv run nox -s docs
```

## License

MIT — see [LICENSE](LICENSE).

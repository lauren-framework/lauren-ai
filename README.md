<p align="center">
  <img src="https://raw.githubusercontent.com/lauren-framework/lauren-framework/refs/heads/main/assets/lauren-logo.png" width=40%></img>
</p>
<div align="center">
  <h1><i>lauren-ai</i></h1>
</div>
<p align="center">
    <em>lauren-ai: the first-party AI/LLM companion to the Lauren web framework. Agents, tools, memory, workflows, RAG, and evaluation in the same decorator-first, DI-driven programming model.</em>
</p>
<p align="center">
<a href="https://github.com/lauren-framework/lauren-ai/actions/workflows/tests.yml?query=branch%3Amain+event%3Apush">
    <img src="https://github.com/lauren-framework/lauren-ai/actions/workflows/tests.yml/badge.svg?branch=main&event=push" alt="Test">
</a>
<a href="https://github.com/lauren-framework/lauren-ai/actions/workflows/lint.yml?query=branch%3Amain+event%3Apush">
    <img src="https://github.com/lauren-framework/lauren-ai/actions/workflows/lint.yml/badge.svg?branch=main&event=push" alt="Lint">
</a>
<a href="https://github.com/lauren-framework/lauren-ai/actions/workflows/codeql.yml?query=branch%3Amain">
    <img src="https://github.com/lauren-framework/lauren-ai/actions/workflows/codeql.yml/badge.svg?branch=main" alt="CodeQL">
</a>
<a href="https://codecov.io/gh/lauren-framework/lauren-ai">
    <img src="https://img.shields.io/codecov/c/github/lauren-framework/lauren-ai?color=%2334D058&label=coverage" alt="Coverage">
</a>
<a href="https://pypi.org/project/lauren-ai/">
    <img src="https://img.shields.io/pypi/v/lauren-ai?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/lauren-ai/">
    <img src="https://img.shields.io/pypi/pyversions/lauren-ai.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://pypi.org/project/lauren-ai/">
    <img src="https://img.shields.io/pypi/dm/lauren-ai.svg?color=%2334D058&label=downloads" alt="Downloads">
</a>
<a href="https://github.com/lauren-framework/lauren-ai/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/lauren-framework/lauren-ai.svg?color=%2334D058" alt="License">
</a>
<a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
</a>
<a href="https://mypy.readthedocs.io/en/stable/">
    <img src="https://img.shields.io/badge/types-mypy-blue.svg" alt="Checked with mypy">
</a>
<a href="https://github.com/j178/prek">
    <img src="https://img.shields.io/badge/pre--commit-prek-FAB040.svg?logo=pre-commit&logoColor=white" alt="prek">
</a>
<a href="https://github.com/lauren-framework/lauren-ai/discussions">
    <img src="https://img.shields.io/github/discussions/lauren-framework/lauren-ai?color=%2334D058&label=discussions" alt="Discussions">
</a>
<a href="https://github.com/lauren-framework/lauren-ai/stargazers">
    <img src="https://img.shields.io/github/stars/lauren-framework/lauren-ai.svg?style=social&label=Star" alt="GitHub Stars">
</a>
</p>

---

**Documentation**: <a href="https://ai.lauren-py.dev" target="_blank">https://github.com/lauren-framework/lauren-ai/tree/main/docs</a>

**Source Code**: <a href="https://github.com/lauren-framework/lauren-ai" target="_blank">https://github.com/lauren-framework/lauren-ai</a>

---

## For AI Agents & Coding Assistants

### Install all skills in one command

```bash
# Claude Code, Cursor, Copilot, Continue, Codex CLI -- auto-detected
npx skills add lauren-framework/lauren-ai
```

This copies the repository's skill packs into your agent's global skills
directory (`~/.claude/skills/`, `~/.cursor/skills/`, etc.). The next time your
agent opens a `lauren-ai` project it has pre-loaded guidance on agents, tools,
memory, evaluation, workflows, and Lauren integration patterns.

| Resource | What it contains |
|---|---|
| [`llms.txt`](https://raw.githubusercontent.com/lauren-framework/lauren-ai/refs/heads/main/llms.txt) | 2 KB overview -- start here |
| [`llms-full.txt`](https://raw.githubusercontent.com/lauren-framework/lauren-ai/refs/heads/main/llms-full.txt) | Complete reference -- all decorators, memory, guardrails, teams, and common errors |
| [`AGENTS.md`](https://github.com/lauren-framework/lauren-ai/blob/main/AGENTS.md) | By-task lookup, common errors, skills index, definition of done |
| [`CLAUDE.md`](https://github.com/lauren-framework/lauren-ai/blob/main/CLAUDE.md) | Architecture invariants, commands, pattern selection, codemap navigation |
| [`skills/`](https://github.com/lauren-framework/lauren-ai/tree/main/skills/) | Copy-paste skill guides covering common `lauren-ai` tasks |

Skills index: [`skills/`](https://github.com/lauren-framework/lauren-ai/tree/main/skills/)

---

`lauren-ai` is the first-party AI/LLM companion to the
[Lauren web framework](https://github.com/lauren-framework/lauren-framework). It
brings large language model agents into the same decorator-first, DI-driven,
module-scoped programming model the rest of Lauren uses. It is built on these
core ideas:

* **Decorator-first metadata.** Agents, tools, teams, guardrails, prompt
  templates, and memory behaviors are declared with decorators and metadata
  rather than custom runtime glue.
* **Module-scoped integration.** `LLMModule.for_root()` and
  `AgentModule.for_root()` plug into Lauren's module system so transports,
  runners, tools, and stores compose like the rest of your application.
* **Provider-agnostic execution.** Anthropic, OpenAI, Ollama, LiteLLM, and
  `MockTransport` all flow through a unified transport layer and consistent
  agent runner API.
* **AI-ready by default.** The public surface is mirrored in `llms.txt`,
  `llms-full.txt`, `AGENTS.md`, `CLAUDE.md`, and `skills/` so both human and AI
  contributors can navigate the package quickly.

## Features

* **Provider-agnostic transport** - Anthropic, OpenAI, Ollama, LiteLLM, and
  `MockTransport` for zero-network-call tests.
* **`@tool()` decorator** - Function-form and class-form (DI-injected), with
  JSON Schema auto-generated from type hints and docstrings.
* **`@agent()` decorator** - Autonomous agentic loop with `use_tools()`,
  lifecycle hooks, retry/error policies, and budget guards.
* **Four memory tiers** - `ShortTermMemory`, `ConversationStore`,
  `UserMemoryStore` plus `@remember()`, and vector-backed retrieval for RAG.
* **Typed extractors** - `Agent[T]`, `Completion[T]`, `Embed[T]`, and
  `StreamCompletion[T]` as Lauren handler parameters.
* **Module factories** - `LLMModule.for_root()` and `AgentModule.for_root()`
  feel like the rest of Lauren's DI-first module configuration.
* **Knowledge Base and agentic RAG** - `KnowledgeBase` with document loaders,
  hybrid retrieval, and `kb.as_tool()`.
* **Structured workflows** - `Workflow`, `Step`, `Parallel`, `Condition`, and
  `Loop` for deterministic multi-agent pipelines.
* **Tool enhancements** - Human-in-the-loop confirmation, pre/post hooks, and
  result caching.
* **Extended thinking** - First-class support for Claude extended thinking and
  OpenAI reasoning models.
* **Evaluation framework** - `AccuracyEval`, `AgentJudge`, `TrajectoryEval`,
  and `PerformanceEval`.
* **Signals** - `ModelCallComplete`, `ToolCallComplete`, and
  `AgentRunComplete` for observability.
* **Pre-built skills** - `WebSearchTool`, `CodeExecutionTool`, and
  `HttpFetchTool`.
* **Testable** - Zero API calls needed in unit tests via `MockTransport`.

## Requirements

Python **3.11**, **3.12**, **3.13**, and **3.14** are supported. Core
dependencies:

* [Lauren](https://github.com/lauren-framework/lauren-framework) - the host web
  framework and DI/module runtime.
* [Pydantic](https://docs.pydantic.dev/) - structured models, schemas, and
  validation.
* [httpx](https://www.python-httpx.org/) - provider HTTP transport.
* [anyio](https://anyio.readthedocs.io/) - concurrency primitives and async
  portability.

## Installation

```bash
# Core (no LLM provider extras)
pip install lauren-ai

# With Anthropic
pip install "lauren-ai[anthropic]"

# With OpenAI
pip install "lauren-ai[openai]"

# With everything
pip install "lauren-ai[all]"
```

## Quick start

```python
import os

from pydantic import BaseModel

from lauren import LaurenFactory, controller, module, post
from lauren.types import Json
from lauren_ai import (
    Agent,
    AgentModule,
    AgentRunner,
    InMemoryConversationStore,
    LLMConfig,
    LLMModule,
    agent,
    tool,
    use_tools,
)


@tool()
async def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: The city name.
    """
    return {"city": city, "temperature_c": 18, "condition": "cloudy"}


# @agent() is outermost; @use_tools() is below it (applied first)
@agent(model="claude-opus-4-6", system="You are a helpful travel assistant.")
@use_tools(get_weather)
class TravelAgent: ...


class AskRequest(BaseModel):
    question: str


@controller("/travel")
class TravelController:
    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    @post("/ask")
    async def ask(self, body: Json[AskRequest], agent: Agent[TravelAgent]) -> dict:
        response = await self._runner.run(agent, body.question)
        return {"answer": response.content, "turns": response.turns}


LLMProviderModule = LLMModule.for_root(
    LLMConfig.for_anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
)
AIAgentModule = AgentModule.for_root(
    agents=[TravelAgent],
    tools=[get_weather],
    conversation_store=InMemoryConversationStore(),  # history persists across requests
)


@module(controllers=[TravelController], imports=[LLMProviderModule, AIAgentModule])
class AppModule: ...


app = LaurenFactory.create(AppModule)
```

```bash
uvicorn main:app --reload
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev --extra anthropic

# Run tests
uv run nox -s tests

# Run linter
uv run nox -s lint

# Run typecheck
uv run nox -s typecheck

# Build docs
uv run nox -s docs
```

## License

MIT - see [LICENSE](https://github.com/lauren-framework/lauren-ai/blob/main/LICENSE).

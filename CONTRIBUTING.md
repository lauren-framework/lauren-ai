# Contributing to lauren-ai

Thank you for contributing! This guide covers everything you need to develop,
test, and release `lauren-ai`.

---

## 1. Prerequisites

| Tool | Version |
|---|---|
| Python | ≥ 3.11 |
| uv | ≥ 0.4 |
| git | any recent |

Install `uv` globally. `nox` is already included in the dev dependencies and is
normally invoked through `uv run`:

```bash
pip install uv
```

---

## 2. Development setup

```bash
git clone https://github.com/lauren-framework/lauren-ai
cd lauren-ai

# Install the project + dev extras
uv sync --extra dev

# Activate the venv
source .venv/bin/activate

# Install pre-commit hooks
uv tool install prek
prek install
```

---

## 3. Running tests

```bash
# Unit tests only (fast)
uv run nox -s tests_unit

# All tests
uv run nox -s tests

# With coverage report
uv run nox -s coverage

# Benchmarks (excluded from default run)
uv run nox -s benchmark
```

Running `uv run nox` with no `-s` now executes the repository's default PR gate
session set: `lint`, `tests`, `format`, `build`, `build_check`, and `prek`.

The default `pytest` run excludes `benchmark` and `eval` marked tests.
Coverage must reach **80%** or the test run fails.

---

## 4. Linting and formatting

```bash
# Check lint + format
uv run nox -s lint

# Auto-fix
uv run nox -s format
```

All code uses **ruff** for linting and formatting. Line length is 100.

---

## 5. Type checking

```bash
uv run nox -s typecheck
```

All public API surfaces must have complete type annotations. Private helpers
should have annotations too unless they are trivially obvious.

---

## 6. Documentation

```bash
# Build docs
uv run nox -s docs

# Serve locally with live reload
uv run nox -s docs_serve
```

Docs live in `docs/` and are built with **MkDocs Material**. All public API
symbols must have RST-style docstrings (`:param:`, `:type:`, `:return:`,
`:rtype:`).

---

## 7. Branch naming

All branches must follow:

```
<type>(#<issue-number>)-<short-description>
```

**Allowed types:** `feat`, `fix`, `chore`, `task`, `refactor`, `docs`, `test`

Examples:
```
feat(#42)-add-streaming-support
fix(#99)-tool-schema-optional-params
docs-update-knowledge-base-guide
```

The pre-commit hook (`prek install`) enforces this automatically.

---

## 8. Commit messages

Commits follow the [Conventional Commits](https://www.conventionalcommits.org/)
specification:

```
feat: add streaming support to AgentRunner
fix: correct ToolSchema optional param handling
docs: update knowledge base guide examples
test: add coverage for MockTransport streaming
```

---

## 9. Adding a new transport provider

1. Create `src/lauren_ai/_transport/_<provider>.py`
2. Implement the `Transport` protocol (all 4 methods: `complete`, `complete_stream`, `embed`, `count_tokens`)
3. Register the provider name in `_module.py` `_build_transport()`
4. Add the provider to `pyproject.toml` optional extras
5. Add tests under `tests/unit/` using `MockTransport` patterns
6. Document in `docs/guides/llm-calls.md`

---

## 10. Adding a new tool skill

1. Add an `@tool()`-decorated async function to `src/lauren_ai/_skills/__init__.py`
2. Export it from `__all__`
3. Keep function-form `@tool()` annotations importable at module import time;
   `from __future__ import annotations` is supported, but unresolved forward refs
   and circular imports still break schema generation
4. Write unit tests mocking any external calls

---

## 11. Release process

Releases are triggered by pushing a version tag:

```bash
git tag v1.2.3
git push origin v1.2.3
```

The GitHub Actions `release.yml` workflow will:
1. Run the full CI matrix
2. Build the wheel and sdist
3. Publish to PyPI via OIDC (no manual token needed)

Development and release workflow details now live in:

- `docs/development/release.md`
- `docs/development/versioning.md`

Versioning is managed by `setuptools-scm`. Never set `version =` manually in
`pyproject.toml`.

---

## 12. Key design invariants

| Rule | Details |
|---|---|
| Decorator parentheses | `@tool()`, `@agent()`, `@team()`, `@guardrail()`, `@remember()`, `@traced()` — never bare |
| Decorator order | `@agent()` outermost, `@use_tools()` below |
| `from __future__ import annotations` | Supported, but tool signature types must resolve when `@tool()` builds the schema |
| `__all__` | Required in every public module |
| RST docstrings | Required on all public symbols |
| Transport protocol | Never import a specific provider at module level |
| `ToolResult` | Always pass `tool_use_id=` kwarg |
| `PydanticOutputParser` | Keep referenced schema types importable in examples and parser wiring |
| `@guardrail()` | Decorator takes `input=[...]` and `output=[...]` lists of guardrail instances |
| `GuardrailDecision.action` | Must be `"pass"`, `"block"`, or `"modify"` |
| `@team()` | Requires `mode="coordinator"` or `mode="collaborate"` |
| `StructuredLLM[T]` | Created via `llm.with_structured_output(ModelClass)`, never constructed directly |
| `UserMemoryStore` | `@runtime_checkable` Protocol — implementations must be injectable singletons |
| `TraceExporter` | `@runtime_checkable` Protocol — `export(trace)` must be async |

---

## 13. New features quick reference

### Prompt templates & chains (Section 32)

```python
from lauren_ai import PromptTemplate, ChatPromptTemplate, Chain, StrOutputParser

# String template with {variable} interpolation
tmpl = PromptTemplate("Translate to {language}: {text}")

# Chain with pipe composition
chain = tmpl | llm_service | StrOutputParser()
result = await chain.invoke(language="French", text="Hello!")
```

### Output parsers (Section 33)

```python
from lauren_ai import PydanticOutputParser, JSONOutputParser, RetryOutputParser

class Sentiment(BaseModel):
    label: str
    score: float

parser = PydanticOutputParser(Sentiment)
with_retry = RetryOutputParser(parser, llm=llm_service, max_retries=2)
```

### Agent teams (Section 34)

```python
from lauren_ai import team, TeamRunner

@team(name="ResearchTeam", mode="coordinator", max_rounds=5)
class ResearchTeam:
    def __init__(self, researcher: ResearchAgent, writer: WriterAgent) -> None: ...

runner = TeamRunner(ResearchTeam, llm=llm_service, agent_runner=agent_runner)
result = await runner.run("Summarise recent AI news")
```

### Tracing (Section 35)

```python
from lauren_ai import traced, InMemoryTraceExporter, SpanKind

exporter = InMemoryTraceExporter()

@traced(name="my_agent_run", kind=SpanKind.AGENT, exporter=exporter)
async def run_agent(task: str) -> str: ...
```

### Persistent user memory (Section 36)

```python
from lauren_ai import remember, InMemoryUserMemoryStore

store = InMemoryUserMemoryStore()

@remember(store=store, extract=True, inject=True, top_k=5)
@agent(model="claude-opus-4-6")
class PersonalAssistant: ...
```

### Structured output (Section 37)

```python
from lauren_ai import LLMService
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float

structured = llm.with_structured_output(Sentiment)
result: Sentiment = await structured.complete([Message(role="user", content="Great!")])
```

### Multimodal inputs (Section 38)

```python
from lauren_ai import ImageContent, Message

img = ImageContent.from_file("/tmp/chart.png")
msg = Message.from_multimodal("user", ["Describe this chart:", img])
```

### Semantic router (Section 39)

```python
from lauren_ai import SemanticRouter, Route

router = SemanticRouter(
    routes=[Route(name="support", examples=["I need help", "broken"]),
            Route(name="sales",   examples=["pricing", "buy"])],
    embed_fn=llm.embed,
)
await router.compile()
match = await router.route("How do I cancel my subscription?")
```

### Cost & rate tracking (Section 40)

```python
from lauren_ai import CostTracker, TokenBudget, default_pricing_table

tracker = CostTracker(pricing=default_pricing_table())
async with tracker.session(conversation_id="c1") as session:
    ...  # usage is accumulated automatically
print(session.total_estimate.total_usd)
```

### Guardrails (Section 41)

```python
from lauren_ai import guardrail, PIIRedactor, LengthFilter, PromptInjectionFilter

@guardrail(
    input=[PromptInjectionFilter(), LengthFilter(max_chars=2000)],
    output=[PIIRedactor(entities=["EMAIL", "PHONE"])],
)
@agent(model="claude-opus-4-6")
class SafeAgent: ...
```

---

## 14. Getting help

- Open an issue: https://github.com/lauren-framework/lauren-ai/issues
- Discussions: https://github.com/lauren-framework/lauren-ai/discussions

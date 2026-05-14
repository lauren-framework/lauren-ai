---
name: system-prompt-templating
description: Builds system prompts dynamically with composable sections, template variable substitution, and fluent builder API. Use when agent system prompts need to vary by user, context, or feature flag; when assembling role + context + constraints programmatically; or when sharing prompt fragments across multiple agents.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# System Prompt Templating & Dynamic Composition

## Built-in support: `PromptTemplate` and `ChatPromptTemplate`

Lauren-AI ships `PromptTemplate` (single message) and `ChatPromptTemplate`
(multi-turn) in `lauren_ai._prompts`.  Use them for simple `{variable}` interpolation:

```python
# NOTE: future annotations are allowed, but tool signature types must resolve at import time.
from lauren_ai import PromptTemplate, ChatPromptTemplate

tpl = PromptTemplate(template="You are a {role}. Context: {context}")
msg = tpl.render(role="financial analyst", context="User is a portfolio manager")
# msg.content == "You are a financial analyst. Context: User is a portfolio manager"
```

## Custom builder for richer composition

When you need conditional sections, structured constraints, or a fluent API,
implement a `SystemPromptBuilder`:

```python
class PromptTemplate:
    def __init__(self, template: str):
        self._template = template

    def render(self, **kwargs) -> str:
        result = self._template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class SystemPromptBuilder:
    def __init__(self):
        self._sections: list[str] = []

    def add_role(self, role: str) -> "SystemPromptBuilder":
        self._sections.append(f"You are {role}.")
        return self

    def add_context(self, context: str) -> "SystemPromptBuilder":
        self._sections.append(f"Context: {context}")
        return self

    def add_instruction(self, instruction: str) -> "SystemPromptBuilder":
        self._sections.append(instruction)
        return self

    def add_constraints(self, *constraints: str) -> "SystemPromptBuilder":
        if constraints:
            self._sections.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
        return self

    def build(self) -> str:
        return "\n\n".join(self._sections)
```

## Full example

```python
system = (
    SystemPromptBuilder()
    .add_role("a financial analyst specializing in risk assessment")
    .add_context("User is a portfolio manager at a hedge fund")
    .add_instruction("Always provide quantitative estimates with confidence intervals.")
    .add_constraints(
        "Never give specific stock picks",
        "Always mention risks",
    )
    .build()
)

from lauren_ai import agent

@agent(model="claude-opus-4-6", system=system)
class FinancialAgent: ...
```

## Conditional sections

```python
import os

builder = SystemPromptBuilder().add_role("a helpful assistant")
if os.getenv("VERBOSE_MODE"):
    builder.add_instruction("Explain each step in detail.")
if user_tier == "enterprise":
    builder.add_context("User has access to premium data feeds.")
system = builder.build()
```

## Template variables with `PromptTemplate`

```python
tpl = PromptTemplate(template="You are {role} working for {company}.")
system = tpl.render(role="a legal advisor", company="Acme Corp")
```

## Pitfalls

- `build()` joins sections with `"\n\n"` — sections should not have trailing
  newlines themselves or you get extra blank lines.
- Template `{key}` replacement is plain string substitution — curly braces in
  the template for non-variable purposes must be avoided or escaped.
- Prefer the built-in `PromptTemplate` for simple cases; the custom builder
  adds no overhead when sections are unconditional.

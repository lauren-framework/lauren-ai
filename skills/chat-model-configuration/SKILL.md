---
name: chat-model-configuration
description: Configures agent behaviour via @agent() parameters and AgentConfig. Use when adjusting max_turns, temperature, max_tokens_per_turn, max_cost_usd, tool_error_policy, memory window, or when the agent must stop after a fixed number of turns.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Chat Model Configuration & Parameter Tuning

## Quick start

```python
# agent.py — from __future__ import annotations is safe here (no @tool)
from __future__ import annotations

from lauren_ai import agent
from lauren_ai._memory._stores import InMemoryConversationStore

@agent(
    model="claude-opus-4-6",
    system="You are a helpful assistant.",
    max_turns=3,
    temperature=0.7,
    max_tokens_per_turn=2048,
    memory_window_tokens=20_000,
    tool_error_policy="return_error",
    conversation_store=InMemoryConversationStore(),
)
class ConfiguredAgent: ...
```

---

## @agent() parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier, e.g. `"claude-opus-4-6"` |
| `system` | `str` | `""` | System prompt for every turn |
| `max_turns` | `int` | `10` | Loop iteration cap; raises `AgentMaxTurnsError` |
| `max_tokens_per_turn` | `int` | `4096` | Output token limit per LLM call |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0–2.0) |
| `memory_window_tokens` | `int` | `40_000` | Sliding context window in tokens |
| `max_cost_usd` | `float \| None` | `None` | Cost cap; raises `AgentBudgetExceededError` |
| `parallel_tool_calls` | `bool` | `False` | Run all tool calls in a turn concurrently |
| `tool_error_policy` | `str` | `"return_error"` | `"raise"` \| `"return_error"` \| `"skip"` |
| `conversation_store` | `ConversationStore \| None` | `None` | Persistent conversation history |
| `memory` | `ShortTermMemory \| None` | `None` | Per-agent shared short-term memory |

---

## max_turns enforcement

```python
from lauren_ai import agent, AgentMaxTurnsError
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._transport._mock import MockTransport

@agent(model="claude-opus-4-6", max_turns=2)
class LimitedAgent: ...

cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
mock = MockTransport()
runner = AgentRunner(transport=mock, tools={}, config=cfg)

# Queue 5 tool-use completions — runner stops at max_turns
# AgentMaxTurnsError is raised once the cap is hit
```

---

## Lifecycle-doc class (using a docstring as system prompt)

```python
@agent(model="claude-opus-4-6", max_turns=5)
class DocAgent:
    """You are an expert Python programmer.
    Always respond with working code.
    """
```

The class docstring becomes the system prompt when no `system=` kwarg is given.

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_config.py` | `AgentConfig` frozen dataclass — all defaults |
| `src/lauren_ai/_agents/__init__.py` | `@agent()` decorator, `AgentMeta` |
| `src/lauren_ai/_agents/_runner.py` | `AgentRunnerBase` loop with `max_turns` check |

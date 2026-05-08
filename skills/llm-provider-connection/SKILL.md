---
name: llm-provider-connection
description: Configures an LLM provider connection in lauren-ai using LLMConfig and LLMModule. Use when setting up a new provider (Anthropic, OpenAI, Ollama), switching models, injecting the LLM into a Lauren DI module, or writing tests that need a zero-network MockTransport.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# LLM Provider Connection

## Quick start

```python
from lauren_ai import LLMConfig, LLMModule

# Anthropic
cfg = LLMConfig.for_anthropic(model="claude-opus-4-6", api_key="sk-ant-...")

# OpenAI
cfg = LLMConfig.for_openai(model="gpt-4o", api_key="sk-...")

# Ollama (local, no API key needed)
cfg = LLMConfig.for_ollama(model="llama3.2")

# Tests — zero network calls
cfg, mock = LLMConfig.for_testing()
```

---

## LLMConfig parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | required | `"anthropic"`, `"openai"`, `"ollama"`, or `"litellm"` |
| `model` | `str` | required | Model identifier, e.g. `"claude-opus-4-6"` |
| `api_key` | `str \| None` | `None` | API key; falls back to env var |
| `base_url` | `str \| None` | `None` | Override default provider URL |
| `max_tokens` | `int` | `4096` | Maximum tokens per completion |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0–2.0) |
| `timeout` | `float` | `60.0` | HTTP timeout in seconds |
| `max_retries` | `int` | `3` | Retries on transient errors |
| `cache_system_prompt` | `bool` | `False` | Anthropic prompt caching for system prompt |
| `cache_tools` | `bool` | `False` | Anthropic prompt caching for tool definitions |

---

## Direct constructor form

```python
from lauren_ai import LLMConfig

# Explicit constructor (all three providers)
cfg_anthropic = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-ant-...")
cfg_openai    = LLMConfig(provider="openai",    model="gpt-4o",          api_key="sk-...")
cfg_ollama    = LLMConfig(provider="ollama",    model="llama3.2")        # no key needed
```

---

## Wire into Lauren DI

```python
from lauren_ai import LLMModule, LLMConfig
from lauren import module

cfg = LLMConfig.for_anthropic(model="claude-opus-4-6")
LLMProvider = LLMModule.for_root(cfg)

@module(imports=[LLMProvider])
class AppModule: ...
```

---

## Testing pattern

```python
from lauren_ai import LLMConfig
from lauren_ai._transport import Completion, TokenUsage

cfg, mock = LLMConfig.for_testing()       # provider="anthropic", model="mock-model"

mock.queue_response(
    Completion(
        id="c1",
        model="mock-model",
        content="Hello!",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
)
# … run your agent code; mock.calls records every request
```

---

## Reference files

| File | Contents |
|------|----------|
| `src/lauren_ai/_config.py` | `LLMConfig`, `AgentConfig` frozen dataclasses |
| `src/lauren_ai/_module.py` | `LLMModule.for_root()` DI registration |
| `src/lauren_ai/_transport/_mock.py` | `MockTransport` queue API |

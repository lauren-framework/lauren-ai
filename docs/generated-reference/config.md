# Configuration

Frozen dataclasses that configure the LLM provider and agent behaviour.

### `LLMConfig`

```python
class LLMConfig(provider: Literal['anthropic', 'openai', 'ollama', 'litellm'], model: str, api_key: str | None = None, base_url: str | None = None, max_tokens: int = 4096, temperature: float = 1.0, timeout: float = 60.0, max_retries: int = 3, cache_system_prompt: bool = False, cache_tools: bool = False, embed_model: str | None = None, embed_dimensions: int | None = None)
```

Immutable configuration for an LLM provider connection.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `provider` | `Literal['anthropic', 'openai', 'ollama', 'litellm']` | The backend provider.  One of `"anthropic"`,
`"openai"`, `"ollama"`, or `"litellm"`. |
| `model` | `str` | The model identifier, e.g. `"claude-opus-4-6"` or
`"gpt-4o"`. |
| `api_key` | `str | None` | Provider API key.  When *None* the provider-specific
environment variable is used (e.g. `ANTHROPIC_API_KEY`). |
| `base_url` | `str | None` | Override the provider's default base URL.  Useful for
proxies, self-hosted deployments, or Ollama. |
| `max_tokens` | `int` | Maximum tokens to generate per completion call. |
| `temperature` | `float` | Sampling temperature (0.0–2.0 for most providers). |
| `timeout` | `float` | HTTP request timeout in seconds. |
| `max_retries` | `int` | Maximum number of automatic retries on transient
errors. |
| `cache_system_prompt` | `bool` | Enable Anthropic prompt caching for the
system prompt.  No-op on other providers. |
| `cache_tools` | `bool` | Enable Anthropic prompt caching for the tool
definitions.  No-op on other providers. |
| `embed_model` | `str | None` | Model to use for embedding calls.  Defaults to
`model` when *None*. |
| `embed_dimensions` | `int | None` | Desired embedding dimensionality.  Passed to
providers that support truncated embeddings. |

#### `LLMConfig.for_anthropic`

```python
def for_anthropic(cls, model: str = 'claude-opus-4-6', api_key: str | None = None, kwargs: Any = {}) -> LLMConfig
```

Create a config pre-wired for Anthropic.

The API key is read from the `ANTHROPIC_API_KEY` environment
variable when *api_key* is `None`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `str` | Anthropic model identifier.
Defaults to `"claude-opus-4-6"`. |
| `api_key` | `str | None` | Anthropic API key.  Falls back to
`os.environ["ANTHROPIC_API_KEY"]`. |
| `kwargs` | `Any` | Additional keyword arguments forwarded verbatim to
the `LLMConfig` constructor. |

**Returns:** `LLMConfig` — A fully-initialised `LLMConfig` for Anthropic.

#### `LLMConfig.for_openai`

```python
def for_openai(cls, model: str = 'gpt-4o', api_key: str | None = None, kwargs: Any = {}) -> LLMConfig
```

Create a config pre-wired for OpenAI.

The API key is read from the `OPENAI_API_KEY` environment
variable when *api_key* is `None`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `str` | OpenAI model identifier.  Defaults to `"gpt-4o"`. |
| `api_key` | `str | None` | OpenAI API key.  Falls back to
`os.environ["OPENAI_API_KEY"]`. |
| `kwargs` | `Any` | Additional keyword arguments forwarded verbatim to
the `LLMConfig` constructor. |

**Returns:** `LLMConfig` — A fully-initialised `LLMConfig` for OpenAI.

#### `LLMConfig.for_ollama`

```python
def for_ollama(cls, model: str = 'llama3.2', base_url: str = 'http://localhost:11434', kwargs: Any = {}) -> LLMConfig
```

Create a config pre-wired for a local Ollama server.

No API key is required.  The default `base_url` points to a
locally-running Ollama instance.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model` | `str` | Ollama model tag, e.g. `"llama3.2"` or
`"mistral"`.  Defaults to `"llama3.2"`. |
| `base_url` | `str` | Ollama server URL.
Defaults to `"http://localhost:11434"`. |
| `kwargs` | `Any` | Additional keyword arguments forwarded verbatim to
the `LLMConfig` constructor. |

**Returns:** `LLMConfig` — A fully-initialised `LLMConfig` for Ollama.

#### `LLMConfig.for_testing`

```python
def for_testing(cls) -> tuple[LLMConfig, MockTransport]
```

Create a test config paired with a `MockTransport`.

No network calls will ever be made.  Queue deterministic responses on
the returned `MockTransport`
instance before running your code under test.

**Returns:** `tuple[LLMConfig, MockTransport]

Example::

    cfg, mock = LLMConfig.for_testing()
    mock.queue_response(
        Completion(
            id="test-1",
            model="mock-model",
            content="Hello!",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
    )` — A 2-tuple of `(LLMConfig, MockTransport)`.

### `AgentConfig`

```python
class AgentConfig(system_prompt: str = 'You are a helpful assistant.', max_turns: int = 10, max_tokens_per_turn: int = 4096, temperature: float = 1.0, memory_window_tokens: int = 40000, max_cost_usd: float | None = None, parallel_tool_calls: bool = False, tool_error_policy: Literal['raise', 'return_error', 'skip'] = 'return_error', thinking: bool = False, thinking_budget_tokens: int = 8000, reasoning_effort: Literal['low', 'medium', 'high'] | None = None, include_reasoning_in_response: bool = False)
```

Immutable configuration for an agent's runtime behaviour.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `system_prompt` | `str` | The system prompt sent to the LLM at the start of
every turn. |
| `max_turns` | `int` | Maximum number of agentic loop iterations before
`AgentMaxTurnsError` is raised. |
| `max_tokens_per_turn` | `int` | Maximum output tokens requested per turn. |
| `temperature` | `float` | Sampling temperature for this agent.  Overrides the
`LLMConfig` temperature when set. |
| `memory_window_tokens` | `int` | Sliding-window size in tokens for
conversation history passed to the model. |
| `max_cost_usd` | `float | None` | Hard cost budget in USD.  The runner checks after
each turn and raises
`AgentBudgetExceededError` when
exceeded.  `None` means unlimited. |
| `parallel_tool_calls` | `bool` | When `True` all tool calls in a single
model turn are executed concurrently.  Defaults to `False` to
preserve deterministic ordering guarantees. |
| `tool_error_policy` | `Literal['raise', 'return_error', 'skip']` | How to handle a tool execution error:

* `"raise"` — re-raise the exception immediately.
* `"return_error"` — send the error message back to the model as
  a tool result so it can decide how to proceed.
* `"skip"` — silently omit the failing tool result. |
| `thinking` | `bool` | Enable extended thinking (Anthropic only). |
| `thinking_budget_tokens` | `int` | Token budget for the thinking phase when
`thinking=True`. |
| `reasoning_effort` | `Literal['low', 'medium', 'high'] | None` | OpenAI reasoning effort for o1/o3 models
(`"low"`, `"medium"`, or `"high"`).  `None` means the
provider default. |
| `include_reasoning_in_response` | `bool` | When `True` thinking blocks are
included in the `Completion` response. |


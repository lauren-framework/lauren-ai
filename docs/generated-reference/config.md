# Configuration

Frozen dataclasses that configure the LLM provider and agent behaviour.

### `LLMConfig`

```python
class LLMConfig(provider: Literal['anthropic', 'openai', 'ollama', 'litellm'], model: str, api_key: str | None = None, base_url: str | None = None, max_tokens: int = 4096, temperature: float = 1.0, timeout: float = 60.0, max_retries: int = 3, cache_system_prompt: bool = False, cache_tools: bool = False, embed_model: str | None = None, embed_dimensions: int | None = None)
```

Immutable configuration for an LLM provider connection.

:param provider: The backend provider.  One of ``"anthropic"``,
    ``"openai"``, ``"ollama"``, or ``"litellm"``.
:type provider: Literal["anthropic", "openai", "ollama", "litellm"]
:param model: The model identifier, e.g. ``"claude-opus-4-6"`` or
    ``"gpt-4o"``.
:type model: str
:param api_key: Provider API key.  When *None* the provider-specific
    environment variable is used (e.g. ``ANTHROPIC_API_KEY``).
:type api_key: str | None
:param base_url: Override the provider's default base URL.  Useful for
    proxies, self-hosted deployments, or Ollama.
:type base_url: str | None
:param max_tokens: Maximum tokens to generate per completion call.
:type max_tokens: int
:param temperature: Sampling temperature (0.0–2.0 for most providers).
:type temperature: float
:param timeout: HTTP request timeout in seconds.
:type timeout: float
:param max_retries: Maximum number of automatic retries on transient
    errors.
:type max_retries: int
:param cache_system_prompt: Enable Anthropic prompt caching for the
    system prompt.  No-op on other providers.
:type cache_system_prompt: bool
:param cache_tools: Enable Anthropic prompt caching for the tool
    definitions.  No-op on other providers.
:type cache_tools: bool
:param embed_model: Model to use for embedding calls.  Defaults to
    ``model`` when *None*.
:type embed_model: str | None
:param embed_dimensions: Desired embedding dimensionality.  Passed to
    providers that support truncated embeddings.
:type embed_dimensions: int | None

#### `LLMConfig.for_anthropic`

```python
def for_anthropic(cls, model: str = 'claude-opus-4-6', api_key: str | None = None, kwargs: Any = {}) -> LLMConfig
```

Create a config pre-wired for Anthropic.

The API key is read from the ``ANTHROPIC_API_KEY`` environment
variable when *api_key* is ``None``.

:param model: Anthropic model identifier.
    Defaults to ``"claude-opus-4-6"``.
:type model: str
:param api_key: Anthropic API key.  Falls back to
    ``os.environ["ANTHROPIC_API_KEY"]``.
:type api_key: str | None
:param kwargs: Additional keyword arguments forwarded verbatim to
    the :class:`LLMConfig` constructor.
:return: A fully-initialised :class:`LLMConfig` for Anthropic.
:rtype: LLMConfig

#### `LLMConfig.for_openai`

```python
def for_openai(cls, model: str = 'gpt-4o', api_key: str | None = None, kwargs: Any = {}) -> LLMConfig
```

Create a config pre-wired for OpenAI.

The API key is read from the ``OPENAI_API_KEY`` environment
variable when *api_key* is ``None``.

:param model: OpenAI model identifier.  Defaults to ``"gpt-4o"``.
:type model: str
:param api_key: OpenAI API key.  Falls back to
    ``os.environ["OPENAI_API_KEY"]``.
:type api_key: str | None
:param kwargs: Additional keyword arguments forwarded verbatim to
    the :class:`LLMConfig` constructor.
:return: A fully-initialised :class:`LLMConfig` for OpenAI.
:rtype: LLMConfig

#### `LLMConfig.for_ollama`

```python
def for_ollama(cls, model: str = 'llama3.2', base_url: str = 'http://localhost:11434', kwargs: Any = {}) -> LLMConfig
```

Create a config pre-wired for a local Ollama server.

No API key is required.  The default ``base_url`` points to a
locally-running Ollama instance.

:param model: Ollama model tag, e.g. ``"llama3.2"`` or
    ``"mistral"``.  Defaults to ``"llama3.2"``.
:type model: str
:param base_url: Ollama server URL.
    Defaults to ``"http://localhost:11434"``.
:type base_url: str
:param kwargs: Additional keyword arguments forwarded verbatim to
    the :class:`LLMConfig` constructor.
:return: A fully-initialised :class:`LLMConfig` for Ollama.
:rtype: LLMConfig

#### `LLMConfig.for_testing`

```python
def for_testing(cls) -> tuple[LLMConfig, MockTransport]
```

Create a test config paired with a :class:`~lauren_ai._transport._mock.MockTransport`.

No network calls will ever be made.  Queue deterministic responses on
the returned :class:`~lauren_ai._transport._mock.MockTransport`
instance before running your code under test.

:return: A 2-tuple of ``(LLMConfig, MockTransport)``.
:rtype: tuple[LLMConfig, MockTransport]

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
    )

### `AgentConfig`

```python
class AgentConfig(system_prompt: str = 'You are a helpful assistant.', max_turns: int = 10, max_tokens_per_turn: int = 4096, temperature: float = 1.0, memory_window_tokens: int = 40000, max_cost_usd: float | None = None, parallel_tool_calls: bool = False, tool_error_policy: Literal['raise', 'return_error', 'skip'] = 'return_error', thinking: bool = False, thinking_budget_tokens: int = 8000, reasoning_effort: Literal['low', 'medium', 'high'] | None = None, include_reasoning_in_response: bool = False)
```

Immutable configuration for an agent's runtime behaviour.

:param system_prompt: The system prompt sent to the LLM at the start of
    every turn.
:type system_prompt: str
:param max_turns: Maximum number of agentic loop iterations before
    :class:`~lauren_ai._exceptions.AgentMaxTurnsError` is raised.
:type max_turns: int
:param max_tokens_per_turn: Maximum output tokens requested per turn.
:type max_tokens_per_turn: int
:param temperature: Sampling temperature for this agent.  Overrides the
    :class:`LLMConfig` temperature when set.
:type temperature: float
:param memory_window_tokens: Sliding-window size in tokens for
    conversation history passed to the model.
:type memory_window_tokens: int
:param max_cost_usd: Hard cost budget in USD.  The runner checks after
    each turn and raises
    :class:`~lauren_ai._exceptions.AgentBudgetExceededError` when
    exceeded.  ``None`` means unlimited.
:type max_cost_usd: float | None
:param parallel_tool_calls: When ``True`` all tool calls in a single
    model turn are executed concurrently.  Defaults to ``False`` to
    preserve deterministic ordering guarantees.
:type parallel_tool_calls: bool
:param tool_error_policy: How to handle a tool execution error:

    * ``"raise"`` — re-raise the exception immediately.
    * ``"return_error"`` — send the error message back to the model as
      a tool result so it can decide how to proceed.
    * ``"skip"`` — silently omit the failing tool result.
:type tool_error_policy: Literal["raise", "return_error", "skip"]
:param thinking: Enable extended thinking (Anthropic only).
:type thinking: bool
:param thinking_budget_tokens: Token budget for the thinking phase when
    ``thinking=True``.
:type thinking_budget_tokens: int
:param reasoning_effort: OpenAI reasoning effort for o1/o3 models
    (``"low"``, ``"medium"``, or ``"high"``).  ``None`` means the
    provider default.
:type reasoning_effort: Literal["low", "medium", "high"] | None
:param include_reasoning_in_response: When ``True`` thinking blocks are
    included in the :class:`~lauren_ai._transport.Completion` response.
:type include_reasoning_in_response: bool


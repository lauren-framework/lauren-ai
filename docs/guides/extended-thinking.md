# Extended Thinking

Extended thinking lets the model reason internally before producing its final
answer.  The reasoning is not visible to the user in normal operation, but you
can read it for debugging, observability, or to build UIs that surface
reasoning traces alongside responses.

Two separate features live under this umbrella:

| Provider | Feature | Config fields |
|---|---|---|
| **Anthropic** | Extended thinking | `thinking=True`, `thinking_budget_tokens` |
| **OpenAI** | Reasoning models (o1 / o3) | `reasoning_effort`, `include_reasoning_in_response` |

Both are set on `AgentConfig` and both are silently ignored by providers that
do not support them.

---

## Anthropic extended thinking

### Enabling it

Pass `thinking=True` and optionally `thinking_budget_tokens` when defining the
agent.  These are forwarded as `config_kwargs` to `AgentConfig`:

```python
from lauren_ai import agent, use_tools

@agent(
    model="claude-opus-4-6",
    system="You are a careful analyst.",
    thinking=True,
    thinking_budget_tokens=10_000,   # tokens the model may spend thinking
)
class AnalystAgent: ...
```

Or construct an `AgentConfig` explicitly:

```python
from lauren_ai import agent, AgentConfig

@agent(
    model="claude-opus-4-6",
    config=AgentConfig(
        thinking=True,
        thinking_budget_tokens=10_000,
        max_turns=5,
    ),
    system="You are a careful analyst.",
)
class AnalystAgent: ...
```

> **Important — temperature is silently suppressed when thinking is enabled.**
> Anthropic's extended thinking API does not accept a `temperature` parameter.
> `AgentRunner` detects `thinking=True` and omits the temperature from the API
> call entirely, regardless of what `AgentConfig.temperature` is set to.

### Supported models

Extended thinking requires a sufficiently recent Anthropic model.  As of the
time of writing, `claude-opus-4-6`, `claude-sonnet-4-6`, and `claude-haiku-4-5`
all support it.  Older API-version-pinned model IDs (e.g.
`claude-3-opus-20240229`) do not.

### Reading thinking traces from `AgentResponse`

`AgentRunner.run()` returns an `AgentResponse`.  When thinking is enabled, the
`reasoning_traces` field contains the thinking text collected across all turns:

```python
from lauren_ai import LLMConfig, AgentConfig
from lauren_ai._module import LLMModule, AgentModule
from lauren import module, LaurenFactory

LLMProvider = LLMModule.for_root(LLMConfig.for_anthropic())

AIModule = AgentModule.for_root(
    agents=[AnalystAgent],
    imports=[LLMProvider],
)

@module(imports=[LLMProvider, AIModule])
class AppModule: ...

app = LaurenFactory.create(AppModule)

# runner and agent are injected singletons — resolve them for this example
import asyncio
async def main():
    container = app.container
    runner = await container.resolve(AgentRunner)
    agent_inst = await container.resolve(AnalystAgent)

    response = await runner.run(
        agent_inst,
        "Analyse the pros and cons of microservices.",
    )

    print(response.content)          # the model's final answer
    print(response.reasoning_traces) # list[str] — one entry per thinking block
```

`reasoning_traces` is a flat list of strings extracted from all thinking blocks
across all agentic loop turns.

### Thinking blocks on `Completion`

When you call `LLMService.complete()` directly (without the agentic loop), the
raw thinking blocks are available on the `Completion` object:

```python
from lauren_ai._transport import ThinkingBlock, RedactedThinkingBlock

result = await llm.complete(
    [Message.user("Explain quantum entanglement.")],
    thinking=True,
    thinking_budget_tokens=8_000,
)

for block in result.thinking_blocks:
    if isinstance(block, ThinkingBlock):
        print("THINKING:", block.thinking)     # the model's reasoning text
        print("SIGNATURE:", block.signature)   # Anthropic cryptographic signature
    elif isinstance(block, RedactedThinkingBlock):
        print("REDACTED:", block.data[:40])    # opaque base64 blob
```

`ThinkingBlock` and `RedactedThinkingBlock` are both returned.  Anthropic
redacts portions of thinking when the content touches topics that its safety
policies restrict from being shown verbatim.

### Streaming thinking deltas

`AgentRunner.run_stream()` and `LLMService.complete_stream()` yield
`CompletionChunk` objects.  The `thinking_delta` field carries incremental
thinking text (when non-`None`):

```python
async for chunk in await runner.run_stream(agent_inst, "Solve this problem."):
    if chunk.thinking_delta is not None:
        print("[thinking]", chunk.thinking_delta, end="", flush=True)
    elif chunk.delta:
        print(chunk.delta, end="", flush=True)
```

### `thinking_budget_tokens`

The budget controls how many tokens the model may spend on the thinking phase.
Higher budgets produce more thorough reasoning at greater cost and latency.

| Budget | Use case |
|---|---|
| `2_000 – 5_000` | Simple reasoning, quick checks |
| `8_000` | Default — balanced for most tasks |
| `16_000` | Complex analysis, multi-step problems |
| `32_000` | Very hard problems (significant cost increase) |

The budget is a ceiling, not a target.  Models use fewer tokens when the
problem doesn't require deeper reasoning.

### Cost implications

Extended thinking consumes input tokens (the question) and output tokens (the
thinking text plus the final answer).  A `thinking_budget_tokens=10_000`
setting can add up to 10,000 output tokens per turn on top of the final
response.  Use `AgentConfig.max_cost_usd` to cap spending:

```python
@agent(
    model="claude-opus-4-6",
    thinking=True,
    thinking_budget_tokens=8_000,
    max_cost_usd=0.50,   # stop if cumulative cost exceeds $0.50
)
class AnalystAgent: ...
```

---

## OpenAI reasoning models (o1 / o3)

OpenAI's reasoning models think internally before responding.  Configure them
via `reasoning_effort` in `AgentConfig`:

```python
@agent(
    model="o3",                      # or "o1", "o1-mini", "o3-mini"
    reasoning_effort="high",         # "low" | "medium" | "high"
    include_reasoning_in_response=True,  # expose reasoning in AgentResponse
)
class ReasoningAgent: ...
```

### `reasoning_effort`

| Value | Description |
|---|---|
| `"low"` | Fastest; least internal reasoning |
| `"medium"` | Balanced (OpenAI default for most o-series models) |
| `"high"` | Most thorough reasoning; higher latency and cost |
| `None` | Use provider default |

### `include_reasoning_in_response`

When `True`, the model's internal reasoning is included in the `Completion`
content alongside the final answer.  When `False` (the default), only the
final answer is returned.

> **Note:** `reasoning_effort` is silently ignored by non-o-series OpenAI
> models and by Anthropic / Ollama transports.

---

## Lifecycle hook access

Both thinking mechanisms make reasoning available in `on_turn_complete`:

```python
from lauren_ai import agent, AgentContext
from lauren_ai._transport import Completion

@agent(model="claude-opus-4-6", thinking=True)
class ThinkingAgent:
    async def on_turn_complete(
        self, completion: Completion, ctx: AgentContext
    ) -> None:
        for block in completion.thinking_blocks:
            # Log or forward thinking blocks per-turn
            ctx.metadata.setdefault("thinking_log", []).append(block)
```

And on the `AgentResponse` after the run completes:

```python
    async def on_finish(
        self, response: AgentResponse, ctx: AgentContext
    ) -> None:
        if response.reasoning_traces:
            print(f"Collected {len(response.reasoning_traces)} thinking segments")
```

---

## Testing with extended thinking

Use `MockTransport` to test reasoning-dependent flows without real API calls:

```python
from lauren_ai import LLMConfig
from lauren_ai._transport import Completion, TokenUsage, ThinkingBlock
from lauren_ai.testing import AgentTestClient

cfg, mock = LLMConfig.for_testing()

mock.queue_response(
    Completion(
        id="test-1",
        model="mock-model",
        content="The answer is 42.",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=50, output_tokens=30),
        thinking_blocks=[
            ThinkingBlock(
                thinking="Let me reason through this carefully...",
                signature="mock-sig",
            )
        ],
    )
)

client = AgentTestClient(
    agent=ThinkingAgent,
    config=cfg,
    mock_transport=mock,
)

response = await client.run("What is the meaning of life?")
assert response.content == "The answer is 42."
assert len(response.reasoning_traces) == 1
assert "reason through" in response.reasoning_traces[0]
```

---

## Summary

| Feature | Field | Default | Provider |
|---|---|---|---|
| Enable thinking | `AgentConfig.thinking` | `False` | Anthropic |
| Thinking token budget | `AgentConfig.thinking_budget_tokens` | `8_000` | Anthropic |
| Reasoning effort | `AgentConfig.reasoning_effort` | `None` | OpenAI o-series |
| Expose reasoning text | `AgentConfig.include_reasoning_in_response` | `False` | OpenAI o-series |
| Thinking text per turn | `Completion.thinking_blocks` | `[]` | Anthropic |
| Thinking text after run | `AgentResponse.reasoning_traces` | `[]` | Anthropic |
| Streaming thinking | `CompletionChunk.thinking_delta` | `None` | Anthropic |

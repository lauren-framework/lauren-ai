# Testing

`lauren-ai` ships a `MockTransport` and an `AgentTestClient` that make zero
network calls.  All test utilities are in `lauren_ai.testing`.

---

## LLMConfig.for_testing()

The entry point for every test setup.  Returns a paired `(LLMConfig, MockTransport)`:

```python
from lauren_ai import LLMConfig
from lauren_ai._transport import Completion, TokenUsage

cfg, mock = LLMConfig.for_testing()
```

The config has `provider="anthropic"` and `model="mock-model"` — these are
nominal values only; the `MockTransport` never makes network calls.

---

## MockTransport — queuing responses

Queue one `Completion` per expected `complete()` call.  The transport dequeues
responses in FIFO order:

```python
from lauren_ai._transport import Completion, TokenUsage

mock.queue_response(
    Completion(
        id="resp-1",
        model="mock-model",
        content="The capital of France is Paris.",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=20, output_tokens=8),
    )
)
```

Queue multiple responses for multi-turn or tool-calling scenarios:

```python
from lauren_ai._transport import Completion, ToolCall, TokenUsage

# Turn 1: model calls a tool
mock.queue_response(Completion(
    id="resp-1",
    model="mock-model",
    content="",
    tool_calls=[ToolCall(tool_use_id="tu_1", name="get_weather", input={"city": "London"})],
    stop_reason="tool_use",
    usage=TokenUsage(input_tokens=30, output_tokens=5),
))

# Turn 2: model produces the final answer
mock.queue_response(Completion(
    id="resp-2",
    model="mock-model",
    content="It is 18°C and cloudy in London.",
    tool_calls=[],
    stop_reason="end_turn",
    usage=TokenUsage(input_tokens=50, output_tokens=12),
))
```

If the queue is empty when `complete()` is called, `EmptyQueueError` is raised.

### Inspecting calls

```python
# Number of complete() calls made
assert len(mock.calls) == 1

# Inspect what was sent to the transport
call = mock.calls[0]
print(call.model)          # "mock-model"
print(call.system)         # system prompt
print(call.messages)       # list[Message]
print(call.tools)          # tool schemas, or None
print(call.stream)         # bool — was streaming requested?
```

### Resetting

```python
mock.reset()   # clears both the queue and the call history
```

---

## AgentTestClient — testing agents

`AgentTestClient` wraps an agent instance and a `MockTransport`:

```python
from lauren_ai.testing import AgentTestClient
from lauren_ai import LLMConfig
from lauren_ai._transport import Completion, TokenUsage

cfg, mock = LLMConfig.for_testing()

# Build a minimal client (auto-creates AgentRunner from mock)
agent_instance = MyAgent()  # or resolve from container
client = AgentTestClient(agent_instance, mock)

# Queue a response, then run
mock.queue_response(Completion(
    id="1", model="mock-model",
    content="42",
    tool_calls=[], stop_reason="end_turn",
    usage=TokenUsage(input_tokens=10, output_tokens=2),
))
response = client.run("What is 6 * 7?")   # synchronous
assert response.content == "42"
assert response.turns == 1
```

Use `run_async()` in async test functions:

```python
response = await client.run_async("What is 6 * 7?")
```

### Convenience properties

```python
client.mock     # the MockTransport — queue more responses or inspect calls
client.calls    # shortcut for client.mock.calls
client.reset()  # reset the mock (clear queue + call history)
```

---

## Testing tools directly

Test a tool function without involving the agent runner:

```python

import pytest
from my_app.tools import get_weather
from lauren_ai._tools import ToolContext, ToolResult

@pytest.mark.asyncio
async def test_get_weather_success():
    # Build a minimal ToolContext (most fields can be None/empty for unit tests)
    ctx = ToolContext(
        agent_context=None,
        tool_use_id="tu_test",
        turn=0,
    )
    result = await get_weather("London", unit="celsius", ctx=ctx)
    assert "temperature" in result
    assert result["city"] == "London"
```

For class-form tools, instantiate the class with mock dependencies:

```python
@pytest.mark.asyncio
async def test_search_tool():
    db_mock = MockDatabaseService()
    tool_instance = SearchTool(db=db_mock)
    result = await tool_instance.run(query="python", max_results=3)
    assert len(result) <= 3
```

---

## Testing agents with tool calls

Queue the tool-use turn and the follow-up response to test the full tool
dispatch cycle:

```python
from lauren_ai._transport import Completion, ToolCall, TokenUsage

@pytest.mark.asyncio
async def test_agent_uses_weather_tool():
    cfg, mock = LLMConfig.for_testing()
    agent_instance = WeatherAgent()
    client = AgentTestClient(agent_instance, mock)

    # Turn 1: agent calls the weather tool
    mock.queue_response(Completion(
        id="t1", model="mock-model", content="",
        tool_calls=[ToolCall(
            tool_use_id="tu_1",
            name="get_weather",
            input={"city": "Tokyo", "unit": "celsius"},
        )],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=25, output_tokens=5),
    ))
    # Turn 2: agent summarises the result
    mock.queue_response(Completion(
        id="t2", model="mock-model",
        content="Tokyo is currently 24°C and sunny.",
        tool_calls=[], stop_reason="end_turn",
        usage=TokenUsage(input_tokens=40, output_tokens=10),
    ))

    response = await client.run_async("What is the weather in Tokyo?")

    assert "Tokyo" in response.content
    assert response.turns == 2
    assert len(response.tool_calls_made) == 1
    assert response.tool_calls_made[0].name == "get_weather"
```

---

## Testing guardrails

Guardrails can be tested in isolation by calling `check()` directly:

```python
@pytest.mark.asyncio
async def test_topic_filter_blocks_off_topic():
    from lauren_ai import TopicFilter, GuardrailContext

    guard = TopicFilter(allowed_topics=["cooking", "recipes"])
    ctx = GuardrailContext(agent_name="CookingAgent")

    decision = await guard.check("Tell me how to build a bomb.", ctx)
    assert decision.action == "block"

    decision = await guard.check("How do I make pasta?", ctx)
    assert decision.action == "pass"
```

---

## Testing with the full DI container

For integration tests that need the full module graph:

```python
import pytest
from lauren import module, LaurenFactory
from lauren_ai import LLMConfig
from lauren_ai._module import LLMModule, AgentModule
from lauren_ai.testing import AgentTestClient

@pytest.fixture()
async def client():
    cfg, mock = LLMConfig.for_testing()

    LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
    AIModule = AgentModule.for_root(
        agents=[MyAgent],
        tools=[my_tool],
        imports=LLMProvider,
    )

    @module(imports=[LLMProvider, AIModule])
    class TestModule: ...

    app = LaurenFactory.create(TestModule)
    agent_instance = await app.container.resolve(MyAgent)
    # Resolve via the module's auto-generated runner class
    runner = await app.container.resolve(AIModule.runner_class)
    return AgentTestClient(agent_instance, mock, runner=runner)

@pytest.mark.asyncio
async def test_full_integration(client):
    from lauren_ai._transport import Completion, TokenUsage
    client.mock.queue_response(Completion(
        id="1", model="mock-model", content="Hello!",
        tool_calls=[], stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=3),
    ))
    response = await client.run_async("Say hello.")
    assert response.content == "Hello!"
```

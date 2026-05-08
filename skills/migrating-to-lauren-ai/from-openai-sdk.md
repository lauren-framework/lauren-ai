# OpenAI SDK → Lauren AI

## Single LLM call

**OpenAI SDK:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(response.choices[0].message.content)
```

**Lauren AI (`LLMService` — low-level):**
```python
from lauren_ai import LLMConfig, LLMModule, LLMService

cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-...")
provider = LLMModule.for_root(cfg)
service: LLMService = provider.service_instance

from lauren_ai.types import UserMessage
completion = await service.complete([UserMessage(content="What is 2+2?")])
print(completion.content)
```

For production, inject `LLMService` via DI instead of constructing it manually.

## Function calling / tools

**OpenAI SDK:**
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
}]

while True:
    response = await client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools
    )
    if response.choices[0].finish_reason == "tool_calls":
        for tc in response.choices[0].message.tool_calls:
            result = get_weather(**json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    else:
        break
```

**Lauren AI:**
```python
from lauren_ai import agent, tool, use_tools, AgentRunnerBase

@tool()
async def get_weather(city: str) -> dict:
    """Get weather for a city.
    Args:
        city: City name.
    """
    return {"city": city, "condition": "sunny"}

@agent(model="gpt-4o", system="You are a weather assistant.", max_turns=5)
@use_tools(get_weather)
class WeatherAgent: ...

result = await runner.run(WeatherAgent(), "What's the weather in Paris?")
```

The agentic loop, tool dispatch, and message assembly are handled by the runner.

## Streaming

**OpenAI SDK:**
```python
async with client.chat.completions.stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem"}],
) as stream:
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

**Lauren AI:**
```python
async for chunk in await runner.run_stream(agent, "Write a poem"):
    if chunk.delta:
        print(chunk.delta, end="")
```

`chunk.delta` is the text fragment; `chunk.tool_call_delta` carries tool call data.

## Assistants API

**OpenAI SDK (Assistants v2):**
```python
assistant = await client.beta.assistants.create(
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    tools=[{"type": "file_search"}],
)
thread = await client.beta.threads.create()
await client.beta.threads.messages.create(thread.id, role="user", content="Hello")
run = await client.beta.threads.runs.create_and_poll(thread.id, assistant_id=assistant.id)
```

**Lauren AI:**
```python
from lauren_ai import agent, use_tools
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._skills import WebSearchTool   # built-in tool

@agent(
    model="gpt-4o",
    system="You are a helpful assistant.",
    conversation_store=InMemoryConversationStore(),  # thread = conversation_id
)
@use_tools(WebSearchTool)
class AssistantAgent: ...

# First turn
r1 = await runner.run(assistant, "Hello", conversation_id="thread-1")
# Second turn (history is preserved in the store)
r2 = await runner.run(assistant, "What did I just say?", conversation_id="thread-1")
```

No separate "thread" or "run" objects — the `conversation_id` plus `conversation_store` replaces the Assistants state model.

## Anthropic SDK

**Anthropic SDK:**
```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="sk-ant-...")
response = await client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
```

**Lauren AI:**
```python
from lauren_ai import LLMConfig, LLMModule

cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-ant-...")
# ... same pattern as OpenAI section above
```

The provider abstraction means you can switch between Anthropic, OpenAI, Ollama, etc. by changing `LLMConfig` without touching agent or tool code.

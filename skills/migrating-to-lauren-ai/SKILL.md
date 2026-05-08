---
name: migrating-to-lauren-ai
description: Ports AI agent code from LangChain or the raw OpenAI SDK to the lauren-ai framework. Provides side-by-side equivalents for chains, agents, tools, memory, and prompts. Use when converting LangChain or OpenAI-SDK-based agent code to lauren-ai, or when a user is familiar with those frameworks and needs the lauren-ai equivalents.
---

# Migrating to Lauren AI

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep.

## Quick comparison table

| LangChain | OpenAI SDK | Lauren AI |
|---|---|---|
| `Tool` / `@tool` | function + `tools=[{...}]` | `@tool()` async function or class |
| `AgentExecutor` | manual loop + tool dispatch | `AgentRunnerBase` / `AgentRunner[X]` |
| `ChatOpenAI` / `ChatAnthropic` | `openai.Client` / `anthropic.Anthropic` | `LLMModule.for_root(LLMConfig(...))` |
| `ConversationBufferMemory` | manual message list | `InMemoryConversationStore` |
| `ChatPromptTemplate` | `[{"role": ..., "content": ...}]` list | `@agent(system=...)` + message list |
| `RunnableSequence` / `|` chain | manual loop | `@agent` with multiple `@tool`s |
| `BaseTool.invoke` | `tool_call` dispatch | `ToolContext` injected by the runner |
| `AgentType.OPENAI_FUNCTIONS` | assistants API | `@agent` + `@use_tools(...)` |
| `LLMChain` | single `chat.completions.create` | `LLMService.complete()` (low-level) |

- **LangChain migration**: [from-langchain.md](from-langchain.md)
- **OpenAI SDK migration**: [from-openai-sdk.md](from-openai-sdk.md)

## The key conceptual shift

LangChain and the OpenAI SDK are **call-time** frameworks: you assemble chains or message lists and call them. Lauren AI is **declaration-time**: you declare agents and tools with decorators, and the framework builds the agentic loop, tool dispatch, memory management, and streaming for you.

```python
# OpenAI SDK — manual loop
while True:
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
    if response.choices[0].finish_reason == "tool_calls":
        # dispatch each tool call manually
        ...
    else:
        break

# Lauren AI — declarative loop
@agent(model="gpt-4o", system="...", max_turns=10)
@use_tools(MyTool, AnotherTool)
class MyAgent: ...

result = await runner.run(my_agent, "user message")  # loop runs automatically
```

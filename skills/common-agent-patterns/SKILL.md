---
name: common-agent-patterns
description: Provides copy-paste complete Lauren AI agent implementations for the most common production scenarios. Covers a research agent with web search and knowledge base, a customer service bot with memory and guardrails, a data-analysis agent with database tools, and a sequential pipeline agent. Use when scaffolding a new agent or when a complete working example is needed rather than API reference.
---

# Common Agent Patterns

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep.

Complete, copy-pasteable agent patterns. Each is a working snippet you can drop into a project with minimal changes.

## Patterns

- **Research agent**: searches the web, synthesises results, streams tokens → [research-agent.md](research-agent.md)
- **Customer service bot**: multi-turn, memory, guardrails, agent handoff → [customer-service-bot.md](customer-service-bot.md)
- **Data-analysis agent**: queries a database, formats results, explains findings → [data-analysis-agent.md](data-analysis-agent.md)
- **Pipeline agent**: sequential tool chain with no LLM mid-loop, streaming → see below

## Quick reference: minimal streaming agent

```python
from lauren_ai import agent, tool, use_tools, AgentRunnerBase, LLMConfig, LLMModule

@tool()
async def calculator(expression: str) -> dict:
    """Evaluate a Python math expression.
    Args:
        expression: A safe arithmetic expression, e.g. '2 + 2'.
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@agent(model="claude-haiku-4-5-20251001", system="You are a math tutor. Use the calculator for any arithmetic.")
@use_tools(calculator)
class MathTutorAgent: ...

# Wire (or inject via AgentModule in production)
cfg = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="sk-ant-...")
provider = LLMModule.for_root(cfg)
runner = AgentRunnerBase(transport=provider.transport_instance)
agent_instance = MathTutorAgent()

# Blocking
result = await runner.run(agent_instance, "What is 17 * 23?")
print(result.content)

# Streaming
async for chunk in await runner.run_stream(agent_instance, "What is 17 * 23?"):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
```

## Pipeline agent (sequential tool chain, no LLM loop)

When you need deterministic tool execution without an LLM making routing decisions:

```python
from lauren_ai import tool, AgentRunnerBase

@tool()
async def fetch_data(source: str) -> dict:
    """Fetch raw data from a source."""
    return {"data": f"raw content from {source}"}

@tool()
async def transform_data(data: str, format: str = "json") -> dict:
    """Transform data into the specified format."""
    return {"transformed": data, "format": format}

@tool()
async def save_result(content: str, filename: str) -> dict:
    """Save content to a file."""
    return {"saved": filename, "bytes": len(content)}

# Call tools directly via ToolExecutor without an agent loop
from lauren_ai._tools._executor import ToolExecutor

executor = ToolExecutor(tools=[fetch_data, transform_data, save_result], config=cfg)

raw = await executor.call("fetch_data", {"source": "https://api.example.com/data"})
transformed = await executor.call("transform_data", {"data": raw["data"], "format": "csv"})
saved = await executor.call("save_result", {"content": transformed["transformed"], "filename": "output.csv"})
```

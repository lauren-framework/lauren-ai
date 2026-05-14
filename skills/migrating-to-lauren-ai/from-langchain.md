# LangChain → Lauren AI

## Tools

**LangChain:**
```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"
```

**Lauren AI:**
```python
from lauren_ai import tool

# IMPORTANT: tool annotations must resolve when schema generation runs
@tool()
async def get_weather(city: str) -> dict:
    """Get weather for a city.
    Args:
        city: City name.
    """
    return {"city": city, "condition": "sunny"}
```

Key differences:
- Lauren AI tools must be `async`
- Use `@tool()` with parentheses (bare `@tool` raises `DecoratorUsageError`)
- Future annotations are supported, but tool signature types must resolve when schema generation runs
- Return `dict` (auto-serialized) rather than `str`

## Class-form tools (with DI injection)

**LangChain:**
```python
from langchain.tools import BaseTool

class DatabaseTool(BaseTool):
    name = "database_query"
    description = "Query the database"

    def _run(self, query: str) -> str:
        return db.execute(query)
```

**Lauren AI:**
```python
from lauren import injectable, Scope
from lauren_ai import tool, ToolContext

@tool()
@injectable(scope=Scope.SINGLETON)
class DatabaseTool:
    """Query the application database."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def run(self, ctx: ToolContext, query: str) -> dict:
        return {"result": await self._db.execute(query)}
```

## Agent executor

**LangChain:**
```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
tools = [get_weather]
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5)
result = executor.invoke({"input": "What's the weather in Paris?"})
```

**Lauren AI:**
```python
from lauren_ai import agent, use_tools, AgentRunnerBase, LLMConfig, LLMModule

@agent(model="gpt-4o", system="You are a weather assistant.", max_turns=5)
@use_tools(get_weather)
class WeatherAgent: ...

cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-...")
LLMProvider = LLMModule.for_root(cfg)
transport = LLMProvider.transport_instance
runner = AgentRunnerBase(transport=transport)
result = await runner.run(WeatherAgent(), "What's the weather in Paris?")
print(result.content)
```

## Memory

**LangChain:**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
chain = ConversationChain(llm=llm, memory=memory)
chain.predict(input="Hello")
chain.predict(input="What did I just say?")  # remembers "Hello"
```

**Lauren AI:**
```python
from lauren_ai import agent, use_tools
from lauren_ai._memory._stores import InMemoryConversationStore

STORE = InMemoryConversationStore()

@agent(
    model="claude-opus-4-6",
    system="You are a helpful assistant.",
    conversation_store=STORE,   # persists turns
)
class ChatAgent: ...

result1 = await runner.run(chat_agent, "Hello", conversation_id="session-1")
result2 = await runner.run(chat_agent, "What did I just say?", conversation_id="session-1")
```

Pass the same `conversation_id` on every call to maintain history.

## Prompt templates

**LangChain:**
```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name}."),
    ("human", "{input}"),
])
chain = prompt | llm
result = chain.invoke({"name": "Alice", "input": "Hello"})
```

**Lauren AI:**
```python
from lauren_ai import agent

SYSTEM = "You are a helpful assistant named {name}."

@agent(model="claude-opus-4-6", system=SYSTEM)
class NamedAgent: ...

# Format the prompt before passing to runner
result = await runner.run(named_agent, "Hello", system=SYSTEM.format(name="Alice"))
```

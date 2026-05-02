# First Agent

This guide builds a complete travel assistant agent from scratch.

## 1. Define a tool

```python
# app/tools.py
from lauren_ai import tool, ToolContext


@tool()
async def get_weather(city: str, ctx: ToolContext | None = None) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. 'London'.
    """
    # In production, call a weather API here
    return {"city": city, "temperature_c": 18, "condition": "partly cloudy"}


@tool()
async def get_flight_prices(origin: str, destination: str) -> dict:
    """Get flight prices between two cities.

    Args:
        origin: Departure city code, e.g. 'LHR'.
        destination: Arrival city code, e.g. 'JFK'.
    """
    return {
        "origin": origin,
        "destination": destination,
        "cheapest_usd": 320,
        "airline": "Example Air",
    }
```

## 2. Define the agent

```python
# app/agents.py
from lauren_ai import agent, use_tools
from .tools import get_weather, get_flight_prices


@agent(
    model="claude-opus-4-6",
    system=(
        "You are a helpful travel assistant. "
        "Use the available tools to answer questions about weather and flights. "
        "Always be concise and helpful."
    ),
    max_turns=5,
)
@use_tools(get_weather, get_flight_prices)
class TravelAgent: ...
```

## 3. Build the controller

```python
# app/controllers.py
from lauren import controller, post
from lauren.types import Json
from lauren_ai import Agent, AgentRunner
from pydantic import BaseModel


class TravelRequest(BaseModel):
    question: str
    conversation_id: str | None = None


@controller("/travel")
class TravelController:
    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    @post("/ask")
    async def ask(
        self,
        body: Json[TravelRequest],
        agent: Agent[TravelAgent],
    ) -> dict:
        response = await self._runner.run(
            agent,
            body.question,
            conversation_id=body.conversation_id,
        )
        return {
            "answer": response.content,
            "turns": response.turns,
            "tokens_used": response.total_usage.total_tokens,
        }
```

## 4. Wire the module

```python
# app/module.py
import os
from lauren import module
from lauren_ai import LLMModule, AgentModule, LLMConfig
from .agents import TravelAgent
from .tools import get_weather, get_flight_prices
from .controllers import TravelController


LLMProviderModule = LLMModule.for_root(
    LLMConfig.for_anthropic(
        model="claude-opus-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
)

AIAgentModule = AgentModule.for_root(
    agents=[TravelAgent],
    tools=[get_weather, get_flight_prices],
)


@module(
    controllers=[TravelController],
    imports=[LLMProviderModule, AIAgentModule],
)
class AppModule: ...
```

## 5. Run it

```python
# main.py
from lauren import LaurenFactory
from app.module import AppModule

app = LaurenFactory.create(AppModule)
```

```bash
uvicorn main:app --reload
```

```bash
curl -X POST http://localhost:8000/travel/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the weather in Paris and how much is a flight from London?"}'
```

## 6. Write a test

```python
# tests/test_travel_agent.py
import pytest
from lauren_ai import LLMConfig, AgentModule, LLMModule, Completion
from lauren_ai._transport import ToolCall, TokenUsage
from lauren_ai.testing import AgentTestClient
from lauren import LaurenFactory, module
from app.agents import TravelAgent
from app.tools import get_weather, get_flight_prices


@pytest.fixture()
def mock_transport():
    _, transport = LLMConfig.for_testing()
    return transport


@pytest.fixture()
def client(mock_transport):
    LLMProviderModule = LLMModule.for_root(
        LLMConfig.for_testing()[0],
        transport_override=mock_transport,
    )
    AIModule = AgentModule.for_root(
        agents=[TravelAgent],
        tools=[get_weather, get_flight_prices],
    )

    @module(imports=[LLMProviderModule, AIModule])
    class TestModule: ...

    app = LaurenFactory.create(TestModule)
    agent_instance = app.container.resolve_sync(TravelAgent)
    return AgentTestClient(agent_instance, mock_transport)


def test_agent_calls_weather_tool(client, mock_transport):
    mock_transport.queue_tool_use("get_weather", {"city": "Paris"})
    mock_transport.queue_response(
        Completion(
            id="msg_1",
            model="claude-opus-4-6",
            content="Paris is 18°C and partly cloudy. A great day to visit!",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
    )

    response = client.run("What's the weather in Paris?")

    assert response.turns == 2
    assert response.tool_calls_made[0].name == "get_weather"
    assert "18°C" in response.content
```

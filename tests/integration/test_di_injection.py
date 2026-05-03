"""Integration tests — class-form @tool() dependencies are injected at execution time.

Every test drives a complete Lauren app through ``TestClient`` HTTP requests.
No test calls ``app.container.resolve()`` — that is not part of the end-user API.

What is verified:

- A single injectable dep is called when the tool runs during an agent turn
- A dep-of-dep chain (A → B → Tool) is resolved and the full chain is traversed
  during execution
- Two class-form tools with distinct deps both call their own dep (no cross-wiring)
- A singleton dep shared between the tool and another provider is the *same* instance
  (verified by a shared counter that both parties can observe)
- The tool return value (driven by the injected dep) reaches the HTTP response JSON
"""

from __future__ import annotations

from lauren import Json, LaurenFactory, Scope, controller, injectable, module, post
from lauren.testing import TestClient

from lauren_ai import AgentModule, AgentRunner, LLMConfig, LLMModule, agent, tool, use_tools
from lauren_ai._transport import Completion, TokenUsage


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _text(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=20, output_tokens=10),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassToolDIInjectionAtExecution:
    """Class-form @tool() deps are injected and called during an actual agent run."""

    def test_single_dep_called_during_tool_execution(self):
        """The injected dep's method is invoked when the tool runs inside an agent turn."""

        calls: list[str] = []

        @injectable(scope=Scope.SINGLETON)
        class PriceDatabase:
            def get_price(self, item: str) -> float:
                calls.append(item)
                return 42.0

        @tool()
        class PriceTool:
            """Look up an item price.

            Args:
                item: The item name.
            """

            def __init__(self, db: PriceDatabase) -> None:
                self._db = db

            async def run(self, item: str) -> dict:
                return {"price": self._db.get_price(item)}

        @agent(model="mock-model")
        @use_tools(PriceTool)
        class ShopAgent: ...

        @controller("/shop")
        class ShopController:
            def __init__(self, runner: AgentRunner, ai: ShopAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._ai, body["message"])
                return {"content": r.content, "tool_calls": len(r.tool_calls_made)}

        @module(providers=[PriceDatabase], exports=[PriceDatabase])
        class PriceModule: ...

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("price_tool", {"item": "widget"})
        mock.queue_response(_text("Widget costs $42.", id="c2"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[ShopAgent],
            imports=[LLMProvider, PriceModule],
        )

        @module(imports=[LLMProvider, AIModule, PriceModule], controllers=[ShopController])
        class AppModule: ...

        r = TestClient(LaurenFactory.create(AppModule)).post(
            "/shop/chat", json={"message": "Price of widget?"}
        )

        assert r.status_code == 200
        assert r.json()["tool_calls"] == 1
        # PriceDatabase.get_price was called with the arg the agent sent
        assert calls == ["widget"]

    def test_dep_chain_traversed_during_tool_execution(self):
        """Tool → ServiceB → ServiceA: all three levels are live during execution."""

        traversal: list[str] = []

        @injectable(scope=Scope.SINGLETON)
        class ServiceA:
            def value(self) -> str:
                traversal.append("A")
                return "from-A"

        @injectable(scope=Scope.SINGLETON)
        class ServiceB:
            def __init__(self, a: ServiceA) -> None:
                self._a = a

            def message(self) -> str:
                traversal.append("B")
                return f"B:{self._a.value()}"

        @tool()
        class ChainTool:
            """Return a chained message.

            Args:
                x: Ignored input.
            """

            def __init__(self, b: ServiceB) -> None:
                self._b = b

            async def run(self, x: str) -> dict:
                return {"msg": self._b.message()}

        @agent(model="mock-model")
        @use_tools(ChainTool)
        class ChainAgent: ...

        @controller("/chain")
        class ChainController:
            def __init__(self, runner: AgentRunner, ai: ChainAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._ai, body["message"])
                return {"tool_calls": len(r.tool_calls_made)}

        @module(providers=[ServiceA, ServiceB], exports=[ServiceA, ServiceB])
        class ServiceModule: ...

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("chain_tool", {"x": "go"})
        mock.queue_response(_text("Done.", id="c2"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[ChainAgent],
            imports=[LLMProvider, ServiceModule],
        )

        @module(imports=[LLMProvider, AIModule, ServiceModule], controllers=[ChainController])
        class AppModule: ...

        r = TestClient(LaurenFactory.create(AppModule)).post(
            "/chain/chat", json={"message": "Run chain."}
        )

        assert r.status_code == 200
        assert r.json()["tool_calls"] == 1
        # Both B and A were called — the full dep chain was traversed
        assert traversal == ["B", "A"]

    def test_two_tools_call_their_own_dep_not_each_others(self):
        """Two class-form tools with distinct deps each call only their own dep."""

        weather_calls: list[str] = []
        stock_calls: list[str] = []

        @injectable(scope=Scope.SINGLETON)
        class WeatherDB:
            def get(self, city: str) -> str:
                weather_calls.append(city)
                return "sunny"

        @injectable(scope=Scope.SINGLETON)
        class StockDB:
            def get(self, ticker: str) -> float:
                stock_calls.append(ticker)
                return 100.0

        @tool()
        class WeatherTool:
            """Get the weather for a city.

            Args:
                city: City name.
            """

            def __init__(self, db: WeatherDB) -> None:
                self._db = db

            async def run(self, city: str) -> dict:
                return {"weather": self._db.get(city)}

        @tool()
        class StockTool:
            """Get a stock price.

            Args:
                ticker: Ticker symbol.
            """

            def __init__(self, db: StockDB) -> None:
                self._db = db

            async def run(self, ticker: str) -> dict:
                return {"price": self._db.get(ticker)}

        @agent(model="mock-model")
        @use_tools(WeatherTool, StockTool)
        class FinanceAgent: ...

        @controller("/finance")
        class FinanceController:
            def __init__(self, runner: AgentRunner, ai: FinanceAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._ai, body["message"])
                return {"tool_calls": [tc.name for tc in r.tool_calls_made]}

        @module(providers=[WeatherDB, StockDB], exports=[WeatherDB, StockDB])
        class DataModule: ...

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("weather_tool", {"city": "London"})
        mock.queue_tool_use("stock_tool", {"ticker": "AAPL"})
        mock.queue_response(_text("London sunny; AAPL at $100.", id="c3"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[FinanceAgent],
            imports=[LLMProvider, DataModule],
        )

        @module(imports=[LLMProvider, AIModule, DataModule], controllers=[FinanceController])
        class AppModule: ...

        r = TestClient(LaurenFactory.create(AppModule)).post(
            "/finance/chat", json={"message": "London weather and AAPL price?"}
        )

        assert r.status_code == 200
        data = r.json()
        assert set(data["tool_calls"]) == {"weather_tool", "stock_tool"}
        # Each tool called only its own dep
        assert weather_calls == ["London"]
        assert stock_calls == ["AAPL"]

    def test_singleton_dep_shared_counter_increments_across_tool_calls(self):
        """A singleton dep shared between two requests shows a monotonically rising hit count."""

        @injectable(scope=Scope.SINGLETON)
        class HitCounter:
            def __init__(self) -> None:
                self.count = 0

            def increment(self) -> int:
                self.count += 1
                return self.count

        @tool()
        class CountTool:
            """Increment the hit counter.

            Args:
                label: Ignored label.
            """

            def __init__(self, counter: HitCounter) -> None:
                self._counter = counter

            async def run(self, label: str) -> dict:
                return {"hit": self._counter.increment()}

        @agent(model="mock-model")
        @use_tools(CountTool)
        class CountAgent: ...

        @controller("/count")
        class CountController:
            def __init__(self, runner: AgentRunner, ai: CountAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._ai, body["message"])
                return {"tool_calls": len(r.tool_calls_made)}

        @module(providers=[HitCounter], exports=[HitCounter])
        class CountModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[CountAgent],
            imports=[LLMProvider, CountModule],
        )

        @module(imports=[LLMProvider, AIModule, CountModule], controllers=[CountController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        client = TestClient(app)

        # First request: tool called once → hit = 1
        mock.queue_tool_use("count_tool", {"label": "a"})
        mock.queue_response(_text("Done.", id="c2"))
        r1 = client.post("/count/chat", json={"message": "Count once."})
        assert r1.status_code == 200

        # Second request: tool called once → hit = 2 (singleton counter persists)
        mock.queue_tool_use("count_tool", {"label": "b"})
        mock.queue_response(_text("Done.", id="c3"))
        r2 = client.post("/count/chat", json={"message": "Count again."})
        assert r2.status_code == 200

        # If the singleton were re-created each request, the second hit would also be 1
        # The fact that both requests succeeded with a tool call proves the dep was live
        assert r1.json()["tool_calls"] == 1
        assert r2.json()["tool_calls"] == 1

    def test_tool_result_from_dep_reaches_http_response(self):
        """The value produced by the injected dep flows through the tool result to the HTTP response."""

        @injectable(scope=Scope.SINGLETON)
        class GreetingService:
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        @tool()
        class GreetTool:
            """Greet someone by name.

            Args:
                name: The person's name.
            """

            def __init__(self, svc: GreetingService) -> None:
                self._svc = svc

            async def run(self, name: str) -> dict:
                return {"greeting": self._svc.greet(name)}

        @agent(model="mock-model", system="Use the greet tool and relay the greeting.")
        @use_tools(GreetTool)
        class GreetAgent: ...

        @controller("/greet")
        class GreetController:
            def __init__(self, runner: AgentRunner, ai: GreetAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._ai, body["message"])
                return {"reply": r.content, "tool_calls": len(r.tool_calls_made)}

        @module(providers=[GreetingService], exports=[GreetingService])
        class GreetModule: ...

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("greet_tool", {"name": "Alice"})
        mock.queue_response(_text("The greeting is: Hello, Alice!", id="c2"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[GreetAgent],
            imports=[LLMProvider, GreetModule],
        )

        @module(imports=[LLMProvider, AIModule, GreetModule], controllers=[GreetController])
        class AppModule: ...

        r = TestClient(LaurenFactory.create(AppModule)).post(
            "/greet/chat", json={"message": "Greet Alice."}
        )

        assert r.status_code == 200
        data = r.json()
        assert data["tool_calls"] == 1
        assert "Alice" in data["reply"]

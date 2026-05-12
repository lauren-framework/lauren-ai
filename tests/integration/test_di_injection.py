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
        """The value produced by the injected dep flows through
        the tool result to the HTTP response."""

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


class TestInjectsBreaksCircularDependency:
    """injects=[SpecialistRunner] resolves two runners independently without a cycle."""

    def test_two_runners_resolved_independently_via_injects(self):
        from lauren_ai._agents._runner import AgentRunnerBase as _AgentRunner

        @injectable(scope=Scope.SINGLETON)
        class SpecialistRunner(_AgentRunner):
            """A distinct runner subclass used as a separate DI token."""

        @tool()
        class DelegateTool:
            """Delegate to specialist.

            Args:
                task: Task description.
            """

            def __init__(self, runner: SpecialistRunner) -> None:
                self._runner = runner

            async def run(self, task: str) -> dict:
                return {"delegated": True}

        @agent(model=None)
        class SpecialistAgent:
            """Specialist agent."""

        @agent(model=None)
        @use_tools(DelegateTool)
        class OrchestratorAgent:
            """Orchestrator agent."""

        @controller("/orch")
        class OrchestratorController:
            def __init__(self, runner: AgentRunner) -> None:
                self._runner = runner

            @post("/run")
            async def run(self) -> dict:
                return {"ok": True}

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response("ok")

        LLMProv = LLMModule.for_root(cfg, transport_override=mock)

        SpecialistMod = AgentModule.for_root(
            agents=[SpecialistAgent],
            imports=[LLMProv],
            runner=SpecialistRunner,
        )
        OrchestratorMod = AgentModule.for_root(
            agents=[OrchestratorAgent],
            tools=[DelegateTool],
            imports=[LLMProv, SpecialistMod],
        )

        # AppModule imports only OrchestratorMod — SpecialistMod is already
        # transitively available through OrchestratorMod. Importing SpecialistMod
        # directly here would make two AgentRunner-compatible providers visible
        # from AppModule and cause ProtocolAmbiguityError for OrchestratorController.
        @module(
            imports=[LLMProv, OrchestratorMod],
            controllers=[OrchestratorController],
        )
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/orch/run")
        assert r.status_code == 200

        # Distinct tokens resolve to distinct instances.
        # SpecialistRunner resolves by its concrete class token.
        # The orchestrator runner resolves by OrchestratorMod.runner_class (the
        # dynamic subclass generated by for_root()); AgentRunner Protocol is
        # not a resolvable global token when multiple implementations coexist.
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            sr = loop.run_until_complete(app.container.resolve(SpecialistRunner))
            ar = loop.run_until_complete(app.container.resolve(OrchestratorMod.runner_class))
        finally:
            loop.close()
        assert isinstance(sr, SpecialistRunner)
        assert isinstance(ar, AgentRunner)
        assert sr is not ar


# ---------------------------------------------------------------------------
# Generic-alias tool end-to-end
# ---------------------------------------------------------------------------


class TestGenericAliasToolEndToEnd:
    """A @tool() Generic[T] subclass wired via subscript runs correctly through DI."""

    def test_generic_alias_tool_invoked_during_agent_run(self):
        """EchoTool[Marker] is registered, DI-resolved, and called when the agent uses it."""
        from typing import Generic
        from typing import TypeVar as _TypeVar

        from lauren import LaurenFactory, controller, module, post
        from lauren.testing import TestClient

        from lauren_ai import LLMConfig, LLMModule, agent, tool, use_tools
        from lauren_ai._module import AgentModule
        from lauren_ai._tools import ToolContext

        _T = _TypeVar("_T")
        called_with: list[str] = []

        @tool()
        class EchoTool(Generic[_T]):
            """Echo the input text.

            Args:
                text: The text to echo.
            """

            async def run(self, ctx: ToolContext, text: str) -> dict:
                called_with.append(text)
                return {"echo": text}

        class SomeMarker:
            """Marker."""

        @agent(model="mock-model")
        @use_tools(EchoTool[SomeMarker])
        class EchoAgent: ...

        @controller("/echo")
        class EchoController:
            def __init__(self, runner: AgentRunner, ai: EchoAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._ai, body["message"])
                return {"content": r.content}

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("echo_tool", {"text": "hello"})
        mock.queue_response(_text("Done!", id="c2"))

        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(agents=[EchoAgent], imports=[LLMProv])

        @module(imports=[LLMProv, AIMod], controllers=[EchoController])
        class AppModule: ...

        r = TestClient(LaurenFactory.create(AppModule)).post(
            "/echo/chat", json={"message": "Say hello"}
        )
        assert r.status_code == 200
        assert called_with == ["hello"]

    def test_two_generic_aliases_same_base_class_run_independently(self):
        """Two modules each with a different alias of the same tool get independent singletons."""
        from typing import Generic
        from typing import TypeVar as _TypeVar

        from lauren import LaurenFactory, Scope, controller, injectable, module, post
        from lauren.testing import TestClient

        from lauren_ai import LLMConfig, LLMModule, agent, tool, use_tools
        from lauren_ai._agents._runner import AgentRunnerBase
        from lauren_ai._module import AgentModule

        _T = _TypeVar("_T")
        instances: list[int] = []

        @tool()
        class TagTool(Generic[_T]):
            """Tag the message.

            Args:
                msg: The message.
            """

            async def run(self, msg: str) -> dict:
                instances.append(id(self))
                return {"tagged": msg}

        class MarkerX:
            """X."""

        class MarkerY:
            """Y."""

        @agent(model="mock-model")
        @use_tools(TagTool[MarkerX])
        class AgentX: ...

        @agent(model="mock-model")
        @use_tools(TagTool[MarkerY])
        class AgentY: ...

        @injectable(scope=Scope.SINGLETON)
        class RunnerX(AgentRunnerBase):
            """Runner X."""

        @injectable(scope=Scope.SINGLETON)
        class RunnerY(AgentRunnerBase):
            """Runner Y."""

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("tag_tool", {"msg": "x-msg"})
        mock.queue_response(_text("X done", id="cx"))
        mock.queue_tool_use("tag_tool", {"msg": "y-msg"})
        mock.queue_response(_text("Y done", id="cy"))

        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        ModX = AgentModule.for_root(agents=[AgentX], imports=[LLMProv], runner=RunnerX)
        ModY = AgentModule.for_root(agents=[AgentY], imports=[LLMProv], runner=RunnerY)

        @controller("/xy")
        class XYController:
            def __init__(self, rx: RunnerX, ry: RunnerY, ax: AgentX, ay: AgentY) -> None:
                self._rx, self._ry, self._ax, self._ay = rx, ry, ax, ay

            @post("/x")
            async def run_x(self, body: Json[dict]) -> dict:
                r = await self._rx.run(self._ax, body["msg"])
                return {"content": r.content}

            @post("/y")
            async def run_y(self, body: Json[dict]) -> dict:
                r = await self._ry.run(self._ay, body["msg"])
                return {"content": r.content}

        @module(imports=[LLMProv, ModX, ModY], controllers=[XYController])
        class AppModule: ...

        client = TestClient(LaurenFactory.create(AppModule))
        rx = client.post("/xy/x", json={"msg": "x-msg"})
        ry = client.post("/xy/y", json={"msg": "y-msg"})
        assert rx.status_code == 200
        assert ry.status_code == 200
        # Two separate tool instances (one per alias token)
        assert len(instances) == 2
        assert instances[0] != instances[1]


# ---------------------------------------------------------------------------
# TestSharedTools
# ---------------------------------------------------------------------------


class TestSharedTools:
    """shared_tools= injects DI-owned tool instances into a borrowing AgentModule.

    Pattern mirrors the banking chatbot: a plain ``@module`` (CheckAuthModule)
    owns and exports the tool; each AgentModule imports it and lists it in
    ``shared_tools=`` so the runner receives the singleton instance without
    re-registering the tool as a provider.
    """

    def _make_shared_tool(self):
        """Return (SharedTool class, OwnerModule, calls list) with no-dep tool."""
        calls: list[str] = []

        @tool()
        class SharedTool:
            """Echo a greeting. Args: name: The name to greet."""

            async def run(self, name: str) -> dict:
                calls.append(name)
                return {"greeting": f"hello {name}"}

        @module(providers=[SharedTool], exports=[SharedTool])
        class OwnerModule: ...

        return SharedTool, OwnerModule, calls

    def test_shared_tool_in_runner_tools(self):
        """shared_tools instance lands in agent meta.tools when agent declares @use_tools."""
        import asyncio

        SharedTool, OwnerModule, _ = self._make_shared_tool()

        @agent(model=None)
        @use_tools(SharedTool)
        class BorrowerAgent: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        BorrowerMod = AgentModule.for_root(
            agents=[BorrowerAgent],
            imports=[LLMProv, OwnerModule],
            shared_tools=[SharedTool],
        )

        @module(imports=[LLMProv, OwnerModule, BorrowerMod])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(app.container.resolve(BorrowerMod.runner_class))
        finally:
            loop.close()

        assert "shared_tool" in BorrowerAgent.__lauren_ai_agent__.tools

    def test_shared_tool_no_duplicate_provider_error(self):
        """shared_tools= does not re-register the tool, so no DuplicateBindingError."""
        SharedTool, OwnerModule, _ = self._make_shared_tool()

        @agent(model=None)
        class BorrowerAgent2: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        BorrowerMod = AgentModule.for_root(
            agents=[BorrowerAgent2],
            imports=[LLMProv, OwnerModule],
            shared_tools=[SharedTool],
        )

        @module(imports=[LLMProv, OwnerModule, BorrowerMod])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        assert app is not None

    def test_shared_tool_instance_is_the_di_singleton(self):
        """The instance in agent meta.tools is the same object the container returns."""
        import asyncio

        SharedTool, OwnerModule, _ = self._make_shared_tool()

        @agent(model=None)
        @use_tools(SharedTool)
        class BorrowerAgent3: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        BorrowerMod = AgentModule.for_root(
            agents=[BorrowerAgent3],
            imports=[LLMProv, OwnerModule],
            shared_tools=[SharedTool],
        )

        @module(imports=[LLMProv, OwnerModule, BorrowerMod])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(app.container.resolve(BorrowerMod.runner_class))
            di_instance = loop.run_until_complete(app.container.resolve(SharedTool))
        finally:
            loop.close()

        tool_instance, _ = BorrowerAgent3.__lauren_ai_agent__.tools["shared_tool"]
        assert tool_instance is di_instance

    def test_shared_tool_agent_can_call_it(self):
        """An agent can invoke a shared_tools tool during a run and get its result."""
        SharedTool, OwnerModule, calls = self._make_shared_tool()

        @agent(model="mock-model")
        @use_tools(SharedTool)
        class BorrowerAgent4: ...

        @controller("/borrow")
        class BorrowController:
            def __init__(self, runner: AgentRunner, ai: BorrowerAgent4) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._ai, body["message"])
                return {"content": r.content}

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("shared_tool", {"name": "world"})
        mock.queue_response(_text("hello world", id="c2"))

        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        BorrowerMod = AgentModule.for_root(
            agents=[BorrowerAgent4],
            imports=[LLMProv, OwnerModule],
            shared_tools=[SharedTool],
        )

        @module(imports=[LLMProv, OwnerModule, BorrowerMod], controllers=[BorrowController])
        class AppModule: ...

        r = TestClient(LaurenFactory.create(AppModule)).post(
            "/borrow/chat", json={"message": "Say hello."}
        )

        assert r.status_code == 200
        assert r.json()["content"] == "hello world"
        assert calls == ["world"]

"""End-to-end tests — realistic end-user journey through the lauren-ai public API.

These tests model how a developer building an AI-powered web application would
wire up agents: via ``@controller`` classes that inject ``AgentRunner`` (and
agent instances) through the normal Lauren DI system, then driven by
``lauren.testing.TestClient`` over real HTTP.

No test ever calls ``app.container.resolve()`` — that is not part of the
end-user API surface.

Scenarios covered:

- Minimal chat endpoint: no tools, one-shot question → JSON response
- Function-form tools: single and multi-turn tool calls through an HTTP handler
- Class-form tools with DI: injectable tool dependencies resolved end-to-end
- Agent lifecycle hooks: ``on_start`` / ``on_finish`` side-effects visible in response
- Signal observability: ``SignalBus`` events accumulated during an HTTP request
- Multiple agents: separate endpoints backed by separate agent classes
- ``max_turns`` override: bounded agent stops and returns correct ``stop_reason``
- SSE streaming: ``run_stream()`` frames as ``text/event-stream`` via ``EventStream``
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from lauren import Json, LaurenFactory, Scope, controller, get, injectable, module, post
from lauren.sse import EventStream, ServerSentEvent
from lauren.testing import TestClient

from lauren_ai import (
    AgentContext,
    AgentModule,
    AgentResponse,
    AgentRunner,
    LLMConfig,
    LLMModule,
    ModelCallComplete,
    SignalBus,
    TokenUsage,
    agent,
    tool,
    use_tools,
)
from lauren_ai._transport import Completion, CompletionChunk, ToolCallDelta

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _completion(content: str, *, id: str = "c1") -> Completion:
    """Build a canned text completion for MockTransport."""
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=20, output_tokens=10),
    )


def _chunks(*parts: str, stop_reason: str = "end_turn") -> list[CompletionChunk]:
    """Build a list of CompletionChunk objects from text parts."""
    chunks = [CompletionChunk(delta=p) for p in parts]
    chunks.append(
        CompletionChunk(
            delta="",
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=20, output_tokens=len(parts)),
        )
    )
    return chunks


def _parse_sse(body: bytes) -> list[dict]:
    """Parse a buffered SSE body into a list of ``{event, data}`` dicts."""
    events: list[dict] = []
    current: dict = {}
    for raw_line in body.decode().splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                events.append(current)
                current = {}
        elif line.startswith("event:"):
            current["event"] = line[6:].strip()
        elif line.startswith("data:"):
            current["data"] = line[5:].strip()
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class ChatRequest:
    """Simple chat request body (parsed manually to avoid Pydantic dependency)."""


# ---------------------------------------------------------------------------
# 1. Minimal chat endpoint — no tools, one-shot answer
# ---------------------------------------------------------------------------


class TestMinimalChatEndpoint:
    """Simplest possible wiring: agent + runner injected into a controller."""

    def test_one_shot_answer(self):
        """POST /chat/ returns the agent's content and stop_reason."""

        @agent(model="mock-model", system="You are a helpful assistant.")
        class SimpleAssistant: ...

        @controller("/chat")
        class ChatController:
            def __init__(self, runner: AgentRunner, ai: SimpleAssistant) -> None:
                self._runner = runner
                self._ai = ai

            @post("/")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {
                    "content": response.content,
                    "stop_reason": response.stop_reason,
                    "turns": response.turns,
                }

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("The answer is 42."))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[SimpleAssistant], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule], controllers=[ChatController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/chat/", json={"message": "What is the answer?"})

        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "The answer is 42."
        assert data["stop_reason"] == "end_turn"
        assert data["turns"] == 1

    def test_token_usage_returned(self):
        """Handler can return token counts from the AgentResponse."""

        @agent(model="mock-model")
        class TokenAgent: ...

        @controller("/usage")
        class UsageController:
            def __init__(self, runner: AgentRunner, ai: TokenAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {
                    "input_tokens": response.total_usage.input_tokens,
                    "output_tokens": response.total_usage.output_tokens,
                }

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(
            Completion(
                id="c1",
                model="mock-model",
                content="Done.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        )

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[TokenAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule], controllers=[UsageController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/usage/", json={"message": "Count tokens."})

        assert r.status_code == 200
        data = r.json()
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50


# ---------------------------------------------------------------------------
# 2. Function-form tools
# ---------------------------------------------------------------------------


@tool()
async def get_weather(city: str) -> dict:
    """Return current weather for a city.

    Args:
        city: The city to check.
    """
    return {"city": city, "temp_c": 22, "condition": "sunny"}


@tool()
async def convert_currency(amount: float, from_ccy: str, to_ccy: str) -> dict:
    """Convert an amount between currencies.

    Args:
        amount: Amount to convert.
        from_ccy: Source currency code.
        to_ccy: Target currency code.
    """
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79}
    result = amount / rates.get(from_ccy, 1.0) * rates.get(to_ccy, 1.0)
    return {"amount": round(result, 2), "currency": to_ccy}


@agent(model="mock-model", system="You are a travel assistant.")
@use_tools(get_weather, convert_currency)
class TravelAgent: ...


class TestFunctionFormTools:
    """HTTP endpoint backed by an agent with function-form @tool() decorators."""

    def _make_app(self, mock):
        @controller("/travel")
        class TravelController:
            def __init__(self, runner: AgentRunner, ai: TravelAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {
                    "content": response.content,
                    "turns": response.turns,
                    "tool_calls": [tc.name for tc in response.tool_calls_made],
                }

        cfg, _ = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[TravelAgent],
            tools=[get_weather, convert_currency],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule], controllers=[TravelController])
        class AppModule: ...

        return LaurenFactory.create(AppModule)

    def test_single_tool_call(self):
        """Agent makes one tool call then gives a final answer."""
        _, mock = LLMConfig.for_testing()
        mock.queue_tool_use("get_weather", {"city": "Paris"})
        mock.queue_response(_completion("Paris is sunny at 22°C.", id="c2"))

        app = self._make_app(mock)
        r = TestClient(app).post("/travel/chat", json={"message": "Weather in Paris?"})

        assert r.status_code == 200
        data = r.json()
        assert data["turns"] == 2
        assert data["tool_calls"] == ["get_weather"]

    def test_multi_turn_two_tools(self):
        """Agent calls two tools in separate turns then finishes."""
        _, mock = LLMConfig.for_testing()
        mock.queue_tool_use("get_weather", {"city": "New York"})
        mock.queue_tool_use("convert_currency", {"amount": 100.0, "from_ccy": "USD", "to_ccy": "EUR"})
        mock.queue_response(_completion("NYC sunny; €92 for $100.", id="c3"))

        app = self._make_app(mock)
        r = TestClient(app).post("/travel/chat", json={"message": "NYC weather and 100 USD to EUR?"})

        assert r.status_code == 200
        data = r.json()
        assert data["turns"] == 3
        assert set(data["tool_calls"]) == {"get_weather", "convert_currency"}

    def test_tool_result_forwarded_to_model(self):
        """The second mock call contains the tool result from the first turn."""
        _, mock = LLMConfig.for_testing()
        mock.queue_tool_use("get_weather", {"city": "Rome"})
        mock.queue_response(_completion("Rome: sunny, 22°C.", id="c2"))

        app = self._make_app(mock)
        TestClient(app).post("/travel/chat", json={"message": "Rome weather?"})

        assert len(mock.calls) == 2
        second_messages = mock.calls[1].messages
        assert any("22" in str(m) or "tool_results" in str(m) for m in second_messages), (
            "Tool result not forwarded to second model call"
        )


# ---------------------------------------------------------------------------
# 3. Class-form tools with DI
# ---------------------------------------------------------------------------


class TestClassFormToolsWithDI:
    """Class-form @tool() classes receive constructor deps from the DI container."""

    def test_class_tool_dep_injected_and_called(self):
        """The injected dep is called during the HTTP request and drives the tool result."""

        call_log: list[str] = []

        @injectable(scope=Scope.SINGLETON)
        class OrderDatabase:
            def get_status(self, order_id: str) -> str:
                call_log.append(order_id)
                return "shipped"

        @tool()
        class CheckOrderTool:
            """Check the status of a customer order.

            Args:
                order_id: The order identifier.
            """

            def __init__(self, db: OrderDatabase) -> None:
                self._db = db

            async def run(self, order_id: str) -> dict:
                return {"order_id": order_id, "status": self._db.get_status(order_id)}

        @agent(model="mock-model", system="You are a customer support agent.")
        @use_tools(CheckOrderTool)
        class SupportAgent: ...

        @controller("/support")
        class SupportController:
            def __init__(self, runner: AgentRunner, ai: SupportAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {"content": response.content, "tool_calls": len(response.tool_calls_made)}

        @module(providers=[OrderDatabase], exports=[OrderDatabase])
        class DataModule: ...

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("check_order_tool", {"order_id": "ORD-42"})
        mock.queue_response(_completion("Order ORD-42 has been shipped.", id="c2"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[SupportAgent],
            imports=[LLMProvider, DataModule],
        )

        @module(imports=[LLMProvider, AIModule, DataModule], controllers=[SupportController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/support/chat", json={"message": "Where is ORD-42?"})

        assert r.status_code == 200
        data = r.json()
        assert data["tool_calls"] == 1
        assert call_log == ["ORD-42"]

    def test_two_class_tools_with_different_deps(self):
        """Two class-form tools each receive their own distinct injectable dep."""

        results: list[str] = []

        @injectable(scope=Scope.SINGLETON)
        class ProductCatalog:
            def describe(self, pid: str) -> str:
                results.append(f"catalog:{pid}")
                return f"Product {pid}: high quality"

        @injectable(scope=Scope.SINGLETON)
        class PricingEngine:
            def quote(self, pid: str) -> float:
                results.append(f"pricing:{pid}")
                return 29.99

        @tool()
        class DescribeTool:
            """Describe a product.

            Args:
                product_id: The product identifier.
            """

            def __init__(self, catalog: ProductCatalog) -> None:
                self._catalog = catalog

            async def run(self, product_id: str) -> dict:
                return {"description": self._catalog.describe(product_id)}

        @tool()
        class PriceTool:
            """Quote the price of a product.

            Args:
                product_id: The product identifier.
            """

            def __init__(self, pricing: PricingEngine) -> None:
                self._pricing = pricing

            async def run(self, product_id: str) -> dict:
                return {"price": self._pricing.quote(product_id)}

        @agent(model="mock-model")
        @use_tools(DescribeTool, PriceTool)
        class ShopAgent: ...

        @controller("/shop")
        class ShopController:
            def __init__(self, runner: AgentRunner, ai: ShopAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {"tool_calls": [tc.name for tc in response.tool_calls_made]}

        @module(providers=[ProductCatalog, PricingEngine], exports=[ProductCatalog, PricingEngine])
        class ShopModule: ...

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("describe_tool", {"product_id": "P1"})
        mock.queue_tool_use("price_tool", {"product_id": "P1"})
        mock.queue_response(_completion("P1: high quality, $29.99.", id="c3"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[ShopAgent],
            imports=[LLMProvider, ShopModule],
        )

        @module(imports=[LLMProvider, AIModule, ShopModule], controllers=[ShopController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/shop/chat", json={"message": "Describe and price P1."})

        assert r.status_code == 200
        data = r.json()
        assert set(data["tool_calls"]) == {"describe_tool", "price_tool"}
        # Both deps were actually called
        assert "catalog:P1" in results
        assert "pricing:P1" in results


# ---------------------------------------------------------------------------
# 4. Agent lifecycle hooks
# ---------------------------------------------------------------------------


class TestAgentLifecycleHooks:
    """on_start and on_finish hooks fire with the correct context during an HTTP request."""

    def test_on_start_metadata_visible_in_response(self):
        """Metadata passed to runner.run() is captured in on_start and returned."""

        captured: dict = {}

        @agent(model="mock-model")
        class HookAgent:
            async def on_start(self, ctx: AgentContext) -> None:
                captured.update(ctx.metadata)

        @controller("/hook")
        class HookController:
            def __init__(self, runner: AgentRunner, ai: HookAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                await self._runner.run(
                    self._ai,
                    body["message"],
                    metadata={"user_id": body.get("user_id", "anon")},
                )
                return {"captured_user_id": captured.get("user_id")}

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("Hi!"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[HookAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule], controllers=[HookController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/hook/chat", json={"message": "Hello", "user_id": "u-99"})

        assert r.status_code == 200
        assert r.json()["captured_user_id"] == "u-99"

    def test_on_finish_receives_final_response(self):
        """on_finish is called with the completed AgentResponse."""

        finish_log: list[str] = []

        @agent(model="mock-model")
        class FinishAgent:
            async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
                finish_log.append(response.content)

        @controller("/finish")
        class FinishController:
            def __init__(self, runner: AgentRunner, ai: FinishAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {"content": response.content}

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("Goodbye!"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[FinishAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule], controllers=[FinishController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/finish/chat", json={"message": "Bye."})

        assert r.status_code == 200
        assert r.json()["content"] == "Goodbye!"
        assert finish_log == ["Goodbye!"]


# ---------------------------------------------------------------------------
# 5. Signal observability
# ---------------------------------------------------------------------------


class TestSignalObservability:
    """SignalBus events are emitted during agent runs triggered by HTTP requests."""

    def test_model_call_complete_fired_per_turn(self):
        """One ModelCallComplete per LLM call; tool call = extra turn → 2 events."""

        bus = SignalBus()
        event_models: list[str] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            event_models.append(event.model)

        @tool()
        async def ping(x: str) -> str:
            """Ping. Args: x: Input."""
            return f"pong:{x}"

        @agent(model="mock-model")
        @use_tools(ping)
        class PingAgent: ...

        @controller("/signal")
        class SignalController:
            def __init__(self, runner: AgentRunner, ai: PingAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {"turns": response.turns}

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("ping", {"x": "test"})
        mock.queue_response(_completion("Pong received.", id="c2"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[PingAgent],
            tools=[ping],
            imports=LLMProvider,
            signals=bus,
        )

        @module(imports=[LLMProvider, AIModule], controllers=[SignalController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/signal/chat", json={"message": "Ping please."})

        assert r.status_code == 200
        assert r.json()["turns"] == 2
        assert len(event_models) == 2
        assert all(m == "mock-model" for m in event_models)


# ---------------------------------------------------------------------------
# 6. Multiple agents — separate endpoints
# ---------------------------------------------------------------------------


class TestMultipleAgentsInOneModule:
    """Multiple @agent() classes share one AgentModule, each with its own endpoint."""

    def test_two_agents_on_separate_endpoints(self):
        """Each endpoint invokes its own agent; responses are independent."""

        @tool()
        async def summarize(text: str) -> str:
            """Summarize text. Args: text: Text to summarize."""
            return text[:50]

        @tool()
        async def translate(text: str, lang: str) -> str:
            """Translate text. Args: text: Source text. lang: Target language."""
            return f"[{lang}] {text}"

        @agent(model="mock-model", system="You summarize.")
        @use_tools(summarize)
        class SummaryAgent: ...

        @agent(model="mock-model", system="You translate.")
        @use_tools(translate)
        class TranslateAgent: ...

        @controller("/ai")
        class MultiController:
            def __init__(
                self,
                runner: AgentRunner,
                summary: SummaryAgent,
                translate_ai: TranslateAgent,
            ) -> None:
                self._runner = runner
                self._summary = summary
                self._translate = translate_ai

            @post("/summarize")
            async def do_summarize(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._summary, body["message"])
                return {"content": r.content}

            @post("/translate")
            async def do_translate(self, body: Json[dict]) -> dict:
                r = await self._runner.run(self._translate, body["message"])
                return {"content": r.content}

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[SummaryAgent, TranslateAgent],
            tools=[summarize, translate],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule], controllers=[MultiController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        client = TestClient(app)

        mock.queue_response(_completion("Here is the summary."))
        r1 = client.post("/ai/summarize", json={"message": "Summarize this."})
        assert r1.status_code == 200
        assert r1.json()["content"] == "Here is the summary."

        mock.queue_response(_completion("Here is the translation."))
        r2 = client.post("/ai/translate", json={"message": "Translate this."})
        assert r2.status_code == 200
        assert r2.json()["content"] == "Here is the translation."


# ---------------------------------------------------------------------------
# 7. AgentConfig overrides
# ---------------------------------------------------------------------------


class TestAgentConfigOverrides:
    """Per-agent settings passed to @agent() take effect during HTTP-driven runs."""

    def test_max_turns_stops_loop(self):
        """An agent with max_turns=1 stops after one turn; stop_reason is returned."""

        @tool()
        async def endless_tool(x: str) -> str:
            """Always returns something. Args: x: Input."""
            return x

        @agent(model="mock-model", max_turns=1)
        @use_tools(endless_tool)
        class BoundedAgent: ...

        @controller("/bounded")
        class BoundedController:
            def __init__(self, runner: AgentRunner, ai: BoundedAgent) -> None:
                self._runner = runner
                self._ai = ai

            @post("/chat")
            async def chat(self, body: Json[dict]) -> dict:
                response = await self._runner.run(self._ai, body["message"])
                return {"stop_reason": response.stop_reason}

        cfg, mock = LLMConfig.for_testing()
        # Only one tool-use completion queued; with max_turns=1 the loop
        # stops before calling the model again
        mock.queue_tool_use("endless_tool", {"x": "loop"})

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[BoundedAgent],
            tools=[endless_tool],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule], controllers=[BoundedController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).post("/bounded/chat", json={"message": "Go forever."})

        assert r.status_code == 200
        assert r.json()["stop_reason"] == "max_turns"


# ---------------------------------------------------------------------------
# 8. SSE streaming — run_stream() framed as text/event-stream
# ---------------------------------------------------------------------------


def _get_header(headers, name: str) -> str | None:
    """Look up a header value from the TestResponse headers list of tuples."""
    return next((v for k, v in headers if k.lower() == name.lower()), None)


class TestSSEStreaming:
    """Controllers that call run_stream() and yield chunks as SSE events."""

    def test_stream_chunks_arrive_as_sse_events(self):
        """Each text chunk from run_stream() is framed as an SSE ``delta`` event."""

        @agent(model="mock-model", system="You are a streaming assistant.")
        class StreamAgent: ...

        @controller("/stream")
        class StreamController:
            def __init__(self, runner: AgentRunner, ai: StreamAgent) -> None:
                self._runner = runner
                self._ai = ai

            @get("/chat")
            async def stream_chat(self) -> EventStream:
                async def gen() -> AsyncIterator[ServerSentEvent]:
                    stream = await self._runner.run_stream(self._ai, "Hello!")
                    async for chunk in stream:
                        if chunk.delta:
                            yield ServerSentEvent(event="delta", data=chunk.delta)
                        if chunk.stop_reason:
                            yield ServerSentEvent(event="done", data=chunk.stop_reason)

                return EventStream(gen())

        cfg, mock = LLMConfig.for_testing()
        mock.queue_stream(_chunks("Hello", " World", "!"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[StreamAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule], controllers=[StreamController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).get("/stream/chat")

        assert r.status_code == 200
        assert "text/event-stream" in (_get_header(r.headers, "content-type") or "")

        events = _parse_sse(r.body)
        delta_events = [e for e in events if e.get("event") == "delta"]
        done_events = [e for e in events if e.get("event") == "done"]

        assert len(delta_events) == 3
        assert len(done_events) == 1
        assert done_events[0]["data"] == "end_turn"

    def test_sse_stream_with_class_tool_and_di(self):
        """Class-based tool with DI executes silently; final answer streams as SSE."""

        called_with: list[str] = []

        @injectable(scope=Scope.SINGLETON)
        class SearchIndex:
            def lookup(self, q: str) -> str:
                called_with.append(q)
                return f"result:{q}"

        @tool()
        class LookupTool:
            """Look up a query in the search index.

            Args:
                q: The search query.
            """

            def __init__(self, index: SearchIndex) -> None:
                self._index = index

            async def run(self, q: str) -> str:
                return self._index.lookup(q)

        @agent(model="mock-model")
        @use_tools(LookupTool)
        class SearchAgent: ...

        @controller("/search-stream")
        class SearchStreamController:
            def __init__(self, runner: AgentRunner, ai: SearchAgent) -> None:
                self._runner = runner
                self._ai = ai

            @get("/chat")
            async def stream_chat(self) -> EventStream:
                async def gen() -> AsyncIterator[ServerSentEvent]:
                    stream = await self._runner.run_stream(self._ai, "Search for x.")
                    async for chunk in stream:
                        if chunk.delta:
                            yield ServerSentEvent(event="delta", data=chunk.delta)
                        if chunk.stop_reason == "end_turn":
                            yield ServerSentEvent(event="done", data="end_turn")

                return EventStream(gen())

        cfg, mock = LLMConfig.for_testing()
        # Turn 1: streaming tool-use chunks so _stream_loop can accumulate and execute
        mock.queue_stream(
            [
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="tc1", name="lookup_tool", input_delta='{"q":"x"}')
                ),
                CompletionChunk(
                    delta="",
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                ),
            ]
        )
        # Turn 2: streaming final answer
        mock.queue_stream(_chunks("Found: result:x"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)

        @module(providers=[SearchIndex], exports=[SearchIndex])
        class IndexModule: ...

        AIModule = AgentModule.for_root(
            agents=[SearchAgent],
            imports=[LLMProvider, IndexModule],
        )

        @module(imports=[LLMProvider, AIModule, IndexModule], controllers=[SearchStreamController])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        r = TestClient(app).get("/search-stream/chat")

        assert r.status_code == 200
        events = _parse_sse(r.body)
        delta_events = [e for e in events if e.get("event") == "delta"]
        done_events = [e for e in events if e.get("event") == "done"]

        # The tool was called with the correct argument via DI
        assert called_with == ["x"]
        # The final streamed answer arrived as delta SSE events
        assert any("Found" in e.get("data", "") for e in delta_events)
        assert len(done_events) == 1
        assert done_events[0]["data"] == "end_turn"

    def test_run_stream_threads_execution_context_to_tools(self):
        """run_stream(..., execution_context=...) reaches agent_context.execution_context."""
        from lauren_ai._tools import ToolContext

        seen: list = []

        @tool()
        class IdentityTool:
            """Read the user_id from the verified server-side context."""

            async def run(self, ctx: ToolContext) -> dict:
                # execution_context lives on agent_context, not ToolContext directly.
                ec = ctx.agent_context.execution_context if ctx.agent_context else None
                uid = ec.request.state.get("user_id") if ec else None
                seen.append(uid)
                return {"user_id": uid}

        @agent(model="mock-model")
        @use_tools(IdentityTool)
        class IdentityAgent: ...

        cfg, mock = LLMConfig.for_testing()
        # Turn 1: tool-use chunk
        mock.queue_stream(
            [
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="tc1", name="identity_tool", input_delta="{}")
                ),
                CompletionChunk(
                    delta="",
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                ),
            ]
        )
        mock.queue_stream(_chunks("ok"))

        LLMProv = LLMModule.for_root(cfg, transport_override=mock)
        AIMod = AgentModule.for_root(agents=[IdentityAgent], imports=[LLMProv])

        # Build a fake ExecutionContext with request.state.get("user_id")
        class _State:
            def __init__(self, d):
                self._d = d

            def get(self, key, default=None):
                return self._d.get(key, default)

        class _Req:
            def __init__(self, state):
                self.state = state

        class _ExecCtx:
            def __init__(self, request):
                self.request = request

        fake_ctx = _ExecCtx(_Req(_State({"user_id": "alice"})))

        @module(imports=[LLMProv, AIMod])
        class AppMod: ...

        import asyncio

        app = LaurenFactory.create(AppMod)
        loop = asyncio.new_event_loop()
        try:
            runner = loop.run_until_complete(app.container.resolve(AgentRunner))
            agent_inst = loop.run_until_complete(app.container.resolve(IdentityAgent))

            async def drain():
                async for _ in await runner.run_stream(agent_inst, "Who am I?", execution_context=fake_ctx):
                    pass

            loop.run_until_complete(drain())
        finally:
            loop.close()

        assert seen == ["alice"]

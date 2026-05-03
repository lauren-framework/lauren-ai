"""End-to-end tests — realistic end-user journey through the lauren-ai public API.

These tests simulate what a developer building an AI-powered application would
write.  Every test drives the full stack from public-API imports through
``LaurenFactory.create()`` to ``AgentRunner.run()`` / ``AgentTestClient``,
using a ``MockTransport`` in place of a real LLM provider.

Scenarios covered:
- Minimal agent: no tools, one-shot question-answer
- Function-form tools: single and multi-turn tool calls
- Class-form tools with DI: injectable dependencies are resolved and called
- Agent lifecycle hooks: ``on_start`` / ``on_finish`` receive the correct context
- Signal observability: ``SignalBus`` events fired during a run
- Agent metadata: metadata dict flows from ``run()`` into ``AgentContext``
- Multiple agents in one module: each gets its own tool set
- ``AgentTestClient``: synchronous and async test-client wrappers
"""

from __future__ import annotations

import pytest
from lauren import LaurenFactory, Scope, injectable, module

# Public API ─ what a real user imports
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
from lauren_ai._transport import Completion
from lauren_ai.testing import AgentTestClient

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


# ---------------------------------------------------------------------------
# 1. Minimal agent — no tools, no DI, one-shot answer
# ---------------------------------------------------------------------------


class TestMinimalAgent:
    """Simplest possible usage: declare an agent, run it, get a response."""

    @pytest.mark.asyncio
    async def test_one_shot_answer(self):
        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("The answer is 42."))

        @agent(model="mock-model", system="You are a helpful assistant.")
        class SimpleAssistant: ...

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[SimpleAssistant], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        response = await runner.run(SimpleAssistant(), "What is the answer?")

        assert response.content == "The answer is 42."
        assert response.stop_reason == "end_turn"
        assert response.turns == 1
        assert len(response.tool_calls_made) == 0

    def test_sync_client_one_shot(self):
        """AgentTestClient.run() wraps the async runner for convenience."""
        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("Hello from sync!"))

        @agent(model="mock-model")
        class EchoAgent: ...

        client = AgentTestClient(EchoAgent(), mock)
        response = client.run("Hi!")

        assert response.content == "Hello from sync!"
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_total_token_usage_reported(self):
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

        @agent(model="mock-model")
        class TokenAgent: ...

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[TokenAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        response = await runner.run(TokenAgent(), "Count tokens.")

        assert response.total_usage.input_tokens == 100
        assert response.total_usage.output_tokens == 50


# ---------------------------------------------------------------------------
# 2. Function-form tools
# ---------------------------------------------------------------------------


# Tools defined at module level (idiomatic for function-form tools)

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
    """Agents that use function-form @tool() decorators."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        """Agent makes one tool call then gives a final answer."""
        cfg, mock = LLMConfig.for_testing()

        mock.queue_tool_use("get_weather", {"city": "Paris"})
        mock.queue_response(_completion("Paris is sunny at 22°C.", id="c2"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[TravelAgent],
            tools=[get_weather, convert_currency],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        response = await runner.run(TravelAgent(), "What's the weather in Paris?")

        assert response.stop_reason == "end_turn"
        assert response.turns == 2
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0].name == "get_weather"
        assert response.tool_calls_made[0].input == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_multi_turn_two_tools(self):
        """Agent calls two different tools in separate turns, then finishes."""
        cfg, mock = LLMConfig.for_testing()

        mock.queue_tool_use("get_weather", {"city": "New York"})
        mock.queue_tool_use("convert_currency", {"amount": 100.0, "from_ccy": "USD", "to_ccy": "EUR"})
        mock.queue_response(_completion("In New York it's sunny. €92 for your $100.", id="c3"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[TravelAgent],
            tools=[get_weather, convert_currency],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        response = await runner.run(
            TravelAgent(),
            "What's the weather in New York and how much is $100 in euros?",
        )

        assert response.turns == 3
        tool_names = [tc.name for tc in response.tool_calls_made]
        assert "get_weather" in tool_names
        assert "convert_currency" in tool_names

    @pytest.mark.asyncio
    async def test_tool_result_reaches_model(self):
        """The tool's return value is included in the subsequent model call."""
        cfg, mock = LLMConfig.for_testing()

        mock.queue_tool_use("get_weather", {"city": "Rome"})
        mock.queue_response(_completion("Rome: sunny, 22°C.", id="c2"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[TravelAgent],
            tools=[get_weather],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        await runner.run(TravelAgent(), "Rome weather?")

        # The second model call must include the tool result in its messages
        assert len(mock.calls) == 2
        second_call_messages = mock.calls[1].messages
        tool_result_found = any(
            getattr(m, "role", None) == "tool"
            or (isinstance(m, dict) and m.get("role") == "tool")
            or "tool_results" in str(m)
            or "22" in str(m)  # temp from the tool return value
            for m in second_call_messages
        )
        assert tool_result_found, "Tool result not forwarded to second model call"

    def test_agent_test_client_with_tool(self):
        """AgentTestClient.run() handles a one-tool flow synchronously."""
        cfg, mock = LLMConfig.for_testing()

        mock.queue_tool_use("get_weather", {"city": "Tokyo"})
        mock.queue_response(_completion("Tokyo: 28°C, humid.", id="c2"))

        client = AgentTestClient(TravelAgent(), mock)
        response = client.run("Tokyo weather?")

        assert response.turns == 2
        assert response.tool_calls_made[0].name == "get_weather"


# ---------------------------------------------------------------------------
# 3. Class-form tools with DI
# ---------------------------------------------------------------------------


class TestClassFormToolsWithDI:
    """Class-form @tool() classes receive their constructor deps from the DI container."""

    @pytest.mark.asyncio
    async def test_class_tool_dep_injected_and_called(self):
        """The injected dep is called during the agent run and drives the tool result."""

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

        @module(providers=[OrderDatabase], exports=[OrderDatabase])
        class DataModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[SupportAgent],
            imports=[LLMProvider, DataModule],
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        mock.queue_tool_use("check_order_tool", {"order_id": "ORD-42"})
        mock.queue_response(_completion("Order ORD-42 has been shipped.", id="c2"))

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        response = await runner.run(SupportAgent(), "Where is my order ORD-42?")

        assert response.stop_reason == "end_turn"
        assert len(response.tool_calls_made) == 1
        # The dep was actually called with the arg the agent passed
        assert call_log == ["ORD-42"]

    @pytest.mark.asyncio
    async def test_two_class_tools_with_different_deps(self):
        """Two class-form tools each receive their own distinct dep."""

        @injectable(scope=Scope.SINGLETON)
        class ProductCatalog:
            def describe(self, pid: str) -> str:
                return f"Product {pid}: high quality"

        @injectable(scope=Scope.SINGLETON)
        class PricingEngine:
            def quote(self, pid: str) -> float:
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

        @module(
            providers=[ProductCatalog, PricingEngine],
            exports=[ProductCatalog, PricingEngine],
        )
        class ShopModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[ShopAgent],
            imports=[LLMProvider, ShopModule],
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        describe_tool = await app.container.resolve(DescribeTool)
        price_tool = await app.container.resolve(PriceTool)

        assert isinstance(describe_tool._catalog, ProductCatalog)
        assert isinstance(price_tool._pricing, PricingEngine)
        # Each tool has its own dep type
        assert not hasattr(describe_tool, "_pricing")
        assert not hasattr(price_tool, "_catalog")

    @pytest.mark.asyncio
    async def test_class_tool_with_scope_request_override(self):
        """@tool() on top of @injectable(scope=REQUEST) keeps REQUEST scope."""
        from lauren import injectable as _injectable

        @_injectable(scope=Scope.REQUEST)
        @tool()
        class RequestScopedTool:
            """A per-request tool. Args: x: Input."""

            async def run(self, x: str) -> str:
                return x

        from lauren._di import INJECTABLE_META
        meta = getattr(RequestScopedTool, INJECTABLE_META)
        assert meta.scope == Scope.REQUEST


# ---------------------------------------------------------------------------
# 4. Agent lifecycle hooks
# ---------------------------------------------------------------------------


class TestAgentLifecycleHooks:
    """on_start and on_finish hooks fire with correct AgentContext / AgentResponse."""

    @pytest.mark.asyncio
    async def test_on_start_receives_agent_context(self):
        captured_ctx: list[AgentContext] = []

        @agent(model="mock-model")
        class HookAgent:
            async def on_start(self, ctx: AgentContext) -> None:
                captured_ctx.append(ctx)

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("Hi!"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[HookAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        await runner.run(HookAgent(), "Hello", metadata={"user_id": "u-99"})

        assert len(captured_ctx) == 1
        ctx = captured_ctx[0]
        assert ctx.metadata["user_id"] == "u-99"
        assert ctx.turn == 0
        assert ctx.agent_class is HookAgent

    @pytest.mark.asyncio
    async def test_on_finish_receives_response(self):
        captured_resp: list[AgentResponse] = []

        @agent(model="mock-model")
        class FinishAgent:
            async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
                captured_resp.append(response)

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("Goodbye!"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[FinishAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        await runner.run(FinishAgent(), "Bye.")

        assert len(captured_resp) == 1
        assert captured_resp[0].content == "Goodbye!"
        assert captured_resp[0].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_metadata_flows_into_on_start(self):
        """Metadata passed to runner.run() is available in the AgentContext."""
        received: dict = {}

        @agent(model="mock-model")
        class MetaAgent:
            async def on_start(self, ctx: AgentContext) -> None:
                received.update(ctx.metadata)

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("OK."))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[MetaAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        await runner.run(
            MetaAgent(),
            "Test.",
            metadata={"tenant": "acme", "plan": "pro"},
        )

        assert received["tenant"] == "acme"
        assert received["plan"] == "pro"


# ---------------------------------------------------------------------------
# 5. Signal observability
# ---------------------------------------------------------------------------


class TestSignalObservability:
    """SignalBus events are emitted during agent runs."""

    @pytest.mark.asyncio
    async def test_model_call_complete_fired(self):
        """ModelCallComplete is emitted once for a no-tool agent run."""
        bus = SignalBus()
        events: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            events.append(event)

        @agent(model="mock-model")
        class ObservedAgent: ...

        cfg, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("Hello!"))

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[ObservedAgent],
            imports=LLMProvider,
            signals=bus,
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        await runner.run(ObservedAgent(), "Hi.")

        assert len(events) == 1
        assert events[0].model == "mock-model"

    @pytest.mark.asyncio
    async def test_model_call_complete_fired_per_turn(self):
        """One ModelCallComplete event per LLM call (tool call = extra turn)."""
        bus = SignalBus()
        model_events: list[ModelCallComplete] = []

        @bus.on(ModelCallComplete)
        async def capture(event: ModelCallComplete) -> None:
            model_events.append(event)

        @tool()
        async def ping(x: str) -> str:
            """Ping. Args: x: Input."""
            return f"pong:{x}"

        @agent(model="mock-model")
        @use_tools(ping)
        class PingAgent: ...

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

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        await runner.run(PingAgent(), "Ping please.")

        # Tool call = one extra turn = two total model calls
        assert len(model_events) == 2


# ---------------------------------------------------------------------------
# 6. Multiple agents in one module
# ---------------------------------------------------------------------------


class TestMultipleAgentsInOneModule:
    """Multiple @agent() classes can share one AgentModule.for_root() call."""

    @pytest.mark.asyncio
    async def test_two_agents_resolved_independently(self):
        """Two agents are both resolvable and run independently via the same runner."""

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

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[SummaryAgent, TranslateAgent],
            tools=[summarize, translate],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        summary_inst = await app.container.resolve(SummaryAgent)
        translate_inst = await app.container.resolve(TranslateAgent)

        # Run SummaryAgent
        mock.queue_response(_completion("Here is the summary."))
        r1 = await runner.run(summary_inst, "Summarize this.")
        assert r1.content == "Here is the summary."

        # Run TranslateAgent
        mock.queue_response(_completion("Here is the translation."))
        r2 = await runner.run(translate_inst, "Translate this.")
        assert r2.content == "Here is the translation."

    @pytest.mark.asyncio
    async def test_agents_are_singletons(self):
        """Resolving the same agent class twice returns the same instance."""

        @agent(model="mock-model")
        class SingletonAgent: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[SingletonAgent], imports=LLMProvider)

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        inst_a = await app.container.resolve(SingletonAgent)
        inst_b = await app.container.resolve(SingletonAgent)

        assert inst_a is inst_b


# ---------------------------------------------------------------------------
# 7. AgentConfig overrides
# ---------------------------------------------------------------------------


class TestAgentConfigOverrides:
    """Per-agent AgentConfig settings take effect at run time."""

    @pytest.mark.asyncio
    async def test_max_turns_stops_loop(self):
        """An agent configured with max_turns=1 stops after one turn (no tool call looping)."""

        @tool()
        async def endless_tool(x: str) -> str:
            """Always returns something. Args: x: Input."""
            return x

        @agent(model="mock-model", max_turns=1)
        @use_tools(endless_tool)
        class BoundedAgent: ...

        cfg, mock = LLMConfig.for_testing()
        # Queue only a tool-use completion — the runner should stop after max_turns
        mock.queue_tool_use("endless_tool", {"x": "loop"})

        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[BoundedAgent],
            tools=[endless_tool],
            imports=LLMProvider,
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        response = await runner.run(BoundedAgent(), "Go forever.")

        assert response.stop_reason == "max_turns"

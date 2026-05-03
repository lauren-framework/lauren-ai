"""Integration tests — class-form @tool() and @guardrail() receive DI-injected deps.

Tests cover:
- Class-form tool with a single injectable dep (dep resolved from the DI graph)
- Class-form tool with a dep-of-dep chain (A → B → Tool)
- Multiple class-form tools with different deps (all resolved correctly)
- Singleton dep is the same instance across tool and another consumer
- Dep is actually used during an agent run (end-to-end spy test)
- @guardrail() class with an injectable dep (dep resolved from DI)
- Guardrail dep used correctly in check() at runtime
- GuardrailWiring pattern: guardrail (with dep) attached to an agent via a
  wiring singleton
"""

from __future__ import annotations

import pytest
from lauren import LaurenFactory, Scope, injectable, module

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunner
from lauren_ai._config import LLMConfig
from lauren_ai._guardrails import (
    USE_GUARDRAILS_META,
    GuardrailContext,
    GuardrailDecision,
    UseGuardrailsMeta,
    guardrail,
    use_guardrails,
)
from lauren_ai._module import AgentModule, LLMModule
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai.testing import AgentTestClient

# ---------------------------------------------------------------------------
# Helpers
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
# Class-form tools with DI dependencies
# ---------------------------------------------------------------------------


class TestClassToolDIInjection:
    """Class-form @tool() constructor deps are resolved by the DI container."""

    @pytest.mark.asyncio
    async def test_single_dep_injected(self):
        """PriceDatabase is injected into PriceTool when the DI container builds it."""

        @injectable(scope=Scope.SINGLETON)
        class PriceDatabase:
            def get_price(self, item: str) -> float:
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

        @module(providers=[PriceDatabase], exports=[PriceDatabase])
        class PriceModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[ShopAgent],
            imports=[LLMProvider, PriceModule],
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        price_tool = await app.container.resolve(PriceTool)

        assert isinstance(price_tool, PriceTool)
        assert isinstance(price_tool._db, PriceDatabase)

    @pytest.mark.asyncio
    async def test_dep_chain_injected(self):
        """Tool → ServiceB → ServiceA: the full dep chain is resolved correctly."""

        @injectable(scope=Scope.SINGLETON)
        class ServiceA:
            def value(self) -> str:
                return "from-A"

        @injectable(scope=Scope.SINGLETON)
        class ServiceB:
            def __init__(self, a: ServiceA) -> None:
                self._a = a

            def message(self) -> str:
                return f"B wraps: {self._a.value()}"

        @tool()
        class ChainTool:
            """Return a chained message.

            Args:
                x: Ignored.
            """

            def __init__(self, b: ServiceB) -> None:
                self._b = b

            async def run(self, x: str) -> dict:
                return {"msg": self._b.message()}

        @agent(model="mock-model")
        @use_tools(ChainTool)
        class ChainAgent: ...

        @module(providers=[ServiceA, ServiceB], exports=[ServiceA, ServiceB])
        class ServiceModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[ChainAgent],
            imports=[LLMProvider, ServiceModule],
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        chain_tool = await app.container.resolve(ChainTool)

        assert isinstance(chain_tool._b, ServiceB)
        assert isinstance(chain_tool._b._a, ServiceA)

    @pytest.mark.asyncio
    async def test_multiple_tools_each_get_own_dep(self):
        """Two class-form tools with distinct deps both get the correct dep injected."""

        @injectable(scope=Scope.SINGLETON)
        class WeatherDB:
            def get(self, city: str) -> str:
                return "sunny"

        @injectable(scope=Scope.SINGLETON)
        class StockDB:
            def get(self, ticker: str) -> float:
                return 100.0

        @tool()
        class WeatherTool:
            """Get the weather. Args: city: City name."""

            def __init__(self, db: WeatherDB) -> None:
                self._db = db

            async def run(self, city: str) -> dict:
                return {"weather": self._db.get(city)}

        @tool()
        class StockTool:
            """Get stock price. Args: ticker: Ticker symbol."""

            def __init__(self, db: StockDB) -> None:
                self._db = db

            async def run(self, ticker: str) -> dict:
                return {"price": self._db.get(ticker)}

        @agent(model="mock-model")
        @use_tools(WeatherTool, StockTool)
        class FinanceAgent: ...

        @module(providers=[WeatherDB, StockDB], exports=[WeatherDB, StockDB])
        class DataModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[FinanceAgent],
            imports=[LLMProvider, DataModule],
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        weather_tool = await app.container.resolve(WeatherTool)
        stock_tool = await app.container.resolve(StockTool)

        assert isinstance(weather_tool._db, WeatherDB)
        assert isinstance(stock_tool._db, StockDB)
        # Each tool got its own dep type — no cross-contamination
        assert not isinstance(weather_tool._db, StockDB)
        assert not isinstance(stock_tool._db, WeatherDB)

    @pytest.mark.asyncio
    async def test_dep_is_singleton_shared_between_tool_and_other_provider(self):
        """The same singleton instance is shared between the tool and another consumer."""

        @injectable(scope=Scope.SINGLETON)
        class SharedCache:
            def __init__(self) -> None:
                self.hits: int = 0

        @tool()
        class CacheTool:
            """Read from cache. Args: key: Cache key."""

            def __init__(self, cache: SharedCache) -> None:
                self._cache = cache

            async def run(self, key: str) -> dict:
                self._cache.hits += 1
                return {"key": key, "hits": self._cache.hits}

        @injectable(scope=Scope.SINGLETON)
        class OtherService:
            def __init__(self, cache: SharedCache) -> None:
                self._cache = cache

        @agent(model="mock-model")
        @use_tools(CacheTool)
        class CacheAgent: ...

        @module(providers=[SharedCache, OtherService], exports=[SharedCache, OtherService])
        class CacheModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[CacheAgent],
            imports=[LLMProvider, CacheModule],
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        cache_tool = await app.container.resolve(CacheTool)
        other = await app.container.resolve(OtherService)

        # Same SharedCache instance injected into both
        assert cache_tool._cache is other._cache

    @pytest.mark.asyncio
    async def test_dep_used_during_agent_run(self):
        """End-to-end: the injected dep is actually called when the agent invokes the tool."""

        calls: list[str] = []

        @injectable(scope=Scope.SINGLETON)
        class SpyDB:
            def lookup(self, key: str) -> str:
                calls.append(key)
                return f"value-for-{key}"

        @tool()
        class SpyTool:
            """Run a lookup.

            Args:
                key: The lookup key.
            """

            def __init__(self, db: SpyDB) -> None:
                self._db = db

            async def run(self, key: str) -> dict:
                return {"result": self._db.lookup(key)}

        @agent(model="mock-model")
        @use_tools(SpyTool)
        class SpyAgent: ...

        @module(providers=[SpyDB], exports=[SpyDB])
        class SpyModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(
            agents=[SpyAgent],
            imports=[LLMProvider, SpyModule],
        )

        @module(imports=[LLMProvider, AIModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        runner = await app.container.resolve(AgentRunner)
        agent_inst = await app.container.resolve(SpyAgent)
        client = AgentTestClient(agent_inst, mock, runner=runner)

        mock.queue_tool_use("spy_tool", {"key": "hello"})
        mock.queue_response(_text("Done.", id="c2"))

        response = await client.run_async("Look up hello.")

        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0].name == "spy_tool"
        # SpyDB.lookup was called with the arg the agent passed — dep was used
        assert calls == ["hello"]


# ---------------------------------------------------------------------------
# Class-form guardrails with DI dependencies
# ---------------------------------------------------------------------------


class TestGuardrailDIInjection:
    """@guardrail() class constructor deps are resolved by the DI container."""

    @pytest.mark.asyncio
    async def test_guardrail_dep_injected(self):
        """BlockList is injected into WordFilter when the DI container resolves it."""

        @injectable(scope=Scope.SINGLETON)
        class BlockList:
            def words(self) -> list[str]:
                return ["spam", "scam"]

        @guardrail(kind="input")
        class WordFilter:
            def __init__(self, blocklist: BlockList) -> None:
                self._blocklist = blocklist

            async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
                for word in self._blocklist.words():
                    if word in message.lower():
                        return GuardrailDecision(
                            action="block",
                            violation=f"Blocked word: {word}",
                            guardrail_name="WordFilter",
                        )
                return GuardrailDecision(action="pass", guardrail_name="WordFilter")

        @module(providers=[BlockList, WordFilter], exports=[BlockList, WordFilter])
        class GuardrailModule: ...

        @module(imports=[GuardrailModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        wf = await app.container.resolve(WordFilter)

        assert isinstance(wf, WordFilter)
        assert isinstance(wf._blocklist, BlockList)

    @pytest.mark.asyncio
    async def test_guardrail_dep_used_in_check(self):
        """The injected dep drives block/pass decisions in check()."""

        @injectable(scope=Scope.SINGLETON)
        class BlockList:
            def words(self) -> list[str]:
                return ["badword"]

        @guardrail(kind="input")
        class BlockFilter:
            def __init__(self, blocklist: BlockList) -> None:
                self._blocklist = blocklist

            async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
                for word in self._blocklist.words():
                    if word in message.lower():
                        return GuardrailDecision(
                            action="block",
                            violation=f"Blocked: {word}",
                            guardrail_name="BlockFilter",
                        )
                return GuardrailDecision(action="pass", guardrail_name="BlockFilter")

        @module(providers=[BlockList, BlockFilter], exports=[BlockList, BlockFilter])
        class GuardrailModule: ...

        @module(imports=[GuardrailModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        bf = await app.container.resolve(BlockFilter)
        ctx = GuardrailContext(agent_name="test")

        blocked = await bf.check("this message contains badword here", ctx)
        allowed = await bf.check("this message is perfectly clean", ctx)

        assert blocked.action == "block"
        assert "badword" in (blocked.violation or "")
        assert allowed.action == "pass"

    @pytest.mark.asyncio
    async def test_guardrail_wiring_pattern(self):
        """GuardrailWiring singleton attaches a dep-injected guardrail to an agent."""

        @injectable(scope=Scope.SINGLETON)
        class TopicList:
            def allowed(self) -> list[str]:
                return ["cooking", "recipes"]

        @guardrail(kind="input")
        class TopicGuard:
            def __init__(self, topics: TopicList) -> None:
                self._topics = topics

            async def check(self, message: str, ctx: GuardrailContext) -> GuardrailDecision:
                for t in self._topics.allowed():
                    if t in message.lower():
                        return GuardrailDecision(action="pass", guardrail_name="TopicGuard")
                return GuardrailDecision(
                    action="block", violation="Off-topic", guardrail_name="TopicGuard"
                )

        @agent(model="mock-model")
        @use_guardrails()
        class CookingAgent: ...

        @injectable(scope=Scope.SINGLETON)
        class GuardrailWiring:
            """Attaches the DI-resolved TopicGuard to CookingAgent."""

            def __init__(self, guard: TopicGuard) -> None:
                meta: UseGuardrailsMeta = getattr(CookingAgent, USE_GUARDRAILS_META)
                meta.input_guardrails.append(guard)

        @module(
            providers=[TopicList, TopicGuard, GuardrailWiring],
            exports=[TopicList, TopicGuard],
        )
        class GuardrailModule: ...

        cfg, mock = LLMConfig.for_testing()
        LLMProvider = LLMModule.for_root(cfg, transport_override=mock)
        AIModule = AgentModule.for_root(agents=[CookingAgent], imports=[LLMProvider])

        @module(imports=[LLMProvider, AIModule, GuardrailModule])
        class AppModule: ...

        app = LaurenFactory.create(AppModule)
        # Resolving GuardrailWiring triggers its __init__ side-effect
        await app.container.resolve(GuardrailWiring)

        guard = await app.container.resolve(TopicGuard)
        meta: UseGuardrailsMeta = getattr(CookingAgent, USE_GUARDRAILS_META)

        # Guard's dep was injected by DI
        assert isinstance(guard._topics, TopicList)
        # Wiring attached the resolved guard to the agent's metadata
        assert guard in meta.input_guardrails

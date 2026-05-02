"""Unit tests for SemanticRouter."""
from __future__ import annotations

import math
import pytest

from lauren_ai._routing import SemanticRouter, Route, RouteMatch, RouterConfigError
from lauren_ai._transport import Message
from lauren_ai._transport._mock import MockTransport
from lauren_ai._module import LLMService
from lauren_ai._config import LLMConfig


def make_embed_fn(mapping: dict[str, list[float]]):
    """Create a mock embed_fn that returns predefined vectors."""

    class FakeEmbedding:
        def __init__(self, vec: list[float]):
            self.vector = vec

    async def embed_fn(texts: list[str]) -> list[FakeEmbedding]:
        results = []
        for text in texts:
            vec = mapping.get(text, [0.5] * 3)
            results.append(FakeEmbedding(vec))
        return results

    return embed_fn


class TestRouterConfig:
    def test_empty_routes_raises(self):
        with pytest.raises(RouterConfigError, match="at least one"):
            SemanticRouter(routes=[], embed_fn=lambda x: [])

    def test_duplicate_names_raises(self):
        with pytest.raises(RouterConfigError, match="duplicate"):
            SemanticRouter(
                routes=[
                    Route("a", "desc"),
                    Route("a", "other"),
                ],
                embed_fn=lambda x: [],
            )

    def test_route_without_description_or_examples_raises(self):
        with pytest.raises(RouterConfigError):
            SemanticRouter(
                routes=[Route("a", "", examples=[])],
                embed_fn=lambda x: [],
            )


class TestSemanticRouter:
    async def test_route_matches_by_embedding(self):
        weather_vec = [1.0, 0.0, 0.0]
        travel_vec = [0.0, 1.0, 0.0]
        query_vec = [0.95, 0.05, 0.0]   # close to weather

        embed_fn = make_embed_fn({
            "Will it rain tomorrow?": query_vec,
            "Questions about weather forecasts and temperature": weather_vec,
            "Travel planning and trip itineraries": travel_vec,
        })

        router = SemanticRouter(
            routes=[
                Route("weather", "Questions about weather forecasts and temperature"),
                Route("travel", "Travel planning and trip itineraries"),
            ],
            embed_fn=embed_fn,
            min_confidence=0.5,
        )
        await router.compile()
        match = await router.route("Will it rain tomorrow?")
        assert match.route_name == "weather"
        assert match.method == "embedding"
        assert match.confidence > 0.5

    async def test_low_confidence_returns_default(self):
        embed_fn = make_embed_fn({
            "Tell me a joke": [0.5, 0.5, 0.0],
            "desc_a": [1.0, 0.0, 0.0],
            "desc_b": [0.0, 1.0, 0.0],
        })
        router = SemanticRouter(
            routes=[
                Route("a", "desc_a"),
                Route("b", "desc_b"),
            ],
            embed_fn=embed_fn,
            min_confidence=0.99,  # very high threshold
            fallback_route="a",
        )
        await router.compile()
        match = await router.route("Tell me a joke")
        assert match.route_name == "a"
        assert match.method in ("default", "llm_fallback")

    async def test_dispatch_calls_correct_handler(self):
        weather_vec = [1.0, 0.0, 0.0]
        embed_fn = make_embed_fn({
            "Weather query": [0.99, 0.01, 0.0],
            "desc_weather": weather_vec,
            "desc_other": [0.0, 1.0, 0.0],
        })
        router = SemanticRouter(
            routes=[
                Route("weather", "desc_weather"),
                Route("other", "desc_other"),
            ],
            embed_fn=embed_fn,
            min_confidence=0.5,
            fallback_route="weather",
        )
        await router.compile()

        called_with: list[str] = []

        async def weather_handler(q: str) -> str:
            called_with.append(q)
            return "sunny"

        async def other_handler(q: str) -> str:
            called_with.append(f"other:{q}")
            return "other"

        result = await router.dispatch("Weather query", {
            "weather": weather_handler,
            "other": other_handler,
        })
        assert result == "sunny"
        assert called_with == ["Weather query"]

    async def test_compile_with_examples(self):
        embed_fn = make_embed_fn({
            "Will it rain?": [1.0, 0.0],
            "Hot today?": [0.9, 0.1],
            "Trip to Rome": [0.0, 1.0],
        })
        router = SemanticRouter(
            routes=[
                Route("weather", "Weather", examples=["Will it rain?", "Hot today?"]),
                Route("travel", "Travel", examples=["Trip to Rome"]),
            ],
            embed_fn=embed_fn,
        )
        await router.compile()
        # Check centroids computed
        for r in router.routes:
            assert r._centroid is not None


# ---------------------------------------------------------------------------
# New spec-described API tests
# ---------------------------------------------------------------------------


class TestRouterNotCompiledError:
    """RouterNotCompiledError must be importable and subclass LaurenAIError."""

    def test_importable_from_module(self):
        from lauren_ai._routing import RouterNotCompiledError  # noqa: F401

    def test_importable_from_top_level(self):
        from lauren_ai import RouterNotCompiledError  # noqa: F401

    def test_is_lauren_ai_error(self):
        from lauren_ai._exceptions import LaurenAIError
        from lauren_ai._routing import RouterNotCompiledError

        err = RouterNotCompiledError("not compiled")
        assert isinstance(err, LaurenAIError)

    def test_message_stored(self):
        from lauren_ai._routing import RouterNotCompiledError

        err = RouterNotCompiledError("call compile() first")
        assert "compile" in str(err)


class TestRouteMatchCompatProps:
    """RouteMatch.route and .matched compat properties."""

    def test_matched_true_for_embedding_method(self):
        m = RouteMatch(route_name="weather", confidence=0.9, method="embedding")
        assert m.matched is True

    def test_matched_false_for_default_method(self):
        m = RouteMatch(route_name="weather", confidence=0.3, method="default")
        assert m.matched is False

    def test_matched_false_for_llm_fallback(self):
        m = RouteMatch(route_name="weather", confidence=0.3, method="llm_fallback")
        assert m.matched is False

    def test_route_prop_returns_name_for_embedding(self):
        m = RouteMatch(route_name="travel", confidence=0.85, method="embedding")
        assert m.route == "travel"

    def test_route_prop_returns_none_for_zero_confidence_default(self):
        m = RouteMatch(route_name="fallback", confidence=0.0, method="default")
        assert m.route is None

    def test_route_prop_returns_name_for_llm_fallback(self):
        m = RouteMatch(route_name="travel", confidence=0.4, method="llm_fallback")
        assert m.route == "travel"

    def test_route_prop_for_non_zero_default(self):
        m = RouteMatch(route_name="fallback", confidence=0.2, method="default")
        # Non-zero default still has a route name
        assert m.route == "fallback"


class TestCosineSimilarityHelper:
    """_cosine_similarity helper function correctness."""

    def test_identical_vectors(self):
        from lauren_ai._routing._router import _cosine_similarity

        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from lauren_ai._routing._router import _cosine_similarity

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from lauren_ai._routing._router import _cosine_similarity

        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_mismatched_length_returns_zero(self):
        from lauren_ai._routing._router import _cosine_similarity

        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        from lauren_ai._routing._router import _cosine_similarity

        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


class TestSemanticRouterAddRoute:
    """SemanticRouter.add_route marks router as uncompiled."""

    async def test_add_route_marks_uncompiled(self):
        embed_fn = make_embed_fn({
            "weather desc": [1.0, 0.0],
        })
        router = SemanticRouter(
            routes=[Route("weather", "weather desc")],
            embed_fn=embed_fn,
            min_confidence=0.5,
        )
        await router.compile()
        assert router._compiled is True

        router.add_route(Route("travel", "travel desc"))
        assert router._compiled is False

    async def test_route_after_add_requires_recompile(self):
        weather_vec = [1.0, 0.0]
        travel_vec = [0.0, 1.0]
        extra_vec = [0.5, 0.5]

        embed_fn = make_embed_fn({
            "weather desc": weather_vec,
            "travel desc": travel_vec,
            "extra desc": extra_vec,
            "query": [0.9, 0.1],
        })
        router = SemanticRouter(
            routes=[Route("weather", "weather desc")],
            embed_fn=embed_fn,
            min_confidence=0.5,
        )
        await router.compile()
        router.add_route(Route("travel", "travel desc"))
        # Compiles automatically inside route() when uncompiled.
        match = await router.route("query")
        assert match.route_name in ("weather", "travel")

    def test_add_route_appends_to_routes_list(self):
        embed_fn = make_embed_fn({})
        router = SemanticRouter(
            routes=[Route("a", "desc_a")],
            embed_fn=embed_fn,
        )
        assert len(router.routes) == 1
        router.add_route(Route("b", "desc_b"))
        assert len(router.routes) == 2
        assert router.routes[1].name == "b"


class TestSemanticRouterRouteMatchIntegration:
    """Integration of matched/route compat props with real routing."""

    async def test_high_confidence_match_has_matched_true(self):
        weather_vec = [1.0, 0.0, 0.0]
        embed_fn = make_embed_fn({
            "rain": [0.99, 0.01, 0.0],
            "weather desc": weather_vec,
            "travel desc": [0.0, 1.0, 0.0],
        })
        router = SemanticRouter(
            routes=[
                Route("weather", "weather desc"),
                Route("travel", "travel desc"),
            ],
            embed_fn=embed_fn,
            min_confidence=0.5,
        )
        await router.compile()
        match = await router.route("rain")
        assert match.matched is True
        assert match.route == "weather"

    async def test_low_confidence_match_has_matched_false(self):
        embed_fn = make_embed_fn({
            "joke": [0.5, 0.5, 0.0],
            "weather desc": [1.0, 0.0, 0.0],
            "travel desc": [0.0, 1.0, 0.0],
        })
        router = SemanticRouter(
            routes=[
                Route("weather", "weather desc"),
                Route("travel", "travel desc"),
            ],
            embed_fn=embed_fn,
            min_confidence=0.99,
            fallback_route="weather",
        )
        await router.compile()
        match = await router.route("joke")
        assert match.matched is False

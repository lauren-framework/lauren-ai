"""Semantic routing — route queries to agents/chains by embedding similarity."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from lauren_ai._exceptions import LaurenAIError


class RouterConfigError(LaurenAIError):
    """Raised at startup when SemanticRouter is misconfigured.

    :param message: Human-readable description of the misconfiguration.
    :type message: str
    """


class RouterNotCompiledError(LaurenAIError):
    """Raised when :meth:`SemanticRouter.route` is called before
    :meth:`SemanticRouter.compile`.

    Call ``await router.compile()`` once at startup before routing queries.

    :param message: Human-readable description of the error.
    :type message: str
    """


@dataclass
class Route:
    """A named route with description and optional example utterances.

    Usage::

        Route(
            name="weather",
            description="Questions about weather, forecasts, temperature",
            examples=["Will it rain?", "What is the temperature in Paris?"],
        )

    :param name: Unique route identifier.
    :type name: str
    :param description: Human-readable description of what this route handles.
    :type description: str
    :param examples: Optional list of example utterances used to compute the
        route centroid embedding.
    :type examples: list[str]
    """

    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    _centroid: list[float] | None = field(default=None, init=False, repr=False)


@dataclass
class RouteMatch:
    """Result of a :meth:`SemanticRouter.route` call.

    :param route_name: Name of the matched route (also accessible as
        :attr:`route` for API compatibility).
    :type route_name: str
    :param confidence: Cosine-similarity score of the best match.
    :type confidence: float
    :param method: How the match was determined: ``"embedding"``,
        ``"llm_fallback"``, or ``"default"``.
    :type method: str
    """

    route_name: str
    confidence: float
    method: Literal["embedding", "llm_fallback", "default"]

    @property
    def route(self) -> str | None:
        """Return the matched route name, or ``None`` when the match was a
        default/fallback with confidence below threshold.

        :return: Route name string, or ``None``.
        :rtype: str | None
        """
        if self.method == "default" and self.confidence == 0.0:
            return None
        return self.route_name

    @property
    def matched(self) -> bool:
        """``True`` when the match was found via embedding similarity above the
        configured ``min_confidence`` threshold.

        :return: Whether a confident match was found.
        :rtype: bool
        """
        return self.method == "embedding"


class SemanticRouter:
    """Route incoming queries to named handlers by embedding similarity.

    Usage::

        router = SemanticRouter(
            routes=[
                Route("weather", "Weather questions", examples=["Will it rain?"]),
                Route("travel", "Travel planning", examples=["Plan a trip to Rome"]),
            ],
            embed_fn=llm_service.embed,
            min_confidence=0.6,
            fallback_route="travel",
        )
        await router.compile()   # precompute embeddings (called at startup)

        match = await router.route("Is it going to snow?")
        print(match.route_name)   # "weather"
    """

    def __init__(
        self,
        routes: list[Route],
        embed_fn: Callable[[list[str]], Any],
        min_confidence: float = 0.6,
        fallback_route: str | None = None,
        llm: Any | None = None,  # LLMService for fallback
    ) -> None:
        self._validate(routes)
        self.routes = routes
        self._embed_fn = embed_fn
        self._min_confidence = min_confidence
        self._fallback_route = fallback_route or routes[0].name
        self._llm = llm
        self._compiled = False

    def _validate(self, routes: list[Route]) -> None:
        if not routes:
            raise RouterConfigError("SemanticRouter requires at least one Route.")
        names = [r.name for r in routes]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise RouterConfigError(
                f"SemanticRouter has duplicate route names: {sorted(set(dupes))}"
            )
        for r in routes:
            if not r.description and not r.examples:
                raise RouterConfigError(f"Route {r.name!r} must have a description or examples.")

    async def compile(self) -> None:
        """Precompute embedding centroids for all routes. Call at startup."""
        for route in self.routes:
            texts = route.examples if route.examples else [route.description]
            embeddings = await self._embed_fn(texts)
            vectors = [e.vector if hasattr(e, "vector") else e for e in embeddings]
            route._centroid = _mean_vector(vectors)
        self._compiled = True

    async def route(self, query: str) -> RouteMatch:
        """Find the best matching route for query using embedding similarity."""
        if not self._compiled:
            await self.compile()

        # Embed the query
        result = await self._embed_fn([query])
        query_vector = result[0].vector if hasattr(result[0], "vector") else result[0]

        # Compute cosine similarity to each route centroid
        best_name = self._fallback_route
        best_score = 0.0

        for route in self.routes:
            if route._centroid is None:
                continue
            score = _cosine_similarity(query_vector, route._centroid)
            if score > best_score:
                best_score = score
                best_name = route.name

        if best_score >= self._min_confidence:
            return RouteMatch(
                route_name=best_name,
                confidence=best_score,
                method="embedding",
            )

        # Try LLM fallback
        if self._llm is not None:
            llm_route = await self._llm_route(query)
            return RouteMatch(
                route_name=llm_route,
                confidence=best_score,
                method="llm_fallback",
            )

        return RouteMatch(
            route_name=self._fallback_route,
            confidence=best_score,
            method="default",
        )

    async def _llm_route(self, query: str) -> str:
        from lauren_ai._transport import Message

        route_list = "\n".join(f"- {r.name}: {r.description}" for r in self.routes)
        prompt = (
            f"You are a routing assistant. Given a user query, pick the most appropriate route.\n\n"
            f"Available routes:\n{route_list}\n\n"
            f"Query: {query!r}\n\n"
            f"Respond with ONLY the route name, nothing else."
        )
        from lauren_ai._transport import Completion

        result = await self._llm.complete([Message(role="user", content=prompt)])  # type: ignore[arg-type]
        if isinstance(result, Completion):
            name = result.content.strip().lower()
        else:
            chunks = []
            async for chunk in result:
                if chunk.delta:
                    chunks.append(chunk.delta)
            name = "".join(chunks).strip().lower()

        # Find matching route (case-insensitive)
        for r in self.routes:
            if r.name.lower() == name:
                return r.name
        return self._fallback_route

    def add_route(self, route: Route) -> None:
        """Add *route* to the router and mark it as requiring recompilation.

        Call :meth:`compile` again (or let :meth:`route` auto-compile) before
        routing new queries.

        :param route: The new route to add.
        :type route: Route
        """
        self.routes.append(route)
        self._compiled = False

    async def dispatch(
        self,
        query: str,
        handlers: dict[str, Any],
    ) -> Any:
        """Route query and call the matching handler.

        :param query: User query string.
        :param handlers: Mapping of route_name → async callable(query) -> Any.
        :return: Whatever the matched handler returns.
        """
        match = await self.route(query)
        handler = handlers.get(match.route_name) or handlers.get(self._fallback_route)
        if handler is None:
            raise RouterConfigError(f"No handler for route {match.route_name!r} and no fallback.")
        import inspect

        if inspect.iscoroutinefunction(handler):
            return await handler(query)
        return handler(query)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    mean = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            mean[i] += x
    n = len(vectors)
    return [x / n for x in mean]

# Semantic Router

`SemanticRouter` routes natural-language queries to named handlers using
embedding similarity. Each route is defined with a name, a description, and
optional example utterances. At startup, `compile()` computes an embedding
centroid per route. At request time, the query is embedded and matched against
those centroids via cosine similarity.

## Quick start

```python
from lauren_ai._routing import SemanticRouter, Route

router = SemanticRouter(
    routes=[
        Route(
            name="weather",
            description="Questions about weather, forecasts, and temperature",
            examples=["Will it rain tomorrow?", "What is the temperature in Paris?"],
        ),
        Route(
            name="travel",
            description="Trip planning, destinations, hotels, and flights",
            examples=["Plan a trip to Rome", "Best hotels in Tokyo"],
        ),
    ],
    embed_fn=llm_service.embed,  # async callable accepting list[str]
    min_confidence=0.6,
    fallback_route="travel",
)

# Call once at startup to precompute embedding centroids
await router.compile()

# Route a query
match = await router.route("Is it going to snow in London?")
print(match.route_name)   # "weather"
print(match.confidence)   # e.g. 0.87
print(match.method)       # "embedding"
```

## Route

```python
from lauren_ai._routing import Route
from dataclasses import dataclass, field

@dataclass
class Route:
    name: str                          # unique identifier for the route
    description: str                   # natural-language description used for embedding
    examples: list[str]                # optional example utterances
```

`examples` takes priority over `description` when computing the centroid. If
both are provided the examples are embedded and averaged; the `description` is
used only when `examples` is empty.

At least one of `description` or `examples` must be non-empty. A route with
both empty raises `RouterConfigError` at construction time.

## SemanticRouter

```python
SemanticRouter(
    routes: list[Route],
    embed_fn: Callable[[list[str]], Awaitable[list[Any]]],
    min_confidence: float = 0.6,
    fallback_route: str | None = None,
    llm: LLMService | None = None,
)
```

- `routes`: must contain at least one `Route` with a unique name.
- `embed_fn`: async function accepting `list[str]` and returning a list of
  objects with a `.vector` attribute (e.g. `list[Embedding]`) or a list of
  `list[float]` directly.
- `min_confidence`: cosine-similarity threshold. Queries that do not exceed
  this score fall through to the LLM fallback or the default route.
- `fallback_route`: name of the route used when no route exceeds
  `min_confidence`. Defaults to the first route in the list.
- `llm`: optional `LLMService` instance. When provided and
  `min_confidence` is not met, the router issues a one-shot prompt to the LLM
  asking it to pick the best route by name.

## RouteMatch

```python
@dataclass
class RouteMatch:
    route_name: str
    confidence: float
    method: Literal["embedding", "llm_fallback", "default"]
```

- `method="embedding"` — chosen by cosine similarity above `min_confidence`.
- `method="llm_fallback"` — cosine similarity was below threshold; the LLM
  made the decision.
- `method="default"` — cosine similarity was below threshold and no `llm` was
  configured; `fallback_route` was used.

## compile()

```python
await router.compile()
```

Embeds the description or examples for every route and stores the mean vector
as the route centroid. You must call `compile()` before calling `route()`.
The router calls `compile()` automatically on the first `route()` call if you
forget, but calling it explicitly at application startup is recommended so
startup validation catches embedding errors early.

## route()

```python
match: RouteMatch = await router.route("some user query")
```

1. Embeds the query using `embed_fn`.
2. Computes cosine similarity between the query vector and every route
   centroid.
3. Returns a `RouteMatch` with the best matching route if similarity exceeds
   `min_confidence`.
4. Falls back to LLM routing or the default route otherwise.

## dispatch()

`dispatch()` routes and calls the matching handler in one step:

```python
result = await router.dispatch(
    query="Will it snow?",
    handlers={
        "weather": weather_agent.run,
        "travel": travel_agent.run,
    },
)
```

Each handler must be a callable accepting a single `str` argument. Both sync
and async callables are supported. If the matched route has no handler and the
fallback route also has no handler, a `RouterConfigError` is raised.

## LLM fallback

When `min_confidence` is not met and an `llm` is provided the router sends a
compact prompt to the LLM:

```
You are a routing assistant. Given a user query, pick the most appropriate
route.

Available routes:
- weather: Questions about weather, forecasts, and temperature
- travel: Trip planning, destinations, hotels, and flights

Query: "What is the capital of France?"

Respond with ONLY the route name, nothing else.
```

The response is matched case-insensitively against the route names. If no
match is found the fallback route is used instead.

## Error handling

| Error | When |
|---|---|
| `RouterConfigError` | Empty route list, duplicate names, route with neither description nor examples, no handler for matched or fallback route in `dispatch()`. |

## Integration with LaurenFactory

Register the router as a singleton in your module:

```python
from lauren import module, use_factory, Scope
from lauren_ai._routing import SemanticRouter, Route
from lauren_ai._module import LLMService

@module(
    providers=[
        use_factory(
            provide=SemanticRouter,
            factory=lambda llm: SemanticRouter(
                routes=[
                    Route("billing", "Billing and subscription questions"),
                    Route("support", "Technical support and troubleshooting"),
                ],
                embed_fn=llm.embed,
                min_confidence=0.65,
                llm=llm,
            ),
            inject=[LLMService],
            scope=Scope.SINGLETON,
        ),
    ],
    exports=[SemanticRouter],
)
class RouterModule: ...
```

Then inject the router into a controller or agent and call `compile()` in a
`@post_construct` hook:

```python
from lauren import post_construct, injectable, Scope

@injectable(scope=Scope.SINGLETON)
class DispatchService:
    def __init__(self, router: SemanticRouter) -> None:
        self._router = router

    @post_construct
    async def _startup(self) -> None:
        await self._router.compile()

    async def handle(self, query: str) -> str:
        return await self._router.dispatch(
            query,
            handlers={
                "billing": self._billing_handler,
                "support": self._support_handler,
            },
        )
```

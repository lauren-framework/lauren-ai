# Semantic Router

Intent-based routing using embedding similarity.

### `SemanticRouter`

Route incoming queries to named handlers by embedding similarity.

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

### `Route`

A named route with description and optional example utterances.

Usage::

    Route(
        name="weather",
        description="Questions about weather, forecasts, temperature",
        examples=["Will it rain?", "What is the temperature in Paris?"],
    )

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | Unique route identifier. |
| `description` | `str` | Human-readable description of what this route handles. |
| `examples` | `list[str]` | Optional list of example utterances used to compute the
route centroid embedding. |

### `RouteMatch`

Result of a `SemanticRouter.route()` call.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `route_name` | `str` | Name of the matched route (also accessible as
`route` for API compatibility). |
| `confidence` | `float` | Cosine-similarity score of the best match. |
| `method` | `Literal['embedding', 'llm_fallback', 'default']` | How the match was determined: `"embedding"`,
`"llm_fallback"`, or `"default"`. |

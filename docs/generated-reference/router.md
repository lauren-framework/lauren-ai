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

:param name: Unique route identifier.
:type name: str
:param description: Human-readable description of what this route handles.
:type description: str
:param examples: Optional list of example utterances used to compute the
    route centroid embedding.
:type examples: list[str]

### `RouteMatch`

Result of a :meth:`SemanticRouter.route` call.

:param route_name: Name of the matched route (also accessible as
    :attr:`route` for API compatibility).
:type route_name: str
:param confidence: Cosine-similarity score of the best match.
:type confidence: float
:param method: How the match was determined: ``"embedding"``,
    ``"llm_fallback"``, or ``"default"``.
:type method: str


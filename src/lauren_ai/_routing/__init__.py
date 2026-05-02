"""Semantic routing subsystem for ``lauren-ai``.

Routes natural-language queries to named handlers by comparing query
embeddings against pre-compiled route centroids.
"""

from __future__ import annotations

from lauren_ai._routing._router import (
    SemanticRouter,
    Route,
    RouteMatch,
    RouterConfigError,
    RouterNotCompiledError,
)

__all__ = [
    "SemanticRouter",
    "Route",
    "RouteMatch",
    "RouterConfigError",
    "RouterNotCompiledError",
]

"""Agent HTTP endpoints module (PRD 11).

Provides typed streaming HTTP endpoint infrastructure for ``lauren-ai`` agents:

- :class:`~lauren_ai._http._events.AgentEvent` — discriminated union of all
  lifecycle events suitable for wire serialisation.
- :class:`~lauren_ai._http._service.AgentStreamService` — injectable service
  that translates ``run_stream()`` output into ``AgentEvent`` items.
- :func:`~lauren_ai._http._module.AgentHttpModule` — factory creating a
  complete Lauren module with streaming HTTP routes.
"""

from __future__ import annotations

from lauren_ai._http._events import (
    AgentDoneEvent,
    AgentEvent,
    AgentHttpTokenEvent,
    AgentToolProgressEvent,
    AgentToolResultEvent,
    AgentToolStartEvent,
)
from lauren_ai._http._module import AgentHttpModule

__all__ = [
    "AgentEvent",
    "AgentDoneEvent",
    "AgentHttpTokenEvent",
    "AgentToolStartEvent",
    "AgentToolResultEvent",
    "AgentToolProgressEvent",
    "AgentHttpModule",
]

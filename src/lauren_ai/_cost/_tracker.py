"""CostTracker -- accumulates token usage and estimates per conversation/user."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lauren_ai._cost._pricing import CostEstimate, PricingTable, default_pricing_table


@dataclass
class CostSession:
    """Context manager result from CostTracker.session()."""

    conversation_id: str | None = None
    user_id: str | None = None
    usage_by_model: dict[str, Any] = field(default_factory=dict)
    total_estimate: CostEstimate = field(default_factory=CostEstimate)

    @property
    def total_tokens(self) -> int:
        total = 0
        for u in self.usage_by_model.values():
            total += getattr(u, "total_tokens", 0)
        return total


@dataclass
class CostReport:
    """Aggregated cost report for a user or conversation."""

    total_estimate: CostEstimate = field(default_factory=CostEstimate)
    by_model: dict[str, CostEstimate] = field(default_factory=dict)
    by_conversation: dict[str, CostEstimate] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_tokens_estimate(self) -> int:
        """Rough total tokens based on cost (approximate)."""
        return 0  # Can be computed from by_model if needed


class CostTracker:
    """Injectable service that accumulates token usage from ModelCallComplete signals.

    Usage::

        # Register in module
        @module(providers=[use_class(CostTracker, scope=Scope.SINGLETON)])
        class AppModule: ...

        # In a controller
        async with self.cost.session(conversation_id=cid, user_id=uid) as session:
            result = await self.runner.run(agent, message)
            print(f"Cost: ${session.total_estimate.total_usd:.6f}")
    """

    def __init__(self, pricing: PricingTable | None = None) -> None:
        self._pricing = pricing or default_pricing_table()
        # conversation_id -> list of (model, usage)
        self._conv_usage: dict[str, list[tuple[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def _on_model_call_complete(self, event: Any) -> None:
        """Signal handler -- accumulate usage from ModelCallComplete events."""
        if not hasattr(event, "usage") or event.usage is None:
            return
        conv_id = getattr(event, "conversation_id", None) or "_global"
        model = getattr(event, "model", "unknown")
        async with self._lock:
            if conv_id not in self._conv_usage:
                self._conv_usage[conv_id] = []
            self._conv_usage[conv_id].append((model, event.usage))

    @asynccontextmanager
    async def session(
        self,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[CostSession, None]:
        """Context manager tracking cost for a single request."""
        session = CostSession(conversation_id=conversation_id, user_id=user_id)

        # Take a snapshot of current usage count before session starts
        key = conversation_id or "_global"
        async with self._lock:
            before_count = len(self._conv_usage.get(key, []))

        yield session

        # Compute cost for entries added during this session
        async with self._lock:
            entries = self._conv_usage.get(key, [])[before_count:]

        for model, usage in entries:
            estimate = self._pricing.estimate(model, usage)
            session.total_estimate = session.total_estimate + estimate

    def record_usage(self, model: str, usage: Any, conversation_id: str | None = None) -> None:
        """Manually record usage (for testing or non-signal-based usage)."""
        key = conversation_id or "_global"
        if key not in self._conv_usage:
            self._conv_usage[key] = []
        self._conv_usage[key].append((model, usage))

    async def report(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> CostReport:
        """Generate a cost report."""
        report = CostReport()

        async with self._lock:
            entries_map = dict(self._conv_usage)

        if conversation_id:
            entries = entries_map.get(conversation_id, [])
            for model, usage in entries:
                estimate = self._pricing.estimate(model, usage)
                report.total_estimate = report.total_estimate + estimate
                if model not in report.by_model:
                    report.by_model[model] = CostEstimate()
                report.by_model[model] = report.by_model[model] + estimate
            report.by_conversation[conversation_id] = report.total_estimate
        else:
            for conv_id, entries in entries_map.items():
                conv_total = CostEstimate()
                for model, usage in entries:
                    estimate = self._pricing.estimate(model, usage)
                    report.total_estimate = report.total_estimate + estimate
                    if model not in report.by_model:
                        report.by_model[model] = CostEstimate()
                    report.by_model[model] = report.by_model[model] + estimate
                    conv_total = conv_total + estimate
                report.by_conversation[conv_id] = conv_total

        return report

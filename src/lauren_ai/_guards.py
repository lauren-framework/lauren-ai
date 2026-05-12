"""Guard factories for Lauren AI applications.

Provides guards for token budget enforcement, model capability checking,
and content safety filtering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

__all__ = [
    "token_budget_guard",
    "requires_capability",
    "safety_guard",
    "BudgetStore",
    "InMemoryBudgetStore",
    "SafetyPolicy",
    "BudgetUsage",
]

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Budget store
# ---------------------------------------------------------------------------


class BudgetUsage:
    """Tracks token and cost usage within a time window.

    :param tokens: Total tokens consumed in the current window.
    :type tokens: int
    :param cost_usd: Total cost in USD within the current window.
    :type cost_usd: float
    """

    def __init__(self) -> None:
        import time

        self.tokens: int = 0
        self.cost_usd: float = 0.0
        self._window_start: float = time.monotonic()

    def reset(self) -> None:
        """Reset usage counters and start a new window."""
        import time

        self.tokens = 0
        self.cost_usd = 0.0
        self._window_start = time.monotonic()

    def is_window_expired(self, window_seconds: int) -> bool:
        """Return True if the current window has elapsed.

        :param window_seconds: Window duration in seconds.
        :type window_seconds: int
        :rtype: bool
        """
        import time

        return time.monotonic() - self._window_start >= window_seconds


class BudgetStore:
    """Protocol for token budget state storage.

    Implementations must track per-key usage within sliding time windows.
    """

    async def get_usage(self, key: str, *, window_seconds: int = 3600) -> BudgetUsage:
        """Return current usage for *key* within the window.

        :param key: Budget key (e.g. user ID, IP address).
        :type key: str
        :param window_seconds: Window duration in seconds.
        :type window_seconds: int
        :rtype: BudgetUsage
        """
        raise NotImplementedError

    async def record_usage(
        self, key: str, *, tokens: int = 0, cost_usd: float = 0.0, window_seconds: int = 3600
    ) -> None:
        """Record token and cost usage for *key*.

        :param key: Budget key.
        :type key: str
        :param tokens: Tokens to add to the running total.
        :type tokens: int
        :param cost_usd: Cost in USD to add to the running total.
        :type cost_usd: float
        :param window_seconds: Window duration in seconds.
        :type window_seconds: int
        """
        raise NotImplementedError


class InMemoryBudgetStore(BudgetStore):
    """In-memory budget store for single-process deployments."""

    def __init__(self) -> None:
        from collections import defaultdict

        self._usage: dict[str, BudgetUsage] = defaultdict(BudgetUsage)

    async def get_usage(self, key: str, *, window_seconds: int = 3600) -> BudgetUsage:
        """Return current usage, resetting expired windows.

        :param key: Budget key.
        :type key: str
        :param window_seconds: Window duration in seconds.
        :type window_seconds: int
        :rtype: BudgetUsage
        """
        usage = self._usage[key]
        if usage.is_window_expired(window_seconds):
            usage.reset()
        return usage

    async def record_usage(
        self, key: str, *, tokens: int = 0, cost_usd: float = 0.0, window_seconds: int = 3600
    ) -> None:
        """Add to usage counters for *key*.

        :param key: Budget key.
        :type key: str
        :param tokens: Tokens to add.
        :type tokens: int
        :param cost_usd: Cost in USD to add.
        :type cost_usd: float
        :param window_seconds: Window duration in seconds.
        :type window_seconds: int
        """
        usage = await self.get_usage(key, window_seconds=window_seconds)
        usage.tokens += tokens
        usage.cost_usd += cost_usd


# ---------------------------------------------------------------------------
# Token budget guard
# ---------------------------------------------------------------------------


def token_budget_guard(
    *,
    max_tokens_per_hour: int,
    max_cost_usd_per_hour: float | None = None,
    key_fn: Callable[[Any], str] | None = None,
    store: BudgetStore | None = None,
    window_seconds: int = 3600,
) -> type:
    """Return a guard that rejects requests when a per-key token budget is exceeded.

    The guard tracks usage per key (default: client IP) within a rolling window.
    When the budget is exceeded, it raises :class:`~lauren_ai.AgentBudgetExceededError`,
    which the framework maps to an HTTP 429 response.

    :param max_tokens_per_hour: Maximum tokens allowed per key per hour.
    :type max_tokens_per_hour: int
    :param max_cost_usd_per_hour: Maximum USD cost allowed per key per hour.
                                   ``None`` disables cost tracking.
    :type max_cost_usd_per_hour: float | None
    :param key_fn: Function that extracts the budget key from an
                   :class:`~lauren.types.ExecutionContext`. Defaults to client IP.
    :type key_fn: Callable[[ExecutionContext], str] | None
    :param store: Budget state backend. Defaults to :class:`InMemoryBudgetStore`.
    :type store: BudgetStore | None
    :param window_seconds: Window duration in seconds (default: 3600 = 1 hour).
    :type window_seconds: int
    :returns: A guard class suitable for ``@use_guards()`` or ``global_guards=``.
    :rtype: type
    """
    _store = store or InMemoryBudgetStore()
    _key_fn = key_fn or _default_ctx_key_fn
    _window = window_seconds

    try:
        from lauren import Scope, injectable

        HAS_LAUREN = True
    except ImportError:
        HAS_LAUREN = False

    class _TokenBudgetGuard:
        """Guard that enforces per-key token and cost budgets."""

        async def can_activate(self, ctx: Any) -> bool:
            """Check whether the request is within the token budget.

            :param ctx: The execution context.
            :returns: ``True`` if the budget allows the request.
            :raises AgentBudgetExceededError: If the budget is exceeded.
            """
            from ._exceptions import AgentBudgetExceededError

            key = _key_fn(ctx)
            usage = await _store.get_usage(key, window_seconds=_window)

            if usage.tokens >= max_tokens_per_hour:
                raise AgentBudgetExceededError(
                    f"Token budget exhausted for key '{key}': "
                    f"{usage.tokens}/{max_tokens_per_hour} tokens used this hour.",
                    budget_type="tokens",
                    used=usage.tokens,
                    limit=max_tokens_per_hour,
                )

            if max_cost_usd_per_hour is not None and usage.cost_usd >= max_cost_usd_per_hour:
                raise AgentBudgetExceededError(
                    f"Cost budget exhausted for key '{key}': "
                    f"${usage.cost_usd:.4f}/${max_cost_usd_per_hour:.4f} used this hour.",
                    budget_type="cost",
                    used=usage.cost_usd,
                    limit=max_cost_usd_per_hour,
                )

            return True

    if HAS_LAUREN:
        try:
            from lauren import Scope, injectable

            _TokenBudgetGuard = injectable(scope=Scope.SINGLETON)(_TokenBudgetGuard)
        except Exception:
            pass

    return _TokenBudgetGuard


# ---------------------------------------------------------------------------
# Model capability guard
# ---------------------------------------------------------------------------

# Known capabilities per model family
_MODEL_CAPABILITIES: dict[str, frozenset[str]] = {
    "claude": frozenset({"tool_use", "vision", "streaming", "extended_thinking"}),
    "claude-haiku": frozenset({"tool_use", "streaming"}),
    "gpt-4o": frozenset({"tool_use", "vision", "streaming"}),
    "gpt-4o-mini": frozenset({"tool_use", "streaming"}),
    "o1": frozenset({"reasoning", "tool_use"}),
    "o3": frozenset({"reasoning", "tool_use"}),
    "llama": frozenset({"streaming"}),
    "gemma": frozenset({"streaming"}),
}


def _get_model_capabilities(model: str) -> frozenset[str]:
    """Return known capabilities for a model string.

    :param model: Model identifier string.
    :type model: str
    :returns: Set of capability strings.
    :rtype: frozenset[str]
    """
    model_lower = model.lower()
    for prefix, caps in _MODEL_CAPABILITIES.items():
        if prefix in model_lower:
            return caps
    # Unknown model — return empty set (guard will block)
    return frozenset()


def requires_capability(*capabilities: str) -> type:
    """Return a guard that rejects requests when the LLM model lacks specified capabilities.

    The guard inspects the model name from the application's :class:`~lauren_ai.LLMConfig`
    and compares it against a known capability table.

    :param capabilities: Capability strings to require, e.g. ``"tool_use"``, ``"vision"``.
    :type capabilities: str
    :returns: A guard class.
    :rtype: type
    :raises ValueError: If no capabilities are specified.
    """
    if not capabilities:
        raise ValueError("At least one capability must be specified.")

    required = frozenset(capabilities)

    class _CapabilityGuard:
        """Guard that checks model capabilities against the LLM config."""

        def __init__(self, llm_config: Any = None) -> None:
            self._config = llm_config

        async def can_activate(self, ctx: Any) -> bool:
            """Check that the configured model supports all required capabilities.

            :param ctx: The execution context.
            :returns: ``True`` if all capabilities are supported.
            :raises AgentConfigError: If the model lacks a required capability.
            """
            from ._exceptions import AgentConfigError

            model = ""
            if self._config and hasattr(self._config, "model"):
                model = self._config.model

            available = _get_model_capabilities(model)

            missing = required - available
            if missing:
                raise AgentConfigError(
                    f"Model '{model}' does not support required capabilities: "
                    f"{', '.join(sorted(missing))}. "
                    f"Available: {', '.join(sorted(available)) or 'unknown'}."
                )

            return True

    return _CapabilityGuard


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------


class SafetyPolicy:
    """Defines content safety rules for the :func:`safety_guard`.

    Override :meth:`is_safe` to implement custom safety logic.

    :param blocked_keywords: List of exact keywords that trigger a block.
    :type blocked_keywords: list[str]
    :param blocked_patterns: List of regex patterns that trigger a block.
    :type blocked_patterns: list[str]
    """

    def __init__(
        self,
        *,
        blocked_keywords: list[str] | None = None,
        blocked_patterns: list[str] | None = None,
    ) -> None:
        self.blocked_keywords = [kw.lower() for kw in (blocked_keywords or [])]
        self.blocked_patterns = blocked_patterns or []
        self._compiled: list[Any] = []
        if self.blocked_patterns:
            import re

            self._compiled = [re.compile(p, re.IGNORECASE) for p in self.blocked_patterns]

    def is_safe(self, text: str) -> bool:
        """Return ``True`` if *text* passes all safety checks.

        :param text: The text to evaluate.
        :type text: str
        :rtype: bool
        """
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.blocked_keywords):
            return False
        return not any(pat.search(text) for pat in self._compiled)


def safety_guard(
    *,
    policy: SafetyPolicy,
    model: str = "claude-haiku-4-5",
    on_violation: Literal["block", "log"] = "block",
) -> type:
    """Return a guard that runs incoming requests through a safety policy.

    For simple keyword/regex policies, evaluation is done locally.
    For LLM-based evaluation, configure via ``policy.is_safe`` override.

    :param policy: The :class:`SafetyPolicy` to apply.
    :type policy: SafetyPolicy
    :param model: Model to use for LLM-based safety screening (when applicable).
    :type model: str
    :param on_violation: Action on policy violation. ``"block"`` returns 403;
                         ``"log"`` allows the request but logs the violation.
    :type on_violation: Literal["block", "log"]
    :returns: A guard class.
    :rtype: type
    """

    class _SafetyGuard:
        """Guard that screens request content against the safety policy."""

        async def can_activate(self, ctx: Any) -> bool:
            """Evaluate the request body against the safety policy.

            :param ctx: The execution context.
            :returns: ``True`` if the request passes safety checks.
            :raises PermissionError: If ``on_violation="block"`` and the check fails.
            """
            import logging

            logger = logging.getLogger("lauren_ai.safety_guard")

            # Extract text content from the request
            text = ""
            try:
                request = getattr(ctx, "request", None)
                if request and hasattr(request, "state"):
                    # Check if body has been parsed into state
                    body = getattr(request.state, "_parsed_body", None)
                    if body and isinstance(body, dict):
                        text = " ".join(str(v) for v in body.values())
            except Exception:
                pass

            if text and not policy.is_safe(text):
                if on_violation == "block":
                    raise PermissionError(
                        "Request blocked by safety policy. The content violates usage guidelines."
                    )
                else:
                    logger.warning(
                        "Safety policy violation detected (on_violation=log). "
                        "Request allowed through."
                    )

            return True

    return _SafetyGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_ctx_key_fn(ctx: Any) -> str:
    """Extract rate-limit key from ExecutionContext (defaults to client IP)."""
    try:
        request = getattr(ctx, "request", None)
        if request:
            client = getattr(request, "client", None)
            if client and hasattr(client, "host"):
                return str(client.host)
    except Exception:
        pass
    return "anonymous"

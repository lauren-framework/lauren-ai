from __future__ import annotations

"""TokenBudget -- enforce token/cost limits per conversation."""

from dataclasses import dataclass

from lauren_ai._exceptions import LaurenAIError


class BudgetExceededError(LaurenAIError):
    """Raised before an LLM call that would exceed the configured budget.

    :param message: Human-readable description of the exceeded limit.
    :type message: str
    :param limit_type: Category of limit (e.g. ``"tokens_per_conversation"``).
    :type limit_type: str
    :param limit: The configured budget ceiling.
    :type limit: float
    :param current: The actual usage at the point the budget was exceeded.
        Also available as :attr:`used` for API compatibility.
    :type current: float
    """

    def __init__(self, message: str, *, limit_type: str, limit: float, current: float) -> None:
        """Initialise the error.

        :param message: Human-readable description.
        :type message: str
        :param limit_type: Budget category identifier.
        :type limit_type: str
        :param limit: Configured limit value.
        :type limit: float
        :param current: Current usage when the limit was hit.
        :type current: float
        """
        super().__init__(message)
        self.limit_type = limit_type
        self.limit = limit
        self.current = current

    @property
    def used(self) -> float:
        """Alias for :attr:`current` — the usage at budget-exceeded time.

        :return: Current usage value.
        :rtype: float
        """
        return self.current


@dataclass
class TokenBudget:
    """Per-conversation and per-user token/cost budget limits.

    Checked BEFORE each LLM call; raises BudgetExceededError if the
    estimated next call would exceed the limit.

    Usage::

        budget = TokenBudget(
            max_tokens_per_conversation=50_000,
            max_usd_per_conversation=0.50,
        )
        config = LLMConfig(..., budget=budget)
    """

    max_tokens_per_conversation: int | None = None
    max_usd_per_conversation: float | None = None
    max_tokens_per_user_per_day: int | None = None

    def check(
        self,
        *,
        conversation_id: str | None = None,
        current_tokens: int = 0,
        current_usd: float = 0.0,
        estimated_tokens: int = 0,
        estimated_usd: float = 0.0,
    ) -> None:
        """Raise BudgetExceededError if projected usage would exceed limits."""
        if self.max_tokens_per_conversation is not None:
            projected = current_tokens + estimated_tokens
            if projected > self.max_tokens_per_conversation:
                raise BudgetExceededError(
                    f"Token budget exceeded for conversation {conversation_id!r}: "
                    f"projected {projected} > limit {self.max_tokens_per_conversation}",
                    limit_type="tokens_per_conversation",
                    limit=self.max_tokens_per_conversation,
                    current=current_tokens,
                )
        if self.max_usd_per_conversation is not None:
            projected_usd = current_usd + estimated_usd
            if projected_usd > self.max_usd_per_conversation:
                raise BudgetExceededError(
                    f"USD budget exceeded for conversation {conversation_id!r}: "
                    f"projected ${projected_usd:.4f} > limit ${self.max_usd_per_conversation:.4f}",
                    limit_type="usd_per_conversation",
                    limit=self.max_usd_per_conversation,
                    current=current_usd,
                )

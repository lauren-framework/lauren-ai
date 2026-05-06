"""Per-model pricing table for cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelPricing:
    """Per-token pricing for a specific model.

    All ``per_1k`` fields express USD cost per **1,000 tokens**.  The
    ``per_m`` aliases expose the same values scaled to per-million tokens
    for backward compatibility.

    :param input_per_1k: USD cost per 1 000 input tokens.
    :type input_per_1k: float
    :param output_per_1k: USD cost per 1 000 output tokens.
    :type output_per_1k: float
    :param cache_read_per_1k: USD cost per 1 000 prompt-cache read tokens.
    :type cache_read_per_1k: float
    """

    # Per-1k fields (canonical — matches spec)
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    cache_read_per_1k: float = 0.0
    # Per-million fields stored separately for backward compat
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    cache_read_per_m: float = 0.0
    cache_write_per_m: float = 0.0

    def __post_init__(self) -> None:
        # Derive per_1k from per_m when per_1k not explicitly provided and
        # per_m is non-zero, so legacy code that passes input_per_m= still
        # works correctly via the estimate() method.
        if self.input_per_m and not self.input_per_1k:
            self.input_per_1k = self.input_per_m / 1_000
        if self.output_per_m and not self.output_per_1k:
            self.output_per_1k = self.output_per_m / 1_000
        if self.cache_read_per_m and not self.cache_read_per_1k:
            self.cache_read_per_1k = self.cache_read_per_m / 1_000
        # Keep per_m in sync for legacy callers
        if self.input_per_1k and not self.input_per_m:
            self.input_per_m = self.input_per_1k * 1_000
        if self.output_per_1k and not self.output_per_m:
            self.output_per_m = self.output_per_1k * 1_000
        if self.cache_read_per_1k and not self.cache_read_per_m:
            self.cache_read_per_m = self.cache_read_per_1k * 1_000


@dataclass
class CostEstimate:
    """USD cost breakdown for a set of token usages.

    :param input_usd: USD cost for input tokens.
    :type input_usd: float
    :param output_usd: USD cost for output tokens.
    :type output_usd: float
    :param cache_read_usd: USD cost for prompt-cache read tokens.
    :type cache_read_usd: float
    :param cache_write_usd: USD cost for prompt-cache write tokens.
    :type cache_write_usd: float
    """

    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_read_usd: float = 0.0
    cache_write_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        """Total USD cost across all token categories.

        :return: Sum of all USD cost fields.
        :rtype: float
        """
        return self.input_usd + self.output_usd + self.cache_read_usd + self.cache_write_usd

    def __add__(self, other: CostEstimate) -> CostEstimate:
        """Add two :class:`CostEstimate` instances together.

        :param other: The other estimate to add.
        :type other: CostEstimate
        :return: A new :class:`CostEstimate` with summed fields.
        :rtype: CostEstimate
        """
        return CostEstimate(
            input_usd=self.input_usd + other.input_usd,
            output_usd=self.output_usd + other.output_usd,
            cache_read_usd=self.cache_read_usd + other.cache_read_usd,
            cache_write_usd=self.cache_write_usd + other.cache_write_usd,
        )

    @classmethod
    def from_usage(
        cls,
        usage: Any,
        model: str,
        pricing: PricingTable,
    ) -> CostEstimate:
        """Construct a :class:`CostEstimate` from a :class:`~lauren_ai._transport.TokenUsage`.

        Looks up *model* in *pricing*; returns a zero-cost estimate when the
        model is not found in the table.

        :param usage: Token usage statistics (must have ``input_tokens`` and
            ``output_tokens`` integer attributes).
        :type usage: Any
        :param model: Model identifier to look up in *pricing*.
        :type model: str
        :param pricing: The pricing table to use for cost calculation.
        :type pricing: PricingTable
        :return: A :class:`CostEstimate` for the given usage.
        :rtype: CostEstimate
        """
        return pricing.estimate(model, usage)


class PricingTable:
    """Mapping of model name to :class:`ModelPricing` for cost estimation.

    Usage::

        table = PricingTable(models={
            "claude-haiku-4-5": ModelPricing(input_per_m=0.80, output_per_m=4.00),
        })
        estimate = table.estimate("claude-haiku-4-5", usage)

    :param models: Mapping of model identifier to :class:`ModelPricing`.
    :type models: dict[str, ModelPricing] | None
    """

    def __init__(self, models: dict[str, ModelPricing] | None = None) -> None:
        """Initialise with an optional model-pricing mapping.

        :param models: Model pricing entries.
        :type models: dict[str, ModelPricing] | None
        """
        self._models = models or {}

    @classmethod
    def default(cls) -> PricingTable:
        """Return the built-in default pricing table.

        :return: A :class:`PricingTable` populated with known model prices.
        :rtype: PricingTable
        """
        return default_pricing_table()

    def price_for(self, model: str) -> ModelPricing | None:
        """Return the :class:`ModelPricing` for *model*, or ``None`` when the
        model is not in the table.

        :param model: Model identifier.
        :type model: str
        :return: The pricing entry, or ``None``.
        :rtype: ModelPricing | None
        """
        return self._models.get(model)

    def estimate(self, model: str, usage: Any) -> CostEstimate:
        """Estimate cost for *usage* under *model*.

        :param model: Model identifier.
        :type model: str
        :param usage: Token usage object (must expose ``input_tokens`` and
            ``output_tokens`` integer attributes).
        :type usage: Any
        :return: USD cost breakdown.
        :rtype: CostEstimate
        """
        pricing = self._models.get(model)
        if pricing is None:
            return CostEstimate()
        return CostEstimate(
            input_usd=usage.input_tokens * pricing.input_per_m / 1_000_000,
            output_usd=usage.output_tokens * pricing.output_per_m / 1_000_000,
            cache_read_usd=getattr(usage, "cache_read_tokens", 0)
            * pricing.cache_read_per_m
            / 1_000_000,
            cache_write_usd=getattr(usage, "cache_write_tokens", 0)
            * pricing.cache_write_per_m
            / 1_000_000,
        )

    def get(self, model: str) -> ModelPricing | None:
        """Alias for :meth:`price_for`.

        :param model: Model identifier.
        :type model: str
        :return: The pricing entry, or ``None``.
        :rtype: ModelPricing | None
        """
        return self._models.get(model)

    def __contains__(self, model: str) -> bool:
        return model in self._models


def default_pricing_table() -> PricingTable:
    """Return the built-in pricing table with current model prices."""
    return PricingTable(
        models={
            # Anthropic Claude
            "claude-haiku-4-5": ModelPricing(
                input_per_m=0.80,
                output_per_m=4.00,
                cache_read_per_m=0.08,
                cache_write_per_m=1.00,
            ),
            "claude-haiku-4-5-20251001": ModelPricing(
                input_per_m=0.80,
                output_per_m=4.00,
                cache_read_per_m=0.08,
                cache_write_per_m=1.00,
            ),
            "claude-sonnet-4-6": ModelPricing(
                input_per_m=3.00,
                output_per_m=15.00,
                cache_read_per_m=0.30,
                cache_write_per_m=3.75,
            ),
            "claude-opus-4-6": ModelPricing(
                input_per_m=15.00,
                output_per_m=75.00,
                cache_read_per_m=1.50,
                cache_write_per_m=18.75,
            ),
            # OpenAI
            "gpt-4o": ModelPricing(input_per_m=5.00, output_per_m=15.00),
            "gpt-4o-mini": ModelPricing(input_per_m=0.15, output_per_m=0.60),
            "gpt-4-turbo": ModelPricing(input_per_m=10.00, output_per_m=30.00),
            "o1": ModelPricing(input_per_m=15.00, output_per_m=60.00),
            "o3-mini": ModelPricing(input_per_m=1.10, output_per_m=4.40),
        }
    )

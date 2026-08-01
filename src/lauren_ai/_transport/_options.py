"""Request options and provider capability helpers.

The common transport intentionally exposes only stable cross-provider fields.
This module contains the bounded escape hatch for provider-specific request
options and SDK transport controls. The objects are immutable at the public
boundary so a request cannot mutate module-level defaults while concurrent
calls are in flight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "RequestOptions",
    "OpenAIRequestOptions",
    "AnthropicRequestOptions",
    "ProviderCapabilities",
    "redact_mapping",
]

_SENSITIVE_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
        "webhook",
    }
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a recursively redacted diagnostic copy of value.

    This helper is deliberately conservative: keys containing common
    credential markers are replaced even when they are vendor-specific, while
    non-sensitive request options remain available for debugging.
    """

    if value is None:
        return {}

    def _redact(item: Any, *, key: object | None = None) -> Any:
        if key is not None and _is_sensitive_key(key):
            return "[REDACTED]"
        if isinstance(item, Mapping):
            return {str(k): _redact(v, key=k) for k, v in item.items()}
        if isinstance(item, list):
            return [_redact(v) for v in item]
        if isinstance(item, tuple):
            return tuple(_redact(v) for v in item)
        return item

    return _redact(value)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return MappingProxyType(dict(value))


def _merge_maps(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if base is None and override is None:
        return None
    merged: dict[str, Any] = {}
    if base:
        merged.update(base)
    if override:
        merged.update(override)
    return MappingProxyType(merged)


@dataclass(frozen=True, slots=True)
class RequestOptions:
    """Per-client or per-call provider request controls.

    extra_headers, extra_query and extra_body mirror the official SDK escape
    hatches. provider is reserved for known, provider-specific request fields
    such as OpenAI reasoning_effort or Anthropic thinking. A transport may
    validate and translate those fields without expanding the common protocol
    for every provider release.

    timeout and max_retries are per-call overrides. They are accepted by the
    common API but are only applied by transports whose underlying SDK supports
    request-level overrides.
    """

    extra_headers: Mapping[str, str] | None = None
    extra_query: Mapping[str, Any] | None = None
    extra_body: Mapping[str, Any] | None = None
    provider: Mapping[str, Any] | None = None
    timeout: float | None = None
    max_retries: int | None = None
    include_raw_response: bool = False

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("RequestOptions.timeout must be greater than zero")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("RequestOptions.max_retries must be non-negative")
        if self.extra_headers is not None:
            for key, value in self.extra_headers.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise TypeError("RequestOptions.extra_headers must map strings to strings")
                if "\r" in value or "\n" in value:
                    raise ValueError("RequestOptions.extra_headers values must not contain newlines")
        object.__setattr__(self, "extra_headers", _freeze_mapping(self.extra_headers))
        object.__setattr__(self, "extra_query", _freeze_mapping(self.extra_query))
        object.__setattr__(self, "extra_body", _freeze_mapping(self.extra_body))
        object.__setattr__(self, "provider", _freeze_mapping(self.provider))

    def merged(self, override: RequestOptions | None) -> RequestOptions:
        """Return a new options object with override taking precedence."""

        if override is None:
            return self
        return RequestOptions(
            extra_headers=_merge_maps(self.extra_headers, override.extra_headers),
            extra_query=_merge_maps(self.extra_query, override.extra_query),
            extra_body=_merge_maps(self.extra_body, override.extra_body),
            provider=_merge_maps(self.provider, override.provider),
            timeout=override.timeout if override.timeout is not None else self.timeout,
            max_retries=override.max_retries if override.max_retries is not None else self.max_retries,
            include_raw_response=self.include_raw_response or override.include_raw_response,
        )

    def with_call_extras(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> RequestOptions:
        """Return options with convenience per-call maps merged into this set."""

        return self.merged(
            RequestOptions(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
            )
        )

    def as_diagnostic(self) -> dict[str, Any]:
        """Return a safe representation suitable for logs and signals."""

        return {
            "extra_headers": redact_mapping(self.extra_headers),
            "extra_query": redact_mapping(self.extra_query),
            "extra_body": redact_mapping(self.extra_body),
            "provider": redact_mapping(self.provider),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "include_raw_response": self.include_raw_response,
        }


@dataclass(frozen=True, slots=True)
class OpenAIRequestOptions:
    """Typed convenience options for high-value OpenAI request fields."""

    reasoning_effort: str | None = None
    response_format: Any | None = None
    parallel_tool_calls: bool | None = None
    metadata: Mapping[str, str] | None = None
    service_tier: str | None = None
    seed: int | None = None
    store: bool | None = None

    def to_request_options(self) -> RequestOptions:
        """Convert the typed values to the common provider namespace."""

        values = {
            key: value
            for key, value in {
                "reasoning_effort": self.reasoning_effort,
                "response_format": self.response_format,
                "parallel_tool_calls": self.parallel_tool_calls,
                "metadata": dict(self.metadata) if self.metadata is not None else None,
                "service_tier": self.service_tier,
                "seed": self.seed,
                "store": self.store,
            }.items()
            if value is not None
        }
        return RequestOptions(provider=values)


@dataclass(frozen=True, slots=True)
class AnthropicRequestOptions:
    """Typed convenience options for high-value Anthropic request fields."""

    thinking: Mapping[str, Any] | None = None
    top_k: int | None = None
    top_p: float | None = None
    metadata: Mapping[str, Any] | None = None
    service_tier: str | None = None
    output_config: Mapping[str, Any] | None = None
    cache_control: Mapping[str, Any] | None = None

    def to_request_options(self) -> RequestOptions:
        """Convert the typed values to the common provider namespace."""

        values = {
            key: value
            for key, value in {
                "thinking": dict(self.thinking) if self.thinking is not None else None,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "metadata": dict(self.metadata) if self.metadata is not None else None,
                "service_tier": self.service_tier,
                "output_config": dict(self.output_config) if self.output_config is not None else None,
                "cache_control": dict(self.cache_control) if self.cache_control is not None else None,
            }.items()
            if value is not None
        }
        return RequestOptions(provider=values)


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Static capability metadata for a provider/model configuration."""

    provider: str
    model: str
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_structured_output: bool = False
    supports_audio: bool = False
    supports_documents: bool = False
    supports_embeddings: bool = False
    supports_responses: bool = False
    supports_realtime: bool = False

    def supports(self, capability: str) -> bool:
        """Return whether a named capability is enabled."""

        field_name = f"supports_{capability}"
        value = getattr(self, field_name, None)
        if value is None:
            raise ValueError(f"Unknown provider capability: {capability!r}")
        return bool(value)

"""Sampling-parameter adaptation for ``anthropic`` SDK 1.x and 2.x (no SDK needed).

The ``anthropic`` SDK changed incompatibly twice:

* **1.0** removed ``temperature``, ``top_p`` and ``top_k`` from
  ``messages.create`` / ``messages.stream`` (passing one is a ``TypeError``) and
  moved its HTTP layer from ``httpx`` to ``httpx2``.  It still sends API version
  ``2023-06-01``, which accepts the parameters through ``extra_body``.
* **2.0** sends API version ``2026-10-21``, which rejects the parameters with
  HTTP 400 even at their default value.

These tests exercise the adaptation logic with plain callables.  The tests in
``test_anthropic_sdk_contract.py`` drive the real SDK instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from lauren_ai import LLMConfig
from lauren_ai._transport import Message
from lauren_ai._transport import _anthropic as anthropic_transport
from lauren_ai._transport._anthropic import (
    AnthropicTransport,
    _adapt_sampling_params,
    _api_rejects_sampling,
    _undeclared_sampling_params,
)


@pytest.fixture(autouse=True)
def _reset_dropped_warnings() -> Iterator[None]:
    anthropic_transport._dropped_sampling_warned.clear()
    yield
    anthropic_transport._dropped_sampling_warned.clear()


# ── Helpers over plain callables (no SDK needed) ─────────────────────────────


def _declares_all(*, model: str, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None: ...


def _declares_none(*, model: str, max_tokens: int) -> None: ...


def _takes_kwargs(**kwargs: object) -> None: ...


def _client(api_version: str | None) -> SimpleNamespace:
    headers = {} if api_version is None else {"anthropic-version": api_version}
    return SimpleNamespace(default_headers=headers)


class TestUndeclaredSamplingParams:
    def test_sdk_0x_declares_every_parameter(self) -> None:
        assert _undeclared_sampling_params(_declares_all) == frozenset()

    def test_sdk_1x_declares_none(self) -> None:
        assert _undeclared_sampling_params(_declares_none) == {"temperature", "top_p", "top_k"}

    def test_partial_declaration(self) -> None:
        def method(*, temperature: float = 1.0) -> None: ...

        assert _undeclared_sampling_params(method) == {"top_p", "top_k"}

    def test_var_keyword_accepts_everything(self) -> None:
        assert _undeclared_sampling_params(_takes_kwargs) == frozenset()

    def test_uninspectable_callable_is_treated_as_accepting_everything(self) -> None:
        class NoSignature:
            @property
            def __signature__(self) -> object:
                raise ValueError("no signature")

            def __call__(self) -> None: ...

        assert _undeclared_sampling_params(NoSignature()) == frozenset()


class TestApiRejectsSampling:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("2023-06-01", False),
            ("2026-10-20", False),
            ("2026-10-21", True),
            ("2027-01-01", True),
        ],
    )
    def test_compares_the_api_version(self, version: str, expected: bool) -> None:
        assert _api_rejects_sampling(_client(version)) is expected

    def test_header_name_is_case_insensitive(self) -> None:
        client = SimpleNamespace(default_headers={"Anthropic-Version": "2026-10-21"})
        assert _api_rejects_sampling(client) is True

    def test_missing_header_or_headers_means_accepted(self) -> None:
        assert _api_rejects_sampling(_client(None)) is False
        assert _api_rejects_sampling(SimpleNamespace()) is False


class TestAdaptSamplingParams:
    def test_declared_parameters_are_left_alone(self) -> None:
        kwargs: dict[str, Any] = {"model": "m", "temperature": 0.2, "top_p": 0.9}
        _adapt_sampling_params(_client("2023-06-01"), _declares_all, kwargs)
        assert kwargs == {"model": "m", "temperature": 0.2, "top_p": 0.9}

    def test_default_temperature_is_omitted(self) -> None:
        kwargs: dict[str, Any] = {"model": "m", "temperature": 1.0}
        _adapt_sampling_params(_client("2023-06-01"), _declares_none, kwargs)
        assert kwargs == {"model": "m"}

    def test_default_temperature_is_omitted_without_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=anthropic_transport.logger.name):
            _adapt_sampling_params(_client("2026-10-21"), _declares_none, {"temperature": 1.0})
        assert caplog.records == []

    def test_non_default_values_are_forwarded_when_the_api_accepts_them(self) -> None:
        kwargs: dict[str, Any] = {"model": "m", "temperature": 0.2, "top_p": 0.9, "top_k": 5}
        _adapt_sampling_params(_client("2023-06-01"), _declares_none, kwargs)
        assert kwargs == {"model": "m", "extra_body": {"temperature": 0.2, "top_p": 0.9, "top_k": 5}}

    def test_forwarding_merges_into_an_existing_extra_body(self) -> None:
        kwargs: dict[str, Any] = {"temperature": 0.0, "extra_body": {"metadata": "x"}}
        _adapt_sampling_params(_client("2023-06-01"), _declares_none, kwargs)
        assert kwargs == {"extra_body": {"metadata": "x", "temperature": 0.0}}

    def test_zero_temperature_is_not_mistaken_for_the_default(self) -> None:
        kwargs: dict[str, Any] = {"temperature": 0.0}
        _adapt_sampling_params(_client("2023-06-01"), _declares_none, kwargs)
        assert kwargs == {"extra_body": {"temperature": 0.0}}

    def test_explicit_top_p_is_always_honoured_even_at_one(self) -> None:
        # The API documents no default for top_p, so an explicit value is kept.
        kwargs: dict[str, Any] = {"top_p": 1.0}
        _adapt_sampling_params(_client("2023-06-01"), _declares_none, kwargs)
        assert kwargs == {"extra_body": {"top_p": 1.0}}

    def test_values_are_dropped_when_the_api_rejects_them(self) -> None:
        kwargs: dict[str, Any] = {"model": "m", "temperature": 0.2, "top_p": 0.9, "top_k": 5}
        _adapt_sampling_params(_client("2026-10-21"), _declares_none, kwargs)
        assert kwargs == {"model": "m"}

    def test_dropped_values_warn_once_per_parameter(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=anthropic_transport.logger.name):
            for _ in range(3):
                _adapt_sampling_params(_client("2026-10-21"), _declares_none, {"temperature": 0.2})
            _adapt_sampling_params(_client("2026-10-21"), _declares_none, {"top_p": 0.9})
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 2
        assert "temperature=0.2" in messages[0]
        assert "top_p=0.9" in messages[1]
        assert "2023-06-01" in messages[0]  # tells the user how to opt back in

    def test_untouched_when_no_sampling_parameters_are_present(self) -> None:
        kwargs: dict[str, Any] = {"model": "m", "max_tokens": 5}
        _adapt_sampling_params(_client("2026-10-21"), _declares_none, kwargs)
        assert kwargs == {"model": "m", "max_tokens": 5}

    def test_unrelated_keys_survive(self) -> None:
        kwargs: dict[str, Any] = {"model": "m", "temperature": 0.2, "timeout": 3, "extra_headers": {"a": "b"}}
        _adapt_sampling_params(_client("2026-10-21"), _declares_none, kwargs)
        assert kwargs == {"model": "m", "timeout": 3, "extra_headers": {"a": "b"}}


class TestTransportWithFakeClients:
    """A fake whose ``create`` takes ``**kwargs`` keeps receiving every parameter."""

    async def test_fake_client_still_receives_temperature(self) -> None:
        seen: list[dict[str, Any]] = []

        async def create(**kwargs: Any) -> Any:
            seen.append(kwargs)
            return SimpleNamespace(
                id="m",
                model="x",
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

        client = SimpleNamespace(
            messages=SimpleNamespace(create=create), default_headers={"anthropic-version": "2026-10-21"}
        )
        config = LLMConfig.for_anthropic(model="claude-x", api_key="k", max_retries=0)
        transport = AnthropicTransport(config, client=client)
        await transport.complete([Message(role="user", content="hi")], model="claude-x", temperature=0.2)
        assert seen[0]["temperature"] == 0.2

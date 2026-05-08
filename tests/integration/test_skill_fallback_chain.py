"""Integration tests for the FallbackChain pattern (Skill 31).

Tests cover:
- First provider succeeds — returns its result
- First provider fails, second succeeds — returns second result
- All providers fail — returns fallback_response
- provider_index reflects which provider was used
- Custom fallback_response is returned on total failure
"""


# ---------------------------------------------------------------------------
# FallbackChain implementation
# ---------------------------------------------------------------------------


class FallbackChain:
    """Tries providers in order, returning first success."""

    def __init__(
        self,
        providers: list,
        fallback_response: str = "I'm sorry, I cannot process this request right now.",
    ):
        self._providers = providers
        self._fallback = fallback_response

    async def execute(self, prompt: str) -> dict:
        last_error = None
        for i, provider in enumerate(self._providers):
            try:
                result = await provider(prompt)
                return {"content": result, "provider_index": i, "success": True}
            except Exception as e:
                last_error = e
                continue
        return {
            "content": self._fallback,
            "provider_index": -1,
            "success": False,
            "error": str(last_error) if last_error else "",
        }


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

_FALLBACK_MSG = "I'm sorry, I cannot process this request right now."
_CUSTOM_FALLBACK = "Service temporarily unavailable."


async def _provider_ok(prompt: str) -> str:
    return f"primary: {prompt}"


async def _provider_secondary(prompt: str) -> str:
    return f"secondary: {prompt}"


async def _provider_fail(prompt: str) -> str:
    raise RuntimeError("Provider unavailable")


async def _provider_p1_fail(prompt: str) -> str:
    raise RuntimeError("p1 failed")


async def _provider_p2_fail(prompt: str) -> str:
    raise RuntimeError("p2 failed")


async def _provider_p3_ok(prompt: str) -> str:
    return "p3 result"


async def _provider_val_fail(prompt: str) -> str:
    raise ValueError("p2 error")


async def _provider_val_fail_1(prompt: str) -> str:
    raise RuntimeError("first failure")


async def _provider_val_fail_2(prompt: str) -> str:
    raise RuntimeError("second failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFallbackChainFirstProviderSucceeds:
    async def test_first_provider_returns_result(self):
        """When the first provider succeeds, its result is returned."""
        chain = FallbackChain(providers=[_provider_ok, _provider_secondary])
        data = await chain.execute("hello")
        assert data["success"] is True
        assert data["content"] == "primary: hello"
        assert data["provider_index"] == 0

    async def test_first_provider_used_when_both_work(self):
        """provider_index is 0 when the first provider works."""
        chain = FallbackChain(providers=[_provider_ok, _provider_secondary])
        data = await chain.execute("test")
        assert data["provider_index"] == 0


class TestFallbackChainSecondaryFallback:
    async def test_second_provider_used_on_primary_failure(self):
        """When the first provider raises, the second provider is tried."""
        chain = FallbackChain(providers=[_provider_fail, _provider_secondary])
        data = await chain.execute("query")
        assert data["success"] is True
        assert data["content"] == "secondary: query"
        assert data["provider_index"] == 1

    async def test_third_provider_used_when_first_two_fail(self):
        """Falls through to the third provider when the first two raise."""
        chain = FallbackChain(providers=[_provider_p1_fail, _provider_p2_fail, _provider_p3_ok])
        data = await chain.execute("x")
        assert data["success"] is True
        assert data["content"] == "p3 result"
        assert data["provider_index"] == 2


class TestFallbackChainAllFail:
    async def test_fallback_response_returned_when_all_providers_fail(self):
        """Returns the fallback_response string when every provider raises."""
        chain = FallbackChain(
            providers=[_provider_p1_fail, _provider_val_fail],
            fallback_response=_CUSTOM_FALLBACK,
        )
        data = await chain.execute("anything")
        assert data["success"] is False
        assert data["content"] == _CUSTOM_FALLBACK
        assert data["provider_index"] == -1

    async def test_error_field_contains_last_exception_message(self):
        """The error field contains the last provider's exception message."""
        chain = FallbackChain(providers=[_provider_val_fail_1, _provider_val_fail_2])
        data = await chain.execute("test")
        assert "second failure" in data["error"]

    async def test_default_fallback_message(self):
        """The default fallback message is returned when not overridden."""
        chain = FallbackChain(providers=[_provider_fail])
        data = await chain.execute("query")
        assert "cannot process" in data["content"].lower()

    async def test_empty_providers_returns_fallback(self):
        """An empty providers list immediately returns the fallback response."""
        chain = FallbackChain(providers=[], fallback_response="no providers")
        data = await chain.execute("x")
        assert data["success"] is False
        assert data["content"] == "no providers"
        assert data["provider_index"] == -1


class TestFallbackChainPromptPropagation:
    async def test_prompt_forwarded_to_provider(self):
        """The original prompt is passed unchanged to the provider."""
        chain = FallbackChain(providers=[_provider_ok])
        data = await chain.execute("my specific prompt")
        assert data["content"] == "primary: my specific prompt"

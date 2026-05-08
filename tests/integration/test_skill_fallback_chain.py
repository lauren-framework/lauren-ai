"""Integration tests for the FallbackChain pattern (Skill 31).

Tests cover:
- First provider succeeds — returns its result
- First provider fails, second succeeds — returns second result
- All providers fail — returns fallback_response
- provider_index reflects which provider was used
- Custom fallback_response is returned on total failure
"""

import pytest


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
            "error": str(last_error),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFallbackChainFirstProviderSucceeds:
    @pytest.mark.asyncio
    async def test_first_provider_returns_result(self):
        """When the first provider succeeds, its result is returned."""

        async def primary(prompt: str) -> str:
            return f"primary: {prompt}"

        async def secondary(prompt: str) -> str:
            return f"secondary: {prompt}"

        chain = FallbackChain(providers=[primary, secondary])
        result = await chain.execute("hello")

        assert result["success"] is True
        assert result["content"] == "primary: hello"
        assert result["provider_index"] == 0

    @pytest.mark.asyncio
    async def test_first_provider_used_when_both_work(self):
        """provider_index is 0 when the first provider works."""
        called = []

        async def primary(prompt: str) -> str:
            called.append("primary")
            return "from primary"

        async def secondary(prompt: str) -> str:
            called.append("secondary")
            return "from secondary"

        chain = FallbackChain(providers=[primary, secondary])
        await chain.execute("test")

        assert called == ["primary"]


class TestFallbackChainSecondaryFallback:
    @pytest.mark.asyncio
    async def test_second_provider_used_on_primary_failure(self):
        """When the first provider raises, the second provider is tried."""

        async def failing_primary(prompt: str) -> str:
            raise ConnectionError("Primary unavailable")

        async def working_secondary(prompt: str) -> str:
            return f"fallback: {prompt}"

        chain = FallbackChain(providers=[failing_primary, working_secondary])
        result = await chain.execute("query")

        assert result["success"] is True
        assert result["content"] == "fallback: query"
        assert result["provider_index"] == 1

    @pytest.mark.asyncio
    async def test_third_provider_used_when_first_two_fail(self):
        """Falls through to the third provider when the first two raise."""
        call_order = []

        async def p1(prompt: str) -> str:
            call_order.append(1)
            raise RuntimeError("p1 failed")

        async def p2(prompt: str) -> str:
            call_order.append(2)
            raise RuntimeError("p2 failed")

        async def p3(prompt: str) -> str:
            call_order.append(3)
            return "p3 result"

        chain = FallbackChain(providers=[p1, p2, p3])
        result = await chain.execute("x")

        assert result["success"] is True
        assert result["content"] == "p3 result"
        assert result["provider_index"] == 2
        assert call_order == [1, 2, 3]


class TestFallbackChainAllFail:
    @pytest.mark.asyncio
    async def test_fallback_response_returned_when_all_providers_fail(self):
        """Returns the fallback_response string when every provider raises."""

        async def p1(prompt: str) -> str:
            raise ValueError("p1 error")

        async def p2(prompt: str) -> str:
            raise ValueError("p2 error")

        chain = FallbackChain(
            providers=[p1, p2],
            fallback_response="Service temporarily unavailable.",
        )
        result = await chain.execute("anything")

        assert result["success"] is False
        assert result["content"] == "Service temporarily unavailable."
        assert result["provider_index"] == -1

    @pytest.mark.asyncio
    async def test_error_field_contains_last_exception_message(self):
        """The error field contains the last provider's exception message."""

        async def p1(prompt: str) -> str:
            raise RuntimeError("first failure")

        async def p2(prompt: str) -> str:
            raise RuntimeError("second failure")

        chain = FallbackChain(providers=[p1, p2])
        result = await chain.execute("test")

        assert "second failure" in result["error"]

    @pytest.mark.asyncio
    async def test_default_fallback_message(self):
        """The default fallback message is returned when not overridden."""

        async def p(prompt: str) -> str:
            raise Exception("all gone")

        chain = FallbackChain(providers=[p])
        result = await chain.execute("query")

        assert "cannot process" in result["content"].lower()

    @pytest.mark.asyncio
    async def test_empty_providers_returns_fallback(self):
        """An empty providers list immediately returns the fallback response."""
        chain = FallbackChain(providers=[], fallback_response="no providers")
        result = await chain.execute("x")

        assert result["success"] is False
        assert result["content"] == "no providers"
        assert result["provider_index"] == -1


class TestFallbackChainPromptPropagation:
    @pytest.mark.asyncio
    async def test_prompt_forwarded_to_provider(self):
        """The original prompt is passed unchanged to the provider."""
        received = []

        async def capture(prompt: str) -> str:
            received.append(prompt)
            return "ok"

        chain = FallbackChain(providers=[capture])
        await chain.execute("my specific prompt")

        assert received == ["my specific prompt"]

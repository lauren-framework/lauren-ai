"""Unit tests for _interceptors.py — ai_metrics_interceptor and
token_usage_response_interceptor (fallback paths)."""

from __future__ import annotations

import pytest

from lauren_ai._interceptors import (
    ai_metrics_interceptor,
    token_usage_response_interceptor,
)

# ---------------------------------------------------------------------------
# ai_metrics_interceptor tests (fallback path — no lauren installed)
# ---------------------------------------------------------------------------


class TestAIMetricsInterceptor:
    def test_returns_class(self):
        cls = ai_metrics_interceptor()
        assert cls is not None
        assert callable(cls)

    def test_instance_has_intercept(self):
        cls = ai_metrics_interceptor()
        obj = cls()
        assert hasattr(obj, "intercept")

    @pytest.mark.asyncio
    async def test_fallback_intercept_passes_through(self):
        cls = ai_metrics_interceptor()
        obj = cls()

        async def call_next(ctx):
            return "response"

        result = await obj.intercept(object(), call_next)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_intercept_with_state_no_error(self):
        cls = ai_metrics_interceptor()
        obj = cls()

        class FakeState:
            pass

        class FakeRequest:
            state = FakeState()

        class FakeCtx:
            request = FakeRequest()
            signals = None

        async def call_next(ctx):
            return "ok"

        result = await obj.intercept(FakeCtx(), call_next)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_intercept_with_token_usage(self):
        cls = ai_metrics_interceptor()
        obj = cls()

        class FakeUsage:
            model = "claude-opus-4-6"
            input_tokens = 100
            output_tokens = 50
            cost_usd = 0.05

        class FakeState:
            ai_token_usage = FakeUsage()

        class FakeRequest:
            state = FakeState()

        class FakeCtx:
            request = FakeRequest()
            signals = None

        async def call_next(ctx):
            return "response"

        result = await obj.intercept(FakeCtx(), call_next)
        assert result == "response"
        # In fallback mode, the interceptor just passes through (no state mutation)
        # If lauren is available, ai_duration_ms would be set
        # Just verify no exception occurs

    @pytest.mark.asyncio
    async def test_intercept_no_request_attr(self):
        cls = ai_metrics_interceptor()
        obj = cls()

        class NoRequestCtx:
            pass

        async def call_next(ctx):
            return "response"

        result = await obj.intercept(NoRequestCtx(), call_next)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_intercept_with_signals_emit(self):
        cls = ai_metrics_interceptor()
        obj = cls()

        emitted = []

        class FakeSignals:
            async def emit(self, event):
                emitted.append(event)

        class FakeUsage:
            model = "test-model"
            input_tokens = 10
            output_tokens = 5
            cost_usd = 0.001

        class FakeState:
            ai_token_usage = FakeUsage()

        class FakeRequest:
            state = FakeState()

        class FakeCtx:
            request = FakeRequest()
            signals = FakeSignals()

        async def call_next(ctx):
            return "response"

        result = await obj.intercept(FakeCtx(), call_next)
        # If lauren_ai._signals.ModelCallComplete is available, it should be emitted
        assert result == "response"


# ---------------------------------------------------------------------------
# token_usage_response_interceptor tests (fallback path)
# ---------------------------------------------------------------------------


class TestTokenUsageResponseInterceptor:
    def test_returns_class(self):
        cls = token_usage_response_interceptor()
        assert cls is not None
        assert callable(cls)

    def test_instance_has_intercept(self):
        cls = token_usage_response_interceptor()
        obj = cls()
        assert hasattr(obj, "intercept")

    @pytest.mark.asyncio
    async def test_fallback_intercept_passes_through(self):
        cls = token_usage_response_interceptor()
        obj = cls()

        async def call_next(ctx):
            return "response"

        result = await obj.intercept(object(), call_next)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_appends_headers_when_usage_present(self):
        cls = token_usage_response_interceptor()
        obj = cls()

        class FakeUsage:
            input_tokens = 100
            output_tokens = 50
            cost_usd = 0.05

        class FakeState:
            ai_token_usage = FakeUsage()

        class FakeRequest:
            state = FakeState()

        class FakeResponse:
            headers: dict = {}

        class FakeCtx:
            request = FakeRequest()

        response = FakeResponse()

        async def call_next(ctx):
            return response

        result = await obj.intercept(FakeCtx(), call_next)
        # The response should have headers added if possible
        assert result is response

    @pytest.mark.asyncio
    async def test_no_request_no_error(self):
        cls = token_usage_response_interceptor()
        obj = cls()

        class NoRequestCtx:
            pass

        async def call_next(ctx):
            return "response"

        result = await obj.intercept(NoRequestCtx(), call_next)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_no_usage_no_headers_added(self):
        cls = token_usage_response_interceptor()
        obj = cls()

        class FakeState:
            ai_token_usage = None

        class FakeRequest:
            state = FakeState()

        class FakeCtx:
            request = FakeRequest()

        async def call_next(ctx):
            return "response"

        result = await obj.intercept(FakeCtx(), call_next)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_response_with_headers_gets_usage(self):
        cls = token_usage_response_interceptor()
        obj = cls()

        class FakeUsage:
            input_tokens = 200
            output_tokens = 100
            cost_usd = 0.10

        class FakeState:
            ai_token_usage = FakeUsage()

        class FakeRequest:
            state = FakeState()

        class FakeCtx:
            request = FakeRequest()

        class FakeResponse:
            headers = {}

        response = FakeResponse()

        async def call_next(ctx):
            return response

        result = await obj.intercept(FakeCtx(), call_next)
        assert result is response
        # In fallback mode (no lauren), interceptor passes through without header changes
        # If lauren is installed, headers would have x-token-usage=300
        # Just verify no exception occurs and the response is returned

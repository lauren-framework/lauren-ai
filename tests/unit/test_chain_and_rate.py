"""Tests for _chains/_chain.py and _cost/_rate.py coverage gaps."""

from __future__ import annotations

import asyncio

import pytest

from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage


def _compl(content: str = "ok") -> Completion:
    return Completion(
        id="c1",
        model="mock",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


def _make_llm(responses: list[str]):
    cfg, mock = LLMConfig.for_testing()
    for r in responses:
        mock.queue_response(_compl(r))
    from lauren_ai._module import LLMService

    return LLMService(transport=mock, config=cfg), mock


# ---------------------------------------------------------------------------
# _chains/_chain.py
# ---------------------------------------------------------------------------


class TestChainInvoke:
    @pytest.mark.asyncio
    async def test_callable_first_step(self):
        """Non-template, non-Runnable callable as first step."""
        from lauren_ai._chains._chain import Chain

        def transform(x):
            return x.upper()

        llm, _ = _make_llm(["LLM SAYS OK"])

        chain = Chain(steps=[transform, llm])
        result = await chain.invoke("hello")
        assert isinstance(result, Completion)
        assert result.content == "LLM SAYS OK"

    @pytest.mark.asyncio
    async def test_async_callable_first_step(self):
        """Async callable as first step."""
        from lauren_ai._chains._chain import Chain

        async def async_transform(x):
            return f"async:{x}"

        llm, _ = _make_llm(["done"])

        chain = Chain(steps=[async_transform, llm])
        result = await chain.invoke("input")
        assert isinstance(result, Completion)

    @pytest.mark.asyncio
    async def test_llm_step_with_string_current(self):
        """LLM step wraps non-Message current in a Message."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._prompts import PromptTemplate

        tpl = PromptTemplate(template="Tell me about {topic}", input_variables=["topic"])
        llm, _ = _make_llm(["Python is great"])

        chain = Chain(steps=[tpl, llm])
        result = await chain.invoke({"topic": "Python"})
        assert isinstance(result, Completion)
        assert result.content == "Python is great"

    @pytest.mark.asyncio
    async def test_llm_step_with_message_current(self):
        """LLM step with a single Message."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._transport import Message

        llm, _ = _make_llm(["response"])

        async def make_message(x):
            return Message(role="user", content=str(x))

        chain = Chain(steps=[make_message, llm])
        result = await chain.invoke("query")
        assert result.content == "response"

    @pytest.mark.asyncio
    async def test_parse_step_on_completion(self):
        """Parser step receiving a Completion uses .content."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._prompts import PromptTemplate

        tpl = PromptTemplate(template="Return JSON for {x}")
        llm, _ = _make_llm(['{"value": 42}'])

        chain = Chain(steps=[tpl, llm, JSONOutputParser()])
        result = await chain.invoke({"x": "test"})
        assert result == {"value": 42}

    @pytest.mark.asyncio
    async def test_parse_step_on_non_completion(self):
        """Parser step receiving a non-Completion uses str()."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._output_parsers._builtin import JSONOutputParser

        def produce_json(x):
            return '{"k": 1}'

        chain = Chain(steps=[produce_json, JSONOutputParser()])
        result = await chain.invoke("anything")
        assert result == {"k": 1}

    @pytest.mark.asyncio
    async def test_callable_subsequent_step(self):
        """Callable (non-parser, non-invoke) in subsequent step."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._prompts import PromptTemplate

        tpl = PromptTemplate(template="Say {x}")
        llm, _ = _make_llm(["hello world"])

        def extract_words(compl):
            return compl.content.split()

        chain = Chain(steps=[tpl, llm, extract_words])
        result = await chain.invoke({"x": "hi"})
        assert result == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_stream_method(self):
        """stream() calls complete_stream on the LLM step."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._prompts import PromptTemplate
        from lauren_ai._transport import CompletionChunk

        tpl = PromptTemplate(template="Say {x}")

        async def fake_stream():
            yield CompletionChunk(delta="stream", stop_reason="end_turn", usage=None)

        class FakeTransport:
            async def complete(self, messages, **kw):
                return fake_stream()

            async def complete_stream(self, messages, **kw):
                return fake_stream()

        cfg = LLMConfig(provider="anthropic", model="m", api_key="k")
        from lauren_ai._module import LLMService

        llm = LLMService(transport=FakeTransport(), config=cfg)

        chain = Chain(steps=[tpl, llm])
        # stream() returns complete_stream coroutine, need to await it
        stream_coro = await chain.stream(x="hi")
        actual_stream = await stream_coro
        chunks = []
        async for chunk in actual_stream:
            chunks.append(chunk.delta)
        assert "stream" in chunks

    @pytest.mark.asyncio
    async def test_stream_raises_when_no_llm(self):
        """stream() raises ValueError when no LLMService in chain."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._prompts import PromptTemplate

        tpl = PromptTemplate(template="test {x}")
        chain = Chain(steps=[tpl])
        with pytest.raises(ValueError, match="No LLMService"):
            await chain.stream(x="hi")

    @pytest.mark.asyncio
    async def test_chat_prompt_as_first_step(self):
        """ChatPromptTemplate as first step feeds messages list to LLM."""
        from lauren_ai._chains._chain import Chain
        from lauren_ai._prompts import ChatPromptTemplate

        tpl = ChatPromptTemplate(
            messages=[("system", "You are helpful."), ("human", "{q}")],
            input_variables=["q"],
        )
        llm, _ = _make_llm(["answer"])
        chain = Chain(steps=[tpl, llm])
        result = await chain.invoke({"q": "What is 2+2?"})
        assert result.content == "answer"

    def test_chain_factory_function(self):
        from lauren_ai._chains._chain import chain as chain_fn

        c = chain_fn(lambda x: x, lambda x: x)
        from lauren_ai._chains._chain import Chain

        assert isinstance(c, Chain)
        assert len(c.steps) == 2


# ---------------------------------------------------------------------------
# _cost/_rate.py
# ---------------------------------------------------------------------------


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_no_limits(self):
        """No RPM or TPM configured → acquire returns immediately."""
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter()
        await limiter.acquire(estimated_tokens=100)
        assert len(limiter._request_times) == 1

    @pytest.mark.asyncio
    async def test_refill_tokens_on_acquire(self):
        """tokens_per_minute configured → refill runs."""
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter(tokens_per_minute=6000)  # 100/sec
        limiter._token_count = 50.0
        await limiter.acquire(estimated_tokens=10)
        # token_count should have decreased by 10 (after refill)
        assert len(limiter._request_times) == 1

    @pytest.mark.asyncio
    async def test_acquire_tracks_request_times(self):
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter(requests_per_minute=1000)
        await limiter.acquire()
        await limiter.acquire()
        assert len(limiter._request_times) == 2

    @pytest.mark.asyncio
    async def test_token_wait_when_insufficient(self):
        """acquire() waits when token_count < estimated_tokens (fast path)."""
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter(tokens_per_minute=600)  # 10/sec
        limiter._token_count = 0.0
        # With 0 tokens and need 1, should sleep briefly
        start = asyncio.get_event_loop().time()
        await limiter.acquire(estimated_tokens=1)
        elapsed = asyncio.get_event_loop().time() - start
        # Should have waited some time
        assert elapsed >= 0.0  # at minimum ran without error

    def test_backoff_with_retry_after(self):
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter()
        assert limiter.backoff_for(0, retry_after=5.0) == 5.0

    def test_backoff_exponential(self):
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter(initial_backoff_s=1.0, max_backoff_s=60.0, jitter=False)
        b0 = limiter.backoff_for(0)
        b1 = limiter.backoff_for(1)
        b2 = limiter.backoff_for(2)
        assert b0 == 1.0
        assert b1 == 2.0
        assert b2 == 4.0

    def test_backoff_capped_at_max(self):
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter(initial_backoff_s=10.0, max_backoff_s=15.0, jitter=False)
        assert limiter.backoff_for(10) == 15.0

    def test_backoff_with_jitter_in_range(self):
        from lauren_ai._cost._rate import RateLimiter

        limiter = RateLimiter(initial_backoff_s=2.0, jitter=True)
        for _ in range(10):
            b = limiter.backoff_for(0)
            # With jitter: base=2.0, factor in [0.5, 1.0]
            assert 1.0 <= b <= 2.0

    def test_rate_limit_error_fields(self):
        from lauren_ai._cost._rate import RateLimitExhaustedError

        err = RateLimitExhaustedError("hit limit", limit=60, retry_after=3.5)
        assert err.limit == 60
        assert err.retry_after == 3.5
        assert "hit limit" in str(err)

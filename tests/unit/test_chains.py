"""Unit tests for Chain, Runnable, RunnableLambda, and the chain() factory."""
from __future__ import annotations

import pytest

from lauren_ai._chains import Chain, Runnable, RunnableLambda, chain
from lauren_ai._chains._chain import _try_get_structured_llm
from lauren_ai._output_parsers import CommaSeparatedListParser, JSONOutputParser, StrOutputParser
from lauren_ai._prompts import ChatPromptTemplate, PromptTemplate
from lauren_ai._transport import Completion, TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(content: str) -> Completion:
    return Completion(
        id="test",
        model="test-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=0, output_tokens=0),
    )


class _EchoRunnable:
    """Simple Runnable that echoes its input back prefixed with 'echo:'."""

    async def invoke(self, input):
        return f"echo:{input}"


class _UpperRunnable:
    """Runnable that upper-cases a string input."""

    async def invoke(self, input):
        return str(input).upper()


class _DoubleRunnable:
    """Runnable that doubles an integer input."""

    async def invoke(self, input):
        return input * 2


# ---------------------------------------------------------------------------
# TestRunnable
# ---------------------------------------------------------------------------


class TestRunnable:
    def test_runnable_protocol_satisfied_by_echo(self):
        assert isinstance(_EchoRunnable(), Runnable)

    def test_runnable_lambda_satisfies_protocol(self):
        r = RunnableLambda(lambda x: x)
        assert isinstance(r, Runnable)

    async def test_echo_runnable_invoke(self):
        r = _EchoRunnable()
        result = await r.invoke("hello")
        assert result == "echo:hello"


# ---------------------------------------------------------------------------
# TestRunnableLambda
# ---------------------------------------------------------------------------


class TestRunnableLambda:
    async def test_sync_lambda_wraps_correctly(self):
        r = RunnableLambda(lambda x: x + 1)
        assert await r.invoke(5) == 6

    async def test_async_lambda_wraps_correctly(self):
        async def double(x):
            return x * 2

        r = RunnableLambda(double)
        assert await r.invoke(4) == 8

    async def test_string_transform(self):
        r = RunnableLambda(str.upper)
        assert await r.invoke("hello") == "HELLO"

    def test_pipe_creates_chain(self):
        r = RunnableLambda(lambda x: x)
        c = r | RunnableLambda(lambda x: x)
        assert isinstance(c, Chain)

    async def test_chained_lambdas(self):
        a = RunnableLambda(lambda x: x + 1)
        b = RunnableLambda(lambda x: x * 3)
        c = a | b
        result = await c.invoke(2)
        assert result == 9  # (2+1)*3


# ---------------------------------------------------------------------------
# TestChain
# ---------------------------------------------------------------------------


class TestChain:
    async def test_single_runnable_step(self):
        c = Chain(steps=[_EchoRunnable()])
        result = await c.invoke("hello")
        assert result == "echo:hello"

    async def test_two_runnable_steps_pipe(self):
        c = Chain(steps=[_EchoRunnable(), _UpperRunnable()])
        result = await c.invoke("hello")
        assert result == "ECHO:HELLO"

    async def test_pipe_operator_appends_step(self):
        c = Chain(steps=[_EchoRunnable()]) | _UpperRunnable()
        assert len(c.steps) == 2
        result = await c.invoke("world")
        assert result == "ECHO:WORLD"

    async def test_ror_operator_prepends_step(self):
        base = Chain(steps=[_UpperRunnable()])
        c = _EchoRunnable() | base
        assert isinstance(c, Chain)
        result = await c.invoke("hi")
        assert result == "ECHO:HI"

    async def test_invoke_with_kwargs(self):
        """Legacy kwargs interface — dict merges into first step."""
        r = RunnableLambda(lambda x: x["value"])
        c = Chain(steps=[r])
        result = await c.invoke({"value": 42})
        assert result == 42

    async def test_invoke_kwargs_compat(self):
        """chain.invoke(key=val) is the legacy template-chain interface."""
        r = RunnableLambda(lambda x: x.get("name", "?") if isinstance(x, dict) else x)
        c = Chain(steps=[r])
        result = await c.invoke(name="Alice")
        assert result == "Alice"

    async def test_three_steps_sequential(self):
        """Each step receives the previous output."""
        a = RunnableLambda(lambda x: x + 10)
        b = RunnableLambda(lambda x: x * 2)
        c_step = RunnableLambda(lambda x: x - 1)
        pipeline = Chain(steps=[a, b, c_step])
        result = await pipeline.invoke(5)
        assert result == (5 + 10) * 2 - 1  # 29

    async def test_chain_with_prompt_template(self):
        tpl = PromptTemplate(template="Hello {name}!")
        upper = RunnableLambda(lambda msg: msg.content.upper())
        pipeline = tpl | upper
        result = await pipeline.invoke({"name": "world"})
        assert result == "HELLO WORLD!"

    async def test_chain_with_chat_prompt_template(self):
        tpl = ChatPromptTemplate(
            messages=[("human", "{q}")],
            input_variables=["q"],
        )
        first_content = RunnableLambda(lambda msgs: msgs[0].content)
        pipeline = tpl | first_content
        result = await pipeline.invoke({"q": "hi"})
        assert result == "hi"

    async def test_chain_with_str_output_parser(self):
        r = RunnableLambda(lambda _: "  hello  ")
        pipeline = r | StrOutputParser()
        result = await pipeline.invoke(None)
        assert result == "hello"

    async def test_chain_with_json_output_parser(self):
        r = RunnableLambda(lambda _: '{"answer": 42}')
        pipeline = r | JSONOutputParser()
        result = await pipeline.invoke(None)
        assert result == {"answer": 42}

    async def test_chain_with_completion_to_str_parser(self):
        """Parser step receives Completion and extracts .content."""
        comp = _make_completion("  trimmed  ")
        r = RunnableLambda(lambda _: comp)
        pipeline = r | StrOutputParser()
        result = await pipeline.invoke(None)
        assert result == "trimmed"

    async def test_chain_steps_list_is_independent(self):
        """Piping does not mutate the original chain."""
        base = Chain(steps=[_EchoRunnable()])
        extended = base | _UpperRunnable()
        assert len(base.steps) == 1
        assert len(extended.steps) == 2

    async def test_callable_step_wraps_in_chain(self):
        """Plain callables should work as non-first steps."""
        first = RunnableLambda(lambda x: x + " world")
        upper_fn = lambda s: s.upper()  # noqa: E731
        pipeline = first | upper_fn
        result = await pipeline.invoke("hello")
        assert result == "HELLO WORLD"

    async def test_invoke_none_input(self):
        r = RunnableLambda(lambda x: "constant" if x is None else x)
        c = Chain(steps=[r])
        result = await c.invoke(None)
        assert result == "constant"


# ---------------------------------------------------------------------------
# TestChainFactory
# ---------------------------------------------------------------------------


class TestChainFactory:
    def test_chain_factory_creates_chain(self):
        r1 = _EchoRunnable()
        r2 = _UpperRunnable()
        c = chain(r1, r2)
        assert isinstance(c, Chain)
        assert len(c.steps) == 2

    async def test_chain_factory_invokes_correctly(self):
        c = chain(
            RunnableLambda(lambda x: x + 1),
            RunnableLambda(lambda x: x * 2),
        )
        result = await c.invoke(3)
        assert result == 8

    def test_chain_factory_empty(self):
        c = chain()
        assert isinstance(c, Chain)
        assert c.steps == []

    async def test_chain_factory_single_step(self):
        c = chain(RunnableLambda(lambda x: x ** 2))
        result = await c.invoke(5)
        assert result == 25


# ---------------------------------------------------------------------------
# TestPipeOperatorComposition
# ---------------------------------------------------------------------------


class TestPipeOperatorComposition:
    def test_template_pipe_returns_chain(self):
        tpl = PromptTemplate(template="{x}")
        c = tpl | StrOutputParser()
        assert isinstance(c, Chain)

    def test_chat_template_pipe_returns_chain(self):
        tpl = ChatPromptTemplate(messages=[("human", "{q}")])
        c = tpl | StrOutputParser()
        assert isinstance(c, Chain)

    def test_parser_pipe_returns_chain(self):
        p = StrOutputParser()
        c = p | RunnableLambda(lambda x: x)
        assert isinstance(c, Chain)

    def test_multi_pipe_chain_length(self):
        tpl = PromptTemplate(template="{x}")
        r1 = RunnableLambda(lambda m: m.content)
        r2 = RunnableLambda(str.upper)
        c = tpl | r1 | r2
        assert len(c.steps) == 3

    async def test_full_pipeline_prompt_to_parsed(self):
        tpl = PromptTemplate(template="{items}")
        extract_content = RunnableLambda(lambda msg: msg.content)
        split = CommaSeparatedListParser()
        pipeline = tpl | extract_content | split
        result = await pipeline.invoke({"items": "a, b, c"})
        # extract_content returns "a, b, c", split splits on commas
        assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# TestRunnableLambdaEdgeCases
# ---------------------------------------------------------------------------


class TestRunnableLambdaEdgeCases:
    async def test_lambda_returning_none(self):
        r = RunnableLambda(lambda x: None)
        result = await r.invoke("anything")
        assert result is None

    async def test_lambda_with_complex_input(self):
        r = RunnableLambda(lambda d: d["a"] + d["b"])
        result = await r.invoke({"a": 1, "b": 2})
        assert result == 3

    async def test_lambda_async_with_side_effects(self):
        calls = []

        async def record(x):
            calls.append(x)
            return x

        r = RunnableLambda(record)
        await r.invoke("test")
        assert calls == ["test"]

    async def test_lambda_exception_propagates(self):
        r = RunnableLambda(lambda x: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            await r.invoke(0)


# ---------------------------------------------------------------------------
# TestTryGetStructuredLLM
# ---------------------------------------------------------------------------


class TestTryGetStructuredLLM:
    def test_none_for_non_structured(self):
        assert _try_get_structured_llm("not a structured llm") is None

    def test_none_for_none(self):
        assert _try_get_structured_llm(None) is None

    def test_none_for_parser(self):
        assert _try_get_structured_llm(StrOutputParser()) is None

"""Tests targeting previously uncovered code paths to push coverage >= 90%."""

from __future__ import annotations

import pytest

from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


def _compl(content: str = "ok", *, model: str = "mock") -> Completion:
    return Completion(
        id="c1",
        model=model,
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


def _make_llm(responses: list[str]) -> tuple[object, MockTransport]:
    cfg, mock = LLMConfig.for_testing()
    for r in responses:
        mock.queue_response(_compl(r))
    from lauren_ai._module import LLMService

    return LLMService(transport=mock, config=cfg), mock


# ---------------------------------------------------------------------------
# _transport/_structured.py
# ---------------------------------------------------------------------------


class TestStructuredLLM:
    def test_build_schema_fallback_for_non_pydantic(self):
        """Non-Pydantic class → _build_schema returns {}."""
        from lauren_ai._transport._structured import StructuredLLM

        class Plain:
            pass

        llm, _ = _make_llm(["ignored"])
        s = StructuredLLM(llm, Plain)
        assert s._schema == {}

    @pytest.mark.asyncio
    async def test_complete_json_fallback_path(self):
        """No tool_calls in completion → falls back to json.loads(content)."""
        from lauren_ai._transport._structured import StructuredLLM

        try:
            from pydantic import BaseModel

            class Out(BaseModel):
                value: int

        except ImportError:
            pytest.skip("pydantic not available")

        llm, mock = _make_llm(['{"value": 42}'])
        s = StructuredLLM(llm, Out)
        result = await s.complete([])
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_complete_json_fallback_raises_on_bad_json(self):
        """Bad JSON in content raises OutputParserError."""
        from lauren_ai._output_parsers._base import OutputParserError
        from lauren_ai._transport._structured import StructuredLLM

        try:
            from pydantic import BaseModel

            class Out(BaseModel):
                value: int

        except ImportError:
            pytest.skip("pydantic not available")

        llm, mock = _make_llm(["not json"])
        s = StructuredLLM(llm, Out)
        with pytest.raises(OutputParserError):
            await s.complete([])

    def test_or_creates_chain(self):
        """__or__ returns a Chain."""
        from lauren_ai._transport._structured import StructuredLLM

        class Plain:
            pass

        llm, _ = _make_llm([])
        s = StructuredLLM(llm, Plain)

        from lauren_ai._output_parsers._base import OutputParser

        class DummyParser(OutputParser):
            def parse(self, text: str) -> str:
                return text

            @property
            def format_instructions(self) -> str:
                return ""

        chain = s | DummyParser()
        from lauren_ai._chains import Chain

        assert isinstance(chain, Chain)


# ---------------------------------------------------------------------------
# _memory/_remember.py
# ---------------------------------------------------------------------------


class TestRememberModule:
    def test_bare_remember_raises(self):
        from lauren_ai._memory._remember import MemoryConfigError, remember

        with pytest.raises(MemoryConfigError, match="parentheses"):
            remember(object())  # type: ignore[arg-type]

    def test_remember_decorator_attaches_meta(self):
        from lauren_ai._memory._remember import REMEMBER_META, RememberMeta, remember

        @remember(store="my_store", extract=False, inject=True, top_k=3)
        class MyAgent:
            pass

        meta: RememberMeta = getattr(MyAgent, REMEMBER_META)
        assert meta.store_token == "my_store"
        assert meta.extract is False
        assert meta.inject is True
        assert meta.top_k == 3

    def test_build_memory_context_empty(self):
        from lauren_ai._memory._remember import build_memory_context

        assert build_memory_context([]) == ""

    def test_build_memory_context_with_facts(self):
        from dataclasses import dataclass

        from lauren_ai._memory._remember import build_memory_context

        @dataclass
        class Fact:
            content: str
            confidence: float

        facts = [
            Fact("User likes Python", 0.9),
            Fact("User works at Acme", 0.6),
            Fact("User prefers dark mode", 0.3),
        ]
        result = build_memory_context(facts)
        assert "User likes Python" in result
        assert "high" in result
        assert "medium" in result
        assert "low" in result

    @pytest.mark.asyncio
    async def test_extract_facts_returns_list(self):
        """extract_facts calls LLM and parses JSON array."""
        from lauren_ai._memory._remember import extract_facts

        llm, _ = _make_llm(['[{"content": "User likes cats", "topics": ["pets"]}]'])
        facts = await extract_facts("user1", "I really love my cat!", llm)
        assert isinstance(facts, list)
        assert len(facts) == 1
        assert facts[0]["content"] == "User likes cats"

    @pytest.mark.asyncio
    async def test_extract_facts_returns_empty_on_bad_json(self):
        from lauren_ai._memory._remember import extract_facts

        llm, _ = _make_llm(["nothing useful here"])
        facts = await extract_facts("user1", "Hello", llm)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_returns_empty_on_bad_json_array(self):
        from lauren_ai._memory._remember import extract_facts

        llm, _ = _make_llm(["[invalid json"])
        facts = await extract_facts("user1", "Hello", llm)
        assert facts == []


# ---------------------------------------------------------------------------
# _output_parsers/_retry.py
# ---------------------------------------------------------------------------


class TestRetryOutputParser:
    def test_invoke_delegates_to_parse(self):
        """invoke() extracts .content and calls parse()."""

        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._output_parsers._retry import RetryOutputParser

        parser = RetryOutputParser(parser=JSONOutputParser(), llm=None)
        compl = _compl('{"key": "val"}')
        import asyncio

        result = asyncio.run(parser.invoke(compl))
        assert result == {"key": "val"}

    def test_invoke_str_input(self):
        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._output_parsers._retry import RetryOutputParser

        parser = RetryOutputParser(parser=JSONOutputParser(), llm=None)
        import asyncio

        result = asyncio.run(parser.invoke('{"k": 1}'))
        assert result == {"k": 1}

    def test_format_instructions_delegates(self):
        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._output_parsers._retry import RetryOutputParser

        inner = JSONOutputParser()
        parser = RetryOutputParser(parser=inner, llm=None)
        assert parser.format_instructions == inner.format_instructions

    @pytest.mark.asyncio
    async def test_parse_with_retry_succeeds_first_try(self):
        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._output_parsers._retry import RetryOutputParser

        llm, _ = _make_llm([])
        parser = RetryOutputParser(parser=JSONOutputParser(), llm=llm)
        from lauren_ai._transport import Message

        result = await parser.parse_with_retry(
            original_messages=[Message(role="user", content="q")],
            completion=_compl('{"answer": 1}'),
        )
        assert result == {"answer": 1}

    @pytest.mark.asyncio
    async def test_parse_with_retry_retries_and_succeeds(self):
        """First attempt fails (bad JSON), retry with corrected LLM response."""
        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._output_parsers._retry import RetryOutputParser

        llm, _ = _make_llm(['{"fixed": true}'])
        parser = RetryOutputParser(parser=JSONOutputParser(), llm=llm, max_retries=2)
        from lauren_ai._transport import Message

        result = await parser.parse_with_retry(
            original_messages=[Message(role="user", content="q")],
            completion=_compl("bad json {"),
        )
        assert result == {"fixed": True}

    @pytest.mark.asyncio
    async def test_parse_with_retry_exhausts_raises(self):
        from lauren_ai._output_parsers._base import MaxRetryError
        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._output_parsers._retry import RetryOutputParser

        llm, _ = _make_llm(["still bad", "still bad", "still bad"])
        parser = RetryOutputParser(parser=JSONOutputParser(), llm=llm, max_retries=2)
        from lauren_ai._transport import Message

        with pytest.raises(MaxRetryError):
            await parser.parse_with_retry(
                original_messages=[Message(role="user", content="q")],
                completion=_compl("not json"),
            )

    def test_or_creates_chain(self):
        from lauren_ai._output_parsers._builtin import JSONOutputParser
        from lauren_ai._output_parsers._retry import RetryOutputParser

        parser = RetryOutputParser(parser=JSONOutputParser(), llm=None)
        chain = parser | JSONOutputParser()
        from lauren_ai._chains import Chain

        assert isinstance(chain, Chain)


# ---------------------------------------------------------------------------
# _prompts/_templates.py
# ---------------------------------------------------------------------------


class TestPromptTemplateInvoke:
    @pytest.mark.asyncio
    async def test_invoke_dict_returns_message(self):
        from lauren_ai._prompts import PromptTemplate

        tpl = PromptTemplate(template="Hello {name}!", input_variables=["name"])
        result = await tpl.invoke({"name": "World"})
        assert result.content == "Hello World!"

    @pytest.mark.asyncio
    async def test_invoke_non_dict_renders_no_vars(self):
        from lauren_ai._prompts import PromptTemplate

        # No variables needed — non-dict input → format(**{}) succeeds
        tpl = PromptTemplate(template="Static message", input_variables=[])
        result = await tpl.invoke("anything")
        # non-dict path returns str (format(**{}))
        assert "Static message" in str(result)

    def test_or_creates_chain(self):
        from lauren_ai._prompts import PromptTemplate

        tpl = PromptTemplate(template="Hello {x}", input_variables=["x"])
        from lauren_ai._output_parsers._builtin import JSONOutputParser

        chain = tpl | JSONOutputParser()
        from lauren_ai._chains import Chain

        assert isinstance(chain, Chain)


class TestChatPromptTemplateInvoke:
    @pytest.mark.asyncio
    async def test_invoke_dict_returns_messages(self):
        from lauren_ai._prompts import ChatPromptTemplate

        tpl = ChatPromptTemplate(
            messages=[("system", "You speak {lang}."), ("human", "{msg}")],
            input_variables=["lang", "msg"],
        )
        result = await tpl.invoke({"lang": "French", "msg": "Hello"})
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_invoke_non_dict_calls_format_messages_no_args(self):
        from lauren_ai._prompts import ChatPromptTemplate

        tpl = ChatPromptTemplate(messages=[("human", "Static")])
        result = await tpl.invoke("ignored")
        assert isinstance(result, list)
        assert len(result) == 1


class TestFewShotPromptTemplate:
    def test_format_with_kwargs(self):
        from lauren_ai._prompts import FewShotExample, FewShotPromptTemplate

        tpl = FewShotPromptTemplate(
            prefix="Examples:",
            examples=[FewShotExample("cat", "animal")],
            example_template="{input} → {output}",
            suffix="Now: {query}",
            input_variables=["query"],
        )
        result = tpl.format(query="dog")
        assert "cat → animal" in result
        assert "dog" in result

    def test_format_missing_var_raises(self):
        from lauren_ai._prompts import FewShotPromptTemplate, PromptRenderError

        tpl = FewShotPromptTemplate(
            prefix="",
            examples=[],
            example_template="{input} → {output}",
            suffix="{missing}",
            input_variables=["missing"],
        )
        with pytest.raises(PromptRenderError):
            tpl.format()

    @pytest.mark.asyncio
    async def test_invoke_dict_calls_render(self):
        from lauren_ai._prompts import FewShotExample, FewShotPromptTemplate

        tpl = FewShotPromptTemplate(
            prefix="",
            examples=[FewShotExample("a", "b")],
            example_template="{input}-{output}",
            suffix="q: {q}",
            input_variables=["q"],
        )
        result = await tpl.invoke({"q": "test"})
        assert "test" in result.content

    @pytest.mark.asyncio
    async def test_invoke_non_dict_calls_render_no_kwargs(self):
        from lauren_ai._prompts import FewShotPromptTemplate

        tpl = FewShotPromptTemplate(
            prefix="Prefix",
            examples=[],
            example_template="{input}-{output}",
            suffix="Static",
        )
        result = await tpl.invoke("whatever")
        assert "Prefix" in result.content or "Static" in result.content

    def test_or_creates_chain(self):
        from lauren_ai._prompts import FewShotPromptTemplate

        tpl = FewShotPromptTemplate(
            prefix="",
            examples=[],
            example_template="{input}",
            suffix="{x}",
        )
        from lauren_ai._output_parsers._builtin import JSONOutputParser

        chain = tpl | JSONOutputParser()
        from lauren_ai._chains import Chain

        assert isinstance(chain, Chain)


# ---------------------------------------------------------------------------
# _guardrails/_builtin.py — embedding check path
# ---------------------------------------------------------------------------


class TestTopicFilterEmbeddingCheck:
    @pytest.mark.asyncio
    async def test_embedding_check_pass_high_similarity(self):
        """embed_fn returns high similarity → pass."""
        from lauren_ai._guardrails._base import GuardrailContext
        from lauren_ai._guardrails._builtin import TopicFilter

        class FakeEmb:
            def __init__(self, v):
                self.vector = v

        async def embed_fn(texts):
            # Query = [1, 0, 0], topic "cooking" = [1, 0, 0] → cos = 1.0
            return [FakeEmb([1.0, 0.0, 0.0]) for _ in texts]

        guard = TopicFilter(
            allowed_topics=["cooking"],
            embed_fn=embed_fn,
            min_similarity=0.5,
        )
        decision = await guard.check("recipe", GuardrailContext(agent_name="A"))
        assert decision.action == "pass"

    @pytest.mark.asyncio
    async def test_embedding_check_block_low_similarity(self):
        from lauren_ai._guardrails._base import GuardrailContext
        from lauren_ai._guardrails._builtin import TopicFilter

        class FakeEmb:
            def __init__(self, v):
                self.vector = v

        async def embed_fn(texts):
            # All zero vectors → similarity undefined → 0.0
            return [FakeEmb([0.0, 0.0, 0.0]) for _ in texts]

        guard = TopicFilter(
            allowed_topics=["cooking"],
            embed_fn=embed_fn,
            min_similarity=0.5,
        )
        decision = await guard.check("unrelated", GuardrailContext(agent_name="A"))
        assert decision.action == "block"


# ---------------------------------------------------------------------------
# _routing/_router.py — LLM fallback and dispatch
# ---------------------------------------------------------------------------


class TestSemanticRouterLLMFallback:
    @pytest.mark.asyncio
    async def test_llm_fallback_used_when_confidence_low(self):
        from lauren_ai._routing import Route, SemanticRouter

        class FakeEmb:
            def __init__(self, v):
                self.vector = v

        async def embed_fn(texts):
            # All same vector → low discrimination, but similarity still 1.0 per route
            return [FakeEmb([0.1, 0.0, 0.0]) for _ in texts]

        llm, mock = _make_llm(["weather"])
        route = Route("weather", "Weather questions", examples=["What is rain?"])
        router = SemanticRouter(routes=[route], embed_fn=embed_fn, min_confidence=0.99, llm=llm)
        await router.compile()

        match = await router.route("What is the forecast?")
        assert match.route_name == "weather"

    @pytest.mark.asyncio
    async def test_llm_route_stream_fallback(self):
        """_llm_route handles streaming response."""
        from lauren_ai._routing import Route, SemanticRouter

        class FakeEmb:
            def __init__(self, v):
                self.vector = v

        async def embed_fn(texts):
            return [FakeEmb([0.0, 0.0, 0.0]) for _ in texts]

        # Queue a streaming response
        from lauren_ai._transport import CompletionChunk

        async def mock_complete(self_or_msgs, messages_or_kw=None, **kwargs):
            async def stream():
                yield CompletionChunk(delta="weather", stop_reason=None, usage=None)

            return stream()

        route = Route("weather", "Weather route", examples=["rain"])
        router = SemanticRouter(routes=[route], embed_fn=embed_fn, min_confidence=0.99)
        router._llm = type("FakeLLM", (), {"complete": mock_complete})()
        await router.compile()

        name = await router._llm_route("What is rain?")
        assert name == "weather"

    @pytest.mark.asyncio
    async def test_dispatch_calls_async_handler(self):
        from lauren_ai._routing import Route, SemanticRouter

        class FakeEmb:
            def __init__(self, v):
                self.vector = v

        async def embed_fn(texts):
            return [FakeEmb([1.0, 0.0] if "cook" in t else [0.0, 1.0]) for t in texts]

        route = Route("cooking", "cooking", examples=["recipe"])
        router = SemanticRouter(routes=[route], embed_fn=embed_fn)
        await router.compile()

        called_with = []

        async def handler(q):
            called_with.append(q)
            return "cooked"

        result = await router.dispatch("cook something", {"cooking": handler})
        assert result == "cooked"
        assert called_with == ["cook something"]

    @pytest.mark.asyncio
    async def test_dispatch_calls_sync_handler(self):
        from lauren_ai._routing import Route, SemanticRouter

        class FakeEmb:
            def __init__(self, v):
                self.vector = v

        async def embed_fn(texts):
            return [FakeEmb([1.0, 0.0]) for _ in texts]

        route = Route("info", "info questions", examples=["what"])
        router = SemanticRouter(routes=[route], embed_fn=embed_fn)
        await router.compile()

        def sync_handler(q):
            return f"sync:{q}"

        result = await router.dispatch("what is X", {"info": sync_handler})
        assert result.startswith("sync:")

    @pytest.mark.asyncio
    async def test_dispatch_raises_no_handler(self):
        from lauren_ai._routing import Route, RouterConfigError, SemanticRouter

        class FakeEmb:
            def __init__(self, v):
                self.vector = v

        async def embed_fn(texts):
            return [FakeEmb([1.0]) for _ in texts]

        route = Route("weather", "weather", examples=["rain"])
        router = SemanticRouter(routes=[route], embed_fn=embed_fn)
        await router.compile()

        with pytest.raises(RouterConfigError, match="No handler"):
            await router.dispatch("rain", {})

    def test_add_route_marks_uncompiled(self):
        from lauren_ai._routing import Route, SemanticRouter

        async def embed_fn(texts):
            return []

        route1 = Route("a", "first", examples=["x"])
        router = SemanticRouter(routes=[route1], embed_fn=embed_fn)
        router._compiled = True

        new_route = Route("b", "second", examples=["y"])
        router.add_route(new_route)
        assert not router._compiled
        assert len(router.routes) == 2


# ---------------------------------------------------------------------------
# _teams/_runner.py — streaming paths and worker streaming
# ---------------------------------------------------------------------------


class TestTeamRunnerStreaming:
    def _make_team(self, mode: str = "collaborate"):
        from lauren_ai._teams._decorator import team
        from lauren_ai._teams._runner import TeamRunner

        @team(mode=mode, model="mock")
        class MyTeam:
            def __init__(self, analyst: str, writer: str) -> None:
                pass

        llm, mock = _make_llm(["analysis output", "writing output", "synthesis"])
        runner = TeamRunner(team_cls=MyTeam, llm=llm, agent_runner=None)
        return runner, mock

    @pytest.mark.asyncio
    async def test_stream_collaborate_yields_events(self):

        runner, _ = self._make_team("collaborate")
        events = []
        async for event in runner.run_stream("do the task"):
            events.append(event)

        types = [type(e).__name__ for e in events]
        assert "TeamWorkerStarted" in types
        assert "TeamWorkerFinished" in types
        assert "TeamFinalAnswer" in types

    @pytest.mark.asyncio
    async def test_stream_coordinator_yields_events(self):

        from lauren_ai._teams._decorator import team
        from lauren_ai._teams._runner import TeamRunner

        @team(mode="coordinator", model="mock", max_rounds=2)
        class CoordTeam:
            def __init__(self, expert: str) -> None:
                pass

        # Coordinator returns DONE immediately
        llm, _ = _make_llm(["DONE: final answer"])
        runner = TeamRunner(team_cls=CoordTeam, llm=llm, agent_runner=None)
        events = []
        async for event in runner.run_stream("solve it"):
            events.append(event)

        types = [type(e).__name__ for e in events]
        assert "TeamCoordinatorDecision" in types
        assert "TeamFinalAnswer" in types

    @pytest.mark.asyncio
    async def test_discover_workers_attribute_error(self):
        """Class without __init__ annotations returns empty list."""
        from lauren_ai._teams._decorator import team
        from lauren_ai._teams._runner import TeamRunner

        @team(mode="collaborate", model="mock")
        class EmptyTeam:
            pass

        llm, _ = _make_llm([])
        runner = TeamRunner(team_cls=EmptyTeam, llm=llm, agent_runner=None)
        assert runner._worker_names == []

    @pytest.mark.asyncio
    async def test_call_worker_streaming(self):
        """_call_worker handles streaming LLM response."""
        from lauren_ai._teams._decorator import team
        from lauren_ai._teams._runner import TeamRunner
        from lauren_ai._transport import CompletionChunk

        @team(mode="collaborate", model="mock")
        class T:
            def __init__(self, w: str) -> None:
                pass

        from lauren_ai._module import LLMService

        class FakeTransport:
            async def complete(self, messages, **kw):
                async def stream():
                    yield CompletionChunk(delta="chunk", stop_reason="end_turn", usage=None)

                return stream()

        cfg = LLMConfig(provider="anthropic", model="mock", api_key="k")
        llm = LLMService(transport=FakeTransport(), config=cfg)
        runner = TeamRunner(team_cls=T, llm=llm, agent_runner=None)
        result = await runner._call_worker("w", "task")
        assert result == "chunk"

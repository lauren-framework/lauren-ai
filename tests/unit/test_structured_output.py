"""Unit tests for StructuredLLM and with_structured_output."""

from __future__ import annotations

from pydantic import BaseModel

from lauren_ai._config import LLMConfig
from lauren_ai._module import LLMService
from lauren_ai._transport import Message
from lauren_ai._transport._mock import MockTransport
from lauren_ai._transport._structured import StructuredLLM


class SentimentResult(BaseModel):
    sentiment: str
    confidence: float


class UserInfo(BaseModel):
    name: str
    age: int


class TestStructuredLLM:
    def _make_llm(self) -> tuple[LLMService, MockTransport]:
        transport = MockTransport()
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=transport, config=config)
        return llm, transport

    async def test_with_structured_output_returns_model(self):
        llm, transport = self._make_llm()
        transport.queue_structured(SentimentResult(sentiment="positive", confidence=0.9))
        structured = llm.with_structured_output(SentimentResult)
        result = await structured.complete([Message(role="user", content="Great!")])  # type: ignore[arg-type]
        assert isinstance(result, SentimentResult)
        assert result.sentiment == "positive"
        assert result.confidence == 0.9

    async def test_structured_llm_is_generic(self):
        llm, transport = self._make_llm()
        transport.queue_structured(UserInfo(name="Alice", age=30))
        structured: StructuredLLM[UserInfo] = llm.with_structured_output(UserInfo)
        result = await structured.complete([Message(role="user", content="Alice")])  # type: ignore[arg-type]
        assert result.name == "Alice"
        assert result.age == 30

    def test_with_structured_output_returns_structured_llm(self):
        llm, _ = self._make_llm()
        structured = llm.with_structured_output(SentimentResult)
        assert isinstance(structured, StructuredLLM)

    def test_schema_built_from_model(self):
        llm, _ = self._make_llm()
        structured = llm.with_structured_output(SentimentResult)
        assert "sentiment" in str(structured._schema)

    async def test_pipe_with_chain(self):
        from lauren_ai._prompts import PromptTemplate

        llm, transport = self._make_llm()
        transport.queue_structured(SentimentResult(sentiment="neutral", confidence=0.5))
        structured = llm.with_structured_output(SentimentResult)
        from lauren_ai._chains import Chain

        chain = PromptTemplate(template="{text}") | structured
        assert isinstance(chain, Chain)

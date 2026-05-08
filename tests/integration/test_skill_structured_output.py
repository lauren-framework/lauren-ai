"""Integration tests for Skill 4: Structured Output Parsing.

Tests cover:
- Manual JSON parse from agent completion content
- PydanticOutputParser parsing valid JSON
- PydanticOutputParser raises on invalid JSON
- StructuredLLM with MockTransport (queue_structured)
- parse_structured_response helper function pattern
"""

import json

import pytest
from pydantic import BaseModel

from lauren_ai import LLMConfig, PydanticOutputParser
from lauren_ai._agents import agent
from lauren_ai._module import LLMService
from lauren_ai._transport import Completion, Message, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Helper functions (pure Python — no HTTP needed)
# ---------------------------------------------------------------------------


def parse_structured_response(content: str, schema: type[BaseModel]) -> BaseModel:
    """Parse LLM text content into a Pydantic model."""
    try:
        data = json.loads(content.strip())
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Agent output is not valid {schema.__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Pydantic models under test
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    label: str
    score: float
    reasoning: str


class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(content: str = "OK") -> Completion:
    return Completion(
        id="c1",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="Return JSON only.")
class JSONAgent: ...


@agent(model="mock-model")
class ProductAgent: ...


# ---------------------------------------------------------------------------
# TestManualJSONParse
# ---------------------------------------------------------------------------


class TestManualJSONParse:
    def test_parse_valid_json_to_sentiment_result(self):
        content = '{"label": "positive", "score": 0.9, "reasoning": "Great product!"}'
        result = parse_structured_response(content, SentimentResult)
        assert isinstance(result, SentimentResult)

    def test_parse_sentiment_label(self):
        content = '{"label": "positive", "score": 0.9, "reasoning": "Great product!"}'
        result = parse_structured_response(content, SentimentResult)
        assert result.label == "positive"

    def test_parse_sentiment_score(self):
        content = '{"label": "positive", "score": 0.9, "reasoning": "Great product!"}'
        result = parse_structured_response(content, SentimentResult)
        assert result.score == pytest.approx(0.9)

    def test_parse_product_info(self):
        content = '{"name": "Widget", "price": 9.99, "in_stock": true}'
        result = parse_structured_response(content, ProductInfo)
        assert result.name == "Widget"
        assert result.price == pytest.approx(9.99)
        assert result.in_stock is True

    def test_parse_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="SentimentResult"):
            parse_structured_response("not json at all", SentimentResult)

    def test_parse_missing_field_raises_value_error(self):
        content = '{"label": "positive"}'  # missing score and reasoning
        with pytest.raises(ValueError):
            parse_structured_response(content, SentimentResult)


# ---------------------------------------------------------------------------
# TestPydanticOutputParser
# ---------------------------------------------------------------------------


class TestPydanticOutputParser:
    def test_parser_parses_valid_json(self):
        parser = PydanticOutputParser(model=SentimentResult)
        content = '{"label": "neutral", "score": 0.5, "reasoning": "Average"}'
        result = parser.parse(content)
        assert isinstance(result, SentimentResult)

    def test_parser_label_field(self):
        parser = PydanticOutputParser(model=SentimentResult)
        result = parser.parse('{"label": "neutral", "score": 0.5, "reasoning": "Average"}')
        assert result.label == "neutral"

    def test_parser_product_name(self):
        parser = PydanticOutputParser(model=ProductInfo)
        result = parser.parse('{"name": "Gadget", "price": 19.99, "in_stock": false}')
        assert result.name == "Gadget"

    def test_parser_raises_on_invalid_json(self):
        from lauren_ai import OutputParserError

        parser = PydanticOutputParser(model=SentimentResult)
        with pytest.raises((OutputParserError, ValueError, Exception)):
            parser.parse("this is not json")


# ---------------------------------------------------------------------------
# TestStructuredLLMWithMock
# ---------------------------------------------------------------------------


def _make_llm_service() -> tuple[LLMService, MockTransport]:
    """Build a test LLMService backed by MockTransport."""
    mock = MockTransport()
    cfg, _ = LLMConfig.for_testing()
    llm = LLMService(transport=mock, config=cfg)
    return llm, mock


class TestStructuredLLMWithMock:
    async def test_structured_llm_returns_pydantic_model(self):
        llm, mock = _make_llm_service()
        mock.queue_structured(SentimentResult(label="positive", score=0.9, reasoning="Great!"))

        structured = llm.with_structured_output(SentimentResult)
        result = await structured.complete([Message(role="user", content="Analyze")])  # type: ignore[arg-type]
        assert isinstance(result, SentimentResult)

    async def test_structured_llm_label_matches(self):
        llm, mock = _make_llm_service()
        mock.queue_structured(SentimentResult(label="negative", score=0.1, reasoning="Bad"))

        structured = llm.with_structured_output(SentimentResult)
        result = await structured.complete([Message(role="user", content="Analyze")])  # type: ignore[arg-type]
        assert result.label == "negative"

    async def test_structured_llm_score_matches(self):
        llm, mock = _make_llm_service()
        mock.queue_structured(SentimentResult(label="positive", score=0.95, reasoning="Excellent"))

        structured = llm.with_structured_output(SentimentResult)
        result = await structured.complete([Message(role="user", content="Analyze")])  # type: ignore[arg-type]
        assert result.score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# TestAgentJSONCompletion
# ---------------------------------------------------------------------------


class TestAgentJSONCompletion:
    def test_agent_json_content_can_be_parsed(self):
        json_content = '{"label": "positive", "score": 0.8, "reasoning": "Sounds happy"}'
        client = TestClient(JSONAgent())
        client.mock.queue_response(_c(json_content))
        result = client.run("Analyze: I love this!")
        parsed = parse_structured_response(result.content, SentimentResult)
        assert parsed.label == "positive"

    def test_agent_product_json_parses_correctly(self):
        client = TestClient(ProductAgent())
        client.mock.queue_response(_c('{"name": "Laptop", "price": 999.0, "in_stock": true}'))
        result = client.run("Describe product")
        parsed = parse_structured_response(result.content, ProductInfo)
        assert parsed.name == "Laptop"
        assert parsed.in_stock is True
